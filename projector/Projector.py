# coding=utf-8
# Copyright 2024 Vital Star Foundation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from os import environ as env
from dotenv import load_dotenv

import pandas as pd
import zipfile
import tempfile
import json
import pickle
import torch
from smallscript import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets
from datetime import datetime
from IPython.display import display
from pathlib import Path
from .operator.Projection import Projection

class Projector(SObject):
    model = Holder().name('model')
    projection = Holder().name('projection')
    console = Holder().name('console')
    figure = Holder().name('figure')
    colorRoll = Holder().name('colorRoll').type('ColorRoll')
    traces = Holder().name('traces').type('Map')
    highlight = Holder().name('highlight')

    def __init__(self):
        self.console(ipywidgets.Output())
        self.figure(go.FigureWidget())

    def filename(self, filename=""):
        if filename == "":
            res = self._getOrSet('filename', filename, nil)
            if res.notNil():
                return res
            defaultname = "unnamed.zip"
            if self.name().notEmpty():
                defaultname = f"{self.name()}.zip"
            filename = defaultname
        self.setValue('filename', filename)
        return filename

    #### File I/O
    def hasFile(self):
        filepath = Path(self.filename())
        return self.asSObj(filepath.exists())

    def saveFile(self):
        with zipfile.ZipFile(self.filename(), "w") as zip:
            # with tempfile.NamedTemporaryFile(
            #         mode="wt", encoding='utf-8', delete=True) as f:
            #     self.log(f'dumping projectionDF to {f.name} ...', Logger.LevelInfo)
            #     json.dump(self.embeddingDF().to_dict(orient='records'), f)
            #     f.flush()
            #     zip.write(f.name, "projection.df", compress_type= zipfile.ZIP_DEFLATED)
            with tempfile.NamedTemporaryFile(mode="wb", delete=True) as f:
                pickle.dump(self.project(), f)
                f.flush()
                zip.write(f.name, "projection.pkl", compress_type= zipfile.ZIP_DEFLATED)
        return self

    def loadFile(self):
        with zipfile.ZipFile(self.filename(), "r") as zip:
            tmpdir = tempfile.TemporaryDirectory()
            self.log(f'tempdir {tmpdir.name}', Logger.LevelInfo)
            # zip.extract("projection.df", tmpdir.name)
            # with open(f"{tmpdir.name}/projection.df", "r", encoding='utf-8') as f:
            #     data = json.load(f)
            #     self.projectionDF(pd.DataFrame(data))
            zip.extract("projection.pkl", tmpdir.name)
            with open(f"{tmpdir.name}/projection.pkl", "rb") as f:
                projection = pickle.load(f)
                self.projection(projection)
        return self

    def deleteFile(self):
        filepath = Path(self.filename())
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
        return self

    #### Projection matrix operations
    def project(self, dim=3):
        projection = self.projection()
        if projection is nil:
            wte = self.model().modelParams().getValue('wte')
            projection = self.model().project(wte, dim)
            self.projection(projection)
            dataframe = pd.DataFrame(projection.projected(), columns=['x', 'y', 'z'])
            words = self.model().words('_')
            dataframe['word'] = list(words)
            projection.df(dataframe)
        return projection

    def projectVector(self, vectors):
        projection = self.project()
        model = self.model()
        projected = model.projectVector(projection, vectors)
        return projected

    def searchTokens(self, spelling, n = 5):
        model = self.model()
        res = model.searchTokens(spelling, n)
        return res

    def encode(self, text):
        tokenizer = self.model().tokenizer()
        codes = tokenizer.encode(text)
        return codes

    def _vectors(self, vectors):
        if isinstance(vectors, list):
            if isinstance(vectors[0], list):  # List of vectors
                vectors = torch.tensor(vectors)
            else:  # Single vector in list
                vectors = torch.tensor(vectors).unsqueeze(0)
        elif isinstance(vectors, torch.Tensor) and vectors.dim() == 1:  # Single vector
            vectors = vectors.unsqueeze(0)
        return vectors

    def closestSimilarities(self, vectors):
        vectors = self._vectors(vectors)
        wte = self.model().modelParams()['wte']
        vnorms = vectors.norm(dim=1, keepdim=True)
        wnorms = wte.norm(dim=1, keepdim=True)
        if vnorms.shape == (1,1) and vnorms == 0:      # zero vector
            return torch.tensor([[0]])
        vectors = vectors / torch.where(vnorms == 0, torch.ones_like(vnorms), vnorms)
        wte = wte / torch.where(wnorms == 0, torch.ones_like(wnorms), wnorms)
        sim = torch.mm(vectors, wte.T)
        return sim

    def closestAngles(self, vectors, k=1):
        sim = self.closestSimilarities(vectors)
        sim = torch.clamp(sim, -1, 1)
        maxSim, top_k_indices = torch.topk(sim, k, dim=1)
        rad = torch.arccos(maxSim)
        deg = rad * (180/torch.pi)
        degrees = torch.round(deg * 10000) / 10000
        return degrees

    def closestIndices(self, vectors, k=1):
        wte = self.model().modelParams()['wte']
        sim = self.closestSimilarities(vectors)
        if sim.shape == (1,1) and sim == 0:      # zero vector
            wnorms = wte.norm(dim=1, keepdim=True)
            nearZero = torch.argmin(wnorms)
            return torch.tensor([[nearZero]])
        sim = torch.mm(vectors, wte.T)
        # distances = torch.cdist(vectors, wte, p=2)  # L2 norm
        # _, top_k_indices = torch.topk(-distances, k, dim=1)
        _, top_k_indices = torch.topk(sim, k, dim=1)
        return top_k_indices

    def closestWords(self, vectors):
        indices = self.closestIndices(vectors)
        indices = indices[:, 0]     # getting the first column, closest words
        selected = self.projection().select(indices.tolist())
        words = selected['word'].tolist()
        return words
        # words = self.model().words()
        # if indices.shape == (1,):
        #     closest = [words[indices.item()]]
        # else:
        #     closest = [words[index] for index in indices]
        # return closest


    #### Plotly Functions
    def updateHighlight(self, tokenVector, k):
        highlight = self.highlight()
        if highlight.notNil():
            highlight.remove()
        tokenTrace = self.newTrace().fromVectors(tokenVector)
        knn_ids = tokenTrace.knn(k).knn_ids()
        highlight = self.newTrace().fromIndices(knn_ids)
        highlight.colorRoll().color('blue')
        self.highlight(highlight)
        return highlight

    def onClick(self, plytrace, points, selector):
        if len(points.point_inds) == 0:
            return
        self.console().outputs = ()
        idx = points.point_inds[0]
        with self.console():
            k = 10
            print(f"{datetime.now()} {plytrace.name}[{idx}]")
            trace = self.traces().getValue(plytrace.name)
            tokenVec = trace.vectors()[idx]
            highlight = self.updateHighlight(tokenVec, k)
            df = highlight.asDF()
            print(df)
            highlight.show()

            # # knn = trace.closestIndices(k)[idx,:].tolist()
            # knn = trace.closestIndices(k)[idx,:]
            # highlight = self.highlight()
            # if highlight.notNil():
            #     highlight.remove()
            # highlight = trace.projector().newTrace().fromIndices(knn)
            # highlight.colorRoll().color('blue')
            # self.highlight(highlight)
            # df = highlight.df()
            # angles = highlight.closestAngles(k)
            # df['angle'] = angles[idx,:]
            # sims = highlight.knn(k)
            # df['sim'] = sims[idx,:]
            # print(df)
            # highlight.show()

    def showEmbedding(self):
        def on_click2(trace, points, selector):
            with self.console():
                print(datetime.now())

        if len(self.figure().data) == 0:
            figwidget = self.figure()
            df = self.projection().df()
            fig = px.scatter_3d(
                            df, x='x', y='y', z='z',
                            hover_data='word')
                # need to use px, instead of go for performance
            fig.data[0].marker.opacity=0.1
            fig.data[0].marker.size=2
            fig.data[0].marker.color='grey'
            figwidget.add_trace(fig.data[0])
            figwidget.update_layout(
                height=800, width=1000,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),  # Increase margins
                xaxis=dict(
                    showline=True,
                    linecolor='black',  # Or any desired color
                    linewidth=1.5       # Adjust thickness
                ),
                yaxis=dict(
                    showline=True,
                    linecolor='black',
                    linewidth=1.5
                )
            )
        return self.figure()

    def showColorband(self):
        def rgb_to_hex(rgb_string):
            rgb_string = rgb_string.strip()
            match = re.search(r"\((.*?)\)", rgb_string)
            if not match:
                return "#000000"
            rgb_values = match.group(1)
            try:
                r, g, b = map(int, rgb_values.split(','))
            except ValueError:
                return "#000000"
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                return "#000000"
            hex_code = "#" + format(r, '02x') + format(g, '02x') + format(b, '02x')
            return hex_code.upper()

        colors = px.colors.convert_colors_to_same_type(px.colors.qualitative.Alphabet_r, colortype='rgb')[0]
        data = [list(range(0, len(colors)))]
        text = [[str(i) for i in range(0, len(colors))]]
        rgb = [[rgb_to_hex(i) for i in px.colors.qualitative.Alphabet_r]]
        fig = go.Figure(data=go.Heatmap(
            z=data,
            text=text,
            hovertext=rgb,
            texttemplate="%{text}",
            hovertemplate='%{hovertext}',
            colorscale=colors,
            opacity=0.9,
            showscale=False
        ))
        fig.update_layout(
            height=30, width=800,
            margin=dict(l=0, r=0, t=0, b=0),
            modebar={'remove': ['zoom', 'zoomin', 'zoomout', 'pan', 'reset', 'autoscale', 'resetscale', 'toImage']},
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange='reversed')
        )
        return fig

    def showConsole(self):
        display(self.console())
        return self

    def printConsole(self):
        print(self.console().outputs[0]['text'])
        return self

    def writeConsole(self, text):
        self.console().append_stdout(text)
        return self

    def cleanAll(self):
        self.figure().data = ()
        return self

    def clearTraces(self):
        for trace in self.traces().values():
            self.removeTrace(trace)
        return self

    def newTrace(self):
        inference = self.model().inference()
        projection = self.projection()
        trace = Trace().projector(self).\
                    projection(projection).\
                    inference(inference).\
                    colorRoll(self.colorRoll())
                    # colorRoll(ColorRoll().reset(self.colorRoll().coloridx()))
        return trace

    def showTrace(self, trace):
        self.removeTrace(trace)     # try to remove trace first.
        self.traces().setValue(trace.name(), trace)
        scatter = trace.scatter()
        self.figure().add_trace(scatter)
        # self.figure().data[-1].on_click(self.onClick)
        addedScatter = List(self.figure().select_traces(selector={'name': trace.name()}))[-1]
        addedScatter.on_click(self.onClick)
        return self

    def removeTrace(self, trace):
        traces = self.traces()
        if trace.name() in traces:
            traces.delValue(trace.name())
            id = trace.name()
            traces = [t for t in self.figure().data if t.name != id]
            self.figure().data = tuple(traces)
        return self

class ColorRoll(SObject):
    color = Holder().name('color')
    colors = Holder().name('colors')
    coloridx= Holder().name('coloridx')

    def __init__(self):
        self.colors(List(px.colors.qualitative.Alphabet_r))
        self.reset()

    def next(self):
        color = self.color()
        colors = self.colors()
        coloridx = self.coloridx() % colors.len()
        next = colors[coloridx]
        self.coloridx(coloridx + 1)
        self.color(next)
        return color

    def reset(self, coloridx=0):
        self.coloridx(coloridx)
        color = self.colors()[coloridx]
        self.color(color)
        return self

class Trace(SObject):
    projector = Holder().name('projector')
    inference = Holder().name('inference')
    projection = Holder().name('projection')
    prompt = Holder().name('prompt')
    vectors = Holder().name('vectors')          # full scale vectors
    projected = Holder().name('projected')      # projected vectors
    colorRoll = Holder().name('colorRoll')
    knn_sims = Holder().name('knn_sims')                # knn similarity to WTE
    knn_ids = Holder().name('knn_ids')                  # knn Ids to tokens

    def __init__(self):
        id = self.idDigits(8)
        self.name(id)

    def fromIndices(self, indices):
        projector = self.projector()
        projection = self.projection()
        wte = projector.model().modelParams()['wte']
        vectors = List()
        indices1 = projector._vectors(indices)
        indices2 = indices1.squeeze(0)
        for id in indices2:
            vectors.append(wte[id])
        vectors = torch.stack(vectors, dim=0)

        projected = projection.projected()[indices2]
        self.vectors(vectors)
        self.projected(projected)
        return self

    def fromPrompt(self, prompt):       # @vectors should bewte vectors
        self.prompt(prompt)
        inference = self.inference()
        inference.prompt(prompt)
        ids = inference.inputs()
        ids = indices = torch.tensor(ids)
        inference.wte()
        return self.fromIndices(ids)

    def fromVectors2(self, vectors):
        projector = self.projector()
        projection = self.projection()
        if vectors is 0:
            # n_embd = projector.model().modelParams().getAsNumber('n_embd')
            n_embd = projector.model().modelParams().getValue('n_embd')
            vectors = torch.zeros(n_embd)
        vectors = projector._vectors(vectors)
        projected = projection.project(vectors)
        self.projected(projected)

        # indices = projector.closestIndices(vectors)[:,0].tolist()
        indices = projector.closestIndices(vectors)[:,0]
        return self.fromIndices(indices)

    ########
    def fromVectors(self, vectors):
        projector = self.projector()
        projection = self.projection()
        if vectors is 0:
            # n_embd = projector.model().modelParams().getAsNumber('n_embd')
            n_embd = projector.model().modelParams().getValue('n_embd')
            vectors = torch.zeros(n_embd)
        vectors = projector._vectors(vectors)
        projected = projection.project(vectors)
        self.vectors(vectors)
        self.projected(projected)

        # indices = self.knn(1).ids()
        # indices = indices[:,0].tolist()
        # # indices = projector.closestIndices(vectors)[:,0].tolist()
        # # indices = torch.tensor(indices)
        # # projected = projection.projected()[indices]
        # df = projection.select(indices)                     # df is a copy
        # df['x'] = projected[:,0]
        # df['y'] = projected[:,1]
        # df['z'] = projected[:,2]
        # text = [String(i) for i in range(vectors.size(0))]
        # df['text'] = text
        # angles = self.closestAngles()
        # df['angle'] = angles[:,0]
        # df['sim'] = self.sim()[:,0]
        # self.vectors(vectors).\
        #         projected(projected).\
        #         df(df)

        return self

    def asDF(self):
        projection = self.projection()
        projected = self.projected()
        indices = self.knn(1).knn_ids()
        indices = indices[:,0].tolist()
        # indices = projector.closestIndices(vectors)[:,0].tolist()
        # indices = torch.tensor(indices)
        # projected = projection.projected()[indices]
        df = projection.select(indices)                     # df is a copy
        df['x'] = projected[:,0]
        df['y'] = projected[:,1]
        df['z'] = projected[:,2]
        text = [String(i) for i in range(projected.size(0))]
        df['text'] = text
        angles = self.closestAngles()
        df['angle'] = angles[:,0]
        df['sims'] = self.knn_sims()[:, 0]
        df['norm'] = self.vectors().norm(dim=1)
        return df

    def knn(self, k=1):
        projector = self.projector()
        # vectors = projector._vectors(self.vectors())
        vectors = self.vectors()
        wte = projector.model().modelParams()['wte']
        vnorms = vectors.norm(dim=1, keepdim=True)
        wnorms = wte.norm(dim=1, keepdim=True)
        if vnorms.shape == (1,1) and vnorms == 0:      # zero vector
            norms, indices = torch.topk(wnorms.T, k, largest=False)
            vecs = wte[indices[0,:]] / wnorms[indices[0,:]]
            average = vecs.mean(dim=0, keepdim=True)
            naverage = average / average.norm()
            maxSim = torch.mm(vecs, naverage.T)
            # indices = torch.tensor([[nearZero]])
        else:
            vectors = vectors / torch.where(vnorms == 0, torch.ones_like(vnorms), vnorms)
            wte = wte / torch.where(wnorms == 0, torch.ones_like(wnorms), wnorms)
            sim = torch.mm(vectors, wte.T)
            maxSim, indices = torch.topk(sim, k, dim=1)
        self.knn_sims(maxSim)
        self.knn_ids(indices)
        return self
        # projector = self.projector()
        # vectors = self.vectors()
        # sims = projector.closestSimilarities(vectors)
        # maxSim, top_k_indices = torch.topk(sims, k, dim=1)
        # return maxSim

    def closestAngles(self, k=1):
        # projector = self.projector()
        # vectors = self.vectors()
        # angles = projector.closestAngles(vectors, k)
        sims = self.knn_sims()
        if sims is nil:
            sims = self.knn(k).knn_sims()
        sims = torch.clamp(sims, -1, 1)
        rad = torch.arccos(sims)
        deg = rad * (180/torch.pi)
        degrees = torch.round(deg * 10000) / 10000
        return degrees

    def closestIndices(self, k=1):
        # projector = self.projector()
        # vectors = self.vectors()
        # indices = projector.closestIndices(vectors, k)
        indices = self.knn_ids()
        if indices is nil:
            indices = self.knn(k).knn_ids()
        return indices

    def closestWords(self, dim=0):
        # projector = self.projector()
        # vectors = self.vectors()
        # words = projector.closestWords(vectors)
        indices = self.closestIndices()
        if dim == 0:
            indices = indices[:, 0]
        else:
            indices = indices[0, :]
        # indices = indices[:, 0]     # getting the first column, closest words
        selected = self.projection().select(indices.tolist())
        words = selected['word'].tolist()
        return words

    def scatter(self):
        trace_name = self.name()
        df = self.asDF()
        n = df.shape[0]
        colorIdx = self.colorRoll().coloridx()
        color = self.colorRoll().next()
        template = "%{hovertext}<br>trace='" + trace_name + "'<br>x=%{x}<br>y=%{y}<br>z=%{z}<extra>%{text}</extra>"
        template = '%{hovertext}<br>idx=%{text}<br>colorIdx=' + str(colorIdx) + '<br>trace=' + trace_name + '<extra></extra>'
        scatter = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'],
                               name=f"{trace_name}",
                               mode='markers + text',
                               text=df['text'],
                               hovertext=df['word'],
                               hovertemplate=template)
        scatter.marker.color = [color] * n
        scatter.marker.size = [15] * n
        scatter.marker.opacity = 0.8
        return scatter

    def color(self, color):
        self.colorRoll().color(color)
        return self

    def show(self):
        self.projector().showTrace(self)
        return self

    def remove(self):
        self.projector().removeTrace(self)
        return self