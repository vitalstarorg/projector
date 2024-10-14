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
from .operator.Similarity import Similarity

class Projector(SObject):
    model = Holder().name('model')
    projection = Holder().name('projection')
    console = Holder().name('console')
    figure = Holder().name('figure')
    view = Holder().name('view')
    colorShape = Holder().name('colorShape').type('ColorShape')
    traces = Holder().name('traces').type('Map')
    selectedTrace = Holder().name('selectedTrace')
    selectedIdx = Holder().name('selectedIdx')
    selectedVector = Holder().name('selectedVector')
    highlight = Holder().name('highlight')

    def __init__(self):
        self.console(ipywidgets.Output())
        self.figure(go.FigureWidget())
        self.view(View())

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

    def similarity(self, similarity=""):
        if similarity != "":
            similarity.model(self.model())
        res = self._getOrSet('similarity', similarity, nil)
        if res is nil:
            res = self.model().getSimilarity()
            self.setValue('similarity', res)
        return res

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
            projection = self.model().projectMatrix(wte, dim)
            self.projection(projection)
            dataframe = pd.DataFrame(projection.projected(), columns=['x', 'y', 'z'])
            words = self.model().words('_')
            dataframe['word'] = list(words)
            projection.df(dataframe)
        return projection

    def projectVector(self, vectors):
        projection = self.project()
        model = self.model()
        projected = projection.projectVector(vectors)
        return projected

    def searchTokens(self, spelling, n = 5):
        model = self.model()
        res = model.searchTokens(spelling, n)
        return res

    def encode(self, text):
        tokenizer = self.model().tokenizer()
        codes = tokenizer.encode(text)
        return codes

    #### Plotly Functions
    def updateHighlight(self, tokenVector, k):
        highlight = self.highlight()
        if highlight.notNil():
            highlight.remove()
        tokenTrace = self.newTrace().fromVectors(tokenVector)
        knn_ids = tokenTrace.knn(k).knn_ids()
        knn_sims = tokenTrace.knn_sims()
        knn_angles = tokenTrace.knn_angles()
        highlight = self.newTrace().\
                            name('highlight').\
                            fromIndices(knn_ids).\
                            knn_ids(knn_ids).\
                            knn_sims(knn_sims).\
                            knn_angles(knn_angles).\
                            colorShape(ColorShape())
        highlight.colorShape().color('blue')
        self.highlight(highlight)
        return highlight

    def onClick(self, plytrace, points, selector):
        if len(points.point_inds) == 0:
            return
        self.console().outputs = ()
        idx = points.point_inds[0]
        self.selectedIdx(idx)
        with self.console():
            k = 10
            print(f"{datetime.now()} {plytrace.name}[{idx}]")
            trace = self.traces().getValue(plytrace.name)
            tokenVec = trace.vectors()[idx]
            self.selectedTrace(trace)
            self.selectedVector(tokenVec)
            highlight = self.updateHighlight(tokenVec, k)
            df = highlight.asDF()
            print(df)
            highlight.show()

    def getColorShape(self): return self.colorShape().clone()

    def getView(self):
        view = self.view()
        scene = self.figure().layout.scene
        xlimits = scene.xaxis.range
        view.xlimits(xlimits)
        ylimits = scene.yaxis.range
        view.ylimits(ylimits)
        zlimits = scene.zaxis.range
        view.zlimits(zlimits)
        if hasattr(scene, 'aspectmode'):
            aspectmode = getattr(scene, 'aspectmode')
            view.aspectmode(aspectmode)
        return view

    def updateView(self, view=nil):
        if view is not nil:
            self.view(view)
        else:
            view = self.view()
        figwidget = self.figure()
        scene = view.asMap()
        figwidget.update_layout(scene=scene)
        return self

    def resetView(self):
        self.view(View())
        self.updateView()
        return self

    def nextColor(self): self.colorShape().next(); return self

    def showEmbedding(self):
        def on_click2(trace, points, selector):
            with self.console():
                print(datetime.now())

        scene = self.view().asMap()
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
                scene=scene,
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

    def clearAll(self):
        self.clearTraces()
        self.figure().data = ()
        return self

    def clearTraces(self):
        self.colorShape().reset()
        for trace in self.traces().values():
            self.removeTrace(trace)
        return self

    def newTrace(self):
        inference = self.model().inference()
        projection = self.projection()
        trace = Trace().projector(self).\
                    projection(projection).\
                    similarity(self.similarity().copy()).\
                    inference(inference).\
                    colorShape(self.colorShape().clone())
        # self.colorShape().next()
        return trace

    def showTrace(self, trace):
        self.removeTrace(trace)     # try to remove trace first.
        self.traces().setValue(trace.name(), trace)
        scatter = trace.scatter()
        self.figure().add_trace(scatter)
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

class View(SObject):
    xlimits = Holder().name('xlimits')
    ylimits = Holder().name('ylimits')
    zlimits = Holder().name('zlimits')
    aspectmode = Holder().name('aspectmode')

    def asMap(self):
        scene = Map()
        aspectmode = self.aspectmode()
        if aspectmode is nil:
            aspectmode = 'auto'
        scene['aspectmode'] = aspectmode
        if self.xlimits() is not nil:
            scene['xaxis'] = dict(range=self.xlimits(), autorange=False)
            scene['yaxis'] = dict(range=self.ylimits(), autorange=False)
            scene['zaxis'] = dict(range=self.zlimits(), autorange=False)
        else:
            scene['xaxis'] = None
            scene['yaxis'] = None
            scene['zaxis'] = None
        return scene

class ColorShape(SObject):
    color = Holder().name('color')
    colors = Holder().name('colors')
    coloridx = Holder().name('coloridx')
    size = Holder().name('size')

    def __init__(self):
        self.colors(List(px.colors.qualitative.Alphabet_r))
        self.reset()

    def shape(self, shape=""):
        if shape == '': return self.getValue('shape')
        shapes = List(["circle", "square", "diamond", "cross", "x", "diamond-open", "square-open", 'circle-open'])
        criteria = shape.split(' ')
        for criterion in criteria:
            shapes = List([s for s in shapes if criterion in s])
        if shapes.len() > 0:
            first = shapes.head()
            self.setValue('shape', first)
        return self

    def reset(self, coloridx=0):
        self.coloridx(coloridx)
        color = self.colors()[coloridx]
        self.color(color)
        self.shape('circle')
        self.size(15)
        return self

    def next(self):
        color = self.color()
        colors = self.colors()
        coloridx = self.coloridx() + 1
        coloridx = coloridx % colors.len()
        next = colors[coloridx]
        self.coloridx(coloridx)
        self.color(next)
        return self

class Trace(SObject):
    projector = Holder().name('projector')
    inference = Holder().name('inference')
    projection = Holder().name('projection')
    prompt = Holder().name('prompt')
    vectors = Holder().name('vectors')          # full scale vectors
    projected = Holder().name('projected')      # projected vectors
    colorShape = Holder().name('colorShape')
    knn_sims = Holder().name('knn_sims')        # knn similarity to WTE
    knn_ids = Holder().name('knn_ids')          # knn Ids to tokens
    knn_angles = Holder().name('knn_angles')    # knn angles
    similarity = Holder().name('similarity')

    def __init__(self):
        id = self.idDigits(8)
        self.name(id)

    def _vectors(self, vectors):
        if isinstance(vectors, list):
            if isinstance(vectors[0], list):  # List of vectors
                vectors = torch.tensor(vectors)
            else:  # Single vector in list
                vectors = torch.tensor(vectors).unsqueeze(0)
        elif isinstance(vectors, torch.Tensor) and vectors.dim() == 1:  # Single vector
            vectors = vectors.unsqueeze(0)
        return vectors

    def fromIndices(self, indices):
        projector = self.projector()
        projection = self.projection()
        wte = projector.model().modelParams()['wte']
        vectors = List()
        indices1 = self._vectors(indices)
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

    def fromVectors(self, vectors):
        projector = self.projector()
        projection = self.projection()
        if vectors is 0:
            n_embd = projector.model().modelParams().getValue('n_embd')
            vectors = torch.zeros(n_embd)
        vectors = self._vectors(vectors)
        projected = projection.projectVector(vectors)
        self.vectors(vectors)
        self.projected(projected)
        return self

    def asDF(self):
        projection = self.projection()
        projected = self.projected()
        vectors = self.vectors()
        indices = self.knn_ids()
        if indices is nil:
            indices = self.knn(1).knn_ids()
        if indices.dim() == 1:
            indices = indices.tolist()
            angles = self.knn_angles().tolist()
            sims = self.knn_sims().tolist()
        else:
            if indices.shape[0] == 1 or indices.shape[1] == 1:
                indices = indices.flatten().tolist()
                angles = self.knn_angles().flatten().tolist()
                sims = self.knn_sims().flatten().tolist()
            else:
                indices = indices[:,0].tolist()
                angles = self.knn_angles()[:,0]
                sims = self.knn_sims()[:,0]
        # indices = projector.closestIndices(vectors)[:,0].tolist()
        # indices = torch.tensor(indices)
        # projected = projection.projected()[indices]
        df = projection.select(indices)                     # df is a copy
        df['x'] = projected[:,0]
        df['y'] = projected[:,1]
        df['z'] = projected[:,2]
        text = [String(i) for i in range(projected.size(0))]
        df['text'] = text
        df['angle'] = angles
        df['sims'] = sims
        df['norm'] = self.vectors().norm(dim=1)
        return df

    def knn(self, k=0):
        vectors = self.vectors()
        if k != 0: self.similarity().k(k)
        self.similarity().knn(vectors)      # if k is not supplied, use similarity() default.
        self.knn_sims(self.similarity().sims())
        self.knn_ids(self.similarity().ids())
        self.knn_angles(self.similarity().angles())
        return self

    def closestAngles(self, k=1):
        angles = self.knn_angles()
        if angles is nil:
            angles = self.knn(k).knn_angles()
        return angles

    def closestIndices(self, k=1):
        vectors = self.vectors()
        indices = self.knn_ids()
        if indices is nil:
            indices = self.knn(k).knn_ids()
        return indices

    def closestWords(self, dim=0):
        indices = self.closestIndices()
        if dim == 0:
            indices = indices[:, 0]
        else:
            indices = indices[0, :]
        selected = self.projection().select(indices.tolist())
        words = selected['word'].tolist()
        return words

    def scatter(self):
        trace_name = self.name()
        df = self.asDF()
        n = df.shape[0]
        colorIdx = self.colorShape().coloridx()
        color = self.colorShape().color()
        template = "%{hovertext}<br>trace='" + trace_name + "'<br>x=%{x}<br>y=%{y}<br>z=%{z}<extra>%{text}</extra>"
        template = '%{hovertext}<br>idx=%{text}<br>colorIdx=' + str(colorIdx) + '<br>trace=' + trace_name + '<extra></extra>'
        scatter = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'],
                               name=f"{trace_name}",
                               mode='markers + text',
                               text=df['text'],
                               hovertext=df['word'],
                               hovertemplate=template)
        scatter.marker = dict(
            color = [color] * n,
            size=[self.colorShape().size().value()] * n,
            opacity=0.8,
            symbol=self.colorShape().shape()
        )
        return scatter

    def nextColor(self):
        self.colorShape().next()
        return self

    def color(self, color):
        self.colorShape().color(color)
        return self

    def shape(self, shape):
        self.colorShape().shape(shape)
        return self

    def size(self, size):
        self.colorShape().size(size)
        return self

    def show(self):
        self.projector().showTrace(self)
        return self

    def showDF(self, precision=4):
        from ipyaggrid import Grid
        grid_options = {
            'defaultColDef': {
                'sortable': True,
                'resizable': True,
                'valueFormatter': f"x % 1 !== 0 ? x.toFixed({precision}) : Math.floor(x)",
                # 'valueFormatter': 'x.toFixed(2)',
            }
        }
        df = self.asDF()
        grid = Grid(grid_data=df
                    , theme='ag-theme-blue'
                    , index=True
                    , columns_fit='auto'
                    , grid_options=grid_options)
                    # , quick_filter=True)
        return grid

    def remove(self):
        self.projector().removeTrace(self)
        return self

    def indices(self):
        ids = self.knn_ids()
        if ids is nil:
            self.knn()
            ids = self.knn_ids()
        ids = self._vectors(ids)
        ids = ids[:,0]
        indices = ids.squeeze().tolist()
        return List(indices)

    def tokens(self):
        indices = self.indices()
        model = self.projector().model()
        tokens = model.tokens().keys()
        tokenList = [tokens[index] for index in indices]
        return List(tokenList)

    def words(self):
        indices = self.indices()
        model = self.projector().model()
        words = model.words()
        wordsList = [words[index] for index in indices]
        return List(wordsList)
