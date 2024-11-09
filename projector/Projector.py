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
import numpy as np
import pandas as pd
import zipfile
import tempfile
import pickle
import random
import torch
from smallscript import *

import plotly.express as px
import plotly.graph_objects as go
import ipywidgets
from datetime import datetime
from IPython.display import display
from pathlib import Path

class Projector(SObject):
    """
    It works like a projector, projecting high dimensional embedding vectors to 3D space for LLM.
    """
    model = Holder().name('model')            # LLM model
    projection = Holder().name('projection')  # 3D projection
    console = Holder().name('console')        # Notebook console
    figure = Holder().name('figure')          # plotly figure
    view = Holder().name('view')              # 3D x,y,z-limits
    camera = Holder().name('camera')          # camera
    colorShape = Holder().name('colorShape').type('ColorShape')
    traces = Holder().name('traces').type('Map')
    selectedTrace = Holder().name('selectedTrace')
    selectedIdx = Holder().name('selectedIdx')
    selectedVector = Holder().name('selectedVector')
    highlight = Holder().name('highlight')    # highlighted trace
    # filename, similarity, wOffset, bOffset

    def __init__(self):
        self.console(ipywidgets.Output())
        self.figure(go.FigureWidget())
        self.view(View())

    #### Attribute holder methods
    def filename(self, filename=""):
        """filename for saving and loading cache"""
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
        """Define the similarity used by the projector to look for closest embedding tokens."""
        if similarity != "":
            similarity.model(self.model())
        res = self._getOrSet('similarity', similarity, nil)
        if res is nil:
            res = self.model().getSimilarity()
            self.setValue('similarity', res)
        return res

    def wOffset(self, wOffset = nil):
        """Define the weight for the wbnorm."""
        # It works equivalently as fnorm() to set ln_f.w as wOffset and ln_f.b as bOffset.
        if wOffset is nil:
            res = self.getValue('wOffset')
            if res is not nil: return res
            wOffset = 1
        if self.model().isNil(): return nil
        n_embd = self.model().modelParams().getValue('n_embd')
        vectors = torch.ones(n_embd) * wOffset
        self.setValue('wOffset', vectors)
        return self

    def bOffset(self, bOffset = nil):
        """Define the bias for the wbnorm."""
        # It works equivalently as fnorm() to set ln_f.b as bOffset.
        if bOffset is nil:
            return self.getValue('bOffset')
            if res is not nil: return res
        if bOffset is 0 or bOffset is nil:
            n_embd = self.model().modelParams().getValue('n_embd')
            bOffset = torch.zeros(n_embd) * bOffset
        self.setValue('bOffset', bOffset)
        return self

    #### File I/O
    def hasCache(self):
        filepath = Path(self.filename())
        return self.asSObj(filepath.exists())

    def saveCache(self):
        """Saving the 3D projection, view and camera as cache."""
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
            with tempfile.NamedTemporaryFile(mode="wb", delete=True) as f:
                pickle.dump(self.view(), f)
                f.flush()
                zip.write(f.name, "view.pkl", compress_type=zipfile.ZIP_DEFLATED)
            with tempfile.NamedTemporaryFile(mode="wb", delete=True) as f:
                pickle.dump(self.camera(), f)
                f.flush()
                zip.write(f.name, "camera.pkl", compress_type=zipfile.ZIP_DEFLATED)
        return self

    def loadCache(self):
        """Loading the 3D projection, view and camera from cache."""
        filepath = Path(self.filename())
        if not filepath.exists() or not filepath.is_file():
            self.log(f"file '{self.filename()}' not found.", Logger.LevelWarning)
            return
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
            zip.extract("view.pkl", tmpdir.name)
            with open(f"{tmpdir.name}/view.pkl", "rb") as f:
                view = pickle.load(f)
                self.view(view)
            zip.extract("camera.pkl", tmpdir.name)
            with open(f"{tmpdir.name}/camera.pkl", "rb") as f:
                camera = pickle.load(f)
                self.camera(camera)
        return self

    def deleteCache(self):
        filepath = Path(self.filename())
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
        return self

    #### Projection matrix operations
    def project(self, dim=3):
        """Calculate the 3D projection using PCA."""
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
        """Project vectors to the 3D projection."""
        projection = self.project()
        model = self.model()
        projected = projection.projectVector(vectors)
        return projected

    def searchTokens(self, spelling, n = 5):
        """Finding n similar spelling tokens."""
        model = self.model()
        res = model.searchTokens(spelling, n)
        return res

    def encode(self, text):
        """Encode @text as tokens."""
        tokenizer = self.model().tokenizer()
        codes = tokenizer.encode(text)
        return codes

    #### Plotly Functions
    def updateHighlight(self, tokenVector, k):
        """Create a trace with k closest embedding vectors to @tokenVector."""
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
        """Plotly onClick method showing selected point in a trace and highlight k closest embedded tokens. If a console is opened, this highlighted trace will be shown as dataframe."""
        if len(points.point_inds) == 0:
            return
        self.console().outputs = ()
        idx = points.point_inds[0]
        self.selectedIdx(idx)
        with self.console():
            k = 10
            print(f"{datetime.now()}; plytrace.name = {plytrace.name}; idx = {idx}")
            traceNames = self.traces().keys()
            traceName = plytrace.name
            if traceName not in traceNames:
                traceName = plytrace.name.rsplit('_', 1)[0]  # remove suffix
                if traceName not in traceNames:
                    return
            print(f"{datetime.now()} {traceName}[{idx}]")
            trace = self.traces().getValue(traceName)
            tokenVec = trace.vectors()[idx]
            self.selectedTrace(trace)
            self.selectedVector(tokenVec)
            highlight = self.updateHighlight(tokenVec, k)
            df = highlight.asDF()
            print(df)
            highlight.show()

    def getColorShape(self): return self.colorShape().clone()

    def getCamera(self):
        figwidget = self.figure()
        if figwidget is nil: return nil
        camera = self.figure().layout.scene.camera
        self.camera(camera)
        return camera

    def updateCamera(self, camera=nil):
        figwidget = self.figure()
        if figwidget is nil: return nil
        if camera is not nil:
            self.camera(camera)
        else:
            camera = self.camera()
            if camera is nil: return nil
        figwidget.update_layout(scene_camera = camera)
        return self

    def resetCamera(self):
        figwidget = self.figure()
        if figwidget is nil: return nil
        figwidget.update_layout(scene_camera = None)
        return self

    def getView(self):
        view = self.view()  # init with View()
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
        figwidget = self.figure()
        if figwidget is nil: return nil
        if view is not nil:
            self.view(view)
        else:
            view = self.view()
        scene = view.asMap()
        figwidget.update_layout(scene=scene)
        return self

    def resetView(self):
        self.view(View())
        self.updateView()
        return self

    def nextColor(self): self.colorShape().next(); return self

    def showEmbedding(self, sample=1.0):
        """Show the embedded token cloud as background. Other traces will be showed as selectable objects."""
        def on_click2(trace, points, selector):
            with self.console():
                print(datetime.now())

        scene = self.view().asMap()
        if len(self.figure().data) == 0:
            figwidget = self.figure()
            df = self.projection().df()
            if sample != 1.0:
                df = df.sample(frac=sample)
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
        """"Show the colorband used in this projector."""
        colorShape = ColorShape()
        colors = colorShape.defaults()
        rgb = colorShape.rgb()
        data = [list(range(0, len(colors)))]
        text = [[str(i) for i in range(0, len(colors))]]
        fig = go.Figure(data=go.Heatmap(
            z=data,
            text=text,
            hovertext=rgb,
            texttemplate="%{text}",
            hovertemplate='%{hovertext}<extra></extra>',
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

    #### Trace related methods
    def clearTraces(self):
        self.colorShape().reset()
        for trace in self.traces().values():
            trace.remove()
        return self

    def newTrace(self):
        """Create a new trace from this projector."""
        inference = self.model().inference()
        projection = self.projection()
        trace = Trace().projector(self).\
                    projection(projection).\
                    similarity(self.similarity().copy()).\
                    inference(inference).\
                    colorShape(self.colorShape().clone()).\
                    wOffset(self.wOffset()).\
                    bOffset(self.bOffset())
        return trace

    def newLines(self):
        """Create a new line from this projector."""
        inference = self.model().inference()
        projection = self.projection()
        lines = Lines().projector(self).\
                    projection(projection).\
                    similarity(self.similarity().copy()).\
                    inference(inference).\
                    colorShape(self.colorShape().clone()).\
                    wOffset(self.wOffset()).\
                    bOffset(self.bOffset())
        return lines

    def showTrace(self, trace):
        self.removeTrace(trace)
        self.traces().setValue(trace.name(), trace)
        scatter = trace.plot()
        self.figure().add_trace(scatter)
        addedScatter = List(self.figure().select_traces(selector={'name': trace.name()}))[-1]
        addedScatter.on_click(self.onClick)
        return self

    def addTrace(self, trace):
        trace.remove()
        self.traces().setValue(trace.name(), trace)
        return self

    def removeTrace(self, trace):
        traces = self.traces()
        if trace.name() in traces:
            traces.delValue(trace.name())
        return self

    def addPlotByName(self, name, scatter, onclick=true_):
        scatter.name = name
        self.figure().add_trace(scatter)
        addedScatter = List(self.figure().select_traces(selector={'name': name}))[-1]
        if onclick:
            addedScatter.on_click(self.onClick)
        return self

    def removePlotByName(self, name):
        traces = [t for t in self.figure().data if t.name != name]
        self.figure().data = tuple(traces)
        return self

class View(SObject):
    # Capture the view limits of a plot.
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
    """
    Define the color scale, size, opacity and shape.
    """
    defaults = Holder().name('defaults').asClassType()
    rgb = Holder().name('rgb')
    colors = Holder().name('colors')
    color = Holder().name('color')
    coloridx = Holder().name('coloridx')
    size = Holder().name('size')
    opacity = Holder().name('opacity')
    # shape

    @Holder().asClassType()
    def metaInit(scope):
        self = scope['self']
        aColors = px.colors.convert_colors_to_same_type(px.colors.qualitative.Alphabet_r, colortype='rgb')[0]
        shuffle = list(range(0, len(aColors)))[::-1]
        random.seed(52)
        random.shuffle(shuffle)
        defaults = []
        for i in shuffle:
            defaults.append(aColors[i])
        self.defaults(defaults)
        return self

    def __init__(self):
        if self.defaults() is nil: return
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

        if self.colors().isNil():
            self.colors(self.defaults())
        rgb = []
        for color in self.colors():
            rgb.append(rgb_to_hex(color))
        rgb = [rgb]
        self.rgb(rgb)
        self.coloridx(coloridx)
        color = self.colors()[coloridx]
        self.color(color)
        self.shape('circle')
        self.size(15)
        self.opacity(0.8)
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

    def show(self):
        fig = go.Figure(data=self.plot())
        fig.update_layout(
            height=30, width=800,
            margin=dict(l=0, r=0, t=0, b=0),
            modebar={'remove': ['zoom', 'zoomin', 'zoomout', 'pan', 'reset', 'autoscale', 'resetscale', 'toImage']},
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange='reversed')
        )
        return fig

    def plot(self):
        # Plot the colorband
        colors = self.colors()
        rgb = self.rgb()
        data = [list(range(0, len(colors)))]
        text = [[str(i) for i in range(0, len(colors))]]
        data=go.Heatmap(
            z=data,
            text=text,
            hovertext=rgb,
            texttemplate="%{text}",
            hovertemplate='%{hovertext}<extra></extra>',
            colorscale=colors,
            opacity=0.9,
            showscale=False
        )
        return data

class Trace(SObject):
    """
    This is an adaptor to a tensor to provide plotting functions.
    """
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
    wOffset = Holder().name('wOffset')
    bOffset = Holder().name('bOffset')
    label = Holder().name('label')

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
        # Show this trace as a dataframe
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

    def isPoint(self): return len(self.asDF()) == 1

    #### Final layer norm used for visualization only.
    # Deal with the final layer norm to allow us to project the embedding vector for better visualization.
    def fnorm(self):
        # original final layer norm
        projector = self.projector()
        model = projector.model()
        fnorm = model.fnorm(self.vectors())
        projected = projector.projectVector(fnorm)
        self.vectors(fnorm)
        self.projected(projected)
        return self

    def wbnorm(self, eps=1e-5):
        # Defined our own norm to replace fnorm().
        # It works equivalently as fnorm() to set ln_f.w as wOffset and ln_f.b as bOffset.
        projector = self.projector()
        model = projector.model()
        vectors = self.vectors()
        mean = vectors.mean(dim=-1, keepdim=True)
        variance = vectors.var(dim=-1, keepdim=True)
        vectors = (vectors - mean) / torch.sqrt(variance + eps)
        if self.wOffset() is not nil:
            vectors = self.wOffset() * vectors
        if self.bOffset() is not nil:
            vectors = self.bOffset() + vectors
        projected = projector.projectVector(vectors)
        self.vectors(vectors)
        self.projected(projected)
        return self

    #### KNN and closest indices, angles & words
    def knn(self, k=0):
        # Trigger the knn calculation on the similarity by using the Similarity object.
        # This is called first for self.closest???() methods.
        vectors = self.vectors()
        if k != 0: self.similarity().k(k)
        self.similarity().knn(vectors)      # if k is not supplied, use similarity() default.
        self.knn_sims(self.similarity().sims())
        self.knn_ids(self.similarity().ids())
        self.knn_angles(self.similarity().angles())
        return self

    def closestIndices(self, k=1):
        vectors = self.vectors()
        indices = self.knn_ids()
        if indices is nil:
            indices = self.knn(k).knn_ids()
        return indices

    def closestAngles(self, k=1):
        angles = self.knn_angles()
        if angles is nil:
            angles = self.knn(k).knn_angles()
        return angles

    def closestWords(self, dim=0):
        indices = self.closestIndices()
        if dim == 0:
            indices = indices[:, 0]
        else:
            indices = indices[0, :]
        selected = self.projection().select(indices.tolist())
        words = selected['word'].tolist()
        return words

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

    #### Plot related methods
    def plot(self):
        trace_name = self.name()
        df = self.asDF()
        n = df.shape[0]
        colorIdx = self.colorShape().coloridx()
        color = self.colorShape().color()
        opacity = self.colorShape().opacity().asFloat()
        template = '%{hovertext}<br>colorIdx=' + str(colorIdx) + '<br>trace=' + trace_name + '<extra></extra>'
        if self.label().notNil():
            text = self.label()
        else:
            if self.isPoint():
                text = ""
            else:
                template = '%{hovertext}<br>idx=%{text}<br>colorIdx=' + str(
                    colorIdx) + '<br>trace=' + trace_name + '<extra></extra>'
                text=df['text']
        scatter = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'],
                               mode='markers + text',
                               text=text,
                               hovertext=df['word'],
                               hovertemplate=template)
        scatter.marker = dict(
            color = [color] * n,
            size=[self.colorShape().size().value()] * n,
            opacity=opacity,
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

    def showDF(self, precision=4):
        # Using Grid to show the dataframe
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
        row_height = 32
        height = min(400, len(df) * row_height)  # Limit the maximum height to 400 pixels
        grid = Grid(grid_data=df
                    , theme='ag-theme-blue'
                    , index=True
                    , columns_fit='auto'
                    , grid_options=grid_options
                    , height=height)
                    # , quick_filter=True)
        return grid

    def show(self):
        self.remove()
        self.projector().addTrace(self)
        self.projector().addPlotByName(self.name(), self.plot())
        return self

    def remove(self):
        self.projector().removeTrace(self)
        self.projector().removePlotByName(self.name())
        return self

class Lines(Trace):
    """
    Same as Trace, instead of showing points, but lines and optional arrows.
    """
    arrow = Holder().name('withArrow').type('False_')
    center = Holder().name('center')
    center3d = Holder().name('center3d')  # list

    def setOrigin(self):
        if self.projector().isNil(): return
        n_embd = self.projector().model().modelParams().getValue('n_embd')
        origin = torch.zeros(n_embd)
        return self.setCenter(origin)

    def setCenter(self, center):
        self.center(center)
        center3d = self.projection().projectVector(center)
        self.center3d(center3d.squeeze().tolist())
        return self

    def withArrow(self): return self.arrow(true_)

    def _line(self, from3d, to3d, template=""):
        x = from3d[0]; y = from3d[1]; z = from3d[2]
        u = to3d[0]; v = to3d[1]; w = to3d[2]
        scatter = go.Scatter3d(
            x=[x, u], y=[y, v], z=[z, w],
            mode='lines',
            line=dict(
                width=3,
                color=self.colorShape().color()),
            hovertemplate=template + '<extra></extra>'
        )
        return scatter

    def _arrow(self, from3d, to3d, template=""):
        x = to3d[0]; y = to3d[1]; z = to3d[2]
        u = x - from3d[0]; v = y - from3d[1]; w = z - from3d[2]
        norm = np.sqrt(u**2 + v**2 + w**2)
        u = u/norm; v = v/norm; w = w/norm
        color = self.colorShape().color()
        scatter = go.Cone(
            x = [x], y = [y], z = [z],
            u = [u], v = [v], w = [w],
            sizemode="absolute",
            sizeref=0.1,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            anchor='tip',
            hovertemplate=template + '<extra></extra>'
        )
        return scatter

    def _coords(self, row):
        coords = List()
        coords.append(row.x).append(row.y).append(row.z)
        return coords

    def _series(self):
        lineName = f"{self.name()}_line"
        arrowName = f"{self.name()}_arrow"
        from3d = nil
        for row in self.asDF().itertuples():
            if from3d is nil:
                from3d = self._coords(row)
                continue
            to3d = self._coords(row)
            self.projector().addPlotByName(lineName, self._line(from3d, to3d), false_)
            if self.arrow():
                self.projector().addPlotByName(arrowName, self._arrow(from3d, to3d), false_)
            from3d = to3d
        return self

    def _radial(self):
        lineName = f"{self.name()}_line"
        arrowName = f"{self.name()}_arrow"
        from3d = self.center3d()
        for row in self.asDF().itertuples():
            to3d = self._coords(row)
            self.projector().addPlotByName(lineName, self._line(from3d, to3d), false_)
            if self.arrow():
                self.projector().addPlotByName(arrowName, self._arrow(from3d, to3d, f"{row.text}"), false_)
        return self

    def show(self):
        self.projector().addTrace(self)
        if self.isPoint():
            if self.center() is nil:
                return super().show()
            self._radial()
            return self
        if self.center() is nil:
            self._series()
        else:
            self._radial()
        return self

    def remove(self):
        self.projector().removeTrace(self)
        self.projector().removePlotByName(f"{self.name()}_line")
        self.projector().removePlotByName(f"{self.name()}_arrow")
        return self
