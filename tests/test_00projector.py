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

from unittest import TestCase
import pytest
from pytest import approx as approx
skip = pytest.mark.skip
skipIf = pytest.mark.skipif

import torch
import numpy as np

import os
import re
from os import environ as env
from math import floor, log10

from projector.Projector import Projector
from projector.operator.GPT2Operator import GPT2Operator
from projector.operator.GPT2OperatorNP import GPT2OperatorNP
import projector.operator.GPT2EncoderNP as GPT2Encoder
from projector.GPT2Inference import GPT2Inference
from projector.operator.Similarity import *
from smallscript import *

class About(SObject):
    __array_priority__ = 10.0  # use higher value to supersede numpy to use About.__eq__()

    precision = Holder().name('precision')
    expected = Holder().name('expected')
    pyapprox = Holder().name('pyapprox')
    always = Holder().name('always')

    def __init__(self):
        self.precision(3)
        self.always(false_)

    def _toNumpy(self, numbers):
        npnum = numbers
        if torch.is_tensor(numbers):
            if numbers.dim() == 0:
                npnum = numbers.item()
            else:
                npnum = numbers.numpy()
        return npnum

    def __call__(self, expected, *args, **kwargs):
        self.expected(expected)
        pyapprox = approx(expected, *args, **kwargs)
        self.pyapprox(pyapprox)
        return self

    def __eq__(self, actual):
        def pctString(expected, actual):
            if abs(actual) < 1e-5:
                diffPct = f"diff = {round(abs(actual - expected), 6)}"
            else:
                diffPct = f"{round(abs((actual - expected) / actual) * 100, 2)}%"
            return diffPct

        def roundPrecision(precision, actual):
            if abs(actual) < 1e-5:
                rounded = actual
            else:
                rounded = round(actual, self.precision() - int(floor(log10(abs(actual)))))
            return rounded

        pyapprox = self.pyapprox()
        actual = self._toNumpy(actual)
        res = actual == pyapprox
        if res: return res
        # actual == pyapprox      # redo the test for debug tracing
        expected = self.expected()
        if not np.isscalar(actual):
            roundedActual = []
            for x in actual:
                rounded = roundPrecision(self.precision(), x)
                roundedActual.append(rounded)
            diffPct = [pctString(expected[i], actual[i]) for i in range(len(actual))]
        else:
            if torch.is_tensor(actual):
                actual = actual.item()
            diffPct = pctString(expected, actual)
            roundedActual = roundPrecision(self.precision(), actual)

        # if isinstance(actual, list):
        #     diffPct = [f"{round(abs((actual[i] - expected[i]) / actual[i]) * 100, 2)}%" for i in range(len(actual))]
        # else:
        #     diffPct = f"{round(abs((actual - expected) / actual) * 100, 2)}%"
        self.log(f"{expected} was expected but actual is {actual} {diffPct}. Try about({roundedActual}, 1e-{self.precision()})", Logger.LevelError)
        return self.always()

    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func.__name__ == 'eq':
            res = cls.__eq__(args[0])
            return res
        return NotImplemented

    def __repr__(self):
        return f"{self.expected()}"

about = approx
about = About().always(true_)   # can be set in mid of debugging
about = About()

# env['SKIPHACK'] = '1'
# env['SKIP'] = '1'
# env['CLEAN_CACHE'] = '1'
class Test_Projector(TestCase):
    import lovely_tensors as lt
    lt.monkey_patch()

    @pytest.fixture(autouse=True, scope='class')
    def setup(self):
        pkg = sscontext.loadPackage('projector')
        pkg = sscontext.loadPackage('tests')
        # model = GPT2Adaptor().name("gpt2").load()
        # pj = Projector().name('pj').model(model)
        if 'CLEAN_CACHE' in env:
            sscontext.log('Remove projector.zip & projector_np.zip.', Logger.LevelInfo)
            Projector().name('projector').deleteFile()
            Projector().name('projector_np').deleteFile()
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test100_basics(self):
        pj = Projector()

    @skipIf('SKIP' in env, reason="disabled")
    def test200_filename(self):
        pj = Projector()
        assert 'unnamed.zip' == pj.filename()
        pj.name('test100')
        assert 'unnamed.zip' == pj.filename()
            # changing name wouldn't change filename after the first call.
        pj = Projector().name('test100')
        assert 'test100.zip' == pj.filename()
        pj.filename('iwantthisname')
        assert 'iwantthisname' == pj.filename()
            # better to suffix it with .zip as it will be saved as zip file.

    @skipIf('SKIP' in env, reason="disabled")
    def test300_np_project(self):
        # 5.213 sec, delete projector_np.zip to measure this time.
        pj = Projector().name('projector_np')
        if not pj.hasFile():
            model = GPT2OperatorNP().name("gpt2-chkpt").loadModel()
            pj.model(model)
            projection = pj.project()
            assert (50257,4) == projection.df().shape
            pj.saveFile()

    # @skipIf('SKIP' in env, reason="disabled")
    def test310_tensor_project(self):
        # model = GPT2Operator().\
        #             org("jtatman").\
        #             name("gpt2-open-instruct-v1-Anthropic-hh-rlhf").\
        #             loadModel()
        # 0.695 sec
        model = GPT2Operator().name("gpt2").loadModel()
        self.__class__.model = model
        projector = Projector().name('pj').model(model)
        projection = projector.project()
        assert (768,3) == projection.components().shape
        self.__class__.projector = projector

    @skipIf('SKIP' in env, reason="disabled")
    def test330_save_load(self):
        projector = self.projector

        projector.name('projector').saveFile()
        pj1 = Projector().name('projector')
        pj1.loadFile()
        df1 = pj1.projection().df()
        assert (768,3) == pj1.projection().components().shape
        assert '\u2588the' == df1.iloc[262].word
        # assert ' the' == df1.iloc[262].word

        # standarized PCA with std
        # x = pj1.projection().mean()[0]
        # assert x == about(-0.002549, 1e-2)
        # x = pj1.projection().std()[0]
        # assert x == about(0.1126, 1e-2)
        #
        # x, y, z = pj1.projection().components()[0]
        # assert x == about(0.02407, 1e-2)
        # assert y == about(0.009943, 1e-2)
        # assert z == about(-0.009700, 1e-2)
        #
        # x, y, z = pj1.projection().projected()[0]
        # assert x == about(9.7107, 1e-2)
        # assert y == about(-4.6005, 1e-2)
        # assert z == about(1.3105, 1e-2)

        # standarized PCA without std
        x = pj1.projection().mean()[0]
        assert x == about(-0.002549, 1e-3)
        x = pj1.projection().std()[0]
        assert x == about(1.0, 1e-3)

        x, y, z = pj1.projection().components()[0]
        assert x == about(0.0184, 1e-3)
        assert y == about(0.0195, 1e-3)
        assert z == about(0.0207, 1e-3)

        x, y, z = pj1.projection().projected()[0]
        assert x == about(1.27, 1e-3)
        assert y == about(-0.414, 1e-3)
        assert z == about(-0.292, 1e-3)

    @skip
    def test340_project_vec_np(self):
        modelNP = GPT2OperatorNP().name("gpt2-chkpt").loadModel()
        self.__class__.modelNP = modelNP
        pj2 = Projector().name('projector_np').model(modelNP).loadFile()
        projection2 = pj2.projection()

        # assert projection2.varianceRatio() == about([0.01985565, 0.01102482, 0.00954412], 1e-3)
        # assert projection2.varianceRatio() == about([0.01985, 0.01102, 0.009544], 1e-3)
        assert projection2.varianceRatio() == about([0.01811, 0.008229, 0.007154], 1e-3)
        tokenSpec = modelNP.searchTokens('Token').head()
        codeToken = tokenSpec['id']
        assert 30642 == codeToken
        vector2 = tokenSpec['vector']
        assert vector2[0] == about(0.01596, 1e-3)

        projected2 = projection2.projectVector(vector2)
        # assert projected2[0][0] == about(-4.2811, 1e-3)
        # assert projected2[0][1] == about(3.52, 1e-3)
        # assert projected2[0][2] == about(0.335, 1e-3)
        assert projected2[0][0] == about(0.01334, 1e-3)
        assert projected2[0][1] == about(0.007604, 1e-3)
        assert projected2[0][2] == about(-0.01539, 1e-3)
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test350_project_vec(self):
        # model = GPT2Operator().name("gpt2").loadModel()
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()
        projection = pj.projection()

        # assert projection.varianceRatio() == approx([0.01986, 0.01102, 0.009544], 1e-3)
        # assert [0.01855, 0.01063, 0.009336] == about(projection.varianceRatio(), 1e-3)
        assert projection.varianceRatio() == about([0.0181, 0.00823, 0.00717], 1e-3)

        tokenSpec = model.searchTokens('Token').head()
        codeToken = tokenSpec['id']
        assert 30642 == codeToken
        vector = tokenSpec['vector']
        assert vector[0] == about(0.01596, 1e-3)

        projected = projection.projectVector(vector)
        # assert 4.281162738800049 == projected[0][0]
        # assert -3.5077905654907227 == approx(projected[0][1], 1e-6)
        # assert 0.3020952641963959 == approx(projected[0][2], 1e-6)
        assert projected[0][0] == about(-0.317, 1e-3)
        assert projected[0][1] == about(-0.389, 1e-3)
        assert projected[0][2] == about(-0.01947, 1e-3)
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test400_show(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()
        pj.showEmbedding()

    @skipIf('SKIP' in env, reason="disabled")
    def test410_zero_cos(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()
        similarity = CosineSimilarity()
        pj.similarity(similarity)

        # Project Zero
        zeros = pj.newTrace().fromVectors(0)
        # assert approx(zeros.projected(), 1e-4) == [[-100.650, 31.910, 33.055]]
        # assert approx(zeros.projected(), 1e-3) == [[11.9618, -3.7794, 2.6324]]
        # assert zeros.projected()[0] == about([11.9618, -3.7794, 2.6324], 1e-3)
        assert zeros.projected()[0] == about([1.37, -0.0445, -0.129], 1e-3)
        words = zeros.closestWords()
        assert zeros.closestIndices().tolist() == [[379]]
        assert zeros.similarity().ids().tolist() == [[379]]
        assert zeros.similarity().sims().squeeze().item() == about(1, 1e-4)
        assert zeros.closestAngles().squeeze().item() == about(0.028, 1e-4)
            # fp32 precision issues
        assert words == ['\u2588at']
        assert zeros.asDF().shape == (1,8)
        zeros.asDF()

        # Project 5 neigbhours around zero
        ids = zeros.similarity().k(5).knn(zeros.vectors()).ids()
        zeroKNN = pj.newTrace().fromIndices(ids)
        assert zeroKNN.asDF().shape == (5,8)
        assert zeroKNN.similarity().ids().squeeze().tolist() == [379, 287, 319, 281, 329]
        assert zeroKNN.similarity().sims().squeeze().tolist() == about([1, 1, 1, 1, 1], 1e-4)
        assert zeroKNN.closestAngles().squeeze().tolist() == about([0.0485, 0.0593, 0.0442, 0., 0.], 1e-4)
            # fp32 precision issues
        words = zeroKNN.closestWords()
        assert words == ['\u2588at', '\u2588in', '\u2588on', '\u2588an', '\u2588for']
        assert zeroKNN.asDF()['word'].tolist() == ['\u2588at', '\u2588in', '\u2588on', '\u2588an', '\u2588for']

    @skipIf('SKIP' in env, reason="disabled")
    def test420_zero_logit(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()

        # Project Zero
        zeros = pj.newTrace().fromVectors(0)
        assert zeros.projected()[0] == about([1.37, -0.0445, -0.129], 1e-3)
        words = zeros.closestWords()
        assert zeros.closestIndices().tolist() == [[379]]
        assert zeros.similarity().ids().tolist() == [[379]]
        assert zeros.similarity().sims().squeeze().item() == about(6.021, 1e-4)
        assert zeros.closestAngles().squeeze().item() == about(0.0396, 1e-3)
            # fp32 precision issues
        assert words == ['\u2588at']
        assert zeros.asDF().shape == (1,8)
        zeros.asDF()

        # Project 5 neigbhours around zero
        ids = zeros.similarity().k(5).knn(zeros.vectors()).ids()
        zeroKNN = pj.newTrace().fromIndices(ids)
        assert zeroKNN.asDF().shape == (5,8)
        assert zeroKNN.similarity().ids().squeeze().tolist() ==about([287, 319, 329, 379, 11])
        assert zeroKNN.similarity().sims().squeeze().tolist() == about([6.074, 6.114, 6.182, 6.021, 9.472], 1e-3)
        assert zeroKNN.closestAngles().squeeze().tolist() == about([0.0593, 0.0442, 0.0, 0.0485, 0.0343], 1e-3)
            # fp32 precision issues
        words = zeroKNN.closestWords()
        assert words == ['\u2588in', '\u2588on', '\u2588for', '\u2588at', ',']
        assert zeroKNN.asDF()['word'].tolist() == ['\u2588in', '\u2588on', '\u2588for', '\u2588at', ',']

    @skipIf('SKIP' in env, reason="disabled")
    def test430_trace_cos(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()
        similarity = CosineSimilarity()
        pj.similarity(similarity)

        # knn for all 4 vectors
        trace = pj.newTrace().fromPrompt("Allen Turing theorized")
        vectors = trace.vectors()
        trace1 = pj.newTrace().fromVectors(vectors)
        # trace1.similarity().k(5).knn(vectors)
        trace1.knn(5)
        ids = trace1.similarity().ids()
        sims = trace1.similarity().sims()
        assert ids.shape == (4,5)
        assert sims.shape == (4,5)
        words = trace1.closestWords()
        assert words == ['Allen', '\u2588Turing', '\u2588theor', 'ized']
        words = trace1.closestWords(dim=1)
        assert words == ['Allen', '\u2588Allen', '\x16', '\\xf7', '\\xfb']
        trace1.asDF()

        ids1 = trace1.similarity().ids()[0,:]
        assert ids1 == about([39989, 9659, 210, 179, 183])
        sims1 = trace1.similarity().sims()[0,:]
        assert sims1 == about([1.0, 0.7447, 0.5195, 0.5187, 0.5185], 1e-3)
        angles1 = trace1.similarity().angles()[0,:]
        knn1 = pj.newTrace().fromIndices(ids1).knn_ids(ids1).knn_sims(sims1).knn_angles(angles1)
        assert knn1.asDF()['word'].tolist() == ['Allen', '\u2588Allen', '\x16', '\\xf7', '\\xfb']

        # knn on 2nd vectors i.e. 'Turing'
        trace2 = pj.newTrace().fromVectors(vectors[1])
        # trace2.similarity().k(5).knn(vectors[1])
        trace2.knn(5)
        words = trace2.closestWords()
        assert words == ['\u2588Turing']
        words = trace2.closestWords(dim = 1)
        assert words == ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588']

        knn2 = pj.newTrace().fromIndices(trace2.similarity().ids())
        assert knn2.asDF()['word'].tolist() == ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588']

        # knn on 3rd vectors i.e. 'theor'
        trace3 = pj.newTrace().fromVectors(vectors[2])
        # trace3.similarity().k(5).knn(vectors[2])
        trace3.knn(5)
        words = trace3.closestWords()
        assert words == ['\u2588theor']
        words = trace3.closestWords(dim = 1)
        assert words == ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588speculated', '\u2588theories']

        knn3 = pj.newTrace().fromIndices(trace3.similarity().ids())
        assert knn3.asDF()['word'].tolist() == ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588speculated', '\u2588theories']

        return

    @skipIf('SKIP' in env, reason="disabled")
    def test440_trace_logit(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()

        # knn for all 4 vectors
        trace = pj.newTrace().fromPrompt("Allen Turing theorized")
        vectors = trace.vectors()
        trace1 = pj.newTrace().fromVectors(vectors)
        trace1.knn(5)
        ids = trace1.similarity().ids()
        sims = trace1.similarity().sims()
        assert ids.shape == (4,5)
        assert sims.shape == (4,5)
        words = trace1.closestWords()
        assert words == ['Allen', '\u2588Turing', '\u2588theor', 'ized']
        words = trace1.closestWords(dim=1)
        assert words == ['Allen', '\u2588Allen', 'Adams', 'sonian', 'Murray']
        trace1.asDF()

        ids1 = trace1.similarity().ids()[0,:]
        assert ids1 == about([39989, 9659, 47462, 35202, 49998])
        sims1 = trace1.similarity().sims()[0,:]
        assert sims1 == about([16.89, 10.6, 8.722, 8.604, 8.581], 1e-3)
        angles1 = trace1.similarity().angles()[0,:]
        knn1 = pj.newTrace().fromIndices(ids1).knn_ids(ids1).knn_sims(sims1).knn_angles(angles1)
        assert knn1.asDF()['word'].tolist() == ['Allen', '\u2588Allen', 'Adams', 'sonian', 'Murray']

        # knn on 2nd vectors i.e. 'Turing'
        trace2 = pj.newTrace().fromVectors(vectors[1])
        trace2.knn(5)
        words = trace2.closestWords()
        assert words == ['\u2588Turing']
        words = trace2.closestWords(dim = 1)
        assert words == ['\u2588Turing', 'ertodd', 'isSpecialOrderable', '\u2588mathemat', '\u2588Canaver']

        knn2 = pj.newTrace().fromIndices(trace2.similarity().ids())
        assert knn2.asDF()['word'].tolist() == ['\u2588Turing', 'ertodd', 'isSpecialOrderable', '\u2588mathemat', '\u2588Canaver']

        # knn on 3rd vectors i.e. 'theor'
        trace3 = pj.newTrace().fromVectors(vectors[2])
        trace3.knn(5)
        words = trace3.closestWords()
        assert words == ['\u2588theor']
        words = trace3.closestWords(dim = 1)
        assert words == ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588mathemat', '\u2588speculate']

        knn3 = pj.newTrace().fromIndices(trace3.similarity().ids())
        assert knn3.asDF()['word'].tolist() == ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588mathemat', '\u2588speculate']

    @skipIf('SKIP' in env, reason="disabled")
    def test445_trace(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()
        trace = pj.newTrace().fromPrompt("Alan Turing theorized")

        x = trace.indices()
        assert x == [36235, 39141, 18765, 1143]
        x = trace.tokens()
        assert x == ['Alan', 'ĠTuring', 'Ġtheor', 'ized']
        x = trace.words()
        assert x == ['Alan', ' Turing', ' theor', 'ized']


    @skipIf('SKIP' in env, reason="disabled")
    def test450_show(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()

        # simulate the show()
        trace = pj.newTrace().fromPrompt("Allen Turing theorized")
        assert len(pj.figure().data) == 0
        trace.show()
        assert len(pj.figure().data) == 1
        trace.remove()
        assert len(pj.figure().data) == 0
        trace.asDF()

        origin = pj.newTrace().name('origin').fromVectors(0).color('black').show()

        vectors = trace.vectors()
        trace1 = pj.newTrace().fromVectors(vectors)
        trace1.asDF()
        trace1.show()

        # simulate the highlight
        tokenVec = trace.vectors()[1]
        highlight = pj.updateHighlight(tokenVec, 10)
        df = highlight.asDF()
        highlight.showDF()

    @skip
    def test900_hack(self):
        model = GPT2Operator().name("gpt2").loadModel()
        pj = Projector().name('projector').model(model).loadFile()
        projection = pj.projection()

        wte = model.modelParams()['wte']
        projected = projection.projected()
        mean = projection.mean()
        std = projection.std()

        a0 = (wte - wte.mean(dim=0)).mean(dim=0).norm()  # 1.171e-07
        a1 = projected.mean(dim=0).norm()  # 8.891e-06
        a2 = projected.std(dim=0).norm()   # 5.572
        a3 = projected.max()               # 16.209
        a4 = projected.min()               # -17.144
            # no way projecting zero would beyond with these min and max.

        idAllen = model.tokenizer().encode('Allen')
        xAllen = pj.newTrace().fromIndices(idAllen)
        id379 = torch.tensor([[379]])
        x379 = pj.newTrace().fromIndices(id379)
        id379 = torch.tensor([379])
        x379 = pj.newTrace().fromIndices(id379)
        t379 = x379.vectors()
        p379 = projection.project(t379)

        t0 = pj.newTrace().fromVectors(0)
        x0 = t0.vectors()
        p0 = projection.project(x0)

        # project using NP, results match with Torch
        # model1 = GPT2OperatorNP().name("gpt2-chkpt").loadModel()
        # pj1 = Projector().name('projector_np').model(model1).loadFile()
        # projection1 = pj1.projection()
        #
        # y1 = projection1.project(t379.numpy())
        # y0 = projection1.project(x0.numpy())
        return

    # @skipIf('SKIP' in env, reason="di sabled")
    @skip
    def test910_hack(self):
        model = GPT2Operator().name("gpt2").loadModel()
        pj = Projector().name('projector').model(model).loadFile()

        # prompt = "Alan Turing theorized that computers would one day become"
        # trace = pj.newTrace().fromPrompt("Allen Turing theorized")

        infer = model.inference().prompt("Allen Turing theorized")
        x0 = infer.x()
        x = infer.ssrun("""self 
                wte | wpe |
                lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum | 
                | layer: 1 | layer: 2 | layer: 3 |
                layer: 4 | layer: 5 | layer: 6 | layer: 7 |
                layer: 8 | layer: 9 | layer: 10 | layer: 11 |
                fnorm""")
        x1 = infer.x()
        trace4 = pj.newTrace().fromVectors(x1)
        df = trace4.asDF()

        return

    def test920_hack(self):
        return
