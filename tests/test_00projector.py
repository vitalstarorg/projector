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
from pytest import approx
skip = pytest.mark.skip
skipIf = pytest.mark.skipif

import torch

import os
import re
from os import environ as env

from projector.Projector import Projector
from projector.operator.GPT2Operator import GPT2Operator
from projector.operator.GPT2OperatorNP import GPT2OperatorNP
import projector.operator.GPT2EncoderNP as GPT2Encoder
from projector.GPT2Inference import GPT2Inference
from smallscript import *

# env['SKIPHACK'] = '1'
# env['SKIP'] = '1'
# env['CLEAN_CACHE'] = '1'
class Test_Projector(TestCase):
    import lovely_tensors as lt
    lt.monkey_patch()

    @pytest.fixture(autouse=True, scope='class')
    def setup(self):
        pkg = sscontext.loadPackage('projector')
        # model = GPT2Adaptor().name("gpt2/gpt2-124M").load()
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
            model = GPT2OperatorNP().name("gpt2/124M").loadModel()
            pj.model(model)
            projection = pj.project()
            assert (50257,4) == projection.df().shape
            pj.saveFile()

    @skipIf('SKIP' in env, reason="disabled")
    def test310_tensor_project(self):
        # 0.695 sec
        model = GPT2Operator().name("gpt2/gpt2-124M").loadModel()
        self.__class__.model = model
        projector = Projector().name('pj').model(model)
        projection = projector.project()
        assert (50257,4) == projection.df().shape
        self.__class__.projector = projector

    @skipIf('SKIP' in env, reason="disabled")
    def test330_save_load(self):
        projector = self.projector

        projector.name('projector').saveFile()
        pj1 = Projector().name('projector')
        pj1.loadFile()
        df1 = pj1.projection().df()
        assert (50257,4) == df1.shape
        assert '\u2588the' == df1.iloc[262].word
        # assert ' the' == df1.iloc[262].word

        x, y, z = pj1.projection().components()[0]
        # assert -0.024070443585515022 == x
        # assert 0.007738918997347355 == y
        # assert 0.008048399351537228 == z
        assert x == approx(0.02407, 1e-2)
        assert y == approx(0.009943, 1e-2)
        assert z == approx(-0.009700, 1e-2)

        x, y, z = pj1.projection().projected()[0]
        # assert -9.118746757507324 == x
        # assert -5.548634052276611 == y
        # assert -1.481603741645813 == z
        assert x == approx(9.7107, 1e-2)
        assert y == approx(-4.6005, 1e-2)
        assert z == approx(1.3105, 1e-2)

        x = pj1.projection().mean()[0]
        assert x == approx(-0.002549, 1e-2)

        x = pj1.projection().std()[0]
        assert x == approx(0.1126, 1e-2)

    # @skipIf('SKIP' in env, reason="disabled")
    # @skip
    def test340_project_vec_np(self):
        modelNP = GPT2OperatorNP().name("gpt2/124M").loadModel()
        self.__class__.modelNP = modelNP
        pj2 = Projector().name('projector_np').model(modelNP).loadFile()
        projection2 = pj2.projection()

        assert [0.01985565, 0.01102482, 0.00954412] == approx(projection2.varianceRatio(), 1e-3)

        tokenSpec = modelNP.searchTokens('Token').head()
        codeToken = tokenSpec['id']
        assert 30642 == codeToken
        vector2 = tokenSpec['vector']
        assert 0.015963181853294373 == vector2[0]

        projected2 = projection2.project(vector2)
        assert -4.281162738800049 == approx(projected2[0][0], 1e-3)
        assert 3.5077905654907227 == approx(projected2[0][1], 1e-2)
        assert 0.3020952641963959 == approx(projected2[0][2], 1e-1)

    @skipIf('SKIP' in env, reason="disabled")
    def test350_project_vec(self):
        # model = GPT2Operator().name("gpt2/gpt2-124M").loadModel()
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()
        projection = pj.projection()

        # assert projection.varianceRatio() == approx([0.01986, 0.01102, 0.009544], 1e-3)
        assert [0.01855, 0.01063, 0.009336] == approx(projection.varianceRatio(), 1e-3)

        tokenSpec = model.searchTokens('Token').head()
        codeToken = tokenSpec['id']
        assert 30642 == codeToken
        vector = tokenSpec['vector']
        assert vector[0] == approx(0.01596, 1e-3)

        projected = projection.project(vector)
        # assert 4.281162738800049 == projected[0][0]
        # assert -3.5077905654907227 == approx(projected[0][1], 1e-6)
        # assert 0.3020952641963959 == approx(projected[0][2], 1e-6)
        assert projected[0][0] == approx(-3.8256, 1e-3)
        assert projected[0][1] == approx(-3.6127, 1e-3)
        assert projected[0][2] == approx(0.2413, 1e-3)


    @skipIf('SKIP' in env, reason="disabled")
    def test400_show(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()
        pj.showEmbedding()

    @skipIf('SKIP' in env, reason="disabled")
    def test410_zero(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()

        # Project Zero
        zeros = pj.newTrace().fromVectors(0)
        # assert approx(zeros.projected(), 1e-4) == [[-100.650, 31.910, 33.055]]
        assert approx(zeros.projected(), 1e-3) == [[11.9618, -3.7794, 2.6324]]
        words = zeros.closestWords()
        assert [[379]] == zeros.closestIndices().tolist()
        assert [[379]] == zeros.knn_ids().tolist()
        assert approx(1, 1e-4) == zeros.knn_sims().squeeze().item()
        assert approx(0.028, 1e-4) == zeros.closestAngles().squeeze().item()
            # fp32 precision issues
        assert ['\u2588at'] == words
        assert (1,8) == zeros.asDF().shape
        zeros.asDF()

        # Project 5 neigbhours around zero
        ids = zeros.knn(5).knn_ids()
        zeroKNN = pj.newTrace().fromIndices(ids)
        assert (5,8) == zeroKNN.asDF().shape
        assert [379, 287, 319, 281, 329] == zeroKNN.knn_ids().squeeze().tolist()
        assert approx([1, 1, 1, 1, 1], 1e-4) == zeroKNN.knn_sims().squeeze().tolist()
        assert approx([0.0485, 0.0593, 0.0442, 0., 0.], 1e-4) == zeroKNN.closestAngles().squeeze().tolist()
            # fp32 precision issues
        words = zeroKNN.closestWords()
        assert ['\u2588at', '\u2588in', '\u2588on', '\u2588an', '\u2588for'] == words
        assert ['\u2588at', '\u2588in', '\u2588on', '\u2588an', '\u2588for'] == zeroKNN.asDF()['word'].tolist()

    @skipIf('SKIP' in env, reason="disabled")
    def test420_trace(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()

        # knn for all 4 vectors
        trace = pj.newTrace().fromPrompt("Allen Turing theorized")
        vectors = trace.vectors()
        trace1 = pj.newTrace().fromVectors(vectors)
        trace1.knn(5)
        ids = trace1.knn_ids()
        sims = trace1.knn_sims()
        assert (4,5) == ids.shape
        assert (4,5) == sims.shape
        words = trace1.closestWords()
        assert ['Allen', '\u2588Turing', '\u2588theor', 'ized'] == words
        words = trace1.closestWords(dim=1)
        assert ['Allen', '\u2588Allen', '\x16', '\\xf7', '\\xfb'] == words

        knn1 = pj.newTrace().fromIndices(trace1.knn_ids()[0,:])
        assert ['Allen', '\u2588Allen', '\x16', '\\xf7', '\\xfb'] == knn1.asDF()['word'].tolist()

        # knn on 2nd vectors i.e. 'Turing'
        trace2 = pj.newTrace().fromVectors(vectors[1])
        trace2.knn(5)
        words = trace2.closestWords()
        assert ['\u2588Turing'] == words
        words = trace2.closestWords(dim = 1)
        assert ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588'] == words

        knn2 = pj.newTrace().fromIndices(trace2.knn_ids())
        assert ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588'] == knn2.asDF()['word'].tolist()

        # knn on 3rd vectors i.e. 'theor'
        trace3 = pj.newTrace().fromVectors(vectors[2])
        trace3.knn(5)
        words = trace3.closestWords()
        assert ['\u2588theor'] == words
        words = trace3.closestWords(dim = 1)
        assert ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588speculated', '\u2588theories'] == words

        knn3 = pj.newTrace().fromIndices(trace3.knn_ids())
        assert ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588speculated', '\u2588theories'] == knn3.asDF()['word'].tolist()

        return

    @skipIf('SKIP' in env, reason="disabled")
    def test430_show(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadFile()

        # simulate the show()
        trace = pj.newTrace().fromPrompt("Allen Turing theorized")
        assert len(pj.figure().data) == 0
        trace.show()
        assert len(pj.figure().data) == 1
        trace.remove()
        assert len(pj.figure().data) == 0

        # simulate the highlight
        tokenVec = trace.vectors()[1]
        highlight = pj.updateHighlight(tokenVec, 10)
        df = highlight.asDF()

    def test900_hack(self):
        model = GPT2Operator().name("gpt2/gpt2-124M").loadModel()
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
        # model1 = GPT2OperatorNP().name("gpt2/124M").loadModel()
        # pj1 = Projector().name('projector_np').model(model1).loadFile()
        # projection1 = pj1.projection()
        #
        # y1 = projection1.project(t379.numpy())
        # y0 = projection1.project(x0.numpy())
        return

    # @skipIf('SKIP' in env, reason="di sabled")
    @skip
    def test910_hack(self):
        model = GPT2Operator().name("gpt2/gpt2-124M").loadModel()
        pj = Projector().name('projector').model(model).loadFile()

        # zeros = pj.newTrace().fromVectors(0)
        # # assert approx(zeros.projected(), 1e-4) == [[-100.650, 31.910, 33.055]]
        # assert approx(zeros.projected(), 1e-3) == [[11.9618, -3.7794, 2.6324]]
        # words = zeros.closestWords()
        # assert [[379]] == zeros.closestIndices().tolist()
        # assert [[379]] == zeros.knn_ids().tolist()
        # assert approx(1, 1e-4) == zeros.knn_sims().squeeze().item()
        # assert approx(0.028, 1e-4) == zeros.closestAngles().squeeze().item()
        #     # fp32 precision issues
        # assert ['\u2588at'] == words
        # assert (1,8) == zeros.asDF().shape
        # zeros.asDF()
        #
        # ids = zeros.knn(5).knn_ids()
        # zeroKNN = pj.newTrace().fromIndices(ids)
        # assert (5,8) == zeroKNN.asDF().shape
        # assert [379, 287, 319, 281, 329] == zeroKNN.knn_ids().squeeze().tolist()
        # assert approx([1, 1, 1, 1, 1], 1e-4) == zeroKNN.knn_sims().squeeze().tolist()
        # assert approx([0.0485, 0.0593, 0.0442, 0., 0.], 1e-4) == zeroKNN.closestAngles().squeeze().tolist()
        #     # fp32 precision issues
        # words = zeroKNN.closestWords()
        # assert ['\u2588at', '\u2588in', '\u2588on', '\u2588an', '\u2588for'] == words
        # assert ['\u2588at', '\u2588in', '\u2588on', '\u2588an', '\u2588for'] == zeroKNN.asDF()['word'].tolist()

        # knn for all 4 vectors
        trace = pj.newTrace().fromPrompt("Allen Turing theorized")
        vectors = trace.vectors()
        trace1 = pj.newTrace().fromVectors(vectors)
        trace1.knn(5)
        ids = trace1.knn_ids()
        sims = trace1.knn_sims()
        assert (4,5) == ids.shape
        assert (4,5) == sims.shape
        words = trace1.closestWords()
        assert ['Allen', '\u2588Turing', '\u2588theor', 'ized'] == words
        words = trace1.closestWords(dim=1)
        assert ['Allen', '\u2588Allen', '\x16', '\\xf7', '\\xfb'] == words

        knn1 = pj.newTrace().fromIndices(trace1.knn_ids()[0,:])
        assert ['Allen', '\u2588Allen', '\x16', '\\xf7', '\\xfb'] == knn1.asDF()['word'].tolist()

        # knn on 2nd vectors i.e. 'Turing'
        trace2 = pj.newTrace().fromVectors(vectors[1])
        trace2.knn(5)
        words = trace2.closestWords()
        assert ['\u2588Turing'] == words
        words = trace2.closestWords(dim = 1)
        assert ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588'] == words

        knn2 = pj.newTrace().fromIndices(trace2.knn_ids())
        assert ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588'] == knn2.asDF()['word'].tolist()

        # knn on 3rd vectors i.e. 'theor'
        trace3 = pj.newTrace().fromVectors(vectors[2])
        trace3.knn(5)
        words = trace3.closestWords()
        assert ['\u2588theor'] == words
        words = trace3.closestWords(dim = 1)
        assert ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588speculated', '\u2588theories'] == words

        knn3 = pj.newTrace().fromIndices(trace3.knn_ids())
        assert ['\u2588theor', '\u2588hypothes', '\u2588hypothesized', '\u2588speculated', '\u2588theories'] == knn3.asDF()['word'].tolist()

        #
        tokenVec = trace.vectors()[1]
        highlight = pj.highlight()
        if highlight.notNil():
            highlight.remove()
        tokenTrace = pj.newTrace().fromVectors(tokenVec)
        knn_ids = tokenTrace.knn(10).knn_ids()
        highlight = pj.newTrace().fromIndices(knn_ids)
        highlight.colorRoll().color('blue')
        pj.highlight(highlight)
        df = highlight.asDF()
        print(df.to_string())
        highlight.show()

        # prompt = "Alan Turing theorized that computers would one day become"
        # trace = pj.newTrace().fromPrompt("Allen Turing theorized")

        infer = model.inference().prompt("Allen Turing theorized")
        x = infer.ssrun("self wte | wpe | x")
        trace4 = pj.newTrace().fromVectors(x)
        df = trace4.asDF()

        return


        return
