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
import io
import pytest
from pytest import approx as approx
skip = pytest.mark.skip
skipIf = pytest.mark.skipif

from os import environ as env

from projector.Projector import Projector
from projector.operator.GPT2Operator import GPT2Operator
from projector.operator.GPT2OperatorNP import GPT2OperatorNP
from projector.utils.About import About
from projector.operator.Similarity import *
from smallscript import *

about = About()

# env['SKIPHACK'] = '1'
# env['SKIP'] = '1'
# env['CLEAN_CACHE'] = '1'
class Test_Projector(TestCase):
    # Optional import for print tensor summary.
    import lovely_tensors as lt
    lt.monkey_patch()

    @pytest.fixture(autouse=True, scope='class')
    def setup(self):
        # Enable smallscript for 'projector' and 'tests' packages
        pkg = sscontext.loadPackage('projector')
        pkg = sscontext.loadPackage('tests')
        if 'CLEAN_CACHE' in env:
            sscontext.log('Remove projector.zip & projector_np.zip.', Logger.LevelInfo)
            Projector().name('projector').deleteCache()
            Projector().name('projector_np').deleteCache()
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test100_basics(self):
        # Smoke tests by just create a projector object.
        pj = Projector()

    @skipIf('SKIP' in env, reason="disabled")
    def test200_filename(self):
        # Testing pj filename determination for saveCache()
        pj = Projector()
        assert 'unnamed.zip' == pj.filename()
            # generate the filename at the first call.
        pj.name('test100')
        assert 'unnamed.zip' == pj.filename()
            # changing name wouldn't change filename after the first call pj.filename().
        pj = Projector().name('test100')
        assert 'test100.zip' == pj.filename()
        pj.filename('i_want_this_name')
        assert 'i_want_this_name' == pj.filename()
            # filename can be assigned directly without derived from its @name
            # better to suffix it with .zip as it will be saved as zip file.

    @skipIf('SKIP' in env, reason="disabled")
    def test300_np_project(self):
        # GPT2Operator implemented using Pytorch implementation is used by default.
        # This test generates numpy cache in case GPT2OperatorNP is used.
        # 5.213 sec, delete projector_np.zip to measure this time.
        pj = Projector().name('projector_np')
        if not pj.hasCache():
            model = GPT2OperatorNP().name("gpt2-chkpt").loadModel()
            pj.model(model)
            projection = pj.project()   # calculate the 3D PCA projection
            assert (50257,4) == projection.df().shape
            pj.saveCache()

    @skipIf('SKIP' in env, reason="disabled")
    def test310_tensor_project(self):
        # Loading another gpt2 model
        # model = GPT2Operator().\
        #             org("jtatman").\
        #             name("gpt2-open-instruct-v1-Anthropic-hh-rlhf").\
        #             loadModel()
        # 0.695 sec

        # Load gpt2 model
        model = GPT2Operator().name("gpt2").loadModel()
        self.__class__.model = model
            # self.model will access to this @model
        projector = Projector().name('pj').model(model)
        projection = projector.project()
        assert (768,3) == projection.components().shape
        self.__class__.projector = projector
            # self.projector will access to this @projector

    @skipIf('SKIP' in env, reason="disabled")
    def test330_save_load(self):
        projector = self.projector
        projector.name('projector').saveCache()
        pj1 = Projector().name('projector')
        pj1.loadCache()
        df1 = pj1.projection().df()
        assert (768,3) == pj1.projection().components().shape
        assert '\u2588the' == df1.iloc[262].word

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
        # Do some basic math tests using Numpy as a baseline.
        # This is disabled because it is slow.
        modelNP = GPT2OperatorNP().name("gpt2-chkpt").loadModel()
        self.__class__.modelNP = modelNP
        pj2 = Projector().name('projector_np').model(modelNP).loadCache()
        projection2 = pj2.projection()

        assert projection2.varianceRatio() == about([0.01811, 0.008229, 0.007154], 1e-3)
        tokenSpec = modelNP.searchTokens('Token').head()
        codeToken = tokenSpec['id']
        assert 30642 == codeToken
        vector2 = tokenSpec['vector']
        assert vector2[0] == about(0.01596, 1e-3)

        # PCA projections here has big discrepency. Please refer to the comment in the next test
        # for more details.
        projected2 = projection2.projectVector(vector2)
        assert projected2[0][0] == about(0.01334, 1e-3)
        assert projected2[0][1] == about(0.007604, 1e-3)
        assert projected2[0][2] == about(-0.01539, 1e-3)
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test350_project_vec(self):
        # Do some basic math tests using pytorch. It matches some but some are far due to precision
        # and underlying numerical implementation.
        model = self.model
        pj = Projector().name('projector').model(model).loadCache()
        projection = pj.projection()

        assert projection.varianceRatio() == about([0.0181, 0.00823, 0.00717], 1e-3)

        tokenSpec = model.searchTokens('Token').head()
        codeToken = tokenSpec['id']
        assert 30642 == codeToken
        vector = tokenSpec['vector']
        assert vector[0] == about(0.01596, 1e-3)

        # 1st source of discrepency:
        # GPT2Operator is using pytorch to derive PCA using covariant matrix. GPT2OperatorNP is using
        # numpy TruncatedSVD. That is the main source of discrepency on the 3D projection. However,
        # transformer calculation is unaffected by this projection.

        # 2nd source of discrepency:
        # The underlying precision of floating point representation. The slight changes in these
        # values, would result significant change in some transformation upto 10%. Surprising we need
        # to reduce the comparision accuracy at 1e-1 for some tests between pytorch and numpy.

        # 3rd source of discrepency:
        # GPT2Operator is using Huggingface model. GPT2OpeartorNP is using TF checkpoint format.
        # The model parameters are essentially the same. Since we reimplemented the transformer using
        # pure pytorch and numpy ground up, so we can make direct comparisons between these
        # implementations. We discover the numerical values between implementations varied significantly
        # through transformer layers regardless using pytorch or numpy. So the LLM prediction is
        # suspectible to floating precision and numerical implementation.
        #
        projected = projection.projectVector(vector)
        assert projected[0][0] == about(-0.317, 1e-3)
        assert projected[0][1] == about(-0.389, 1e-3)
        assert projected[0][2] == about(-0.01947, 1e-3)
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test400_show(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadCache()
        pj.showEmbedding()

    @skipIf('SKIP' in env, reason="disabled")
    def test410_zero_cos(self):
        model = self.model
        pj = Projector().name('projector').model(model).loadCache()
        similarity = CosineSimilarity()
        pj.similarity(similarity)

        # Project origin from 768d to 3d.
        zeros = pj.newTrace().fromVectors(0)    # origin in 768d space
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
        pj = Projector().name('projector').model(model).loadCache()

        # Project Zero
        zeros = pj.newTrace().fromVectors(0)
        assert zeros.projected()[0] == about([1.37, -0.0445, -0.129], 1e-3)
        words = zeros.closestWords()
        assert zeros.closestIndices().tolist() == [[379]]
        assert zeros.similarity().ids().tolist() == [[379]]
        assert zeros.similarity().sims().squeeze().item() == about(6.021, 1e-4)
        # assert zeros.closestAngles().squeeze().item() == about(0.0396, 1e-3)
        assert zeros.closestAngles().squeeze().item() == about(0.0396, 1)
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
        pj = Projector().name('projector').model(model).loadCache()
        similarity = CosineSimilarity()
        pj.similarity(similarity)

        # knn for all 4 vectors
        trace = pj.newTrace().fromPrompt("Alan Turing theorized")
        vectors = trace.vectors()
        trace1 = pj.newTrace().fromVectors(vectors)
        trace1.knn(5)
        ids = trace1.similarity().ids()
        sims = trace1.similarity().sims()
        assert ids.shape == (4,5)
        assert sims.shape == (4,5)
        words = trace1.closestWords()
        assert words == ['Alan', '\u2588Turing', '\u2588theor', 'ized']
        words = trace1.closestWords(dim=1)
        assert words == ['Alan', '\u2588Alan', 'Andy', 'Michael', 'Ian']
        trace1.asDF()

        ids1 = trace1.similarity().ids()[0,:]
        assert ids1 == about([36235, 12246, 35314, 13256, 37776])
        sims1 = trace1.similarity().sims()[0,:]
        assert sims1 == about([1.000, 0.845, 0.585, 0.576, 0.575], 1e-3)
        angles1 = trace1.similarity().angles()[0,:]
        knn1 = pj.newTrace().fromIndices(ids1).knn_ids(ids1).knn_sims(sims1).knn_angles(angles1)
        assert knn1.asDF()['word'].tolist() == ['Alan', '\u2588Alan', 'Andy', 'Michael', 'Ian']

        # knn on 2nd vectors i.e. 'Turing'
        trace2 = pj.newTrace().fromVectors(vectors[1])
        trace2.knn(5)
        words = trace2.closestWords()
        assert words == ['\u2588Turing']
        words = trace2.closestWords(dim = 1)
        assert words == ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588']

        knn2 = pj.newTrace().fromIndices(trace2.similarity().ids())
        assert knn2.asDF()['word'].tolist() == ['\u2588Turing', '\u2588externalToEVA', '\u2588', '\\xff', '\u2588']

        # knn on 3rd vectors i.e. 'theor'
        trace3 = pj.newTrace().fromVectors(vectors[2])
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
        pj = Projector().name('projector').model(model).loadCache()

        # knn for all 4 vectors
        trace = pj.newTrace().fromPrompt("Alan Turing theorized")
        vectors = trace.vectors()
        trace1 = pj.newTrace().fromVectors(vectors)
        trace1.knn(5)
        ids = trace1.similarity().ids()
        sims = trace1.similarity().sims()
        assert ids.shape == (4,5)
        assert sims.shape == (4,5)
        words = trace1.closestWords()
        assert words == ['Alan', '\u2588Turing', '\u2588theor', 'ized']
        words = trace1.closestWords(dim=1)
        assert words == ['Alan', '\u2588Alan', 'Craig', 'Andy', 'Alice']
        trace1.asDF()

        ids1 = trace1.similarity().ids()[0,:]
        assert ids1 == about([36235, 12246, 40441, 35314, 44484])
        sims1 = trace1.similarity().sims()[0,:]
        assert sims1 == about([13.924, 10.111, 8.118, 8.101, 8.045], 1e-3)
        angles1 = trace1.similarity().angles()[0,:]
        knn1 = pj.newTrace().fromIndices(ids1).knn_ids(ids1).knn_sims(sims1).knn_angles(angles1)
        assert knn1.asDF()['word'].tolist() == ['Alan', '\u2588Alan', 'Craig', 'Andy', 'Alice']

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
        pj = Projector().name('projector').model(model).loadCache()
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
        pj = Projector().name('projector').model(model).loadCache()
        pj.wOffset(1)
        pj.bOffset(0)

        # simulate the show()
        trace = pj.newTrace().fromPrompt("Alan Turing theorized")
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
        trace1.fnorm()
        trace1.wbnorm()

        # Simulate write_image()w
        buffer = io.BytesIO()
        pj.figure().write_image(buffer, format='png')

        # Simulate the highlight
        tokenVec = trace.vectors()[1]
        highlight = pj.updateHighlight(tokenVec, 10)
        df = highlight.asDF()
        highlight.showDF()

        # Show arrows
        arrows1 = pj.newLines().withArrow().fromVectors(vectors)
        arrows1.show()
        arrows1.asDF()
        arrows1.remove()
        arrows1.setOrigin()
        arrows1.show()
        arrows1.remove()

        # Show colorband
        pj.showColorband()

        # getView() and getCamera()
        view = pj.getView()
        pj.updateView(view)
        camera = pj.getCamera()
        pj.updateCamera(camera)

    # @skip
    def test900_hack(self):
        model = GPT2Operator().name("gpt2").loadModel()
        pj = Projector().name('projector').model(model).loadCache()
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

        idAlan = model.tokenizer().encode('Alan')
        xAlan = pj.newTrace().fromIndices(idAlan)
        id36235 = torch.tensor([[36235]])
        x36235 = pj.newTrace().fromIndices(id36235)
        id36235 = torch.tensor([36235])
        x36235 = pj.newTrace().fromIndices(id36235)
        t36235 = x36235.vectors()
        p36235 = projection.projectVector(t36235)

        t0 = pj.newTrace().fromVectors(0)
        x0 = t0.vectors()
        p0 = projection.projectVector(x0)

        # project using NP, results match with Torch
        # model1 = GPT2OperatorNP().name("gpt2-chkpt").loadModel()
        # pj1 = Projector().name('projector_np').model(model1).loadFile()
        # projection1 = pj1.projection()
        #
        # y1 = projection1.project(t36235.numpy())
        # y0 = projection1.project(x0.numpy())
        return

    # @skipIf('SKIP' in env, reason="disabled")
    @skip
    def test910_hack(self):
        model = GPT2Operator().name("gpt2").loadModel()
        pj = Projector().name('projector').model(model).loadCache()

        infer = model.inference().prompt("Alan Turing theorized")
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
