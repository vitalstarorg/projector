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
import smallscript
from pytest import approx
skip = pytest.mark.skip
skipIf = pytest.mark.skipif

import numpy as np
from os import environ as env

from projector.operator.GPT2OperatorNP import *
from projector.utils.About import About
from smallscript import *

# env['SKIPHACK'] = '1'
# env['SKIP'] = '1'

about = About()

#### Following tests are the first implementation to establish the baseline math results using numpy.
class Test_GPT2NP(TestCase):

    @pytest.fixture(autouse=True, scope='class')
    def setup(self):
        self.__class__.prompt1 = "Alan Turing theorized that computers would one day become"
        self.__class__.prompt2 = "Alan Turing theorized that computers would one day become the most powerful"
        pkg = sscontext.loadPackage('projector')

    @skip
    def test100_download_model(self):
        # https://openaipublic.blob.core.windows.net/gpt-2/models
        pass

    @skipIf('SKIP' in env, reason="disabled")
    def test110_load_model(self):
        #### Load the model locally
        # Make sure LLM_MODEL_PATH is defined in env or in .env file
        model = GPT2OperatorNP().name("gpt2-chkpt").loadModel()
        self.__class__.model = model
        assert len(model.blockParams().keys()) > 0

    @skipIf('SKIP' in env, reason="disabled")
    def test120_encode_prompt(self):
        input_ids = self.model.tokenizer().encode(self.prompt2)
        assert len(input_ids) > 0

    @skipIf('SKIP' in env, reason="disabled")
    def test130_decode(self):
        token = self.model.tokenizer().decode([36235])
        assert token == "Alan"
        token = self.model.tokenizer().decode([1716])
        assert token == " become"
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test140_wte_wpe(self):
        #### wte and wpe transformation
        # both wte and wpe are not unit vector
        s = self.model
        wte = s.findParamBySpecs('wte')
        wpe = s.findParamBySpecs('wpe')
        assert wte is not nil
        assert wpe is not nil

        input_ids = self.model.tokenizer().encode(self.prompt2)
        x = wte[input_ids[0]]
        words = self.model.closestWords(x)
        self.assertEqual('Alan', words[0])

        # token + positional embeddings
        x1 = wte[input_ids]
        x2 = wpe[range(len(input_ids))]
        self.__class__.x = x1 + x2

    @skipIf('SKIP' in env, reason="disabled")
    def test150_block_math(self):
        #### Delineate all math in a transformer block
        s = self.model
        x = self.x
        n = self.x.shape[0]

        # multi-head causal self attention
        # Layer Norm with ln_1
        w = s.findParamBySpecs("ln_1.w")
        b = s.findParamBySpecs("ln_1.b")
        eps = 1e-5
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x1 = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
        x2 = w * x1 + b  # scale and offset with gamma/beta params

        # numbers from np implementation,
        # https://openaipublic.blob.core.windows.net/gpt-2/models
        # looks gpt2 tf is a low precision model even it is fp32.
        # The discrepency propagates through calculation and down to 0.1 accuracy using pytorch.
        assert w[0] == about(0.22322033)
        assert b[0] == about(-0.003677325)
        assert x1[0][0] == about(0.08182119)
        assert x2[0][0] == about(0.014586827)

        # Multi-Head Attention
        # qkv projection
        w = s.findParamBySpecs('attn.w')
        b = s.findParamBySpecs('attn.b')
        x3 = x2 @ w + b
        assert (n, 2304) == x3.shape
        assert 0.010422617 == about(x3[0][0], 1e-4)

        # split into qkv
        qkv = np.split(x3, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]
        q, k, v = qkv
        self.assertEqual((n, 768), q.shape)
        self.assertEqual((n, 768), k.shape)
        self.assertEqual((n, 768), v.shape)
        assert 0.010422617 == about(q[0][0])
        assert -1.3149338 == about(k[0][0])
        assert -0.06303753 == about(v[0][0])

        # split into heads
        n_head = s.config().getValue('n_head')
        qkv_heads = list(
            map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
        assert 3 == len(qkv_heads)
        q_heads = qkv_heads[0]
        k_heads = qkv_heads[1]
        v_heads = qkv_heads[2]
        assert 12 == len(q_heads)
        assert (13,64) == q_heads[0].shape
        assert 0.010422617 == about(q_heads[0][0][0])
        assert -1.3149338 == about(k_heads[0][0][0])
        assert -0.06303753 == about(v[0][0])

        # causal mask to hide future inputs from being attended to
        causal_mask = (1 - np.tri(x1.shape[0], dtype=x1.dtype)) * -1e10  # [n_seq, n_seq]
        assert (13,13) == causal_mask.shape
        assert 0.0 == about(causal_mask[0][0])
        assert -1e10 == about(causal_mask[0][1])
        assert 0.0 == about(causal_mask[12].sum())
        # [0 1 1]
        # [0 0 1]
        # [0 0 0]

        # a = [[1,2,3],[4,5,6],[7,8,9]]
        # 1  2  3
        # 4  5  6
        # 7  8  9
        # b = zip(*a)
        # for c1,c2,c3 in b:
        #     print(f"{c1} {c2} {c3}")
        # 1  4  7
        # 2  5  8
        # 3  6  9

        # perform attention over each head
        # first iteration
        out_heads = []
        z = [(q, k, v) for q, k, v in zip(*qkv_heads)]
        z0 = z[0]
        q = z0[0]
        k = z0[1]
        v = z0[2]
        assert (13,64) == q.shape
        assert (13,64) == k.shape
        assert (13,64) == v.shape

        # out = G.softmax(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask) @ v
        out = q @ k.T / np.sqrt(q.shape[-1]) + causal_mask
        assert 0.35876253 == about(out[0][0])
        assert 0.27021736 == about(out[1][0])
        out = s.softmax(out)
        assert 1.0 == about(out[0][0], 1e-2)
        assert 0.52973026 == about(out[1][0])
        out = out @ v
        assert -0.06303753 == about(out[0][0])
        assert -0.06342241 == about(out[1][0])
        out_heads.append(out)
        for i in range(1, len(z)):
            q = z[i][0]
            k = z[i][1]
            v = z[i][2]
            out = s.softmax(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask) @ v
            out_heads.append(out)

        # merge heads
        x4 = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
        assert (13,768) == x4.shape

        # MHA: out projection
        # x1 = linear(x1, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
        w = s.findParamBySpecs('attn | proj.w')
        b = s.findParamBySpecs('attn | proj.b')
        x5 = x4 @ w + b
        assert -0.21653405 == about(x5[0][0])

        x = x + x5
        assert -0.19048978 == about(x[0][0])

        # position-wise feed forward network
        # Layer Norm with ln_2
        w = s.findParamBySpecs("ln_2.w")
        b = s.findParamBySpecs("ln_2.b")
        eps = 1e-5
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x6 = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
        x6 = w * x6 + b  # scale and offset with gamma/beta params
        assert 0.019723138 == about(x6[0][0])

        # x = x + ffn(x2, **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]
        # project up
        w = s.findParamBySpecs("fc.w")
        b = s.findParamBySpecs("fc.b")
        x7 = x6 @ w + b
        assert 0.26455063 == about(x7[0][0])

        # a = gelu(linear(x2, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]
        x8 = 0.5 * x7 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x7 + 0.044715 * x7 ** 3)))
        assert 0.15987226 == about(x8[0][0])

        # project back down
        w = s.findParamBySpecs("mlp | proj.w")
        b = s.findParamBySpecs("mlp | proj.b")
        x9 = x8 @ w + b
        assert 0.52171534 == about(x9[0][0])

        x = x + x9
        assert 0.33122557 == about(x[0][0])
        assert -1.0868139 == about(x[0][767])

    @skipIf('SKIP' in env, reason="disabled")
    def test160_block_model(self):
        #### Assemble these math in different method e.g. layerNorm(), attention(), feedforward(), etc.
        s = self.model
        x = self.x

        # Layer Norm & Attention
        w = s.findParamBySpecs("ln_1.w")
        b = s.findParamBySpecs("ln_1.b")
        x2 = s.layerNorm(x, w, b)
        assert x2[0][0] == about(0.014586827)
        n_head = s.config().getValue('n_head')
        attnw = s.findParamBySpecs('attn.w')
        attnb = s.findParamBySpecs('attn.b')
        projw = s.findParamBySpecs('attn | proj.w')
        projb = s.findParamBySpecs('attn | proj.b')
        x5 = s.attention(x2, attnw, attnb,
                             projw, projb, n_head)
        assert -0.21653405 == about(x5[0][0], 1e-4)

        x = x + x5

        # Layer Norm & Feedforward
        w = s.findParamBySpecs("ln_2.w")
        b = s.findParamBySpecs("ln_2.b")
        x6 = s.layerNorm(x, w, b)
        assert 0.019723138 == about(x6[0][0])
        fcw = s.findParamBySpecs("fc.w")
        fcb = s.findParamBySpecs("fc.b")
        projw = s.findParamBySpecs("mlp | proj.w")
        projb = s.findParamBySpecs("mlp | proj.b")
        x9 = s.feedforward(x6, fcw, fcb, projw, projb)
        assert 0.52171534 == about(x9[0][0])

        x = x + x9
        assert 0.33122557 == about(x[0][0], 1e-2)
        assert -1.0868139 == about(x[0][767], 1e-3)
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test170_block_inference(self):
        #### Manipulate these model method using inference object
        model = self.model
        infer = model.inference().prompt(self.prompt2)
        infer.wte()
        infer.wpe()

        infer.lnorm1(0)
        x2 = infer.delta()
        assert 0.014586827 == about(x2[0][0], 1e-4)
        infer.attn(0)
        x5 = infer.delta()
        assert -0.21653405 == about(x5[0][0], 1e-4)

        infer.sum()
        infer.lnorm2(0)
        x6 = infer.delta()
        assert 0.019723138 == about(x6[0][0])
        infer.ffn(0)
        x9 = infer.delta()
        assert 0.52171534 == about(x9[0][0])

        infer.sum()
        x = infer.x()
        assert 0.33122557 == about(x[0][0])
        assert -1.0868139 == about(x[0][767])

    @skipIf('SKIP' in env, reason="disabled")
    def test180_block_layers(self):
        #### Using an inference object to make a more complicate manipulation
        # forward pass through n_layer transformer blocks
        infer = self.model.inference().prompt(self.prompt2)
        infer.wte().wpe()
        for layer in range(infer.nlayer()):
            infer.lnorm1(layer).attn(layer).sum()
            infer.lnorm2(layer).ffn(layer).sum()
        infer.fnorm()
        assert -127.84109 == about(infer.logits()[-1][0])
        assert 8217 == infer.argmax()[-1]       # greedy sampling
        assert ' machines' == infer.generation()[-1]
            # infer.x should be a unit vector, but wte is not
            # the NNs must learn the scale down factor. Don't know the advantage for this.

        infer.add2inputs()
        infer.wte().wpe()
        for layer in range(infer.nlayer()):
            infer.lnorm1(layer).attn(layer).sum()
            infer.lnorm2(layer).ffn(layer).sum()
        infer.fnorm()
        assert ' on' == infer.generation()[-1]
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test200_smallscript1(self):
        #### Using smallscript for the same manipulation
        infer = self.model.inference().prompt(self.prompt2)
        x2 = infer.ssrun("self wte wpe lnorm1: 0 | delta")
        assert 0.014586827 == about(x2[0][0])
        x5 = infer.ssrun("self attn: 0 | delta")
        assert -0.21653405 == about(x5[0][0], 1e-4)
        x6 = infer.ssrun("self sum lnorm2: 0 | delta")
        assert 0.019723138 == about(x6[0][0])
        x9 = infer.ssrun("self ffn: 0 | delta")
        assert 0.52171534 == about(x9[0][0])
        x = infer.ssrun("self sum x")
        assert 0.33122557 == about(x[0][0])
        assert -1.0868139 == about(x[0][767])

        # inferencing with smallscript
        infer = self.model.inference().prompt(self.prompt2)
        # closure = sscontext.interpret("wte wpe lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum x")
        x = infer.ssrun("self wte | wpe | lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum | x")
        assert 0.33122557 == about(x[0][0])
        assert -1.0868139 == about(x[0][767])

    @skip
    def test210_smallscript2(self):
        #### Fully express two GPT generations end-to-end using smallscript
        # takes 3.009sec vs 667ms using PyTorch
        infer = self.model.inference().prompt(self.prompt2)
        x = infer.ssrun("""self 
                wte | wpe |
                lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum | 
                | layer: 1 | layer: 2 | layer: 3 |
                layer: 4 | layer: 5 | layer: 6 | layer: 7 |
                layer: 8 | layer: 9 | layer: 10 | layer: 11 |
                fnorm""")
        assert -127.84109 == about(infer.logits()[-1][0], 1e-3)
        assert 8217 == infer.argmax()[-1]
        assert ' machines' == infer.generation()[-1]

        geneneration = infer.ssrun("""self 
                add2inputs | wte | wpe |
                layer: 0 | layer: 1 | layer: 2 | layer: 3 |
                layer: 4 | layer: 5 | layer: 6 | layer: 7 |
                layer: 8 | layer: 9 | layer: 10 | layer: 11 |
                fnorm | generation""")
        assert [',', ',', 'ized', ' that', ' the', ' could', ' be', ' day', ' be', ' the', ' most', ' powerful', ' machines', ' on'] == geneneration

    @skip
    def test300_pca(self):
        #### Test the projection calculation
        # takes 6.632 sec vs 275ms using PyTorch
        wte = self.model.modelParams().getValue('wte')
        pca = self.model.project(wte, 3)
        assert (wte.shape[0], 3) == pca.shape

    @skipIf('SKIP' in env, reason="disabled")
    def test310_vocabs(self):
        #### Test model.words()
        vocabs = self.model.words()
        self.assertEqual(' the', vocabs[262])

        # Showing UTF-8 encode and decode
        token = "!¢@"
        nbytes = token.encode('utf-8')
        hex_string = ''.join(f"\\x{byte:02x}" for byte in nbytes)
        self.assertEqual('\\x21\\xc2\\xa2\\x40', hex_string)
        utf8_bytes = bytes.fromhex(hex_string.replace('\\x', ''))
        string = utf8_bytes.decode('utf-8')
        self.assertEqual(token, string)
