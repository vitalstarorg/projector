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

from projector.operator.GPT2Operator import GPT2Operator
from projector.GPT2Inference import GPT2Inference
from smallscript import *

# env['SKIPHACK'] = '1'
# env['SKIP'] = '1'
class Test_GPT2(TestCase):
    import lovely_tensors as lt
    lt.monkey_patch()

    @pytest.fixture(autouse=True, scope='class')
    def setup(self):
        self.__class__.prompt1 = "Alan Turing theorized that computers would one day become"
        self.__class__.prompt2 = "Alan Turing theorized that computers would one day become the most powerful"
        from dotenv import load_dotenv
        pkg = sscontext.loadPackage('projector')
        load_dotenv("../.env")
        return

    @skip
    def test100_delete_download_save(self):
        model = GPT2Operator().name("gpt2")     # smallest GPT model from HF
        assert model.path() is not nil
        model.deleteModel()
        assert not model.path().exists()
        model.downloadModel()
        self.__class__.smodel = model
        model.saveModel()

        model = GPT2Operator().org("jtatman").name("gpt2-open-instruct-v1-Anthropic-hh-rlhf")
        model.deleteModel()
        model.downloadModel()
        model.saveModel()

    @skip
    def test105_download_model(self):
        from transformers import AutoModel, AutoTokenizer

        # download and save model locally
        # model_id = "gpt2-124M"
        # model_name = f"gpt2/{model_id}"
        # https://huggingface.co/microsoft/phi-1_5/tree/main?show_file_info=model.safetensors
        model_id = 'phi-1_5'
        model_name = f"microsoft/{model_id}"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(f"./{model_id}")  # Save model locally
        tokenizer.save_pretrained(f"./{model_id}")  # Save tokenizer locally

    # @skip
    def test110_load_model(self):
        smodel = GPT2Operator().name("gpt2").loadModel()
        self.__class__.smodel = smodel
        assert smodel.notNil()
        assert len(smodel.state().keys()) > 0

    @skipIf('SKIP' in env, reason="disabled")
    def test120_encode_prompt(self):
        input_ids = self.smodel.tokenizer().encode(self.prompt2)
        assert len(input_ids) > 0

    @skipIf('SKIP' in env, reason="disabled")
    def test130_decode(self):
        token = self.smodel.tokenizer().decode([36235])
        assert token == "Alan"
        token = self.smodel.tokenizer().decode([1716])
        assert token == " become"
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test140_wte_wpe(self):
        s = self.smodel
        wte = s.findParamBySpecs('wte')
        wpe = s.findParamBySpecs('wpe')
        assert wte != nil
        assert wpe != nil

        input_ids = self.smodel.tokenizer().encode(self.prompt2)
        x = wte[input_ids[0]]
        words = self.smodel.closestWords(x)
        self.assertEqual('Alan', words[0])

        x1 = wte[input_ids]
        x2 = wpe[range(len(input_ids))]
        self.__class__.x = x1 + x2

    @skipIf('SKIP' in env, reason="disabled")
    def test150_block_math1(self):
        s = self.smodel
        x = self.x
        n = self.x.shape[0]

        # Layer Norm with ln_1
        w = s.findParamBySpecs("ln_1.w")
        b = s.findParamBySpecs("ln_1.b")
        eps = 1e-5
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        x1 = (x - mean) / torch.sqrt(variance + eps)
        x2 = w * x1 + b

        # numbers from np implementation,
        # https://openaipublic.blob.core.windows.net/gpt-2/models
        # looks gpt2 tf is a low precision model even it is fp32.
        # The discrepency propagates through calculation and down to 0.1 accuracy.
        # These tests are good enough to confirm the calculation using PyTorch.
        assert w[0] == approx(0.22322033)
        assert b[0] == approx(-0.003677325)
        assert x1[0][0] == approx(0.08182119, 1e-3)
        assert x2[0][0] == approx(0.014586827, 1e-3)

        # Multi-Head Attention
        # qkv projection
        w = s.findParamBySpecs('attn.w')
        b = s.findParamBySpecs('attn.b')
        x3 = x2 @ w + b
        assert (n, 2304) == x3.shape
        assert 0.010422617 == approx(x3[0][0], 1e-1)

        # split into qkv
        qkv = List(torch.split(x3, x3.shape[-1] // 3, dim=-1))
        q, k, v = qkv
        self.assertEqual((n, 768), q.shape)
        self.assertEqual((n, 768), k.shape)
        self.assertEqual((n, 768), v.shape)
        assert 0.010422617 == approx(q[0][0], 1e-1)
        assert -1.3149338 == approx(k[0][0], 1e-3)
        assert -0.06303753 == approx(v[0][0], 1e-3)

        # split into heads
        n_head = s.config().getValue('n_head')
        qkv_heads = list(map(lambda x: List(torch.split(x, x.size(-1) // n_head, dim=-1)), qkv))
        assert 3 == len(qkv_heads)
        q_heads = qkv_heads[0]
        k_heads = qkv_heads[1]
        v_heads = qkv_heads[2]
        assert 12 == q_heads.len()
        assert (13,64) == q_heads[0].shape
        assert 0.010422617 == approx(q_heads[0][0][0], 1e-1)
        assert -1.3149338 == approx(k_heads[0][0][0], 1e-3)
        assert -0.06303753 == approx(v[0][0], 1e-3)

        # causal mask to hide future inputs from being attended to
        triMatrix = torch.tril(torch.ones(x3.shape[0], x3.shape[0], dtype=x3.dtype))
        causal_mask = (1 - triMatrix) * -1e10
        assert (13,13) == causal_mask.shape
        assert 0.0 == approx(causal_mask[0][0])
        assert -1e10 == approx(causal_mask[0][1])
        assert 0.0 == approx(causal_mask[12].sum())

        # perform attention over each head
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
        scaling = torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype))
        out = (q @ k.transpose(-1,-2) / scaling + causal_mask)
        assert 0.35876253 == approx(out[0][0], 1e-2)
        assert 0.27021736 == approx(out[1][0], 1e-2)
        out = s.softmax(out)
        assert 1.0 == approx(out[0][0], 1e-2)
        assert 0.52973026 == approx(out[1][0], 1e-3)
        out = out @ v
        assert -0.06303753 == approx(out[0][0], 1e-3)
        assert -0.06342241 == approx(out[1][0], 1e-3)
        out_heads.append(out)
        for i in range(1, len(z)):
            q = z[i][0]
            k = z[i][1]
            v = z[i][2]
            out = s.softmax(q @ k.transpose(-1,-2) / scaling + causal_mask) @ v
            out_heads.append(out)

        # merge heads
        x4 = torch.cat(out_heads, dim=1)
        assert (13,768) == x4.shape

        # MHA: out projection
        w = s.findParamBySpecs('attn | proj.w')
        b = s.findParamBySpecs('attn | proj.b')
        x5 = x4 @ w + b
        assert -0.21653405 == approx(x5[0][0], 1e-2)

        x = x + x5
        assert -0.19048978 == approx(x[0][0], 1e-2)

        # position-wise feed forward network
        # Layer Norm with ln_2
        w = s.findParamBySpecs("ln_2.w")
        b = s.findParamBySpecs("ln_2.b")
        eps = 1e-5
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        x6 = (x - mean) / torch.sqrt(variance + eps)
        x6 = w * x6 + b
        assert 0.019723138 == approx(x6[0][0], 1e-2)

        w = s.findParamBySpecs("fc.w")
        b = s.findParamBySpecs("fc.b")
        x7 = x6 @ w + b
        assert 0.26455063 == approx(x7[0][0], 1e-2)

        sqrt_2_pi = torch.sqrt(torch.tensor(2 / torch.pi))  # Equivalent to np.sqrt(2 / np.pi)
        term = x7 + 0.044715 * x7 ** 3
        tanh_term = torch.tanh(sqrt_2_pi * term)
        x8 = 0.5 * x7 * (1 + tanh_term)
        assert 0.15987226 == approx(x8[0][0], 1e-3)

        # project back down
        w = s.findParamBySpecs("mlp | proj.w")
        b = s.findParamBySpecs("mlp | proj.b")
        x9 = x8 @ w + b
        assert 0.52171534 == approx(x9[0][0], 1e-2)

        x = x + x9
        assert 0.33122557 == approx(x[0][0], 1e-2)
        assert -1.0868139 == approx(x[0][767], 1e-3)

    @skipIf('SKIP' in env, reason="disabled")
    def test160_block_model(self):
        s = self.smodel
        x = self.x

        # Layer Norm & Attention
        w = s.findParamBySpecs("ln_1.w")
        b = s.findParamBySpecs("ln_1.b")
        x2 = s.layerNorm(x, w, b)
        assert x2[0][0] == approx(0.014586827, 1e-3)
        n_head = s.config().getValue('n_head')
        attnw = s.findParamBySpecs('attn.w')
        attnb = s.findParamBySpecs('attn.b')
        projw = s.findParamBySpecs('attn | proj.w')
        projb = s.findParamBySpecs('attn | proj.b')
        x5 = s.attention(x2, attnw, attnb,
                             projw, projb, n_head)
        assert -0.21653405 == approx(x5[0][0], 1e-2)

        x = x + x5

        # Layer Norm & Feedforward
        w = s.findParamBySpecs("ln_2.w")
        b = s.findParamBySpecs("ln_2.b")
        x6 = s.layerNorm(x, w, b)
        assert 0.019723138 == approx(x6[0][0], 1e-2)
        fcw = s.findParamBySpecs("fc.w")
        fcb = s.findParamBySpecs("fc.b")
        projw = s.findParamBySpecs("mlp | proj.w")
        projb = s.findParamBySpecs("mlp | proj.b")
        x9 = s.feedforward(x6, fcw, fcb, projw, projb)
        assert 0.52171534 == approx(x9[0][0], 1e-2)

        x = x + x9
        assert 0.33122557 == approx(x[0][0], 1e-2)
        assert -1.0868139 == approx(x[0][767], 1e-3)

    @skipIf('SKIP' in env, reason="disabled")
    def test170_block_inference(self):
        smodel = self.smodel

        infer = smodel.inference().prompt(self.prompt2)
        infer.wte()
        infer.wpe()

        infer.lnorm1(0)
        x2 = infer.delta()
        assert 0.014586827 == approx(x2[0][0], 1e-3)
        infer.attn(0)
        x5 = infer.delta()
        assert -0.21653405 == approx(x5[0][0], 1e-2)

        infer.sum()
        infer.lnorm2(0)
        x6 = infer.delta()
        assert 0.019723138 == approx(x6[0][0], 1e-2)
        infer.ffn(0)
        x9 = infer.delta()
        assert 0.52171534 == approx(x9[0][0], 1e-2)

        infer.sum()
        x = infer.x()
        assert 0.33122557 == approx(x[0][0], 1e-2)
        assert -1.0868139 == approx(x[0][767], 1e-3)

    @skipIf('SKIP' in env, reason="disabled")
    def test180_block_layers(self):
        smodel = self.smodel

        infer = smodel.inference().prompt(self.prompt2)
        infer.wte().wpe()
        for layer in range(infer.nlayer()):
            infer.lnorm1(layer).attn(layer).sum()
            infer.lnorm2(layer).ffn(layer).sum()
        infer.fnorm()
        assert -127.84109 == approx(infer.logits()[-1][0], 1e-3)
        assert 8217 == infer.argmax()[-1]
        assert ' machines' == infer.generation()[-1]

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
        smodel = self.smodel

        infer = smodel.inference().prompt(self.prompt2)
        x2 = infer.ssrun("self wte wpe lnorm1: 0 | delta")
        assert 0.014586827 == approx(x2[0][0], 1e-3)
        x5 = infer.ssrun("self attn: 0 | delta")
        assert -0.21653405 == approx(x5[0][0], 1e-2)
        x6 = infer.ssrun("self sum lnorm2: 0 | delta")
        assert 0.019723138 == approx(x6[0][0], 1e-2)
        x9 = infer.ssrun("self ffn: 0 | delta")
        assert 0.52171534 == approx(x9[0][0], 1e-2)
        x = infer.ssrun("self sum x")
        assert 0.33122557 == approx(x[0][0], 1e-2)
        assert -1.0868139 == approx(x[0][767], 1e-3)

        # inferencing with smallscript
        infer = smodel.inference().prompt(self.prompt2)
        # closure = sscontext.interpret("wte wpe lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum x")
        x = infer.ssrun("self wte | wpe | lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum | x")
        assert 0.33122557 == approx(x[0][0], 1e-2)
        assert -1.0868139 == approx(x[0][767], 1e-3)

    @skipIf('SKIP' in env, reason="disabled")
    def test210_smallscript2(self):
        smodel = self.smodel

        infer = smodel.inference().prompt(self.prompt2)
        x = infer.ssrun("""self 
                wte | wpe |
                lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum | 
                | layer: 1 | layer: 2 | layer: 3 |
                layer: 4 | layer: 5 | layer: 6 | layer: 7 |
                layer: 8 | layer: 9 | layer: 10 | layer: 11 |
                fnorm""")
        assert -127.84109 == approx(infer.logits()[-1][0], 1e-3)
        assert 8217 == infer.argmax()[-1]
        assert ' machines' == infer.generation()[-1]

        geneneration = infer.ssrun("""self 
                add2inputs | wte | wpe |
                layer: 0 | layer: 1 | layer: 2 | layer: 3 |
                layer: 4 | layer: 5 | layer: 6 | layer: 7 |
                layer: 8 | layer: 9 | layer: 10 | layer: 11 |
                fnorm | generation""")
        assert [',', ',', 'ized', ' that', ' the', ' could', ' be', ' day', ' be', ' the', ' most', ' powerful', ' machines', ' on'] == geneneration

    @skipIf('SKIP' in env, reason="disabled")
    def test300_pca(self):
        wte = self.smodel.modelParams().getValue('wte')
        projection = self.smodel.projectMatrix(wte, 3)
        assert (wte.shape[0], 3) == projection.projected().shape
        return

    @skipIf('SKIP' in env, reason="disabled")
    def test310_vocabs(self):
        vocabs = self.smodel.words()
        self.assertEqual(' the', vocabs[262])

    @skipIf('SKIP' in env, reason="disabled")
    def test320_tokens(self):
        infer = self.smodel.inference().prompt(self.prompt2)
        infer.wte().wpe()
        x = infer.inputs()
        assert x == [36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716, 262, 749, 3665]
        x = infer.tokens()
        assert x == ['Alan', 'ĠTuring', 'Ġtheor', 'ized', 'Ġthat', 'Ġcomputers', 'Ġwould', 'Ġone', 'Ġday', 'Ġbecome', 'Ġthe', 'Ġmost', 'Ġpowerful']
        x = infer.words()
        assert x == ['Alan', ' Turing', ' theor', 'ized', ' that', ' computers', ' would', ' one', ' day', ' become', ' the', ' most', ' powerful']

