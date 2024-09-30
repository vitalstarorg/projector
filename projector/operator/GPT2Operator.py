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

import torch
from transformers import AutoTokenizer, PretrainedConfig, GPT2LMHeadModel, AutoModelWithLMHead
from smallscript import *
from projector.GPT2Inference import GPT2Inference
from pathlib import Path
from dotenv import load_dotenv
from .Projection import Projection, GPT2Projection
from .GPT2EncoderNP import Encoder, bytes_to_unicode
from .Similarity import Similarity
import json
import difflib

class GPT2Operator(SObject):
    state = Holder().name('state')
    tokenizer = Holder().name('tokenizer')
    config = Holder().name('config')
    byte_decoder = Holder().name('byte_decoder')

    blockParams = Holder().name('blockParams').type('Map')
    modelParams = Holder().name('modelParams').type('Map')
    blockfilter = Holder().name('blockfilter')
    modelkeys = Holder().name('modelkeys').type('List')
    blockkeys = Holder().name('blockkeys').type('List')

    def __init__(self):
        self.blockfilter("h.{}.")
        self.modelkeys().\
            append('wte').\
            append('wpe').\
            append('ln_f.w').\
            append('ln_f.b')
        self.blockkeys().\
            append('ln_1.w'). \
            append('ln_1.b'). \
            append('attn.w'). \
            append('attn.b'). \
            append('attn | proj.w'). \
            append('attn | proj.b'). \
            append('ln_2.w'). \
            append('ln_2.b'). \
            append('fc.w'). \
            append('fc.b'). \
            append('mlp | proj.w'). \
            append('mlp | proj.b')

    def model(self, model=""):  # Hugging Face model
        res = self._getOrSet('model', model, nil)
        if model == "": return res
        self.state(model.state_dict())
        return self

    def readParams(self):
        nlayer = self.config().getValue('n_layer')
        modelParams = self.modelParams()
        modelParams.\
            setValue('n_layer', nlayer).\
            setValue('n_head', self.config().getValue('n_head')).\
            setValue('n_embd', self.config().getValue('n_embd')).\
            setValue('n_vocab', self.config().getValue('vocab_size'))
        for key in self.modelkeys():
            param = self.findParamBySpecs(key)
            modelParams.setValue(key, param)
        for i in range(nlayer):
            filter = self.blockfilter().format(i)
            block = Map()
            self.blockParams().setValue(i, block)
            for key in self.blockkeys():
                param = self.findParamBySpecs(f"{filter} | {key}")
                block.setValue(key, param)
        return self

    def findParamNameBySpecs(self, specs, delimiter='|'):
        keys = self.state().keys()
        specs = [spec.strip() for spec in specs.split('|')]
        found = nil
        for key in keys:
            if all(re.search(spec, key) for spec in specs):
                found = String(key)
                break
        return found

    def findParamBySpecs(self, specs, delimiter='|'):
        key = self.findParamNameBySpecs(specs)
        if key.isNil(): return nil
        res = self.state().get(key)
        return res

    def path(self, modelname=""):
        if modelname == "":
            modelname = self.name()
        if 'LLM_MODEL_PATH' not in env:
            self.log(f"env LLM_MODEL_PATH not set.", Logger.LevelError)
            return nil
        llmModelPath = Path(env['LLM_MODEL_PATH'])
        path = Path.home() / llmModelPath / modelname
        return path

    def loadEnv(self, dotEnv='.env'):
        path = Path.cwd()
        home = Path.home()
        while path != path.root:
            filepath = path / dotEnv
            if filepath.exists():
                load_dotenv(filepath)
                break
            path = path.parent
            if path == home: break
        return self

    def loadModel(self, modelname=""):
        self.loadEnv()
        if modelname != "":
            self.name(modelname)
        modelpath = self.path(modelname)
        if modelpath == nil: return nil
        model = AutoModelWithLMHead.from_pretrained(modelpath)
        tokenizer = AutoTokenizer.from_pretrained(modelpath)
        config = Map(PretrainedConfig.get_config_dict(modelpath)[0])
        self.model(model).config(config).tokenizer(tokenizer).readParams()

        # Direct preprocessing vocab.json to bypass expensive tokenizer.decode() in words()
        with open(modelpath / 'vocab.json', "r") as f:
            byte_encoder = bytes_to_unicode()
            byte_decoder = {v: k for k, v in byte_encoder.items()}
            self.byte_decoder(byte_decoder)
        return self

    def searchTokens(self, spelling, n = 5):
        tokens = self.tokens()
        wte = self.modelParams()['wte']
        closest = difflib.get_close_matches(spelling, tokens.keys(), n)
        res = Map()
        for token in closest:
            id = tokens[token]
            res[token] = {'id': id, 'vector': wte[id]}
        return res

    def tokens(self):
        tokens = self.tokenizer().get_vocab()
        index_to_token = {index: token for token, index in tokens.items()}
        token_to_index = { token: index for index, token in sorted(index_to_token.items())}
        return Map(token_to_index)

    def _words(self, labels, byte_decoder, replacement):
        whitespace = r"^\s{1}"
        vocabs = List()
        if replacement == '':
            for token in labels:
                word = bytearray([byte_decoder[c] for c in token]).decode("utf-8", errors="backslashreplace")
                vocabs.append(word)
        else:
            for token in labels:
                word = bytearray([byte_decoder[c] for c in token]).decode("utf-8", errors="backslashreplace")
                # word = re.sub(whitespace, "_", word)
                word = re.sub(whitespace, "\u2588", word)
                vocabs.append(word)
        return vocabs

    def words(self, replacement=''):
        labels = self.tokens().keys()
        byte_decoder = self.byte_decoder()
        vocabs = self._words(labels, byte_decoder, replacement)
        return vocabs

    def projectMatrix(self, matrix, ndim=3):
        emptyModel = self.createEmpty()         # visitor.projectVector()
        projection = GPT2Projection().emptyModel(emptyModel)
        projection.projectMatrix(matrix, ndim=3)
        return projection

    def getSimilarity(self):
        similarity = Similarity().model(self)
        return similarity

    def closestWords(self, vector, nwords=1):
        wte = self.findParamBySpecs('wte')
        indices = self.closestIndices(vector, wte, nwords)
        lst = List(self.tokenizer().convert_ids_to_tokens(indices))
        return lst

    def closestIndices(self, vector, matrix, nwords=1):
        vec_n = vector / vector.norm(dim=0)
        mat_n = matrix / matrix.norm(dim=1, keepdim=True)
        sim = torch.matmul(mat_n, vec_n)
        top_k = torch.topk(sim, k=nwords)
        indices = top_k.indices
        return indices

    def softmax(self, x):
        max = torch.max(x, dim=-1, keepdim=True).values
        exp = torch.exp(x - max)
        exp = exp / exp.sum(dim=-1, keepdim=True)
        return exp

    def layerNorm(self, x, w, b, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        x1 = (x - mean) / torch.sqrt(variance + eps)
        x2 = w * x1 + b
        return x2

    def attention(self, x, attnw, attnb, projw, projb, nHead):
        # qkv projection
        x3 = x @ attnw + attnb

        # multi-head
        qkv = List(torch.split(x3, x3.shape[-1] // 3, dim=-1))
        qkv_heads = list(
            map(
                lambda x: torch.split(x, x.size(-1) // nHead, dim=-1),
                qkv
            )
        )
        triMatrix = torch.tril(torch.ones(x3.shape[0], x3.shape[0], dtype=x3.dtype))
        causal_mask = (1 - triMatrix) * -1e10
        q0 = qkv_heads[0][0]
        scaling = torch.sqrt(torch.tensor(q0.shape[-1], dtype=q0.dtype))
        out_heads = []
        for q, k, v in zip(*qkv_heads):
            out = self.softmax(
                    q @ k.transpose(-1, -2) / scaling + causal_mask
                  ) @ v
            out_heads.append(out)
        x4 = torch.cat(out_heads, dim=1)
            # merge heads

        # MHA: out projection
        x5 = x4 @ projw + projb
        return x5

    def feedforward(self, x, fcw, fcb, projw, projb):
        # project up
        x1 = x @ fcw + fcb

        # gelu
        sqrt_2_pi = torch.sqrt(torch.tensor(2 / torch.pi))  # Equivalent to np.sqrt(2 / np.pi)
        term = x1 + 0.044715 * x1 ** 3
        tanh_term = torch.tanh(sqrt_2_pi * term)
        x2 = 0.5 * x1 * (1 + tanh_term)

        # project back down
        x3 = x2 @ projw + projb
        return x3

    def getDevice(self):
        if torch.cuda.is_available(): return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'

    def logits(self, x, wte):
        x11 = x @ wte.transpose(-1, -2)
        return x11

    def argmax(self, x):
        argmax = x.argmax(dim=-1)
        return argmax

    def generation(self, x):
        generation = [self.tokenizer().decode(id) for id in x]
        return generation

    def zeros(self, nrow, ncolumn):
        zeros = torch.zeros(nrow, ncolumn)
        return zeros

    def inference(self):
        inference = GPT2Inference().model(self)
        modelParams = self.modelParams()
        inference.nlayer(modelParams.getAsNumber('n_layer'))
        inference.nhead(modelParams.getAsNumber('n_head'))
        inference.nembd(modelParams.getAsNumber('n_embd'))
        return inference
