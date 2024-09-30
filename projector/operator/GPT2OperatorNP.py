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

# Based on https://github.com/jaymody/picoGPT/tree/main

import json
import os
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

import numpy as np
import tensorflow as tf
from smallscript import *
from .GPT2Operator import GPT2Operator
from .GPT2EncoderNP import get_encoder
from .Projection import GPT2ProjectionNP

# GPT2 Adaptor using Numpy
class GPT2OperatorNP(GPT2Operator):

    def model(self, model=""):
        res = self._getOrSet('model', model, nil)
        return res

    def readParams(self):
        nlayer = self.config().getValue('n_layer')
        modelParams = self.modelParams()
        modelParams.setValue('n_layer', nlayer)
        modelParams.setValue('n_head', self.config().getValue('n_head'))
        modelParams.setValue('n_embd', self.config().getValue('n_embd'))
        modelParams.setValue('n_vocab', self.config().getValue('n_vocab'))
        modelParams.setValue('wte', self.model().getValue('wte'))
        modelParams.setValue('wpe', self.model().getValue('wpe'))
        modelParams.setValue('ln_f.w', self.model().getValue('ln_f').get('g'))
        modelParams.setValue('ln_f.b', self.model().getValue('ln_f').get('b'))
        blocks = self.model().getValue('blocks')
        for i in range(nlayer):
            block = blocks[i]
            blk = Map()
            self.blockParams().setValue(i, blk)
            blk.setValue('ln_1.w', block['ln_1']['g'])
            blk.setValue('ln_1.b', block['ln_1']['b'])
            blk.setValue('attn.w', block['attn']['c_attn']['w'])
            blk.setValue('attn.b', block['attn']['c_attn']['b'])
            blk.setValue('attn | proj.w', block['attn']['c_proj']['w'])
            blk.setValue('attn | proj.b', block['attn']['c_proj']['b'])
            blk.setValue('ln_2.w', block['ln_2']['g'])
            blk.setValue('ln_2.b', block['ln_2']['b'])
            blk.setValue('fc.w', block['mlp']['c_fc']['w'])
            blk.setValue('fc.b', block['mlp']['c_fc']['b'])
            blk.setValue('mlp | proj.w', block['mlp']['c_proj']['w'])
            blk.setValue('mlp | proj.b', block['mlp']['c_proj']['b'])
        return self

    def findParamBySpecs(self, specs, delimiter='|'):
        param = self.modelParams().getValue(specs, nil)
        if param is not nil: return param
        param = self.blockParams()[0].get(specs, nil)
        return param

    def loadModel(self, modelname=""):
        self.loadEnv()
        if modelname != "":
            self.name(modelname)
        modelpath = self.path(modelname)
        if modelpath is nil: return nil
        encoder, hparams, params = self.load_encoder_hparams_and_params("", modelpath)
        self.model(Map(params)).config(Map(hparams)).tokenizer(encoder).readParams()
        return self

    def load_gpt2_params_from_tf_ckpt(self, tf_ckpt_path, hparams):
        def set_in_nested_dict(d, keys, val):
            if not keys:
                return val
            if keys[0] not in d:
                d[keys[0]] = {}
            d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
            return d

        params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
        for name, _ in tf.train.list_variables(tf_ckpt_path):
            tfvars = tf.train.load_variable(tf_ckpt_path, name)
            array = np.squeeze(tfvars)
            name = name[len("model/"):]
            if name.startswith("h"):
                m = re.match(r"h([0-9]+)/(.*)", name)
                n = int(m[1])
                sub_name = m[2]
                set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
            else:
                set_in_nested_dict(params, name.split("/"), array)
        return params

    def load_encoder_hparams_and_params(self, model_size, modelpath):
        model_dir = os.path.join(modelpath, model_size)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        encoder = get_encoder(model_size, modelpath)
        with open(os.path.join(model_dir, "hparams.json")) as f:
            hparams = json.load(f)
        params = self.load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)
        return encoder, hparams, params

    def tokens(self):
        tokens = self.tokenizer().encoder
        return Map(tokens)

    def words(self, replacement=''):
        encoder = self.tokenizer()
        labels = list(encoder.encoder.keys())
        byte_decoder = encoder.byte_decoder
        vocabs = self._words(labels, byte_decoder, replacement)
        return vocabs

    # def projectVector(self, projection, vectors):
    #     if isinstance(vectors, list):
    #         if isinstance(vectors[0], list):  # List of vectors (list of lists)
    #             vectors = np.array(vectors)
    #         else:  # Single vector in list
    #             vectors = np.array(vectors).reshape(1, -1)
    #     elif isinstance(vectors, np.ndarray) and vectors.ndim == 1:  # Single vector (1D array)
    #         vectors = vectors.reshape(1, -1)
    #     mean = projection.mean()
    #     std = projection.std()
    #     z = (vectors - mean)/std
    #     components = projection.components()
    #     projected = np.dot(z, components.T)
    #     return projected

    def projectMatrix(self, matrix, ndim=3):
        emptyModel = self.createEmpty()         # visitor.projectVector()
        projection = GPT2ProjectionNP().emptyModel(emptyModel)
        projection.projectMatrix(matrix, ndim=3)
        return projection

    # def projectMatrix1(self, matrix, ndim=3):
    #     scaler = StandardScaler()
    #     scaled = scaler.fit_transform(matrix)
    #     svd = TruncatedSVD(n_components=ndim, random_state=42)  # minimize randomness for tests
    #     projected = svd.fit_transform(scaled)
    #     mean = scaler.mean_
    #     std = scaler.scale_
    #     components = svd.components_
    #     emptyModel = self.createEmpty()    # visitor.projectVector()
    #     projection = Projection().emptyModel(emptyModel).projected(projected).components(components).mean(mean).std(std)
    #     variance_ratio = svd.explained_variance_ratio_
    #     projection.varianceRatio(variance_ratio)
    #     return projection

    def closestWords(self, vector, nwords=1):
        wte = self.findParamBySpecs('wte')
        indices = self.closestIndices(vector, wte, nwords)
        words = []
        for i in range(0, nwords):
            words.append(self.tokenizer().decode([indices[i][0]]))
        return words

    def closestIndices(self, vector, wte, nwords=1):
        similarities = np.dot(wte, vector) / (
                np.linalg.norm(wte, axis=1) * np.linalg.norm(vector))
        sorted = np.argsort(-similarities)
        indices = []
        for i in range(0, nwords):
            closest_word_index = sorted[i]
            indices.append((closest_word_index, similarities[closest_word_index]))
        return indices

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def layerNorm(self, x, w, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
        return w * x + b  # scale and offset with gamma/beta params

    def attentionQKV(self, q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
        return self.softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

    def attention(self, x, attnw, attnb, projw, projb, nHead):
        # qkv projection
        x3 = x @ attnw + attnb  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

        # split into qkv
        qkv = np.split(x3, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

        # split into heads
        qkv_heads = list(
            map(lambda x: np.split(x, nHead, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

        # causal mask to hide future inputs from being attended to
        causal_mask = (1 - np.tri(x3.shape[0], dtype=x3.dtype)) * -1e10  # [n_seq, n_seq]

        # perform attention over each head
        out_heads = [self.attentionQKV(q, k, v, causal_mask) for q, k, v in
                     zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

        # merge heads
        x4 = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

        # out projection
        x5 = x4 @ projw + projb  # [n_seq, n_embd] -> [n_seq, n_embd]

        return x5

    def feedforward(self, x, fcw, fcb, projw, projb):
        # project up
        x1 = x @ fcw + fcb

        # gelu
        x2 = 0.5 * x1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x1 + 0.044715 * x1 ** 3)))

        # project back down
        x3 = x2 @ projw + projb
        return x3

    def logits(self, x, wte):
        x11 = x @ wte.T
        return x11

    def argmax(self, x):
        argmax = np.argmax(x, axis=-1)
        return argmax

    def generation(self, x):
        generation = [self.tokenizer().decode([id]) for id in x]
        return generation

    def zeros(self, nrow, ncolumn):
        zeros = np.zeros((nrow, ncolumn))
        return zeros
