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

from smallscript import *

# Keep the state during inferencing. Could be a potential target for more generalization for other LLM models.
class GPT2Inference(SObject):
    model = Holder().name('model')      # LLMModel
    nlayer = Holder().name('nlayer')
    nhead = Holder().name('nhead')
    nembd = Holder().name('nembd')
    delta = Holder().name('delta')
    x = Holder().name('x')
    logits = Holder().name('logits')
    argmax = Holder().name('argmax')
    generation = Holder().name('generation')

    #### Attributes
    def prompt(self, prompt=''):
        prompt = self.asSObj(prompt)
        res = self._getOrSet('prompt', prompt, nil)
        if prompt == "": return res
        inputs = self.model().tokenizer().encode(prompt)
        self.inputs(List(inputs))
        return self

    def inputs(self, inputs=''):
        inputs = self.asSObj(inputs)
        res = self._getOrSet('inputs', inputs, nil)
        if inputs == "": return res
        self.reset()
        return self

    #### Helper
    def reset(self):
        inputs = self.inputs()
        if inputs.isNil(): return self
        zeros = self.model().zeros(inputs.len(), self.nembd())
        self.x(zeros)
        self.delta(zeros)
        return self

    def indices(self):
        return self.inputs()

    def tokens(self):
        indices = self.inputs()
        tokens = self.model().tokens().keys()
        tokenList = [tokens[index] for index in indices]
        return List(tokenList)

    def words(self):
        indices = self.inputs()
        words = self.model().words()
        wordsList = [words[index] for index in indices]
        return List(wordsList)

    #### Main methods
    def add2inputs(self):
        inputs = self.inputs()
        inputs.append(self.argmax()[-1].item())
        self.inputs(inputs)
        return self

    def wte(self):
        wte = self.model().modelParams().getValue('wte')
        self.x(wte[self.inputs()])
        return self

    def wpe(self):
        wpe = self.model().modelParams().getValue('wpe')
        pe = wpe[range(self.inputs().len())]
        self.x(self.x() + pe)
        return self

    def lnorm1(self, blkn):
        model = self.model()
        blk = model.blockParams().getValue(blkn)
        w = blk.getValue('ln_1.w')
        b = blk.getValue('ln_1.b')
        x2 = model.layerNorm(self.x(), w, b)
        self.delta(x2)
        return self

    def attn(self, blkn):
        model = self.model()
        blk = model.blockParams().getValue(blkn)
        n_head = self.nhead()
        attnw = blk.getValue('attn.w')
        attnb = blk.getValue('attn.b')
        projw = blk.getValue('attn | proj.w')
        projb = blk.getValue('attn | proj.b')
        x5 = model.attention(self.delta(),
                        attnw, attnb,
                        projw, projb, n_head)
        self.delta(x5)
        return self

    def sum(self):
        x = self.x() + self.delta()
        self.x(x)
        return self

    def lnorm2(self, blkn):
        model = self.model()
        blk = model.blockParams().getValue(blkn)
        w = blk.getValue('ln_2.w')
        b = blk.getValue('ln_2.b')
        x2 = model.layerNorm(self.x(), w, b)
        self.delta(x2)
        return self

    def ffn(self, blkn):
        model = self.model()
        blk = model.blockParams().getValue(blkn)
        fcw = blk.getValue("fc.w")
        fcb = blk.getValue("fc.b")
        projw = blk.getValue("mlp | proj.w")
        projb = blk.getValue("mlp | proj.b")
        x9 = model.feedforward(self.delta(),
                          fcw, fcb, projw, projb)
        self.delta(x9)
        return self

    def layer(self, blkn):
        self.\
            lnorm1(blkn).\
            attn(blkn).\
            sum().\
            lnorm2(blkn).\
            ffn(blkn).\
            sum()
        return self

    def fnorm(self):
        # ln_f norm
        model = self.model()
        w = model.modelParams().getValue('ln_f.w')
        b = model.modelParams().getValue('ln_f.b')
        x10 = model.layerNorm(self.x(), w, b)
        self.delta(x10)
        # self.x(x10)

        # logits
        wte = model.modelParams().getValue('wte')
        x11 = model.logits(x10, wte)
        self.logits(x11)

        # argmax
        argmax = model.argmax(x11)
        self.argmax(argmax)

        # generation
        generation = model.generation(argmax)
        self.generation(generation)
        return self