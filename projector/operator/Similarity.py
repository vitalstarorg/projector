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

import torch
from smallscript import *

class Similarity(SObject):
    model = Holder().name('model')

    k = Holder().name('k')
    ids = Holder().name('ids')
    sims = Holder().name('sims')
    angles = Holder().name('angles')

    def __init__(self):
        self.k(1)

    def isEmpty(self): return true_ if self.sims() is nil else false_
    def copy(self): return self.createEmpty().model(self.model())

    def degrees(self, sims):
        sims = torch.clamp(sims, -1, 1)
        rad = torch.arccos(sims)
        deg = rad * (180/torch.pi)
        degrees = torch.round(deg * 10000) / 10000
        return degrees

    def knn(self, vectors):
        wte = self.model().modelParams()['wte']
        vnorms = vectors.norm(dim=1, keepdim=True)
        wnorms = wte.norm(dim=1, keepdim=True)
        if vnorms.shape == (1,1) and vnorms == 0:      # zero vector
            norms, indices = torch.topk(wnorms.T, self.k(), largest=False)
            vecs = wte[indices[0,:]]
            vectors = vecs.mean(dim=0, keepdim=True)
        sim = torch.mm(vectors, wte.T)
        maxSim, indices = torch.topk(sim, self.k(), dim=1)
        self.sims(maxSim)
        self.ids(indices)

        vnorms = vectors.norm(dim=1, keepdim=True)
        vectors = vectors / torch.where(vnorms == 0, torch.ones_like(vnorms), vnorms)
        wte = wte / torch.where(wnorms == 0, torch.ones_like(wnorms), wnorms)
        sim = torch.mm(vectors, wte.T)
        degrees = self.degrees(sim).flatten()[indices]
        self.angles(degrees)
        return self

class CosineSimilarity(Similarity):
    def knn(self, vectors):
        wte = self.model().modelParams()['wte']
        vnorms = vectors.norm(dim=1, keepdim=True)
        wnorms = wte.norm(dim=1, keepdim=True)
        if vnorms.shape == (1,1) and vnorms == 0:      # zero vector
            norms, indices = torch.topk(wnorms.T, self.k(), largest=False)
            vecs = wte[indices[0,:]] / wnorms[indices[0,:]]
            average = vecs.mean(dim=0, keepdim=True)
            naverage = average / average.norm()
            maxSim = torch.mm(vecs, naverage.T)
            degrees = self.degrees(maxSim)
        else:
            vectors = vectors / torch.where(vnorms == 0, torch.ones_like(vnorms), vnorms)
            wte = wte / torch.where(wnorms == 0, torch.ones_like(wnorms), wnorms)
            sim = torch.mm(vectors, wte.T)
            maxSim, indices = torch.topk(sim, self.k(), dim=1)
            degrees = self.degrees(maxSim)
        self.sims(maxSim)
        self.ids(indices)
        self.angles(degrees)
        return self
