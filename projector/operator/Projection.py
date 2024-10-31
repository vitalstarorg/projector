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
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from smallscript import *

class Projection(SObject):
    """
    Abstract two different PCA projection implementations: pytorch and numpy.
    """
    emptyModel = Holder().name('emptyModel')        # it is empty model, used as visitor
    projected = Holder().name('projected')          # projected wte to lower dimension
    df = Holder().name('df')                        # projected wte in dataframe
    components = Holder().name('components')        # PCA
    varianceRatio = Holder().name('varianceRatio')  # Variance ratio
    mean = Holder().name('mean')                    # Scaling factors
    std = Holder().name('std')
    epsilon = Holder().name('epsilon')

    def __init__(self):
        self.epsilon(1e-4)

    def select(self, ids):
        selected = self.df().iloc[ids]
        return selected

    def adjustComponents(self, components):
        # Make the PCA a bit deterministic by reversing axis to make first element to be positive.
        first = components[0,:]
        for i in range(first.shape[0]):
            element = first[i]
            if element > 0: continue
            components[:,i] = -components[:,i]
        return components

class GPT2Projection(Projection):
    """
    Pytorch implementation for Projection
    """
    def projectMatrix(self, matrix, ndim=3):
        """Project matrix to 3D space using PCA."""
        mean = matrix.mean(dim=0)
        # var = matrix.var(dim=0)
        # std = torch.sqrt(var + self.epsilon().value())
        # std = matrix.std(dim=0, unbiased=False)
        std = torch.ones_like(mean)
        z = (matrix - mean) / std
        covariance = torch.mm(z.T, z) / (z.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        components = eigenvectors[:, :ndim]
        # Not scalable
        # U, S, V_T = torch.linalg.svd(z)
        # comp = V_T.T
        # comp = comp[:, :ndim]
        # emb = torch.mm(z, comp)

        components = self.adjustComponents(components)
        projected = torch.mm(z, components)
        self. projected(projected).components(components).mean(mean).std(std)

        # calculate the variance ratio
        total_variance = torch.sum(eigenvalues)
        top_eigenvalues = eigenvalues[:ndim]
        explained_variance_ratio = top_eigenvalues / total_variance
        self.varianceRatio(explained_variance_ratio)
        return self

    def projectVector(self, vectors):
        """Project vectors to 3D space."""
        if isinstance(vectors, list):
            if isinstance(vectors[0], list):  # List of vectors
                vectors = torch.tensor(vectors)
            else:  # Single vector in list
                vectors = torch.tensor(vectors).unsqueeze(0)
        elif isinstance(vectors, torch.Tensor) and vectors.dim() == 1:  # Single vector
            vectors = vectors.unsqueeze(0)
        mean = self.mean()
        std = self.std()
        z = (vectors - mean)/std
        components = self.components()
        projected = torch.mm(z, components)
        return projected

class GPT2ProjectionNP(Projection):
    """
    Numpy implementation for Projection
    """
    def projectMatrix(self, matrix, ndim=3):
        scaler = StandardScaler(with_mean=True, with_std=False)
        scaled = scaler.fit_transform(matrix)
        svd = TruncatedSVD(n_components=ndim, random_state=42)  # minimize randomness for tests
        projected = svd.fit_transform(scaled)
        mean = scaler.mean_
        std = scaler.scale_
        if std is None:
            std = np.ones_like(mean)
        components = self.adjustComponents(svd.components_)
        self.projected(projected).components(components).mean(mean).std(std)
        variance_ratio = svd.explained_variance_ratio_
        self.varianceRatio(variance_ratio)
        return self

    def projectVector(self, vectors):
        if isinstance(vectors, list):
            if isinstance(vectors[0], list):  # List of vectors (list of lists)
                vectors = np.array(vectors)
            else:  # Single vector in list
                vectors = np.array(vectors).reshape(1, -1)
        elif isinstance(vectors, np.ndarray) and vectors.ndim == 1:  # Single vector (1D array)
            vectors = vectors.reshape(1, -1)
        mean = self.mean()
        std = self.std()
        z = (vectors - mean)/std
        components = self.components()
        projected = np.dot(z, components.T)
        return projected
