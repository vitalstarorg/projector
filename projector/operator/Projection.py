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
from dotenv import load_dotenv

import pandas as pd
import zipfile
import tempfile
import json
import pickle
import torch
import numpy as np
from smallscript import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets
from datetime import datetime
from IPython.display import display

class Projection(SObject):
    emptyModel = Holder().name('emptyModel')        # it is empty model, used as visitor
    projected = Holder().name('projected')          # projected wte to lower dimension
    df = Holder().name('df')                        # projected wte in dataframe
    components = Holder().name('components')        # PCA
    varianceRatio = Holder().name('varianceRatio')  # Variance ratio
    mean = Holder().name('mean')                    # Scaling factors
    std = Holder().name('std')

    def project(self, vectors):
        emptyModel = self.emptyModel()
        vector = emptyModel.projectVector(self, vectors)
        return vector

    def select(self, ids):
        selected = self.df().iloc[ids]
        return selected
