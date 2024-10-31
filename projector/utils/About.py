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
from math import floor, log10
import pytest
from pytest import approx as approx
from smallscript import *


class About(SObject):
    """
    Used as a replacement for pytest.approx for comparing torch and numpy tensors.
    Setup:
        about = approx
            # Use approx for the following pytest comparisons.

        about = About()
            # Use About for the following pytest comparisons.

        about = About().always(true_)
            # Use About for the following pytest comparisons even test failed. Expect and actual
            # comparison will be shown in the log.
            # This flag can be set in mid of debugging by doing `about.always(true_)`.

    Example1:
        assert similarity == about([13.924, 10.111, 8.118, 8.101, 8.045], 1e-3)
            # compare a list with tensor or array of numbers with 0.001 relative precision.

        assert ids1 == about([36235, 12246, 40441, 35314, 44484])
            # compare a list with tensor or array of indices default precision.
    """
    __array_priority__ = 10.0  # use higher value to supersede numpy to use About.__eq__()

    precision = Holder().name('precision')
    expected = Holder().name('expected')
    pyapprox = Holder().name('pyapprox')
    always = Holder().name('always')        # continue if test failed

    def __init__(self):
        self.precision(3)
        self.always(false_)

    def _toNumpy(self, numbers):
        npnum = numbers
        if torch.is_tensor(numbers):
            if numbers.dim() == 0:
                npnum = numbers.item()
            else:
                npnum = numbers.numpy()
        return npnum

    def __call__(self, expected, *args, **kwargs):
        self.expected(expected)
        pyapprox = approx(expected, *args, **kwargs)
        self.pyapprox(pyapprox)
        return self

    def __eq__(self, actual):
        def pctString(expected, actual):
            if abs(actual) < 1e-5:
                diffPct = f"diff = {round(abs(actual - expected), 6)}"
            else:
                diffPct = f"{round(abs((actual - expected) / actual) * 100, 2)}%"
            return diffPct

        def roundPrecision(precision, actual):
            if abs(actual) < 1e-5:
                rounded = actual
            else:
                rounded = round(actual, self.precision() - int(floor(log10(abs(actual)))))
            return rounded

        pyapprox = self.pyapprox()
        actual = self._toNumpy(actual)
        res = actual == pyapprox
        if res: return res
        # actual == pyapprox      # redo the test for debug tracing
        expected = self.expected()
        if not np.isscalar(actual):
            roundedActual = []
            for x in actual:
                rounded = roundPrecision(self.precision(), x)
                roundedActual.append(rounded)
            diffPct = [pctString(expected[i], actual[i]) for i in range(len(actual))]
        else:
            if torch.is_tensor(actual):
                actual = actual.item()
            diffPct = pctString(expected, actual)
            roundedActual = roundPrecision(self.precision(), actual)

        self.log(f"{expected} was expected but actual is {actual} {diffPct}. Try about({roundedActual}, 1e-{self.precision()})", Logger.LevelError)
        return self.always()

    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func.__name__ == 'eq':
            res = cls.__eq__(args[0])
            return res
        return NotImplemented

    def __repr__(self):
        return f"{self.expected()}"
