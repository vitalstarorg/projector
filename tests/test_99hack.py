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
skip = pytest.mark.skip
skipIf = pytest.mark.skipif

import os
from os import environ as env
env['SKIPHACK'] = '1'
# env['LLM_MODEL_PATH'] = 'data/prj/model'

from smallscript import *
from projector.operator.GPT2Operator import GPT2Operator
from projector.Projector import *

class Test_Hack(TestCase):
    @pytest.fixture(autouse=True, scope='class')
    def setup(self):
        from dotenv import load_dotenv
        pkg = sscontext.loadPackage('projector')
        load_dotenv("../.env")
        # pkg = sscontext.getOrNewPackage('projector')
        # pkg.findPath("projector")
        # pkg.loadSObjects()
        return

    @skipIf('SKIPHACK' in env, reason="disabled")
    def test100_helpers(self):
        scope = sscontext.createScope()
        closure = sscontext.compile("os.environ.SHELL")
        res = closure(scope)
        self.assertEqual('/bin/bash', res)

        res = sscontext.ssrun("a := 'hello'; scope")
        return

    # @skipIf('SKIPHACK' in env, reason="disabled")
    @skip
    def test900_hack(self):
        model = GPT2Operator().name("gpt2").loadModel()
        pj = Projector().name('pj').model(model)

        return
