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
skip = pytest.mark.skip; skipIf = pytest.mark.skipif

from smallscript import *
from os import environ as env
# env['SKIP'] = '1'

class Test_Smoke(TestCase):
    @pytest.fixture(autouse=True, scope='class')
    # @pytest.fixture(autouse=True, scope='function')
    def setup(self):
        from dotenv import load_dotenv
        pkg = sscontext.loadPackage('projector')
        load_dotenv("../.env")

        self.var1 = 'start'
        yield
        self.var1 = 'done'

    @skipIf('SKIP' in env, reason="disabled")
    def test100_helpers(self):
        print("hello")
