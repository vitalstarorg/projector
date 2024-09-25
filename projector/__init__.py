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

import os
import logging

loglevel = os.environ.get('LOG_LEVEL', 'WARNING')
logger = logging.getLogger(__name__)
logger.setLevel(loglevel)
# print(f"=== {logger}")
# print(f"=== {logger.getEffectiveLevel()}")
logger.propagate = True

handler = logging.StreamHandler()
handler.setLevel(loglevel)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
