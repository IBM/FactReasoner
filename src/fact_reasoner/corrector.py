# coding=utf-8
# Copyright 2023-present the International Business Machines.
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

# FactCorrector pipeline

import json
import math
import os
import time
import subprocess
import uuid
import logging

from typing import Any, Dict

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.global_vars import logger
from pgmpy.models import MarkovNetwork
from pgmpy.readwrite import UAIWriter
from mellea.backends import ModelOption
from mellea.core import FancyLogger

class FactCorrector:
    def __init__(self):
        pass