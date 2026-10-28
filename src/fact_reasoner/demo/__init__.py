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

"""Interactive Logical Coherence demo (``fact-reasoner-lcs-demo``).

A FastAPI backend (:mod:`fact_reasoner.demo.app`) serving a Carbon-styled page
that extracts claims, highlights their spans, draws the mined relation graph and
reports the LCS readouts. See ``docs/examples/lcs/demo.md``.
"""
