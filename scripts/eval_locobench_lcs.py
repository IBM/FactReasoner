#!/usr/bin/env python
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

"""Evaluate the LCS pipeline on a generated LoCoBench dataset.

A thin wrapper over ``fact_reasoner.locoeval.cli`` (installed as
``locobench-lcs-eval``), so the evaluation runs from a checkout with no editable
install. See that module's docstring for the options and outputs.

Run::

    python scripts/eval_locobench_lcs.py \\
        --data-dir data/locobench-claude-5 \\
        --out-dir results/locobench_claude_5_lcs \\
        --merlin-path /path/to/merlin
"""

import os
import sys

# Make `src/` importable when run from a checkout without an editable install.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from fact_reasoner.locoeval.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
