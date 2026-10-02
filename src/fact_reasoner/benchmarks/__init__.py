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

"""Benchmark harness.

Builds frozen per-backend datasets so that retrieval backends can be compared on
identical queries, and evaluates those frozen datasets. This is benchmarking
tooling, kept separate from the production retrieval path::

    python -m fact_reasoner.benchmarks.build_ntrs_benchmark --help
    python -m fact_reasoner.benchmarks.eval_ntrs_benchmark --help
"""

from fact_reasoner.benchmarks.build_ntrs_benchmark import build_datasets
from fact_reasoner.benchmarks.eval_ntrs_benchmark import evaluate, summarize

__all__ = ["build_datasets", "evaluate", "summarize"]
