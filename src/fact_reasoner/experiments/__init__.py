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

"""LCS experiment harness.

Systematically mines the ``data/lcs`` examples across models and
conditional-strength UQ methods, computes all four LCS scores per mined MRF,
saves results to JSON, and renders a LaTeX report (tables + pgfplots figures).

Run offline first to see the shape of the outputs::

    python -m fact_reasoner.experiments.run_experiments --dry-run \
        --output-dir results/lcs_dryrun
"""

from fact_reasoner.experiments.config import (
    DEFAULT_MODELS,
    ExperimentConfig,
    ModelSpec,
)
from fact_reasoner.experiments.dataset import load_examples
from fact_reasoner.experiments.report import write_report
from fact_reasoner.experiments.runner import ExperimentRunner, run_experiment
from fact_reasoner.experiments.scoring import score_all_lcs

__all__ = [
    "ExperimentConfig",
    "ModelSpec",
    "DEFAULT_MODELS",
    "ExperimentRunner",
    "run_experiment",
    "load_examples",
    "score_all_lcs",
    "write_report",
]
