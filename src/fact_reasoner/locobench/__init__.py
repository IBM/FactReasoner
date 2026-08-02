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

"""LoCoBench: the data-generation harness for the logical-coherence benchmark.

Implements Phase 2 of ``docs/ideation/coherence/benchmark/locobench_phase2.tex``. The
design it executes -- facets, topics, family types, operators, prompts, schema and
thresholds -- is Phase 1 (``locobench_phase1.tex``) and is treated as fixed here.

Quick start::

    # Offline: the whole pipeline with no LLM and no Merlin.
    locobench-generate --dry-run --limit 2 --out /tmp/locobench

    # A real run. Re-running the same command resumes it.
    locobench-generate --config locobench.json

    # Coverage and cost against the current state, generating nothing.
    locobench-generate --config locobench.json --report

Three properties, in the order they matter:

* **Resume is the default.** There is no ``--resume`` flag because there is no
  non-resuming mode: every run reads the output directory first, skips completed
  families, and retries only gate failures. A completed run is a fixed point.
* **Dry-run covers the whole pipeline.** ``--dry-run`` swaps a deterministic offline
  generator for the LLM, so parsers, gates, ladders, schema assertions and storage
  all execute without credentials.
* **Gate failures are recorded, not silently retried.** A rejected family is written
  to ``rejected/`` with the validator, the threshold and the observed value, because
  the per-gate rejection rate is a finding about the prompts.

The unit of work is a *family*: five items (five responses) that are minimal edits of
each other, ordered least to most coherent. An *item* is one response plus its
annotations. The corpus target is 120 families x 5 rungs = 600 items.
"""

from fact_reasoner.locobench.config import (
    DEFAULT_COMMITTEE_MIN,
    GenConfig,
    load_config,
)
from fact_reasoner.locobench.perturb import (
    FAMILY_TYPES,
    LADDERS,
    OPERATOR_CALLS,
    OPERATORS,
    RUNG_NAMES,
    expectations_for,
    ladder_for,
)
from fact_reasoner.locobench.pipeline import FamilyResult, generate_family
from fact_reasoner.locobench.schema import (
    SchemaError,
    validate_item,
    validate_manifest_entry,
)
from fact_reasoner.locobench.store import Store
from fact_reasoner.locobench.topics import (
    DOMAINS,
    TOPICS,
    allocate,
    domain_of,
    is_topic,
)
from fact_reasoner.locobench.validate import (
    THRESHOLDS,
    committee_for,
    stratified_sample,
)

__all__ = [
    "DEFAULT_COMMITTEE_MIN",
    "DOMAINS",
    "FAMILY_TYPES",
    "LADDERS",
    "OPERATORS",
    "OPERATOR_CALLS",
    "RUNG_NAMES",
    "THRESHOLDS",
    "FamilyResult",
    "GenConfig",
    "SchemaError",
    "Store",
    "TOPICS",
    "allocate",
    "committee_for",
    "domain_of",
    "expectations_for",
    "generate_family",
    "is_topic",
    "ladder_for",
    "load_config",
    "stratified_sample",
    "validate_item",
    "validate_manifest_entry",
]
