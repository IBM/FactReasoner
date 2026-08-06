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

Where ``fact_reasoner.locobench`` *generates* items and
``fact_reasoner.experiments`` sweeps the *miner* over hand-written examples, this
subpackage scores generated items from **the relations the item file already
carries**. No LLM is involved: the gold labels are compiled straight into the
coherence MRF, so the evaluation is deterministic and reproducible offline (Merlin
is the only subprocess).

Three modelling choices define the gold arm, all recorded in each result's
``config`` (see :mod:`fact_reasoner.locoeval.gold_graph`):

* an atom's unary prior is **0.9** when the item marks it ``factual``, **0.1**
  otherwise;
* an edge's factor probability is the **midpoint of its ``strength_range``**
  (strong 0.925 / moderate 0.72 / weak 0.47), with ``type_confidence = 1.0``
  because gold is a label rather than an estimate;
* a resolved concession is discounted by ``lambda = 0.45`` using the item's own
  ``resolver_atom_id``, never the miner's text heuristic.

Ordering-only relations (``Precedence`` / ``Succession``, coupling ``none``)
produce no factor, exactly as ``lcs.taxonomy.compile_sense`` dictates.

Quick start::

    from fact_reasoner.locoeval import GoldEvalRunner, write_report

    runner = GoldEvalRunner(
        data_dir="data/locobench-claude-5",
        output_dir="results/locobench_claude_5_lcs",
        merlin_path="/path/to/merlin",
    )
    results = runner.run()          # results.json + records/ + by_item/
    write_report(results, "results/locobench_claude_5_lcs")   # report.tex (+ PDF)

Each item is scored under two arms: ``gold`` (every edge-producing gold relation,
including the deliberately-planted invalid ones) and ``gold_valid`` (only
``validity == "valid"`` edges). The runner also checks each family's
``ordering_constraints`` from ``families.json``.

A caveat the runner measures rather than assumes: in datasets generated before the
gold-relation duplication fix, every rung of a family carries the SAME gold
relations while its response text differs, so a gold-only ladder check is vacuous.
:meth:`GoldEvalRunner.run` records this per family as
``gold_relations_identical_across_rungs`` and the report states it.
"""

from fact_reasoner.locoeval.gold_graph import (
    BAND_RANGES,
    DEFAULT_CONCESSION_DISCOUNT,
    PRIOR_FACTUAL,
    PRIOR_NOT_FACTUAL,
    GoldGraphError,
    atom_priors,
    atom_texts,
    band_probability,
    build_gold_result,
    gold_relations,
    item_atoms,
)
from fact_reasoner.locoeval.report import build_pdf, write_report
from fact_reasoner.locoeval.runner import (
    GOLD_ARMS,
    GRADED_READOUTS,
    GoldEvalRunner,
    evaluate_constraints,
    load_families,
    load_items,
    run_gold_eval,
    summarize_constraints,
)

__all__ = [
    "BAND_RANGES",
    "DEFAULT_CONCESSION_DISCOUNT",
    "GOLD_ARMS",
    "GRADED_READOUTS",
    "GoldEvalRunner",
    "GoldGraphError",
    "PRIOR_FACTUAL",
    "PRIOR_NOT_FACTUAL",
    "atom_priors",
    "atom_texts",
    "band_probability",
    "build_gold_result",
    "build_pdf",
    "evaluate_constraints",
    "gold_relations",
    "item_atoms",
    "load_families",
    "load_items",
    "run_gold_eval",
    "summarize_constraints",
    "write_report",
]
