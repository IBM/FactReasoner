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
subpackage scores generated items under one or more **arms**. A *gold* arm compiles
the relations the item file already carries straight into the coherence MRF, so it
is deterministic and reproducible offline (Merlin the only subprocess). A *mined*
arm runs the real relation miner over the item's response prose instead (see
:mod:`fact_reasoner.locoeval.mined_graph`), holding the atoms and the priors fixed.

The pair answers two different questions. A gold arm asks whether the readouts
behave correctly *given* a correct graph; a mined arm asks whether the pipeline can
*recover* that graph from text. Because everything except the relation source is
held identical -- same atoms with the same ids, same 0.9/0.1 priors, same readouts,
same scorer settings -- any difference between the two is attributable to relation
mining.

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
        data_dir="data/locobench-claude-5-test",
        output_dir="results/locobench_claude_5_lcs",
        merlin_path="/path/to/merlin",
    )
    results = runner.run()          # results.json + records/ + by_item/
    write_report(results, "results/locobench_claude_5_lcs")   # report.tex (+ PDF)

By default each item is scored under two gold arms: ``gold`` (every edge-producing
gold relation, including the deliberately-planted invalid ones) and ``gold_valid``
(only ``validity == "valid"`` edges). Adding a mined arm is a matter of naming it
and supplying an inventory to resolve the model against::

    from fact_reasoner.locoeval import load_model_specs

    runner = GoldEvalRunner(
        data_dir="data/locobench-claude-5-test",
        output_dir="results/locobench_claude_5_mined_lcs",
        merlin_path="/path/to/merlin",
        arms=("gold", "gold_valid", "mined:llama-3.3-70b-instruct:windowed"),
        model_specs=load_model_specs(),      # configs/rits_models.json
        resume=True,                          # reuse per-cell records
    )

The runner checks each family's ``ordering_constraints`` from ``families.json`` for
every arm, so the ladder result is directly comparable across arms.

A caveat the runner measures rather than assumes: in datasets generated before the
gold-relation duplication fix, every rung of a family carries the SAME gold
relations while its response text differs, so a gold-only ladder check is vacuous.
:meth:`GoldEvalRunner.run` records this per family as
``gold_relations_identical_across_rungs`` and the report states it.

Two properties of a mined arm are likewise measured rather than assumed, because
both would otherwise be mistaken for miner quality:

* the ``windowed`` policy selects only FORWARD pairs, so a *directed* gold edge
  running backward in atom order is unreachable under it whatever the prose says.
  ``compare_to_gold`` matches undirected couplings on the unordered pair and
  reports recall split by direction, keeping policy reach separable from accuracy.
* a failed LLM call is captured, not raised, and the miner's parser maps it to the
  same ``None`` a genuine "unrelated" produces. The failure rate is therefore
  counted per cell, and a cell exceeding ``max_call_error_rate`` is refused --
  otherwise a throttled endpoint yields a sparse graph and confident-looking
  scores.
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
from fact_reasoner.locoeval.mined_graph import (
    DEFAULT_MAX_CALL_ERROR_RATE,
    MINED_PREFIX,
    MinedArm,
    MinedArmError,
    abuild_mined_result,
    aggregate_comparisons,
    compare_to_gold,
    count_duplicate_unordered_pairs,
    format_arm,
    parse_arm,
)
from fact_reasoner.locoeval.models import (
    DEFAULT_MODELS_FILE,
    ModelSpec,
    load_model_specs,
    resolve_model,
)
from fact_reasoner.locoeval.report import build_pdf, write_report
from fact_reasoner.locoeval.runner import (
    GOLD_ARMS,
    GRADED_READOUTS,
    TIE_TOLERANCE,
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
    "DEFAULT_MAX_CALL_ERROR_RATE",
    "DEFAULT_MODELS_FILE",
    "GOLD_ARMS",
    "GRADED_READOUTS",
    "GoldEvalRunner",
    "GoldGraphError",
    "MINED_PREFIX",
    "MinedArm",
    "MinedArmError",
    "ModelSpec",
    "PRIOR_FACTUAL",
    "PRIOR_NOT_FACTUAL",
    "TIE_TOLERANCE",
    "abuild_mined_result",
    "aggregate_comparisons",
    "atom_priors",
    "atom_texts",
    "band_probability",
    "build_gold_result",
    "build_pdf",
    "compare_to_gold",
    "count_duplicate_unordered_pairs",
    "evaluate_constraints",
    "format_arm",
    "gold_relations",
    "item_atoms",
    "load_families",
    "load_items",
    "load_model_specs",
    "parse_arm",
    "resolve_model",
    "run_gold_eval",
    "summarize_constraints",
    "write_report",
]
