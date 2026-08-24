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

"""Baselines the Logical Coherence Score is compared against.

Every baseline here answers the same question the LCS answers -- "how well does
this response hang together?" -- and returns one scalar in ``[0, 1]``, higher
meaning more coherent, so the ladder scorer can read a baseline column exactly
as it reads an LCS column.

Two conventions make the comparison an *ablation* rather than a different
experiment, and both are load-bearing:

* **Shared decomposition.** A baseline is handed the same atom texts the LCS was
  scored on. It never atomizes for itself. Anything it does worse, it does worse
  on identical input.
* **No evidence.** Like the coherence MRF (which contains claim variables only),
  no baseline here consults retrieved passages. Metrics that need a source
  document are not comparable and are deliberately absent.

What each baseline is *for* -- the diagnostic argument, not just the number:

``nli_contradiction``
    Flat local contradiction counting: no relation typing, no graph, no
    propagation. The gap to LCS is what typed relations plus joint inference buy.
``roscoe_sc``
    ROSCOE's Self-Consistency, which aggregates pairwise contradiction with a
    ``max`` over *forward* pairs only. Isolates three MRF properties at once:
    accumulation (a second conflict should matter), direction (a backward
    relation should be reachable), and typing.
``discourse``
    Entity/lexical cohesion (DiscoScore's single-document metrics, and with them
    the entity grid). They see noun repetition and nothing else, so they separate
    *cohesion* from *coherence*: a self-contradictory response scores at the
    ceiling if its nouns recur.
``judges``
    LLM raters -- a G-Eval-style rubric judge and a bare 1-5 rating. The practical
    incumbent. Report them through ``judge_with_variance``, since a judge's
    run-to-run spread bounds what the comparison can conclude.
``controls``
    Ladder-blind quantities -- claim count, length, edit distance. If one of
    these reproduces a declared ordering, the ordering was never evidence of
    coherence in the first place.

See ``scripts/run_coherence_baselines.py`` for the driver.
"""

from fact_reasoner.coherence_baselines.base import (
    BaselineScore,
    CoherenceBaseline,
    unordered_pairs,
)
from fact_reasoner.coherence_baselines.controls import (
    CONTROL_BASELINES,
    ClaimCountControl,
    EditDistanceControl,
    ResponseLengthControl,
)
from fact_reasoner.coherence_baselines.discourse import (
    DISCOURSE_BASELINES,
    DiscoScoreLC,
    DiscoScoreRC,
    EntityGraphCoherence,
)
from fact_reasoner.coherence_baselines.judges import (
    DirectCoherenceRating,
    GEvalCoherence,
    judge_with_variance,
    make_backend_generate,
)
from fact_reasoner.coherence_baselines.nli_contradiction import (
    PairwiseNLIContradiction,
)
from fact_reasoner.coherence_baselines.roscoe_sc import RoscoeSelfConsistency

__all__ = [
    "BaselineScore",
    "CoherenceBaseline",
    "unordered_pairs",
    "PairwiseNLIContradiction",
    "RoscoeSelfConsistency",
    "ClaimCountControl",
    "ResponseLengthControl",
    "EditDistanceControl",
    "CONTROL_BASELINES",
    "DiscoScoreRC",
    "DiscoScoreLC",
    "EntityGraphCoherence",
    "DISCOURSE_BASELINES",
    "GEvalCoherence",
    "DirectCoherenceRating",
    "judge_with_variance",
    "make_backend_generate",
]
