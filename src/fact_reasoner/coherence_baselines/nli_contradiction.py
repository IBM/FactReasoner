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

"""Local contradiction detection with no global model.

This is the baseline the coherence MRF has to beat, and the most informative
single number in the comparison. It runs the *same* NLI extractor the factuality
pipeline uses, over the *same* atoms, with no evidence -- and then simply counts
contradictions instead of building a graph:

.. math::
    \\text{score} = 1 - \\frac{|\\{\\text{pairs labelled contradiction}\\}|}
                              {|\\{\\text{pairs scored}\\}|}

Everything the LCS adds over this is: relation *typing* (five couplings, not one),
and *propagation* (a conflict depresses its endpoints, and through them whatever
those endpoints support). So the gap between the two is a direct measurement of
what those two mechanisms are worth, with the decomposition, the extractor and the
absence of evidence all held fixed.

Two predictions worth recording before the numbers land, since a baseline that
fails in a *predicted* way is more informative than one that merely loses:

* It should do creditably where the defect is a **present contradiction** -- a
  planted contradictory pair is exactly what it looks for.
* It should do poorly where the defect is a **missing entailment**. A response
  whose support chain is broken has no contradictory pair anywhere in it, so this
  baseline cannot see the defect at all, while the MRF sees claims left unsupported.

A note on what is *not* done here. The obvious "improvement" -- weighting each
pair by its contradiction probability rather than counting labels -- is available
via ``soft=True``, but the label-counting form is the headline because it is the
form the literature uses (SelfCheckGPT's NLI variant, and the flat
contradiction-rate controls in discourse-coherence work), and comparing against a
metric nobody runs would be a straw man.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from fact_reasoner.coherence_baselines.base import BaselineScore, unordered_pairs
from fact_reasoner.coherence_baselines.batching import CALL_FAILED, run_pairs

#: The label the NLI extractor emits for a contradiction.
CONTRADICTION = "contradiction"


class PairwiseNLIContradiction:
    """One minus the fraction of claim pairs labelled contradictory.

    Args:
        nli_extractor: A :class:`~fact_reasoner.core.nli.NLIExtractor` (or
            anything exposing ``run(premise, hypothesis) -> {"label", "probability"}``).
            Injected rather than constructed so the caller controls the backend
            and so tests can pass a stub.
        soft: When True, accumulate each pair's contradiction *probability*
            instead of counting labels. Reported as a secondary column: it
            separates "the baseline mislabels pairs" from "the baseline labels
            them right but cannot aggregate", which are different failures.
        ground_in_response: When True, prepend the response to the premise so the
            extractor judges the pair in context, matching the relation miner's
            response-grounded regime. Off by default: the point of this baseline
            is to be the *ungrounded local* comparison, and grounding it would
            quietly turn it into a different (better) method.
        name: Report column name.
    """

    def __init__(
        self,
        nli_extractor: Any,
        *,
        soft: bool = False,
        ground_in_response: bool = False,
        show_progress: bool = False,
        throttle: dict | None = None,
        name: str | None = None,
    ):
        self.nli = nli_extractor
        self.soft = soft
        self.ground_in_response = ground_in_response
        self.show_progress = show_progress
        # Extra kwargs for `run_pairs` (rate_per_minute, max_concurrency). Empty
        # means the pipeline-wide defaults, i.e. 1500 requests/minute.
        self.throttle = dict(throttle or {})
        self.name = name or ("nli_contradiction_soft" if soft else "nli_contradiction")

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score one response by its pairwise contradiction rate.

        Args:
            atoms: Atom texts in assertion order.
            response: The response the atoms came from; used only when
                ``ground_in_response`` is set.

        Returns:
            The :class:`BaselineScore`. ``score`` is None when there are fewer
            than two atoms, since a contradiction rate over zero pairs is
            undefined -- and returning 1.0 there would score a single-claim
            response as perfectly coherent, the same vacuity trap the paper
            identifies for the "all relations satisfied" readout.
        """
        started = time.time()
        texts = [t for t in atoms if t and t.strip()]
        n = len(texts)
        if n < 2:
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n,
                pairs_scored=0,
                diagnostics={"reason": "fewer than two atoms; rate undefined"},
            )

        prefix = f"{response.strip()}\n\n" if self.ground_in_response else ""
        contradictions = 0
        soft_mass = 0.0
        scored = 0
        failures = 0
        found: list[dict[str, Any]] = []

        # One throttled batch rather than n(n-1)/2 sequential calls: the shared
        # 1500/min limiter plus bounded concurrency, with failures preserved as
        # CALL_FAILED so a throttled call cannot masquerade as "no contradiction".
        index = list(unordered_pairs(n))
        verdicts = run_pairs(
            self.nli,
            [(prefix + texts[i], texts[j]) for i, j in index],
            show_progress=self.show_progress,
            **self.throttle,
        )

        for (i, j), verdict in zip(index, verdicts):
            if verdict is CALL_FAILED or not isinstance(verdict, dict):
                failures += 1
                continue
            label = str(verdict.get("label") or "").strip().lower()
            prob = verdict.get("probability")
            if not label:
                # An unparseable verdict is a measurement failure, not evidence
                # of consistency. Count it separately and exclude it from the
                # denominator rather than letting it read as "no contradiction".
                failures += 1
                continue
            scored += 1
            if label == CONTRADICTION:
                contradictions += 1
                found.append({"i": i, "j": j, "probability": prob})
                soft_mass += float(prob) if isinstance(prob, (int, float)) else 1.0

        if scored == 0:
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n,
                pairs_scored=0,
                diagnostics={
                    "reason": "every NLI call failed to parse",
                    "call_failures": failures,
                },
            )

        rate = (soft_mass if self.soft else contradictions) / scored
        return BaselineScore(
            name=self.name,
            score=max(0.0, min(1.0, 1.0 - rate)),
            atoms_scored=n,
            pairs_scored=scored,
            diagnostics={
                "contradiction_pairs": contradictions,
                "contradiction_rate": rate,
                "call_failures": failures,
                "pairs": found,
                "soft": self.soft,
                "grounded": self.ground_in_response,
                "seconds": round(time.time() - started, 3),
            },
        )
