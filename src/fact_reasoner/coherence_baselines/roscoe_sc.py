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

"""ROSCOE's Self-Consistency score, adapted to claims.

From the ROSCOE suite (Golovneva et al., ICLR 2023), logical-inference group:

.. math::
    \\text{SC} = 1 - \\max_{i=2..N}\\ \\max_{j<i}\\ p_{\\text{contr}}(h_i, h_j)

ROSCOE was built for step-by-step reasoning chains, where :math:`h_i` is the
*i*-th reasoning step. We apply it to atomic claims, which is a repurposing and is
labelled as such wherever it is reported: a long-form response is not a chain, and
its claims are frequently mutually consistent rather than sequentially derived.

The reason this is the sharpest available competitor is that it is *structured* --
it is not a bag-of-words metric, it uses an entailment model, and it needs no
evidence, exactly like the coherence MRF -- yet it differs from the MRF in three
specific, nameable ways, each of which the MRF claims to improve on:

1. **It is a max, so it cannot accumulate.** One strong contradiction saturates the
   score; a second, third and fourth change nothing. The MRF's marginals fall
   further with each additional conflict.
2. **It is forward-only** (``j < i``), so a relation running backward in claim
   order is unreachable. This is not hypothetical for this corpus: the paper
   measures 20 of 86 gold relations as both directed *and* backward.
3. **It is untyped.** Only contradiction registers. Entailment, equivalence,
   exclusivity and co-necessity are invisible, so a response whose support
   structure has collapsed scores identically to one whose support is intact.

Because those three are separable, this module also exposes the aggregation and
the direction as options. Running ``aggregate="mean"`` and ``symmetric=True``
alongside the faithful configuration turns a single baseline into an ablation that
says *which* of the three differences is doing the work -- which is more useful
than one number that merely loses.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from fact_reasoner.coherence_baselines.base import BaselineScore
from fact_reasoner.coherence_baselines.batching import CALL_FAILED, run_pairs

#: Aggregations over the pairwise contradiction probabilities.
#: ``max`` is ROSCOE as published; ``mean`` is the accumulating variant.
AGGREGATIONS = ("max", "mean")


class RoscoeSelfConsistency:
    """ROSCOE Self-Consistency over claims (adapted).

    Args:
        nli_extractor: Anything exposing
            ``run(premise, hypothesis) -> {"label", "probability"}``. The
            contradiction probability is read from it as described in
            :meth:`_contradiction_probability`.
        aggregate: ``"max"`` for ROSCOE as published (one conflict saturates), or
            ``"mean"`` for the accumulating variant. Report both.
        symmetric: When False (the default, faithful to ROSCOE) only pairs with
            ``j < i`` are scored. When True, both arcs are scored, which removes
            the structural blindness to backward relations.
        show_progress: Draw a progress bar over the throttled batch.
        throttle: Optional ``{"rate_per_minute": int, "max_concurrency": int}``
            overrides for the batched call path. Defaults to the pipeline-wide
            1500 requests/minute.
        name: Report column name. Defaults to a name encoding the configuration,
            so an ablation's columns cannot collide.
    """

    def __init__(
        self,
        nli_extractor: Any,
        *,
        aggregate: str = "max",
        symmetric: bool = False,
        show_progress: bool = False,
        throttle: dict | None = None,
        name: str | None = None,
    ):
        if aggregate not in AGGREGATIONS:
            raise ValueError(
                f"aggregate must be one of {AGGREGATIONS}, got {aggregate!r}."
            )
        self.nli = nli_extractor
        self.aggregate = aggregate
        self.symmetric = symmetric
        self.show_progress = show_progress
        # Extra kwargs for `run_pairs` (rate_per_minute, max_concurrency). Empty
        # means the pipeline-wide defaults, i.e. 1500 requests/minute.
        self.throttle = dict(throttle or {})
        suffix = "" if aggregate == "max" else f"_{aggregate}"
        suffix += "_sym" if symmetric else ""
        self.name = name or f"roscoe_sc{suffix}"

    @staticmethod
    def _contradiction_probability(verdict: dict[str, Any]) -> float | None:
        """Read a contradiction probability from an NLI verdict.

        The extractor reports the probability *of the label it emitted*, not a
        distribution over all three labels, so there is no contradiction mass to
        read when the label is entailment or neutral. Treating those as
        ``p_contr = 0`` is the only sound reading available: the model's stated
        verdict is "not a contradiction", and inventing a residual would fabricate
        a number the extractor never produced.

        Args:
            verdict: The extractor's output.

        Returns:
            The contradiction probability, or None when the verdict is unusable
            (no label at all), so callers can count the failure rather than score it.
        """
        label = str(verdict.get("label") or "").strip().lower()
        if not label:
            return None
        if label != "contradiction":
            return 0.0
        prob = verdict.get("probability")
        return float(prob) if isinstance(prob, (int, float)) else 1.0

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score one response.

        Args:
            atoms: Atom texts in assertion order. Order matters here: the
                faithful configuration only looks backward from each claim.
            response: Unused; accepted so every baseline shares one signature.
                ROSCOE reads the steps, never the surrounding prose.

        Returns:
            The :class:`BaselineScore`, with ``score`` None when there are fewer
            than two atoms or every call failed.
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
                diagnostics={"reason": "fewer than two atoms; SC undefined"},
            )

        probs: list[float] = []
        failures = 0
        worst = {"probability": -1.0, "i": None, "j": None}

        # Enumerate the arcs this configuration asks about, then issue them as one
        # throttled batch. Faithful ROSCOE only looks backward (j < i); the
        # symmetric arm adds the forward arc so the backward-blindness can be
        # ablated rather than merely described.
        index: list[tuple[int, int]] = []
        for i in range(1, n):
            for j in range(i):
                index.append((i, j))
                if self.symmetric:
                    index.append((j, i))

        verdicts = run_pairs(
            self.nli,
            [(texts[src], texts[trg]) for src, trg in index],
            show_progress=self.show_progress,
            **self.throttle,
        )

        for (src, trg), verdict in zip(index, verdicts):
            if verdict is CALL_FAILED or not isinstance(verdict, dict):
                failures += 1
                continue
            p = self._contradiction_probability(verdict)
            if p is None:
                failures += 1
                continue
            probs.append(p)
            if p > worst["probability"]:
                worst = {"probability": p, "i": src, "j": trg}

        if not probs:
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

        conflict = max(probs) if self.aggregate == "max" else sum(probs) / len(probs)
        return BaselineScore(
            name=self.name,
            score=max(0.0, min(1.0, 1.0 - conflict)),
            atoms_scored=n,
            pairs_scored=len(probs),
            diagnostics={
                "aggregate": self.aggregate,
                "symmetric": self.symmetric,
                "conflict_term": conflict,
                "worst_pair": worst if worst["i"] is not None else None,
                "call_failures": failures,
                "seconds": round(time.time() - started, 3),
            },
        )
