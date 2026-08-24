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

"""Ladder-blind controls: quantities that cannot possibly measure coherence.

These are the cheapest and most important baselines in the suite. Each is a
surface property of the response -- how many claims it has, how long it is, how
far it was edited from a reference -- computed with no model at all. None of them
can see whether an argument hangs together.

That is the point. If a control reproduces a declared coherence ordering, then
satisfying that ordering was never evidence of measuring coherence, and a
sophisticated metric that agrees with the ordering may be agreeing with the
confound instead. Steen and Markert (COLING 2022) show this is a real failure
mode rather than a theoretical one: coherence metrics can rank systems correctly
while tracking length or surface form, which is why they argue for exactly this
kind of control and for reporting intra-system rather than pooled correlation.

Interpretation, stated plainly because it is easy to get backwards: a control
*passing* a ladder is bad news for the ladder, not good news for the control.

A note on directionality. A control has to be mapped into "higher = more
coherent" to sit in the same table as the other baselines, and for most of these
there is no defensible direction -- there is no reason longer responses should be
more coherent, or less. Rather than pick one and quietly encode a hypothesis, each
control reports its raw quantity in ``diagnostics`` and a *normalized* score whose
only guarantee is monotonicity in that quantity. The ladder scorer should be read
for whether the control *tracks* the declared ordering, in either direction.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from fact_reasoner.coherence_baselines.base import BaselineScore

#: Saturation point for the claim-count control's normalization. Chosen well
#: above the corpus median (the ladder items carry sixteen claims) so the
#: normalization is not clipping in the range that matters.
_CLAIM_SATURATION = 64.0

#: Saturation point, in tokens, for the length control.
_TOKEN_SATURATION = 1024.0


def _tokens(text: str) -> list[str]:
    """Split on non-word characters; good enough for a length control."""
    return [t for t in re.split(r"\W+", text or "") if t]


class ClaimCountControl:
    """Number of atoms, normalized. Sees nothing about how they relate.

    This is the control that matters most for the paper's ordering ladders,
    because several perturbation operators *add or remove* a claim, and a metric
    that merely counted claims would follow those rungs perfectly.
    """

    name = "control_claim_count"

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score by atom count."""
        n = len([t for t in atoms if t and t.strip()])
        return BaselineScore(
            name=self.name,
            score=min(1.0, n / _CLAIM_SATURATION),
            atoms_scored=n,
            diagnostics={"claim_count": n, "raw": n},
        )


class ResponseLengthControl:
    """Response length in tokens, normalized.

    The confound Steen and Markert single out. Also the one an LLM judge is most
    often accused of following, which makes it the natural companion column to a
    judge baseline.
    """

    name = "control_length"

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score by response token count."""
        n_tok = len(_tokens(response))
        return BaselineScore(
            name=self.name,
            score=min(1.0, n_tok / _TOKEN_SATURATION),
            atoms_scored=len([t for t in atoms if t and t.strip()]),
            diagnostics={"tokens": n_tok, "raw": n_tok},
        )


class EditDistanceControl:
    """Similarity to a reference response, as a proxy for "how much was edited".

    A coherence ladder is built by applying perturbation operators to a base
    response, so edit distance from the base is almost a count of the operators
    applied. A metric that tracked it would reproduce the ladder ordering while
    knowing nothing about coherence -- which is the single most likely way for a
    ladder result to be spurious.

    Args:
        reference: The base response to compare against. When None, the control
            reports no score (there is nothing to be distant from); this keeps it
            usable on the standalone fixtures, where it simply abstains rather
            than inventing a baseline.
    """

    name = "control_edit_distance"

    def __init__(self, reference: str | None = None):
        self.reference = reference

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score by similarity ratio to ``reference``."""
        n = len([t for t in atoms if t and t.strip()])
        if not self.reference:
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n,
                diagnostics={"reason": "no reference response supplied"},
            )
        ratio = SequenceMatcher(None, self.reference, response or "").ratio()
        return BaselineScore(
            name=self.name,
            score=max(0.0, min(1.0, ratio)),
            atoms_scored=n,
            diagnostics={"similarity_to_reference": ratio, "raw": ratio},
        )


#: The controls that need no configuration, ready to run.
CONTROL_BASELINES = (ClaimCountControl(), ResponseLengthControl())
