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

# The two-level relation taxonomy for logical coherence.
#
# This module encodes the relation scheme committed to in
# ``docs/ideation/coherence_mrf_deepdive.pdf`` (Sections 2-3) and
# ``docs/ideation/research_plan.md`` (Section 3):
#
#   * Level 1 -- the inferential *coupling* the Markov network understands:
#     exactly FactReasoner's NLI edge types {entailment, contradiction,
#     equivalence} plus NONE (no factor). Each maps 1:1 to a with-priors factor
#     table in ``network_builder.edge_factor_values``.
#
#   * Level 2 -- the interpretable *discourse sense* (PDTB 3.0 top classes / RST).
#     A Level-2 sense is not a new factor type: it *compiles down* to a Level-1
#     coupling via the deterministic map ``COMPILE`` (deep-dive Table 2), which
#     also carries a *strength prior* (a starting point for the confidence, e.g.
#     Restatement starts near 0.90) and marks the one structural case that needs
#     special handling: Concession, a contradiction the text itself resolves.

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ----------------------------------------------------------------------------
# Level 1 -- inferential couplings (the factor types).
# ----------------------------------------------------------------------------

# The Level-1 coupling string values are exactly the relation ``type`` values
# that ``fact_graph.Edge`` / ``core.base.Relation`` accept, so a mined relation
# round-trips into the existing graph machinery unchanged. NONE means "no edge".
LEVEL1_ENTAILMENT = "entailment"
LEVEL1_CONTRADICTION = "contradiction"
LEVEL1_EQUIVALENCE = "equivalence"
LEVEL1_NONE = "none"

LEVEL1_COUPLINGS = (
    LEVEL1_ENTAILMENT,
    LEVEL1_CONTRADICTION,
    LEVEL1_EQUIVALENCE,
    LEVEL1_NONE,
)

# Couplings that produce an actual pairwise factor (everything except NONE).
LEVEL1_EDGE_COUPLINGS = (
    LEVEL1_ENTAILMENT,
    LEVEL1_CONTRADICTION,
    LEVEL1_EQUIVALENCE,
)


# ----------------------------------------------------------------------------
# Level 2 -- discourse senses (the interpretable inventory).
# ----------------------------------------------------------------------------


class Level2Sense(str, Enum):
    """PDTB/RST-grounded discourse senses mined between two atoms.

    Inherits from ``str`` so the enum values are usable directly as strings
    (JSON serialization, dict keys, prompt substitution). The string values are
    the exact tokens the mining prompt is asked to emit (see ``prompts.py``).
    """

    CAUSE_EFFECT = "Cause-Effect"
    EFFECT_CAUSE = "Effect-Cause"
    EVIDENCE = "Evidence"
    CONDITION = "Condition"
    RESTATEMENT = "Restatement"
    INSTANTIATION = "Instantiation"
    CONTRAST = "Contrast"
    CONCESSION = "Concession"
    PRECEDENCE = "Precedence"
    SUCCESSION = "Succession"
    NONE = "None"

    @classmethod
    def from_string(cls, value: str) -> "Level2Sense":
        """Parse a (possibly noisy) sense string to a :class:`Level2Sense`.

        Matching is case-insensitive and tolerant of separators (``cause_effect``,
        ``cause effect``, ``Cause-Effect`` all match ``CAUSE_EFFECT``). Unknown or
        empty values map to :attr:`NONE`, so a mis-formatted LLM answer degrades to
        "no relation" rather than crashing.

        Args:
            value: The raw sense string from the LLM (or a caller).

        Returns:
            The matching :class:`Level2Sense`, or :attr:`NONE` if unrecognized.
        """
        if not value:
            return cls.NONE
        norm = value.strip().lower().replace("_", "-").replace(" ", "-")
        for sense in cls:
            if sense.value.lower() == norm:
                return sense
        # Also accept the bare enum-member name, e.g. "cause_effect".
        key = value.strip().upper().replace("-", "_").replace(" ", "_")
        return cls.__members__.get(key, cls.NONE)


# ----------------------------------------------------------------------------
# The compile map C : Level-2 sense -> Level-1 coupling (deep-dive Table 2).
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SenseSpec:
    """How a Level-2 discourse sense compiles to a Level-1 coupling.

    Attributes:
        level1: The Level-1 coupling this sense maps to (one of
            ``LEVEL1_COUPLINGS``).
        strength_prior: A starting point for the relation strength. ``None``
            means "use the miner's own confidence" (e.g. a Cause-Effect's
            strength is whatever the strength prompt reports); a float is a fixed
            prior the estimate refines toward (e.g. Restatement starts near 0.90).
        directed: Whether the sense is inherently directed (source->target
            matters). Symmetric senses (Restatement, Contrast) are undirected.
        is_concession: Whether this is the Concession case -- a contradiction the
            text may itself resolve, which the network builder discounts when a
            resolving holding atom is present (deep-dive Eq. 2).
        ordering_only: Whether the sense carries no truth coupling (Level-1 NONE)
            but does record source/target order (Temporal Precedence/Succession).
    """

    level1: str
    strength_prior: Optional[float]
    directed: bool
    is_concession: bool = False
    ordering_only: bool = False


# Deep-dive Table 2: "Compiling Level-2 discourse senses to Level-1 factors."
COMPILE = {
    Level2Sense.CAUSE_EFFECT: SenseSpec(LEVEL1_ENTAILMENT, None, directed=True),
    Level2Sense.EFFECT_CAUSE: SenseSpec(LEVEL1_ENTAILMENT, None, directed=True),
    Level2Sense.EVIDENCE: SenseSpec(LEVEL1_ENTAILMENT, None, directed=True),
    Level2Sense.CONDITION: SenseSpec(LEVEL1_ENTAILMENT, None, directed=True),
    Level2Sense.INSTANTIATION: SenseSpec(LEVEL1_ENTAILMENT, None, directed=True),
    Level2Sense.RESTATEMENT: SenseSpec(LEVEL1_EQUIVALENCE, 0.90, directed=False),
    Level2Sense.CONTRAST: SenseSpec(LEVEL1_CONTRADICTION, None, directed=False),
    Level2Sense.CONCESSION: SenseSpec(
        LEVEL1_CONTRADICTION, None, directed=True, is_concession=True
    ),
    Level2Sense.PRECEDENCE: SenseSpec(
        LEVEL1_NONE, None, directed=True, ordering_only=True
    ),
    Level2Sense.SUCCESSION: SenseSpec(
        LEVEL1_NONE, None, directed=True, ordering_only=True
    ),
    Level2Sense.NONE: SenseSpec(LEVEL1_NONE, None, directed=False),
}


def compile_sense(sense, raw_p: Optional[float] = None):
    """Compile a Level-2 sense to a Level-1 coupling and effective strength.

    Applies the deterministic map ``COMPILE`` (deep-dive Table 2). The effective
    strength is the miner's own confidence (``raw_p``) unless the sense carries a
    fixed strength prior, in which case that prior is used when ``raw_p`` is not
    supplied (and otherwise the prior seeds but the miner's estimate wins).

    Args:
        sense: A :class:`Level2Sense` or a sense string (parsed via
            :meth:`Level2Sense.from_string`).
        raw_p: The miner's strength estimate for this relation, if available.

    Returns:
        A tuple ``(level1_coupling, effective_strength, spec)`` where
        ``effective_strength`` is ``None`` when the coupling is NONE (no factor).
    """
    if not isinstance(sense, Level2Sense):
        sense = Level2Sense.from_string(str(sense))
    spec = COMPILE[sense]

    if spec.level1 == LEVEL1_NONE:
        return LEVEL1_NONE, None, spec

    if raw_p is not None:
        effective = raw_p
    elif spec.strength_prior is not None:
        effective = spec.strength_prior
    else:
        # No estimate and no prior: fall back to an uninformative 0.5.
        effective = 0.5

    return spec.level1, effective, spec


def coupling_from_string(value: str) -> str:
    """Normalize a raw coupling string to one of ``LEVEL1_COUPLINGS``.

    Accepts the exact coupling tokens as well as common variants (``neutral`` and
    ``independent`` map to ``none``). Unknown values map to ``none`` so a noisy
    LLM answer degrades to "no relation".

    Args:
        value: The raw coupling string from the LLM.

    Returns:
        One of ``LEVEL1_COUPLINGS``.
    """
    if not value:
        return LEVEL1_NONE
    v = value.strip().lower()
    if "entail" in v:
        return LEVEL1_ENTAILMENT
    if "contradict" in v:
        return LEVEL1_CONTRADICTION
    if "equiv" in v:
        return LEVEL1_EQUIVALENCE
    # neutral / none / independent / no relation -> NONE
    return LEVEL1_NONE
