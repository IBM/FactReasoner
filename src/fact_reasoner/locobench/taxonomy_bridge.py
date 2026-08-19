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

# The single import point for sense/coupling facts.
#
# `lcs.taxonomy.COMPILE` is the authority: a gold edge whose coupling disagrees with
# COMPILE[sense].level1 is malformed, and `schema.validate_item` asserts exactly that.
# Everything in the harness that needs to know about senses comes through here, so the
# taxonomy is imported in one place and the harness cannot drift from the shipped code.

from __future__ import annotations

from fact_reasoner.lcs.taxonomy import (
    COMPILE,
    LEVEL1_CONFLICT_COUPLINGS,
    LEVEL1_COUPLINGS,
    Level2Sense,
)

# The 13 senses, in taxonomy declaration order.
LEGAL_SENSES: tuple[str, ...] = tuple(s.value for s in Level2Sense)

# The 5 couplings plus `none`.
LEGAL_COUPLINGS: tuple[str, ...] = tuple(LEVEL1_COUPLINGS)

# The couplings with a both-true world to soften -- which is what the concession
# discount needs, and why `exclusive` belongs here alongside `contradiction`.
#
# NOTE this is no longer the same set as the `consistency` readout's conflict event:
# that readout keys on `contradiction` alone and credits `exclusive` in its support
# term instead (see `lcs/lcs_scorer.py`). Do not treat this alias as "what
# consistency counts as a conflict".
CONFLICT_COUPLINGS: tuple[str, ...] = tuple(LEVEL1_CONFLICT_COUPLINGS)

# The senses with no gold data in any existing dataset -- the reason the benchmark
# exists (Phase 1 Section 3.1). Phase 2's `--report` counts edges against these.
NEW_SENSES: tuple[str, ...] = (
    "Instantiation",
    "Restatement",
    "Contrast",
    "Concession",
    "Alternative",
    "Disjunction",
)

# The senses P3 must place at least one of in every plan (Phase 1 P3 instruction 5).
# Precedence/Succession count as one requirement, satisfied by either.
REQUIRED_SENSES: tuple[str, ...] = (
    "Alternative",
    "Disjunction",
    "Restatement",
    "Concession",
)
REQUIRED_EITHER: tuple[str, ...] = ("Precedence", "Succession")


def coupling_for_sense(sense: str) -> str:
    """Return the Level-1 coupling a Level-2 sense compiles to.

    Args:
        sense: One of :data:`LEGAL_SENSES`.

    Returns:
        The coupling string, e.g. ``"entailment"``.

    Raises:
        ValueError: If the sense is unknown. Unlike
            ``Level2Sense.from_string``, which falls back to ``None`` for unrecognized
            input, this raises -- a typo'd sense in gold data must not silently become
            a no-factor edge.
    """
    for s in Level2Sense:
        if s.value == sense:
            return COMPILE[s].level1
    raise ValueError(
        f"Unknown sense: {sense!r} (expected one of {list(LEGAL_SENSES)})."
    )


def spec_for_sense(sense: str):
    """Return the full ``SenseSpec`` for a sense (directedness, concession, ordering).

    Args:
        sense: One of :data:`LEGAL_SENSES`.

    Returns:
        The :class:`~fact_reasoner.lcs.taxonomy.SenseSpec`.

    Raises:
        ValueError: If the sense is unknown.
    """
    for s in Level2Sense:
        if s.value == sense:
            return COMPILE[s]
    raise ValueError(
        f"Unknown sense: {sense!r} (expected one of {list(LEGAL_SENSES)})."
    )


def is_directed(sense: str) -> bool:
    """Whether a sense is inherently directed (source -> target matters)."""
    return bool(spec_for_sense(sense).directed)


def is_ordering_only(sense: str) -> bool:
    """Whether a sense carries no truth coupling (Precedence/Succession)."""
    return bool(spec_for_sense(sense).ordering_only)


def is_conflict(sense: str) -> bool:
    """Whether a sense compiles to a conflict coupling (contradiction or exclusive)."""
    return coupling_for_sense(sense) in CONFLICT_COUPLINGS


__all__ = [
    "CONFLICT_COUPLINGS",
    "LEGAL_COUPLINGS",
    "LEGAL_SENSES",
    "NEW_SENSES",
    "REQUIRED_EITHER",
    "REQUIRED_SENSES",
    "coupling_for_sense",
    "is_conflict",
    "is_directed",
    "is_ordering_only",
    "spec_for_sense",
]
