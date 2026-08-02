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

# The six operators, the ten P5 calls they dispatch, the four family ladders, and the
# per-readout ordering expectations.
#
# Everything here is DATA. The ladders are declarative recipes and the expectations are
# derived from them, so `store.py` emits the `expected` blocks mechanically and there is
# no second place where a rung's meaning is decided.
#
# DEFECT 1 (Phase 2, Section 1.3). Phase 1's C1 contract demands a strict increase in
# `log_partition` across all four adjacent rung pairs, but its own reference ladder
# reports log Z = -9.64 at BOTH rung 1 and rung 2: the concession edit changes an edge
# WEIGHT, not the edge SET, and the normalized log-partition readout measures against a
# ceiling and floor built from the same skeleton, so it can be invariant there. That one
# pair is therefore a predicted INVARIANCE, and expectations are emitted per readout per
# adjacent pair rather than as one chain-wide flag.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The three readouts LoCoBench grades (Phase 1 Section 5.1).
READOUTS = ("mean_marginal", "consistency", "log_partition")

# The six operator groups a reader needs.
OPERATORS = {
    "O1": "relabel",
    "O2": "add conflict",
    "O3": "break chain",
    "O4": "drop relation",
    "O5": "resolve / unresolve",
    "O6": "reorder",
}

# The ten parameterized calls the P5 prompt accepts, grouped by operator. Note
# `add_resolution`: the family manifest names it, Phase 1 Section 4.2 mentions only its
# inverse, and every CONFLICT ladder's resolved rung needs it (Phase 2, ambiguity 4).
OPERATOR_CALLS: dict[str, tuple[str, ...]] = {
    "O1": ("wrong_sense", "direction_reversal", "exhaustiveness_flip"),
    "O2": ("inject_contradiction", "spurious_relation"),
    "O3": ("break_chain",),
    "O4": ("drop_relation",),
    "O5": ("remove_resolution", "add_resolution"),
    "O6": ("shuffle_order", "ordering_only"),
}

# call -> operator, for the reverse lookup the store needs.
CALL_TO_OPERATOR: dict[str, str] = {
    call: op for op, calls in OPERATOR_CALLS.items() for call in calls
}

ALL_CALLS: tuple[str, ...] = tuple(CALL_TO_OPERATOR)

FAMILY_TYPES = ("CONFLICT", "CHAIN", "ORDER", "CONTROL")

# Families per type at the Phase-2 target of 120 (Phase 1 tab:families).
FAMILY_COUNTS = {"CONFLICT": 55, "CHAIN": 25, "ORDER": 25, "CONTROL": 15}

RUNG_NAMES = ("worse", "base", "concession_resolved", "fix_one_conflict", "coherent")

# Expectation vocabulary for `expected.readout_directions`.
INCREASE, DECREASE, INVARIANT, UNCONSTRAINED = (
    "increase",
    "decrease",
    "invariant",
    "unconstrained",
)


@dataclass
class Rung:
    """One rung of a ladder: how to build it, and what it is called.

    Attributes:
        index: 0 (least coherent) to 4 (most).
        name: The rung label recorded in the item.
        calls: The P5 calls applied to the parent, in order. Empty means "this rung is
            the base response", which costs no LLM call.
        parent: The rung index this one is derived from, or None for the base.
    """

    index: int
    name: str
    calls: tuple[str, ...] = ()
    parent: int | None = None

    @property
    def is_base(self) -> bool:
        """Whether this rung is the unperturbed response."""
        return not self.calls


@dataclass
class Ladder:
    """A family type's five rungs, plus the expectations they imply.

    Attributes:
        family: One of :data:`FAMILY_TYPES`.
        rungs: Exactly five, index 0..4.
        flat: Readouts expected not to move anywhere in this ladder. CONTROL sets all
            three, which is what makes it a control.
        notes: Free text carried into the manifest.
    """

    family: str
    rungs: tuple[Rung, ...]
    flat: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.rungs) != 5:
            raise ValueError(
                f"{self.family}: a ladder has exactly 5 rungs, got {len(self.rungs)}."
            )
        if [r.index for r in self.rungs] != [0, 1, 2, 3, 4]:
            raise ValueError(f"{self.family}: rung indices must be 0..4 in order.")

    def base_index(self) -> int:
        """The index of the unperturbed rung."""
        for r in self.rungs:
            if r.is_base:
                return r.index
        raise ValueError(f"{self.family}: no base rung (one rung must have no calls).")


# ----------------------------------------------------------------------------
# The four ladders.
# ----------------------------------------------------------------------------

LADDERS: dict[str, Ladder] = {
    # The reference ladder, whose exact values Phase 1 publishes:
    # 0.586 < 0.607 < 0.612 < 0.613 < 0.620 on mean_marginal.
    "CONFLICT": Ladder(
        family="CONFLICT",
        rungs=(
            Rung(0, "worse", ("spurious_relation",), parent=1),
            Rung(1, "base"),
            Rung(2, "concession_resolved", ("add_resolution",), parent=1),
            Rung(3, "fix_one_conflict", ("add_resolution", "drop_relation"), parent=1),
            Rung(
                4,
                "coherent",
                ("add_resolution", "drop_relation", "drop_relation"),
                parent=1,
            ),
        ),
        notes=(
            "The reference ladder. Only the conflict structure varies; atoms, topic and "
            "every other relation are held fixed, which is what makes the ordering "
            "attributable to coherence."
        ),
    ),
    # Chain integrity: break N links, then repair.
    "CHAIN": Ladder(
        family="CHAIN",
        rungs=(
            Rung(0, "break_3", ("break_chain", "break_chain", "break_chain"), parent=3),
            Rung(1, "break_2", ("break_chain", "break_chain"), parent=3),
            Rung(2, "break_1", ("break_chain",), parent=3),
            Rung(3, "base"),
            Rung(4, "coherent", ("drop_relation",), parent=3),
        ),
        notes="Entailment-chain integrity at decreasing break depth.",
    ),
    # Order sensitivity, plus one rung that must be flat.
    "ORDER": Ladder(
        family="ORDER",
        rungs=(
            Rung(0, "shuffle_full", ("shuffle_order",), parent=3),
            Rung(1, "shuffle_block", ("shuffle_order",), parent=3),
            Rung(2, "shuffle_adjacent", ("shuffle_order",), parent=3),
            Rung(3, "base"),
            Rung(4, "ordering_only", ("ordering_only",), parent=3),
        ),
        notes=(
            "Dual purpose: four ordering rungs plus one ordering-only rung that must be "
            "flat, since a Precedence<->Succession edit adds no factor."
        ),
    ),
    # The control: every rung is meaning-preserving, so the correct result is no trend.
    "CONTROL": Ladder(
        family="CONTROL",
        rungs=(
            Rung(0, "ordering_only_a", ("ordering_only",), parent=2),
            Rung(1, "direction_reversal_a", ("direction_reversal",), parent=2),
            Rung(2, "base"),
            Rung(3, "direction_reversal_b", ("direction_reversal",), parent=2),
            Rung(4, "ordering_only_b", ("ordering_only",), parent=2),
        ),
        flat=READOUTS,
        notes=(
            "The control. Every edit preserves meaning and adds no factor, so the "
            "required pattern is a flat line; any monotone trend here is a false "
            "positive and indicates a score tracking edit count rather than coherence."
        ),
    ),
}


def ladder_for(family: str) -> Ladder:
    """Return a family type's ladder.

    Args:
        family: One of :data:`FAMILY_TYPES`.

    Returns:
        The ladder.

    Raises:
        ValueError: If the family type is unknown.
    """
    try:
        return LADDERS[family]
    except KeyError:
        raise ValueError(
            f"Unknown family type: {family!r} (expected one of {list(FAMILY_TYPES)})."
        ) from None


# ----------------------------------------------------------------------------
# Expectations.
# ----------------------------------------------------------------------------

# The pairs where a readout is predicted NOT to move, or to move against the trend.
# Keyed by (family, from_rung, to_rung, readout). This table IS Defect 1's resolution
# plus Phase 1's Finding 1, and it is the only place either is encoded.
_SPECIAL: dict[tuple[str, int, int, str], str] = {
    # Finding 1: the concession discount raises the both-true cell mass, so a readout
    # that counts conflict ACTIVITY dips while belief-readers rise.
    ("CONFLICT", 1, 2, "consistency"): DECREASE,
    ("CONFLICT", 2, 3, "consistency"): UNCONSTRAINED,
    # Defect 1: the same edit is weight-only, so the normalized log partition is
    # invariant -- Phase 1's reference ladder shows -9.64 at both rungs.
    ("CONFLICT", 1, 2, "log_partition"): INVARIANT,
}


def expectations_for(family: str) -> dict[str, dict[str, str]]:
    """Return the per-readout, per-adjacent-pair ordering expectations.

    Args:
        family: One of :data:`FAMILY_TYPES`.

    Returns:
        ``{"0->1": {readout: direction, ...}, ...}`` for the four adjacent pairs.
        Directions are :data:`INCREASE`, :data:`DECREASE`, :data:`INVARIANT` or
        :data:`UNCONSTRAINED`.
    """
    lad = ladder_for(family)
    out: dict[str, dict[str, str]] = {}
    for i in range(4):
        key = f"{i}->{i + 1}"
        out[key] = {}
        for ro in READOUTS:
            if ro in lad.flat:
                out[key][ro] = INVARIANT
            else:
                out[key][ro] = _SPECIAL.get((family, i, i + 1, ro), INCREASE)
    return out


def readout_directions(family: str, rung_index: int) -> dict[str, str] | None:
    """The expectations for one rung, relative to the rung below it.

    This is what an item's ``expected.readout_directions`` carries.

    Args:
        family: One of :data:`FAMILY_TYPES`.
        rung_index: 0..4.

    Returns:
        ``{readout: direction}``, or None for rung 0, which has nothing below it.
    """
    if rung_index == 0:
        return None
    return expectations_for(family)[f"{rung_index - 1}->{rung_index}"]


def ordering_constraints(family: str) -> list[dict[str, Any]]:
    """Build the family manifest's ordering constraints.

    Three classes, following Phase 1 Section 5.3 with Defect 1 applied:

    * **C1** -- strict increase, listed per readout per pair, so the one invariant pair
      is simply absent from it rather than requiring a chain-wide exemption.
    * **C2** -- the predicted inversions and invariances, asserted positively: a system
      that is monotone here is not implementing the concession discount.
    * **C3** -- endpoint separation, the robust headline claim, for all three readouts.

    Args:
        family: One of :data:`FAMILY_TYPES`.

    Returns:
        A list of constraint dicts ready for the manifest.
    """
    exp = expectations_for(family)
    lad = ladder_for(family)

    c1: list[dict[str, Any]] = []
    c2: list[dict[str, Any]] = []
    for pair, per_readout in exp.items():
        lo, hi = (int(x) for x in pair.split("->"))
        for ro, direction in per_readout.items():
            entry = {"readout": ro, "pair": [lo, hi]}
            if direction == INCREASE:
                c1.append(entry)
            elif direction in (DECREASE, INVARIANT):
                c2.append({**entry, "expect": direction})
            # UNCONSTRAINED pairs are deliberately in neither list.

    constraints: list[dict[str, Any]] = []
    if c1:
        constraints.append({"id": "c1", "class": "C1", "strict": True, "pairs": c1})
    if c2:
        constraints.append({"id": "c2", "class": "C2", "strict": False, "pairs": c2})
    if not lad.flat:
        # A control family makes no endpoint claim -- its endpoints must be equal.
        constraints.append(
            {
                "id": "c3",
                "class": "C3",
                "strict": True,
                "readouts": list(READOUTS),
                "required": [[0, 4]],
            }
        )
    else:
        constraints.append(
            {
                "id": "c3-flat",
                "class": "C3",
                "strict": False,
                "readouts": list(lad.flat),
                "invariant": [[0, 4]],
            }
        )
    return constraints


def plan_rungs(family: str) -> list[dict[str, Any]]:
    """Describe a family's five rungs for the manifest.

    Args:
        family: One of :data:`FAMILY_TYPES`.

    Returns:
        One dict per rung: ``index``, ``name``, ``calls``, ``parent``, ``is_base``.
    """
    return [
        {
            "index": r.index,
            "name": r.name,
            "calls": list(r.calls),
            "parent": r.parent,
            "is_base": r.is_base,
        }
        for r in ladder_for(family).rungs
    ]


def p5_calls_for(family: str) -> int:
    """Count the P5 calls a family's ladder dispatches.

    Not a constant, and not uniform across family types: a single rung may compose
    several operator calls, so CONFLICT and CHAIN cost 7 while ORDER and CONTROL cost 4.
    Summing the ladder is the only reliable way to count them -- a hard-coded 4 (one per
    non-base rung) undercounts CONFLICT by three calls per family, which is 360 calls
    over a 120-family corpus.

    Args:
        family: One of :data:`FAMILY_TYPES`.

    Returns:
        The number of P5 calls.
    """
    return sum(len(r.calls) for r in ladder_for(family).rungs)


def call_budget(
    families: "list[str] | dict[str, int]",
    *,
    n_voters: int = 4,
    conflict_edges_per_family: int = 3,
    inline_responses: int = 1,
) -> dict[str, int]:
    """Project the LLM call budget for a set of families, per prompt identifier.

    Derived from :data:`LADDERS` rather than hard-coded, so the projection cannot drift
    from the ladders actually executed.

    Args:
        families: Either a list of family-type labels (one entry per family) or a
            ``{family_type: count}`` mapping.
        n_voters: Committee models per item, excluding the generator (R3).
        conflict_edges_per_family: V2 runs per conflict edge per *family*, not per item.
        inline_responses: Responses the inline V1/V3/V4 gate gets run on. The harness
            currently audits the base response only (1); Phase 1's V3 scope is all five.

    Returns:
        Calls per prompt id, plus ``generation``, ``committee`` and ``total`` subtotals.
    """
    if isinstance(families, dict):
        counts = dict(families)
    else:
        counts = {}
        for f in families:
            counts[f] = counts.get(f, 0) + 1

    n_fam = sum(counts.values())
    n_items = n_fam * len(RUNG_NAMES)
    budget = {
        "P1": n_fam,
        "P2": n_fam,
        "P3": n_fam,
        "P4": n_fam,
        "P5": sum(p5_calls_for(f) * n for f, n in counts.items()),
        "V1": n_fam * inline_responses + n_items * n_voters,
        "V2": n_fam * conflict_edges_per_family * n_voters,
        "V3": n_fam * inline_responses,
        "V4": n_fam * inline_responses + n_items * n_voters,
    }
    committee = n_items * n_voters * 2 + budget["V2"]
    budget["committee"] = committee
    budget["total"] = sum(budget[k] for k in ("P1", "P2", "P3", "P4", "P5")) + sum(
        budget[k] for k in ("V1", "V2", "V3", "V4")
    )
    budget["generation"] = budget["total"] - committee
    return budget


def family_type_slots(n_families: int) -> list[str]:
    """Assign family types to family slots, proportional to :data:`FAMILY_COUNTS`.

    Deterministic, so a resumed run assigns the same type to the same slot.

    Args:
        n_families: Total families.

    Returns:
        A list of ``n_families`` family-type labels.
    """
    total = sum(FAMILY_COUNTS.values())
    slots: list[str] = []
    for fam in FAMILY_TYPES:
        share = FAMILY_COUNTS[fam] * n_families / total
        slots.extend([fam] * int(round(share)))
    # Rounding can leave the list a little short or long; pad or trim with the largest
    # type, which is CONFLICT.
    while len(slots) < n_families:
        slots.append("CONFLICT")
    return slots[:n_families]


__all__ = [
    "ALL_CALLS",
    "CALL_TO_OPERATOR",
    "DECREASE",
    "FAMILY_COUNTS",
    "FAMILY_TYPES",
    "INCREASE",
    "INVARIANT",
    "LADDERS",
    "OPERATORS",
    "OPERATOR_CALLS",
    "READOUTS",
    "RUNG_NAMES",
    "UNCONSTRAINED",
    "Ladder",
    "Rung",
    "call_budget",
    "expectations_for",
    "family_type_slots",
    "ladder_for",
    "ordering_constraints",
    "p5_calls_for",
    "plan_rungs",
    "readout_directions",
]
