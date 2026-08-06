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
#
# DEFECT 2. A rung's TEXT was perturbed but its GOLD RELATIONS were not: every rung of a
# family was emitted with the base plan's relation list, so the labels described the base
# ladder rather than the rung they shipped with. Every readout was then identical across a
# family, and the C1/C3 strict-increase constraints could not hold by construction. The
# operator-to-edge-set transforms below (`apply_calls`) close that gap: a rung's gold
# relations are the base relations with its own perturbations applied. Two things were
# wrong and both are fixed here -- the relations were never transformed at all, and every
# P5 call targeted the hardcoded edge `r000`, so a two-`drop_relation` rung dropped the
# same edge twice (see `plan_targets`).

from __future__ import annotations

import re
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
    inline_responses: int = 1,
) -> dict[str, int]:
    """Project the LLM call budget for a set of families, per prompt identifier.

    Derived from :data:`LADDERS` rather than hard-coded, so the projection cannot drift
    from the ladders actually executed.

    Phase 1's V2 exhaustiveness adjudicator carried its own per-conflict-edge term here. It
    is gone with the prompt: ``exclusive`` and ``co_necessity`` are derived from the sense by
    ``taxonomy_bridge.COMPILE``, so no model call is needed to assign them.

    Args:
        families: Either a list of family-type labels (one entry per family) or a
            ``{family_type: count}`` mapping.
        n_voters: Committee models per item, excluding the generator (R3).
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
        "V3": n_fam * inline_responses,
        "V4": n_fam * inline_responses + n_items * n_voters,
    }
    committee = n_items * n_voters * 2
    budget["committee"] = committee
    budget["total"] = sum(budget[k] for k in ("P1", "P2", "P3", "P4", "P5")) + sum(
        budget[k] for k in ("V1", "V3", "V4")
    )
    budget["generation"] = budget["total"] - committee
    return budget


# ----------------------------------------------------------------------------
# Operator -> edge-set transforms (Defect 2).
# ----------------------------------------------------------------------------

# The senses a `wrong_sense` relabel maps to, per current sense. Chosen so the relabel
# changes the Level-1 coupling (a relabel that compiled to the same factor would be
# invisible to every readout, and so untestable).
_WRONG_SENSE_MAP: dict[str, str] = {
    "Cause-Effect": "Contrast",
    "Effect-Cause": "Contrast",
    "Evidence": "Contrast",
    "Condition": "Contrast",
    "Instantiation": "Contrast",
    "Restatement": "Contrast",
    "Contrast": "Restatement",
    "Concession": "Restatement",
    "Alternative": "Disjunction",
    "Disjunction": "Alternative",
    "Precedence": "Cause-Effect",
    "Succession": "Cause-Effect",
}

# `exhaustiveness_flip` toggles the exhaustive/non-exhaustive reading of an opposition:
# Alternative (exactly one holds -> `exclusive`) <-> Contrast (mere opposition ->
# `contradiction`). This is the boundary the corpus exists to grade.
_EXHAUSTIVENESS_FLIP: dict[str, str] = {
    "Alternative": "Contrast",
    "Contrast": "Alternative",
}

# `ordering_only` swaps the direction of a temporal edge without adding any coupling.
_ORDERING_FLIP: dict[str, str] = {
    "Precedence": "Succession",
    "Succession": "Precedence",
}

# Which senses each call can legally target. A call whose ladder finds no eligible edge is
# a no-op on the edge set, recorded rather than silently skipped.
_ELIGIBLE_SENSES: dict[str, tuple[str, ...]] = {
    "exhaustiveness_flip": tuple(_EXHAUSTIVENESS_FLIP),
    "ordering_only": tuple(_ORDERING_FLIP),
}

# Calls that leave the MRF's factor set unchanged BY DESIGN, so a rung built only from them
# is *supposed* to score the same as its parent:
#   * shuffle_order  -- moves sentences; adds, removes and retypes nothing.
#   * ordering_only  -- swaps Precedence for Succession, both of which compile to Level-1
#                       `none` and so contribute no factor either way.
# The ORDER and CONTROL ladders are built on exactly this invariance -- a score that moves
# here is tracking edit count rather than coherence -- so the per-rung edge-effect gate must
# exempt them instead of rejecting the families that test it.
EDGE_INVARIANT_CALLS: tuple[str, ...] = ("shuffle_order", "ordering_only")


def _edge_key(edge: dict[str, Any]) -> tuple[str, str]:
    """The (source, target) identity of an edge."""
    return (str(edge.get("source_id")), str(edge.get("target_id")))


def _conflict_edges(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Edges a `drop_relation` may remove to raise coherence, worst-first.

    Two constraints make this ordering, not an arbitrary one:

    * Only CONFLICT couplings are eligible. Dropping an entailment removes a
      mass-concentrating backbone edge and *lowers* coherence, which would invert the very
      rung order the ladder asserts.
    * A resolved concession is skipped: it is already softened, and removing it outright
      would discard the structure rungs 2-4 exist to exercise.

    Ordering: planted-INVALID conflicts first, since those are the ones a "fix one
    conflict" rung most plausibly removes, then the strongest bands. Valid conflicts remain
    eligible -- P5 is asked to delete the connective realizing the edge, and once the text
    no longer draws the relation the label must follow it, whether or not the relation was
    a planted error. Excluding them would make `drop_relation` a no-op on any plan without
    planted conflicts, which is silently no ladder at all.
    """
    band_rank = {"strong": 0, "moderate": 1, "weak": 2}
    conflicts = [
        e
        for e in relations
        if e.get("level1_coupling") in ("contradiction", "exclusive")
        and not e.get("is_resolved_concession")
    ]
    return sorted(
        conflicts,
        key=lambda e: (
            0 if e.get("validity") != "valid" else 1,
            band_rank.get(str(e.get("intended_strength_band")), 3),
            str(e.get("id")),
        ),
    )


def _resolvable_edges(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conflict edges an `add_resolution` may resolve, most-damaging first.

    A resolution attaches an adjudicating holding to a conflict the text already draws, so
    a VALID conflict is the right target -- resolving a planted-invalid edge would dress up
    an error rather than settle a real tension. Concession-sensed edges come first (the
    sense already says the text conceded the tension), then other valid conflicts.
    """
    band_rank = {"strong": 0, "moderate": 1, "weak": 2}
    eligible = [
        e
        for e in relations
        if e.get("level1_coupling") in ("contradiction", "exclusive")
        and e.get("validity") == "valid"
        and not e.get("is_resolved_concession")
    ]
    return sorted(
        eligible,
        key=lambda e: (
            0 if e.get("is_concession") else 1,
            band_rank.get(str(e.get("intended_strength_band")), 3),
            str(e.get("id")),
        ),
    )


def plan_targets(family: str, relations: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Choose which edge each rung's each call targets.

    The harness previously passed the literal edge ``r000`` to every P5 call, so a rung
    composing two ``drop_relation`` calls asked twice for the same edge and the second
    call had nothing left to do. Targets are assigned here, per rung, without repetition
    within a rung.

    Args:
        family: One of :data:`FAMILY_TYPES`.
        relations: The base plan's gold edges (schema shape).

    Returns:
        ``{rung_index: [edge_id per call]}`` for the non-base rungs. An entry is the empty
        string when no eligible edge exists for that call.
    """
    lad = ladder_for(family)
    drop_order = [str(e["id"]) for e in _conflict_edges(relations)]
    resolvable = [str(e["id"]) for e in _resolvable_edges(relations)]
    by_id = {str(e["id"]): e for e in relations}

    out: dict[int, list[str]] = {}
    for rung in lad.rungs:
        if rung.is_base:
            continue
        used: set[str] = set()
        targets: list[str] = []
        drops = 0
        for call in rung.calls:
            target = ""
            if call == "drop_relation":
                # Successive drops walk down the conflict list.
                for eid in drop_order:
                    if eid not in used:
                        target = eid
                        break
                drops += 1
            elif call == "add_resolution":
                for eid in resolvable:
                    if eid not in used:
                        target = eid
                        break
            elif call == "remove_resolution":
                # The inverse: it needs an ALREADY-resolved concession.
                for eid, edge in by_id.items():
                    if eid not in used and edge.get("is_resolved_concession"):
                        target = eid
                        break
            elif call in _ELIGIBLE_SENSES:
                for eid, edge in by_id.items():
                    if eid in used:
                        continue
                    if edge.get("level2_sense") in _ELIGIBLE_SENSES[call]:
                        target = eid
                        break
            elif call == "spurious_relation":
                # Adds a NEW edge between two unrelated atoms; no existing edge target.
                target = ""
            else:
                for eid in by_id:
                    if eid not in used:
                        target = eid
                        break
            if target:
                used.add(target)
            targets.append(target)
        out[rung.index] = targets
    return out


def _next_edge_id(relations: list[dict[str, Any]]) -> str:
    """A fresh ``rNNN`` id, one past the highest currently present."""
    highest = -1
    for e in relations:
        m = re.fullmatch(r"r(\d+)", str(e.get("id", "")))
        if m:
            highest = max(highest, int(m.group(1)))
    return f"r{highest + 1:03d}"


def _spurious_pair(
    relations: list[dict[str, Any]], non_relations: list[dict[str, Any]]
) -> tuple[str, str] | None:
    """Pick an atom pair the base response kept separate, for `spurious_relation`.

    Prefers a declared non-relation -- the plan's own statement that these two atoms are
    unrelated is exactly what makes a link between them spurious.
    """
    existing = {_edge_key(e) for e in relations}
    for nr in non_relations or []:
        pair = (str(nr.get("source_id")), str(nr.get("target_id")))
        if pair not in existing and tuple(reversed(pair)) not in existing:
            return pair
    return None


def apply_calls(
    relations: list[dict[str, Any]],
    calls: "tuple[str, ...] | list[str]",
    *,
    targets: "list[str] | None" = None,
    non_relations: "list[dict[str, Any]] | None" = None,
    generator: str = "perturbation",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply a rung's P5 calls to the base gold edge set.

    This is the label-side counterpart of the P5 text edit: a rung's gold relations are
    the base relations with its own perturbations applied, so the labels describe the rung
    that ships with them. Derived fields (coupling, directedness, ordering-only,
    exhaustiveness) are always recomputed from the sense via ``taxonomy_bridge``, so a
    transformed edge cannot contradict ``COMPILE``.

    Args:
        relations: The base gold edges (schema shape). Not mutated.
        calls: The rung's calls, in order.
        targets: Edge id per call (from :func:`plan_targets`). Defaults to the first
            eligible edge per call.
        non_relations: The plan's declared non-relations, used to site a
            ``spurious_relation``.
        generator: Recorded in the provenance of an added or edited edge.

    Returns:
        ``(relations, non_relations, log)`` -- the transformed edge list (deep-copied), the
        non-relations with any pair that gained an edge removed, and one log entry per
        call, each with ``call``, ``target``, ``effect`` and a ``detail``.

        The non-relations are returned because adding an edge between a declared
        non-relation pair would otherwise leave the item asserting that the same pair both
        is and is not related -- which ``schema.validate_item`` rejects, correctly. A pair
        that gains an edge stops being a declared non-relation for that rung.

    Raises:
        ValueError: If a call is not one of :data:`ALL_CALLS`.
    """
    import copy

    from fact_reasoner.locobench.taxonomy_bridge import (
        coupling_for_sense,
        is_directed,
        is_ordering_only,
    )

    out = copy.deepcopy(list(relations))
    nons = copy.deepcopy(list(non_relations or []))
    log: list[dict[str, Any]] = []
    targets = list(targets or [])

    def _drop_non_relation(pair: tuple[str, str]) -> None:
        """Forget a declared non-relation whose pair just gained an edge."""
        nonlocal nons
        nons = [
            nr
            for nr in nons
            if (str(nr.get("source_id")), str(nr.get("target_id")))
            not in (pair, pair[::-1])
        ]

    def _retype(edge: dict[str, Any], sense: str) -> None:
        """Re-derive every taxonomy-governed field from a new sense."""
        coupling = coupling_for_sense(sense)
        edge["level2_sense"] = sense
        edge["level1_coupling"] = coupling
        edge["directed"] = is_directed(sense)
        edge["ordering_only"] = is_ordering_only(sense)
        edge["is_concession"] = sense == "Concession"
        if coupling in ("contradiction", "exclusive"):
            edge["exhaustive"] = coupling == "exclusive"
        else:
            edge.pop("exhaustive", None)
            # Only a conflict edge can carry a resolution.
            edge["is_resolved_concession"] = False
            edge["resolver_atom_id"] = None
        edge.setdefault("provenance", {})["perturbed_by"] = generator

    def _find(eid: str) -> dict[str, Any] | None:
        return next((e for e in out if str(e.get("id")) == eid), None)

    for i, call in enumerate(calls):
        if call not in ALL_CALLS:
            raise ValueError(
                f"Unknown perturbation call: {call!r} (expected one of {list(ALL_CALLS)})."
            )
        target = targets[i] if i < len(targets) else ""
        edge = _find(target) if target else None
        entry: dict[str, Any] = {"call": call, "target": target or None}

        if call == "drop_relation":
            if edge is None:
                entry.update(effect="noop", detail="no eligible edge to drop")
            else:
                out.remove(edge)
                entry.update(
                    effect="removed",
                    detail=f"{_edge_key(edge)[0]}->{_edge_key(edge)[1]} "
                    f"({edge.get('level2_sense')})",
                )

        elif call == "add_resolution":
            if edge is None:
                entry.update(effect="noop", detail="no conflict edge to resolve")
            else:
                # A resolution adds an adjudicating holding that SOFTENS the conflict. It
                # must not retype the edge: turning an `exclusive` into a `contradiction`
                # would change which factor table the MRF builds, so the rung would differ
                # from its parent by a coupling change rather than by a resolution --
                # a different experiment than the ladder claims to run.
                edge["is_resolved_concession"] = True
                edge.setdefault("provenance", {})["perturbed_by"] = generator
                entry.update(
                    effect="resolved",
                    detail=f"{edge.get('id')} ({edge.get('level1_coupling')}, "
                    f"sense unchanged)",
                )

        elif call == "remove_resolution":
            if edge is None or not edge.get("is_resolved_concession"):
                entry.update(effect="noop", detail="no resolved concession present")
            else:
                edge["is_resolved_concession"] = False
                edge["resolver_atom_id"] = None
                entry.update(effect="unresolved", detail=str(edge.get("id")))

        elif call == "spurious_relation":
            pair = _spurious_pair(out, nons)
            if pair is None:
                entry.update(effect="noop", detail="no unrelated atom pair available")
            else:
                _drop_non_relation(pair)
                new = {
                    "id": _next_edge_id(out),
                    "source_id": pair[0],
                    "target_id": pair[1],
                    "level2_sense": "Contrast",
                    "intended_strength_band": "moderate",
                    "strength_range": [0.6, 0.84],
                    # A spurious link is by definition not a real relation.
                    "validity": "invalid",
                    "error_kind": "spurious",
                    "is_resolved_concession": False,
                    "resolver_atom_id": None,
                    "provenance": {"planned_by": generator, "spurious": True},
                }
                _retype(new, "Contrast")
                out.append(new)
                entry.update(effect="added", detail=f"{pair[0]}->{pair[1]}")

        elif call == "inject_contradiction":
            # Adds a contradiction against an existing atom. Sited like a spurious edge,
            # but labelled as a genuine (if unwanted) contradiction of that atom.
            pair = _spurious_pair(out, nons)
            if pair is None:
                entry.update(effect="noop", detail="no atom pair available")
            else:
                _drop_non_relation(pair)
                new = {
                    "id": _next_edge_id(out),
                    "source_id": pair[0],
                    "target_id": pair[1],
                    "level2_sense": "Contrast",
                    "intended_strength_band": "strong",
                    "strength_range": [0.85, 1.0],
                    "validity": "valid",
                    "error_kind": None,
                    "is_resolved_concession": False,
                    "resolver_atom_id": None,
                    "provenance": {"planned_by": generator, "injected": True},
                }
                _retype(new, "Contrast")
                out.append(new)
                entry.update(effect="added", detail=f"{pair[0]}->{pair[1]}")

        elif call == "break_chain":
            if edge is None:
                entry.update(effect="noop", detail="no eligible edge to break")
            else:
                out.remove(edge)
                entry.update(
                    effect="removed",
                    detail=f"chain link {_edge_key(edge)[0]}->{_edge_key(edge)[1]}",
                )

        elif call == "wrong_sense":
            if edge is None:
                entry.update(effect="noop", detail="no eligible edge to relabel")
            else:
                old = str(edge.get("level2_sense"))
                new_sense = _WRONG_SENSE_MAP.get(old)
                if not new_sense:
                    entry.update(effect="noop", detail=f"no relabel for {old!r}")
                else:
                    _retype(edge, new_sense)
                    edge["validity"] = "invalid"
                    edge["error_kind"] = "wrong_sense"
                    entry.update(effect="relabeled", detail=f"{old} -> {new_sense}")

        elif call == "direction_reversal":
            if edge is None:
                entry.update(effect="noop", detail="no eligible edge to reverse")
            else:
                edge["source_id"], edge["target_id"] = (
                    edge["target_id"],
                    edge["source_id"],
                )
                edge.setdefault("provenance", {})["perturbed_by"] = generator
                # A reversal of a SYMMETRIC edge is meaning-preserving, which is exactly
                # what the CONTROL ladder relies on -- so it stays valid there.
                if is_directed(str(edge.get("level2_sense"))):
                    edge["validity"] = "invalid"
                    edge["error_kind"] = "wrong_direction"
                entry.update(
                    effect="reversed",
                    detail=f"{_edge_key(edge)[0]}->{_edge_key(edge)[1]}",
                )

        elif call == "exhaustiveness_flip":
            if edge is None:
                entry.update(effect="noop", detail="no Alternative/Contrast edge")
            else:
                old = str(edge.get("level2_sense"))
                new_sense = _EXHAUSTIVENESS_FLIP.get(old)
                if not new_sense:
                    entry.update(effect="noop", detail=f"cannot flip {old!r}")
                else:
                    _retype(edge, new_sense)
                    entry.update(effect="relabeled", detail=f"{old} -> {new_sense}")

        elif call == "ordering_only":
            if edge is None:
                entry.update(effect="noop", detail="no Precedence/Succession edge")
            else:
                old = str(edge.get("level2_sense"))
                new_sense = _ORDERING_FLIP.get(old)
                if not new_sense:
                    entry.update(effect="noop", detail=f"cannot flip {old!r}")
                else:
                    # Both compile to Level-1 `none`, so the edge set's FACTORS are
                    # unchanged. That invariance is the point of the ORDER/CONTROL rungs.
                    _retype(edge, new_sense)
                    edge["source_id"], edge["target_id"] = (
                        edge["target_id"],
                        edge["source_id"],
                    )
                    entry.update(effect="reordered", detail=f"{old} -> {new_sense}")

        elif call == "shuffle_order":
            # Sentence order only: no edge is added, removed or retyped. Atom positions
            # move, so `position_distance` is no longer meaningful for this rung and is
            # dropped rather than left stale.
            for e in out:
                e.pop("position_distance", None)
            entry.update(
                effect="reordered", detail="sentence order only; edge set unchanged"
            )

        else:  # pragma: no cover - ALL_CALLS is exhaustive above
            entry.update(effect="noop", detail="call has no edge-set effect")

        log.append(entry)

    return out, nons, log


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
    "apply_calls",
    "call_budget",
    "expectations_for",
    "family_type_slots",
    "ladder_for",
    "ordering_constraints",
    "p5_calls_for",
    "plan_rungs",
    "plan_targets",
    "readout_directions",
]
