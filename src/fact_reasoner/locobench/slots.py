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

"""The slot3 design: three-rung ladders spliced from one template.

The deterministic core of ``.claude/plans/locobench-slot-ladders.md``: no LLM here.

A family's response is written ONCE, as a template with marked SLOTS. A slot is a pair of
claims (source, target) at consecutive positions plus FILLERS: one or two sentences that
assert the slot's own claims and draw exactly one link between them -- none, a support or a
conflict relative to the committed world. A rung is a splice of the template with one filler
per slot (and, for ORDER, a paragraph permutation), so two rungs differ only where their
fillers differ, and "nothing else moved" is a text diff rather than a judgment.

Truth values are the claims' P2 tags (verdict polarity, label-only, so flat priors). Gold
relations are compiled with taxonomy v2: Refutation -> contradiction, Contrast/Concession ->
none. Every ladder's declared order is checked EXACTLY on the gold graphs before any prose is
written (coherence-order dominance at x*, plus a minimum Lambda step), so a family that could
not carry its ranking claim is caught at plan time.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from fact_reasoner.locobench.taxonomy_bridge import (
    coupling_for_sense,
    is_directed,
    is_ordering_only,
)

TAXONOMY = "v2"
DESIGN = "slot3"

# Gold weights: the midpoint of each strength band, as in the ARR gold arm.
BAND_RANGE = {"strong": (0.85, 1.00), "moderate": (0.60, 0.84), "weak": (0.35, 0.59)}
BAND_MID = {k: round((lo + hi) / 2, 4) for k, (lo, hi) in BAND_RANGE.items()}

ENTAILMENT_SENSES = ("Cause-Effect", "Effect-Cause", "Evidence", "Condition", "Instantiation")
# Entailment senses a slot3 plan may use. Effect-Cause and Instantiation are left out because
# their direction is the one planners and readers disagree on: on the first live pilot family
# (f001) all three raters read "general claim; for instance, specific case" as the instance
# supporting the generalization, on three filler versions in a row, the reverse of the plan.
# Condition is left out because it fits almost any topical pair (absent from v1 gold).
SLOT_SENSES = ("Cause-Effect", "Evidence")
# The senses a plan may use for its fixed (non-slot) relations, taxonomy v2.
PLAN_SENSES = SLOT_SENSES + (
    "Restatement",
    "Refutation",
    "Contrast",
    "Concession",
    "Alternative",
    "Disjunction",
    "Precedence",
    "Succession",
)
# Rare senses are spread over the corpus by an allocator rather than demanded of every plan
# (the v1 plan's all-rare-senses quota is one reason its prose read as assembled).
RARE_SENSES = ("Alternative", "Disjunction", "Restatement", "Refutation", "Precedence")

# Violated cells per coupling, over (x_source, x_target) -- ARR paper Def. 2.
VIOLATED = {
    "entailment": {(1, 0)},
    "equivalence": {(0, 1), (1, 0)},
    "co_necessity": {(0, 0)},
    "contradiction": {(1, 1)},
    "exclusive": {(0, 0), (1, 1)},
}
SYMMETRIC = {"equivalence", "co_necessity", "contradiction", "exclusive"}

# Minimum |Lambda| step between adjacent rungs of an increase ladder, in nats. A support that
# the rest of the graph already forces gains almost nothing (saturation), so a ladder whose
# step falls below this is ordered in theory but empty in practice.
MIN_STEP = 0.05


# ----------------------------------------------------------------------------
# Slot roles and ladders
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """What a slot of this role connects and which fillers it carries.

    Attributes:
        name: The role id used in plans.
        truth: Required committed values of (source, target): 1 true, 0 false.
        kinds: The filler kinds a slot of this role carries.
        rest: The kind an unused (spare) slot of this role is fixed at in every rung.
        needs_sense: Whether the plan names an entailment sense for the slot (the
            support of CHAIN, the unsound inference of SIGN, the link of CONTROL).
    """

    name: str
    truth: tuple[int, int]
    kinds: tuple[str, ...]
    rest: str
    needs_sense: bool


ROLES: dict[str, Role] = {
    # Two truths, set against each other or left unlinked.
    "conflict_tt": Role("conflict_tt", (1, 1), ("conflict", "none"), "none", False),
    # A link of a chain between two truths: drawn, or left unlinked.
    "chain_link": Role("chain_link", (1, 1), ("support", "none"), "support", True),
    # A truth and a falsehood: the falsehood derived from the truth (conflict), left unlinked,
    # or refuted by it (support). The one polarity-sensitive role.
    "sign_tf": Role("sign_tf", (1, 0), ("conflict", "none", "support"), "none", True),
    # Two truths joined by one entailment, in three wordings.
    "control": Role("control", (1, 1), ("para1", "para2", "para3"), "para1", True),
}


@dataclass(frozen=True)
class LadderSpec:
    """A three-rung ladder over slot roles.

    Attributes:
        type: The ladder type.
        roles: One role per slot the ladder edits, in order.
        rungs: Per rung (0 least coherent), the filler kind of each role instance.
        declared: ``"increase"`` or ``"invariant"``.
        permute: Whether rungs are paragraph permutations (ORDER).
        spares: Spare slots per role the plan must add, so one failing slot does not
            cost the ladder.
        polarity_invariant: Whether the declared order also holds under assertion
            polarity (every slot endpoint true).
    """

    type: str
    roles: tuple[str, ...]
    rungs: tuple[tuple[str, ...], ...]
    declared: str
    permute: bool = False
    spares: Mapping[str, int] | None = None
    polarity_invariant: bool = True


LADDERS: dict[str, LadderSpec] = {
    "CONFLICT": LadderSpec(
        "CONFLICT",
        ("conflict_tt", "conflict_tt"),
        (("conflict", "conflict"), ("conflict", "none"), ("none", "none")),
        "increase",
        spares={"conflict_tt": 1},
    ),
    "CHAIN": LadderSpec(
        "CHAIN",
        ("chain_link", "chain_link"),
        (("none", "none"), ("support", "none"), ("support", "support")),
        "increase",
    ),
    "SIGN": LadderSpec(
        "SIGN",
        ("sign_tf",),
        (("conflict",), ("none",), ("support",)),
        "increase",
        spares={"sign_tf": 1},
        polarity_invariant=False,
    ),
    "ORDER": LadderSpec("ORDER", (), ((), (), ()), "invariant", permute=True),
    "CONTROL": LadderSpec(
        "CONTROL", ("control",), (("para1",), ("para2",), ("para3",)), "invariant"
    ),
}
PRIMARY_TYPES = ("CONFLICT", "CHAIN", "SIGN", "ORDER")
# Every family also carries a CONTROL ladder on one extra slot. Its rung 0 is the primary
# ladder's rung 1 text, so it costs two items, not three.
PIGGYBACK = "CONTROL"
ANCHOR_RUNG = 1


def required_slots(primary: str) -> dict[str, int]:
    """How many slots of each role a family of this primary type plans (spares included)."""
    out: dict[str, int] = {}
    spec = LADDERS[primary]
    for role in spec.roles:
        out[role] = out.get(role, 0) + 1
    for role, k in (spec.spares or {}).items():
        out[role] = out.get(role, 0) + k
    out["control"] = out.get("control", 0) + 1
    return out


def rare_senses_for(index: int, k: int = 2) -> list[str]:
    """Deterministic round-robin allocation of rare senses to the index-th family."""
    n = len(RARE_SENSES)
    return [RARE_SENSES[(index * k + j) % n] for j in range(k)]


# ----------------------------------------------------------------------------
# Exact Lambda / LCS (ARR paper Eq. 4-5) and the coherence order
# ----------------------------------------------------------------------------

Rel = tuple[int, int, str, float]  # (source index, target index, coupling, weight)


def _violates(coupling: str, u: int, v: int) -> bool:
    return (u, v) in VIOLATED[coupling]


def lambda_lcs(n: int, rels: Sequence[Rel], xstar: Sequence[int],
               priors: Sequence[float] | None = None) -> tuple[float, float]:
    """Exact Lambda and LCS by enumeration.

    Args:
        n: Number of claims.
        rels: Factor-bearing relations ``(s, t, coupling, p)``; ``p`` in [0, 1).
        xstar: The committed world.
        priors: Per-claim priors in (0, 1); flat when omitted.

    Returns:
        ``(Lambda, LCS)``.
    """
    pri = list(priors) if priors is not None else [0.5] * n
    z = 0.0
    m_star = 0.0
    xs = tuple(int(v) for v in xstar)
    for w in itertools.product((0, 1), repeat=n):
        m = 1.0
        for i in range(n):
            m *= pri[i] if w[i] else 1.0 - pri[i]
        for s, t, c, p in rels:
            if _violates(c, w[s], w[t]):
                m *= 1.0 - p
        z += m
        if w == xs:
            m_star = m
    pi_star = 1.0
    for i in range(n):
        pi_star *= pri[i] if xs[i] else 1.0 - pri[i]
    post = m_star / z
    lam = math.log(post / pi_star)
    w_ev = math.log(post / (1 - post)) - math.log(pi_star / (1 - pi_star))
    return lam, 1.0 / (1.0 + math.exp(-w_ev))


def canonical(rels: Iterable[Rel]) -> dict[tuple, float]:
    """Relation multiset -> {(scope, coupling): combined weight}, parallel copies merged."""
    out: dict[tuple, float] = {}
    for s, t, c, p in rels:
        key = (tuple(sorted((s, t))) if c in SYMMETRIC else (s, t), c)
        q = out.get(key, 0.0)
        out[key] = 1 - (1 - q) * (1 - p)
    return out


def compare(hi: Iterable[Rel], lo: Iterable[Rel], xstar: Sequence[int],
            tol: float = 1e-12) -> str:
    """Compare two graphs in the coherence order at ``xstar``.

    Returns:
        ``"strict"`` (hi above lo), ``"equal"``, or ``"none"`` when hi is not reached from
        lo by improving edits (incomparable, or lo above hi).
    """
    a, b = canonical(hi), canonical(lo)
    strict = False
    for key in set(a) | set(b):
        (s, t), c = key
        pa, pb = a.get(key, 0.0), b.get(key, 0.0)
        support = not _violates(c, xstar[s], xstar[t])
        if support:
            if pa < pb - tol:
                return "none"
            strict |= pa > pb + tol
        else:
            if pa > pb + tol:
                return "none"
            strict |= pa < pb - tol
    return "strict" if strict else "equal"


# ----------------------------------------------------------------------------
# Plan model
# ----------------------------------------------------------------------------


def atom_id(pos: int) -> str:
    """Atoms are ``a0..a{n-1}`` in position order, as in the v1 corpora."""
    return f"a{pos - 1}"


def slot_links(slot: Mapping[str, Any]) -> dict[str, str | None]:
    """The sense each filler kind of a slot realizes (None for the unlinked kind)."""
    role = slot["role"]
    sense = slot.get("sense")
    if role == "conflict_tt":
        return {"conflict": "Refutation", "none": None}
    if role == "chain_link":
        return {"support": sense, "none": None}
    if role == "sign_tf":
        return {"conflict": sense, "none": None, "support": "Refutation"}
    if role == "control":
        return {"para1": sense, "para2": sense, "para3": sense}
    raise ValueError(f"unknown slot role {role!r}")


def slot_owns(plan: Mapping[str, Any]) -> dict[str, list[int]]:
    """The positions each slot's fillers assert: its endpoints minus earlier slots' claims.

    A CHAIN's second link shares its source with the first link's target, which the first
    filler already asserts, so the second filler asserts only its target.
    """
    seen: set[int] = set()
    out: dict[str, list[int]] = {}
    for slot in sorted(plan.get("slots", []), key=lambda s: (s["source_pos"], s["id"])):
        own = [p for p in (slot["source_pos"], slot["target_pos"]) if p not in seen]
        seen.update(own)
        out[slot["id"]] = own
    return out


def _sense_rel(source_pos: int, target_pos: int, sense: str, band: str,
               **extra: Any) -> dict[str, Any]:
    coupling = coupling_for_sense(sense, TAXONOMY)
    return {
        "source_id": atom_id(source_pos),
        "target_id": atom_id(target_pos),
        "level2_sense": sense,
        "level1_coupling": coupling,
        "directed": is_directed(sense, TAXONOMY),
        "ordering_only": is_ordering_only(sense, TAXONOMY),
        "intended_strength_band": band,
        "strength_range": list(BAND_RANGE[band]),
        **extra,
    }


def xstar_of(plan: Mapping[str, Any]) -> list[int]:
    """The committed world: each atom's factuality label, in position order."""
    atoms = sorted(plan["atoms"], key=lambda a: a["pos"])
    return [1 if a.get("factual", True) else 0 for a in atoms]


def gold_relations(plan: Mapping[str, Any], assignment: Mapping[str, str],
                   fixed: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The gold relation list of one rung: fixed relations plus each slot's filler link.

    Args:
        plan: The plan.
        assignment: ``{slot id: filler kind}`` for every slot.
        fixed: The fixed relations to use (default: the plan's), already in gold form or
            in plan form.
    """
    xs = xstar_of(plan)
    # The asserted world: 1 for every claim the rung asserts, 0 for one it denies (the false
    # target of a SIGN slot's refuting filler). Before 2026-11-20 this was the all-true world,
    # which mis-signed the refutation on the 60 refuting SIGN rungs.
    denied = stance_for(plan, assignment)
    xa = [0 if denied.get(i + 1) == "denied" else 1 for i in range(len(xs))]
    out: list[dict[str, Any]] = []
    for r in (fixed if fixed is not None else plan.get("relations", [])):
        if "source_id" in r:
            out.append(dict(r))
        else:
            out.append(
                _sense_rel(r["source_pos"], r["target_pos"], r["sense"],
                           r.get("strength_band", "moderate"), slot_id=None, filler=None,
                           provenance={"source": "plan"})
            )
    for slot in plan.get("slots", []):
        kind = assignment[slot["id"]]
        sense = slot_links(slot)[kind]
        if sense is None:
            continue
        out.append(
            _sense_rel(slot["source_pos"], slot["target_pos"], sense,
                       slot.get("strength_band", "strong"), slot_id=slot["id"], filler=kind,
                       provenance={"source": "slot"})
        )
    for i, r in enumerate(out):
        r["id"] = f"r{i:03d}"
        r["sign_verdict"] = relation_sign(r, xs)
        r["sign_assertion"] = relation_sign(r, xa)
        r["position_distance"] = abs(int(r["source_id"][1:]) - int(r["target_id"][1:]))
    return out


def relation_sign(rel: Mapping[str, Any], xstar: Sequence[int]) -> str:
    """support / conflict / none of one gold relation at a committed world."""
    c = rel["level1_coupling"]
    if c not in VIOLATED:
        return "none"
    s, t = int(rel["source_id"][1:]), int(rel["target_id"][1:])
    return "conflict" if _violates(c, xstar[s], xstar[t]) else "support"


def factor_rels(relations: Sequence[Mapping[str, Any]]) -> list[Rel]:
    """Gold relations -> the factor list Lambda is computed on (band midpoints)."""
    out: list[Rel] = []
    for r in relations:
        c = r["level1_coupling"]
        if c not in VIOLATED:
            continue
        p = BAND_MID[r.get("intended_strength_band", "moderate")]
        out.append((int(r["source_id"][1:]), int(r["target_id"][1:]), c, min(p, 0.999)))
    return out


# ----------------------------------------------------------------------------
# Ladder realization: which plan slot plays which role, and the rung assignments
# ----------------------------------------------------------------------------


@dataclass
class Casting:
    """Which plan slots play the primary ladder's roles and the CONTROL role.

    Attributes:
        primary: Slot ids, one per role instance of the primary ladder, in order.
        control: The CONTROL slot id, or None when it is not castable.
        rest: ``{slot id: kind}`` for every slot no ladder edits.
    """

    primary: list[str]
    control: str | None
    rest: dict[str, str]


def default_casting(plan: Mapping[str, Any], primary: str,
                    passing: Mapping[str, set[str]] | None = None,
                    allow_partial: bool = False) -> Casting | None:
    """Cast slots to roles, preferring plan order and slots whose fillers all passed.

    Args:
        plan: The plan.
        primary: The primary ladder type.
        passing: ``{slot id: set of filler kinds that passed V5}``. When None every kind
            is assumed to pass (plan-time casting).
        allow_partial: When the primary roles cannot be filled, return a casting with an
            empty ``primary`` (its slots fixed at a passing kind) rather than None, so the
            CONTROL ladder can still be admitted on its own.

    Returns:
        The casting, or None when it cannot be built.
    """
    spec = LADDERS[primary]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for s in sorted(plan.get("slots", []), key=lambda s: s["id"]):
        by_role.setdefault(s["role"], []).append(s)

    def ok(slot: Mapping[str, Any], kinds: Iterable[str]) -> bool:
        if passing is None:
            return True
        return set(kinds) <= passing.get(slot["id"], set())

    used: set[str] = set()
    cast: list[str] = []
    if primary == "CHAIN":
        # The two links must form a chain: second.source == first.target.
        links = by_role.get("chain_link", [])
        found = None
        for a, b in itertools.permutations(links, 2):
            if a["target_pos"] == b["source_pos"] and ok(a, ROLES["chain_link"].kinds) and ok(
                b, ROLES["chain_link"].kinds
            ):
                found = (a, b)
                break
        if found is None:
            if not allow_partial:
                return None
        else:
            cast = [found[0]["id"], found[1]["id"]]
            used.update(cast)
    else:
        for role in spec.roles:
            cand = next(
                (s for s in by_role.get(role, []) if s["id"] not in used
                 and ok(s, ROLES[role].kinds)),
                None,
            )
            if cand is None:
                if not allow_partial:
                    return None
                cast, used = [], set()
                break
            cast.append(cand["id"])
            used.add(cand["id"])
    control = next(
        (s["id"] for s in by_role.get("control", []) if ok(s, ROLES["control"].kinds)), None
    )
    if control:
        used.add(control)
    rest: dict[str, str] = {}
    for s in plan.get("slots", []):
        if s["id"] in used:
            continue
        role = ROLES[s["role"]]
        order = [role.rest] + [k for k in role.kinds if k != role.rest]
        kind = next((k for k in order if passing is None or k in passing.get(s["id"], set())),
                    None)
        if kind is None:
            return None  # a slot whose every filler failed cannot sit in the text
        rest[s["id"]] = kind
    return Casting(cast, control, rest)


def assignments(plan: Mapping[str, Any], primary: str, casting: Casting) -> dict[str, Any]:
    """Every ladder's per-rung slot assignment.

    Returns:
        ``{"primary": [assign_r0, assign_r1, assign_r2], "control": [...] or None}``, each
        assignment a ``{slot id: kind}`` over ALL slots.
    """
    spec = LADDERS[primary]
    ctrl_base = {casting.control: "para1"} if casting.control else {}
    cast_ok = len(casting.primary) == len(spec.roles)
    prim = []
    for rung in spec.rungs:
        a = dict(casting.rest)
        a.update(ctrl_base)
        if cast_ok:
            for sid, kind in zip(casting.primary, rung):
                a[sid] = kind
        prim.append(a)
    ctrl = None
    if casting.control:
        ctrl = []
        for (kind,) in LADDERS["CONTROL"].rungs:
            a = dict(prim[ANCHOR_RUNG])
            a[casting.control] = kind
            ctrl.append(a)
    return {"primary": prim if cast_ok else None, "control": ctrl, "base": prim[ANCHOR_RUNG]}


def ladder_gold_check(plan: Mapping[str, Any], primary: str, casting: Casting,
                      fixed: Sequence[Mapping[str, Any]] | None = None) -> tuple[str | None, dict]:
    """Check every ladder's declared order exactly on the gold graphs.

    Returns:
        ``(reason or None, report)``; the report holds Lambda/LCS per rung per ladder and the
        order under assertion polarity.
    """
    xs = xstar_of(plan)
    n = len(xs)
    asg = assignments(plan, primary, casting)
    report: dict[str, Any] = {}
    reasons: list[str] = []
    for name, rungs, ltype in (("primary", asg["primary"], primary),
                               ("control", asg["control"], "CONTROL")):
        if rungs is None:
            continue
        spec = LADDERS[ltype]
        graphs = [factor_rels(gold_relations(plan, a, fixed)) for a in rungs]
        scores = [lambda_lcs(n, g, xs) for g in graphs]
        rep = {
            "type": ltype,
            "lambda": [round(s[0], 4) for s in scores],
            "lcs": [round(s[1], 4) for s in scores],
        }
        for lo, hi in zip(range(len(graphs)), range(1, len(graphs))):
            rel = compare(graphs[hi], graphs[lo], xs)
            step = scores[hi][0] - scores[lo][0]
            if spec.declared == "increase":
                if rel != "strict":
                    reasons.append(f"{ltype} rungs {lo}->{hi} are not ordered in the "
                                   f"coherence order at x* ({rel})")
                elif step < MIN_STEP:
                    reasons.append(f"{ltype} rungs {lo}->{hi} differ by only {step:.3f} "
                                   f"nats (need {MIN_STEP}): the edit is saturated away")
            elif rel != "equal":
                reasons.append(f"{ltype} rungs {lo}->{hi} build different graphs")
        # The same ladder read at x* = 1 (every slot claim taken as asserted and true).
        ones = [1] * n
        rep["assertion_lambda"] = [round(lambda_lcs(n, g, ones)[0], 4) for g in graphs]
        rep["assertion_order"] = [compare(graphs[i + 1], graphs[i], ones)
                                  for i in range(len(graphs) - 1)]
        report[name] = rep
    return ("; ".join(reasons) or None), report


# ----------------------------------------------------------------------------
# The plan gate (deterministic)
# ----------------------------------------------------------------------------

N_ATOMS = (10, 14)
N_RELATIONS = (6, 8)
N_NON_RELATIONS = (3, 5)
MIN_FALSE = 2
MAX_FIXED_DISTANCE = 5


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def check_plan(plan: Mapping[str, Any], claims: Sequence[Mapping[str, str]], primary: str,
               rare: Sequence[str]) -> tuple[str | None, dict]:
    """Every structural rule of the slot3 plan, then the exact gold-ladder check.

    The plan's ``factual`` fields are OVERWRITTEN from P2's tags first (the committed world
    must come from the claims, not from the planner's restatement of them).

    Returns:
        ``(reason or None, report)``.
    """
    errs: list[str] = []
    atoms = plan.get("atoms") or []
    tags = {_norm(c["text"]): c["tag"] for c in claims}
    pos = sorted(a.get("pos") for a in atoms if isinstance(a.get("pos"), int))
    if not (N_ATOMS[0] <= len(atoms) <= N_ATOMS[1]):
        errs.append(f"select {N_ATOMS[0]}-{N_ATOMS[1]} claims, got {len(atoms)}")
    if pos != list(range(1, len(atoms) + 1)):
        errs.append("atom positions must be 1..N with no gaps or repeats")
    for a in atoms:
        tag = tags.get(_norm(str(a.get("text", ""))))
        if tag is None:
            errs.append(f"atom {a.get('pos')} is not verbatim one of the given claims: "
                        f"{str(a.get('text'))[:80]!r}")
        else:
            a["factual"] = tag != "incorrect"
    n_false = sum(1 for a in atoms if a.get("factual") is False)
    if n_false < MIN_FALSE:
        errs.append(f"select at least {MIN_FALSE} [incorrect] claims, got {n_false}")
    if errs:
        return "; ".join(errs), {}

    n = len(atoms)
    fact = {a["pos"]: bool(a["factual"]) for a in atoms}
    used_pairs: set[frozenset] = set()

    def pair_ok(s: Any, t: Any, what: str) -> bool:
        if not (isinstance(s, int) and isinstance(t, int) and 1 <= s <= n and 1 <= t <= n):
            errs.append(f"{what}: endpoints must be atom positions 1..{n}")
            return False
        if s == t:
            errs.append(f"{what}: a claim cannot relate to itself")
            return False
        key = frozenset((s, t))
        if key in used_pairs:
            errs.append(f"{what}: the pair ({s},{t}) is already used; every relation, slot "
                        "and non-relation needs its own pair")
            return False
        used_pairs.add(key)
        return True

    rels = plan.get("relations") or []
    if not (N_RELATIONS[0] <= len(rels) <= N_RELATIONS[1]):
        errs.append(f"plan {N_RELATIONS[0]}-{N_RELATIONS[1]} relations, got {len(rels)}")
    for r in rels:
        what = f"relation {r.get('source_pos')}->{r.get('target_pos')}"
        if r.get("sense") not in PLAN_SENSES:
            errs.append(f"{what}: sense {r.get('sense')!r} is not one of {list(PLAN_SENSES)}")
        if r.get("strength_band") not in BAND_RANGE:
            r["strength_band"] = "moderate"
        if pair_ok(r.get("source_pos"), r.get("target_pos"), what) and abs(
            r["source_pos"] - r["target_pos"]
        ) > MAX_FIXED_DISTANCE:
            errs.append(f"{what}: keep relations within {MAX_FIXED_DISTANCE} positions")
    senses = {r.get("sense") for r in rels}
    for want in rare:
        if want == "Precedence":
            if not senses & {"Precedence", "Succession"}:
                errs.append("include at least one Precedence or Succession relation")
        elif want not in senses:
            errs.append(f"include at least one {want} relation")

    slots = plan.get("slots") or []
    need = required_slots(primary)
    have: dict[str, int] = {}
    ids = [s.get("id") for s in slots]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and re.fullmatch(r"S\d+", i)
                                            for i in ids):
        errs.append('slot ids must be distinct and of the form "S1", "S2", ...')
    for s in slots:
        role = ROLES.get(s.get("role"))
        what = f"slot {s.get('id')}"
        if role is None:
            errs.append(f"{what}: role {s.get('role')!r} is not one of {list(need)}")
            continue
        have[role.name] = have.get(role.name, 0) + 1
        src, tgt = s.get("source_pos"), s.get("target_pos")
        chain_shared = False
        if role.name == "chain_link":
            chain_shared = any(o is not s and o.get("role") == "chain_link"
                               and o.get("target_pos") == src for o in slots)
        if chain_shared:
            # The shared claim legitimately appears in two chain slots.
            if not (isinstance(tgt, int) and 1 <= tgt <= n and src != tgt):
                errs.append(f"{what}: bad target")
            else:
                used_pairs.add(frozenset((src, tgt)))
        elif not pair_ok(src, tgt, what):
            continue
        if isinstance(src, int) and isinstance(tgt, int):
            if tgt != src + 1:
                errs.append(f"{what}: a slot joins two CONSECUTIVE positions "
                            f"(target_pos = source_pos + 1), got {src}->{tgt}")
            want_s, want_t = role.truth
            if fact.get(src) != bool(want_s) or fact.get(tgt) != bool(want_t):
                lab = {1: "[correct]", 0: "[incorrect]"}
                errs.append(f"{what} ({role.name}): source must be {lab[want_s]} and target "
                            f"{lab[want_t]}")
        if role.needs_sense:
            if s.get("sense") not in SLOT_SENSES:
                errs.append(f"{what}: sense must be one of {list(SLOT_SENSES)}")
        else:
            s["sense"] = "Refutation"
        if s.get("strength_band") not in BAND_RANGE:
            s["strength_band"] = "strong"
    for role, k in need.items():
        if have.get(role, 0) != k:
            errs.append(f"plan exactly {k} slot(s) of role {role}, got {have.get(role, 0)}")
    if have.keys() - need.keys():
        errs.append(f"no slots of role(s) {sorted(have.keys() - need.keys())} for this ladder")
    if primary == "CHAIN":
        links = [s for s in slots if s.get("role") == "chain_link"]
        if len(links) == 2 and not any(a.get("target_pos") == b.get("source_pos")
                                       for a, b in itertools.permutations(links, 2)):
            errs.append("the two chain_link slots must form a chain: the second slot's "
                        "source_pos is the first slot's target_pos")

    nons = plan.get("non_relations") or []
    if not (N_NON_RELATIONS[0] <= len(nons) <= N_NON_RELATIONS[1]):
        errs.append(f"list {N_NON_RELATIONS[0]}-{N_NON_RELATIONS[1]} non-relations, "
                    f"got {len(nons)}")
    for nr in nons:
        pair_ok(nr.get("source_pos"), nr.get("target_pos"),
                f"non-relation {nr.get('source_pos')}-{nr.get('target_pos')}")
    if errs:
        return "; ".join(dict.fromkeys(errs)), {}

    casting = default_casting(plan, primary)
    if casting is None:
        return "the slots cannot be cast to the ladder's roles", {}
    reason, report = ladder_gold_check(plan, primary, casting)
    # Spares must be usable too, or they are not spares.
    if reason is None and LADDERS[primary].spares:
        for role in LADDERS[primary].spares:
            for spare in [s for s in slots if s["role"] == role and s["id"] not in casting.primary]:
                alt = Casting(list(casting.primary), casting.control, dict(casting.rest))
                swap = next(i for i, sid in enumerate(alt.primary)
                            if next(x for x in slots if x["id"] == sid)["role"] == role)
                alt.rest[alt.primary[swap]] = ROLES[role].rest
                alt.rest.pop(spare["id"], None)
                alt.primary[swap] = spare["id"]
                r2, _ = ladder_gold_check(plan, primary, alt)
                if r2:
                    reason = f"spare slot {spare['id']} cannot stand in: {r2}"
                    break
    return reason, {"casting": casting.__dict__, "ladders": report}


# ----------------------------------------------------------------------------
# Templates and splicing
# ----------------------------------------------------------------------------

MARKER_RE = re.compile(r"\{\{(S\d+)\}\}")
# Phrases that ANNOUNCE that two claims are unconnected. A "none" filler carrying one marks
# its rung as the unlinked one, which a judge can pick up on without reading any logic;
# measured on the first live pilot family (f004: "It is separately established that ...").
ANNOUNCING_RE = re.compile(
    r"\b(separately|independently|in a separate|unrelated(ly)?|on a different note|"
    r"it is (also )?(documented|established|recorded|noted) that)\b",
    re.IGNORECASE,
)


# Fillers that talk ABOUT the claims ("these two propositions cannot both be true") read as
# annotation, and V3 calls them leakage. Measured on the first live CONFLICT family (f002):
# every conflict filler ended "; these two claims/propositions/accounts cannot both be true".
META_RE = re.compile(
    r"\b(these|those|the|both|two|the two|these two|those two)\s+(two\s+)?"
    r"(claims|propositions|statements|assertions)\b",
    re.IGNORECASE,
)


def _norm_words(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()


def restated(filler: str, claims: Iterable[str]) -> str | None:
    """The first claim a filler states TWICE, or None.

    Measured on the pilot (f006): banning "these claims cannot both be true" led one
    conflict filler to restate both claims in full inside the refutation instead.
    """
    hay = " ".join(_norm_words(filler))
    for c in claims:
        w = _norm_words(c)
        key = " ".join(w[1:7] if len(w) > 7 else w)
        if key and hay.count(key) >= 2:
            return c
    return None


def coverage_assignments(plan: Mapping[str, Any], base: Mapping[str, str]) -> list[dict[str, str]]:
    """Extra splices that, with ``base``, put every filler of every slot in some text.

    V3 audits these too, so no rung text carries a filler that no auditor read.
    """
    kinds = {s["id"]: [k for k in ROLES[s["role"]].kinds if k != base[s["id"]]]
             for s in plan.get("slots", [])}
    n = max((len(v) for v in kinds.values()), default=0)
    out = []
    for j in range(n):
        a = dict(base)
        for sid, ks in kinds.items():
            if ks:
                a[sid] = ks[min(j, len(ks) - 1)]
        out.append(a)
    return out


def check_template(template: str, fillers: Mapping[str, Mapping[str, str]],
                   plan: Mapping[str, Any]) -> str | None:
    """Structural checks on P4's template and fillers."""
    errs = []
    text_at = {a["pos"]: a["text"] for a in plan.get("atoms", [])}
    found = MARKER_RE.findall(template)
    want = [s["id"] for s in plan.get("slots", [])]
    for sid in want:
        k = found.count(sid)
        if k != 1:
            errs.append(f"marker {{{{{sid}}}}} must appear exactly once in the template "
                        f"(found {k})")
    extra = set(found) - set(want)
    if extra:
        errs.append(f"the template has markers for unknown slots {sorted(extra)}")
    for s in plan.get("slots", []):
        got = fillers.get(s["id"]) or {}
        for kind in ROLES[s["role"]].kinds:
            text = got.get(kind)
            if not isinstance(text, str) or not text.strip():
                errs.append(f"slot {s['id']} needs a non-empty {kind!r} filler")
            elif MARKER_RE.search(text) or "\n\n" in text.strip():
                errs.append(f"slot {s['id']} {kind!r} filler must be plain sentences "
                            "within one paragraph")
        ends = [text_at[p] for p in (s.get("source_pos"), s.get("target_pos")) if p in text_at]
        for kind, text in got.items():
            twice = restated(text, ends) if isinstance(text, str) else None
            if twice:
                errs.append(f"slot {s['id']} {kind!r} filler states a claim twice "
                            f"({twice[:60]!r}...); assert each claim once")
            m = META_RE.search(text) if isinstance(text, str) else None
            if m:
                errs.append(f"slot {s['id']} {kind!r} filler talks about the text's claims "
                            f"({m.group(0)!r}); state the link in terms of the subject "
                            "matter, e.g. 'X cannot be squared with Y', 'if X, Y cannot hold'")
        none_text = got.get("none")
        if isinstance(none_text, str):
            m = ANNOUNCING_RE.search(none_text)
            if m:
                errs.append(f"slot {s['id']} 'none' filler announces that the claims are "
                            f"unconnected ({m.group(0)!r}); just state them, with no "
                            "connective and no comment on their relation")
        if s["role"] == "control":
            texts = [_norm(got.get(k, "")) for k in ROLES["control"].kinds]
            if len(set(texts)) < 3:
                errs.append(f"slot {s['id']}: the three para fillers must be worded "
                            "differently")
    return "; ".join(errs) or None


def splice(template: str, fillers: Mapping[str, Mapping[str, str]],
           assignment: Mapping[str, str]) -> str:
    """Replace every marker by its assigned filler; normalize spacing only."""

    def rep(m: re.Match) -> str:
        return fillers[m.group(1)][assignment[m.group(1)]].strip()

    out = MARKER_RE.sub(rep, template)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" +([.,;:])", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def permutations_for(n_par: int, seed: int, k: int = 6) -> list[list[int]]:
    """Candidate paragraph orders for ORDER rungs, most different from identity first."""
    ident = list(range(n_par))
    cands = [list(reversed(ident)), ident[1:] + ident[:1], ident[-1:] + ident[:-1]]
    rng = random.Random(seed)
    for _ in range(50):
        p = ident[:]
        rng.shuffle(p)
        cands.append(p)
    out: list[list[int]] = []
    for p in cands:
        if p != ident and p not in out and sum(a != b for a, b in zip(p, ident)) >= 2:
            out.append(p)
    return out[:k]


def permute(text: str, order: Sequence[int]) -> str:
    pars = paragraphs(text)
    return "\n\n".join(pars[i] for i in order)


def word_count(text: str) -> int:
    return len(text.split())


# ----------------------------------------------------------------------------
# The committee readings: expected answers for V1-verify and V5-slot
# ----------------------------------------------------------------------------

# V1-verify labels. `i`/`j` are the listed pair's first and second atom.
VERIFY_LABELS = (
    "i_supports_j", "j_supports_i", "incompatible", "equivalent", "exactly_one",
    "at_least_one", "precedes", "follows", "contrast_only", "concedes", "unlinked",
)
LABEL_COUPLING = {
    "i_supports_j": "entailment", "j_supports_i": "entailment",
    "incompatible": "contradiction", "equivalent": "equivalence",
    "exactly_one": "exclusive", "at_least_one": "co_necessity",
    "precedes": "none", "follows": "none", "contrast_only": "none", "concedes": "none",
    "unlinked": "none",
}
LABEL_SENSE = {
    "i_supports_j": "Evidence", "j_supports_i": "Evidence", "incompatible": "Refutation",
    "equivalent": "Restatement", "exactly_one": "Alternative",
    "at_least_one": "Disjunction",
}


def expected_label(sense: str | None, forward: bool = True) -> str:
    """The V1-verify / V5-slot label a sense should be read as (``forward``: i = source)."""
    if sense is None:
        return "unlinked"
    if sense in ENTAILMENT_SENSES:
        return "i_supports_j" if forward else "j_supports_i"
    if sense in ("Precedence", "Succession"):
        first = sense == "Precedence"
        return "precedes" if first == forward else "follows"
    return {
        "Restatement": "equivalent", "Refutation": "incompatible",
        "Alternative": "exactly_one", "Disjunction": "at_least_one",
        "Contrast": "contrast_only", "Concession": "concedes",
    }[sense]


def label_matches(expected: str, got: str) -> bool:
    """Whether a reading agrees with the expected label at the level gold depends on.

    Factor-bearing links must match exactly (coupling and, for entailment, direction). The
    no-factor links are interchangeable for scoring, so any no-factor reading matches one.
    """
    if got not in LABEL_COUPLING:
        return False
    ce, cg = LABEL_COUPLING[expected], LABEL_COUPLING[got]
    if ce == "none" or cg == "none":
        return ce == cg
    return expected == got


def majority(labels: Sequence[str]) -> str | None:
    if not labels:
        return None
    top, k = max(((lab, labels.count(lab)) for lab in set(labels)), key=lambda x: x[1])
    return top if k * 2 > len(labels) else None


# ----------------------------------------------------------------------------
# Items and manifest
# ----------------------------------------------------------------------------


def build_atoms(plan: Mapping[str, Any], stance: Mapping[int, str] | None = None
                ) -> list[dict[str, Any]]:
    out = []
    for a in sorted(plan["atoms"], key=lambda a: a["pos"]):
        out.append({
            "id": atom_id(a["pos"]),
            "pos": a["pos"],
            "text": a["text"],
            "factual": bool(a["factual"]),
            "stance": (stance or {}).get(a["pos"], "asserted"),
        })
    return out


def stance_for(plan: Mapping[str, Any], assignment: Mapping[str, str]) -> dict[int, str]:
    """A SIGN slot's support filler refutes its false target, so that claim is denied."""
    out: dict[int, str] = {}
    for s in plan.get("slots", []):
        if s["role"] == "sign_tf" and assignment.get(s["id"]) == "support":
            out[s["target_pos"]] = "denied"
    return out


def validate_item(item: Mapping[str, Any]) -> None:
    """Schema of a slot3 item. Raises ValueError on the first violation."""

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"{item.get('id')}: {msg}")

    for key in ("id", "response", "atoms", "relations", "non_relations", "expected", "meta"):
        need(key in item, f"missing {key!r}")
    need(item["meta"].get("taxonomy") == TAXONOMY, "meta.taxonomy must be v2")
    need(item["expected"].get("design") == DESIGN, "expected.design must be slot3")
    ids = {a["id"] for a in item["atoms"]}
    need(len(ids) == len(item["atoms"]) == item.get("num_atoms"), "atom ids / num_atoms")
    seen = set()
    for r in item["relations"]:
        need(r["source_id"] in ids and r["target_id"] in ids, f"relation {r['id']} endpoints")
        need(r["source_id"] != r["target_id"], f"relation {r['id']} is a self-loop")
        key = (r["source_id"], r["target_id"], r.get("slot_id"))
        need(key not in seen, f"duplicate relation {key}")
        seen.add(key)
        need(r["level1_coupling"] == coupling_for_sense(r["level2_sense"], TAXONOMY),
             f"relation {r['id']} contradicts the v2 compile map")
        need(r["intended_strength_band"] in BAND_RANGE, f"relation {r['id']} band")
    need(bool(item["expected"].get("ladders")), "an item belongs to at least one ladder")


__all__ = [
    "BAND_MID",
    "DESIGN",
    "LADDERS",
    "MIN_STEP",
    "PIGGYBACK",
    "PLAN_SENSES",
    "PRIMARY_TYPES",
    "RARE_SENSES",
    "ROLES",
    "SLOT_SENSES",
    "TAXONOMY",
    "Casting",
    "assignments",
    "check_plan",
    "check_template",
    "coverage_assignments",
    "compare",
    "default_casting",
    "expected_label",
    "gold_relations",
    "label_matches",
    "lambda_lcs",
    "ladder_gold_check",
    "permute",
    "permutations_for",
    "rare_senses_for",
    "required_slots",
    "splice",
    "validate_item",
    "xstar_of",
]
