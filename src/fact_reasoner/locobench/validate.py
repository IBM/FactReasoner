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

# The gates and the committee.
#
# EVERY acceptance threshold in the harness lives in THRESHOLDS below and nowhere else.
# A run's behaviour is entirely determined by that dict, so it should be auditable in one
# screen -- and a reviewer should be able to diff it against Phase 1's table directly.
#
# Three rules here are not conveniences:
#
#   * GENERATOR EXCLUSION (R3). The model that wrote an item may not vote on it. A model
#     asked to recover relations from its own prose recovers its own lexical
#     fingerprints, which inflates every Target-A number.
#   * PLANTED ERRORS ARE MEASURED, NOT GRADED. P3 plants 4 deliberately-invalid relations
#     per plan. Scoring them as recovery failures made V1 unsatisfiable (ceiling 0.60
#     against a 0.80 threshold), so `gate_recovery` grades the valid relations and reports
#     the planted ones as the non-gating `V1.planted`.
#   * A CLOSED LIST IS CLOSED IN CODE. V3's hedge list and leakage exemptions are enforced
#     by `_filter_spans`, not left to the auditor's goodwill: live auditors flagged spans
#     the prompt exempts verbatim, and `parse_response` has already applied the same hedge
#     check as a rejection-sampling predicate before V3 ever runs.
#
# Phase 1's V2 exhaustiveness adjudicator and V5 agreement statistics (Fleiss/Cohen/
# Krippendorff) are deliberately absent. V2's `exclusive`/`co_necessity` labels are derived
# from the sense by `taxonomy_bridge.COMPILE`, which the rest of the harness already treats
# as authoritative, and neither V2 nor the kappa functions were ever wired to a call site.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fact_reasoner.locobench.taxonomy_bridge import (
    REQUIRED_EITHER,
    REQUIRED_SENSES,
)

# ----------------------------------------------------------------------------
# The single threshold table (Phase 1 Section 5.2 / Phase 2 tab:thresholds).
# ----------------------------------------------------------------------------

THRESHOLDS: dict[str, Any] = {
    # -- per-item admission --
    #
    # Both V1 rates are taken over the VALID planned relations only (see `gate_recovery`),
    # so with P3's mandated 6-valid-of-10 the denominator is 6 and these are integer gates:
    # 0.80 needs 5 of 6 and 0.70 needs 5 of 6 as well, since 4/6 = 0.667 misses both. One
    # relation therefore decides admission. DO NOT RAISE `v1_coupling` ABOVE 0.834: it
    # silently becomes unanimity-of-6, which is a fresh unsatisfiable gate of exactly the
    # species that scoring the planted errors used to be. Measured across 7 raters and 2
    # items, the valid-only rates ranged 0.833-1.00, so these clear with one relation spare.
    "v1_coupling": 0.80,  # fraction recovered with the correct coupling
    "v1_sense": 0.70,  # ... and with the correct sense
    "v1_rule": "majority",  # of the committee must meet both rates
    "v3_min_score": 4,  # fluency / formality / organization, on 1..5
    "v3_empty_spans": ("leakage", "hedging"),  # must be empty; artifacts recorded only
    "v3_rule": "majority",  # of the auditors must fail a facet for it to reject
    "v4_coverage": 1.00,  # every atom present, over the gating statuses below
    # WHICH V4 statuses reject. `altered` is deliberately absent: it is the judgment most
    # sensitive to the surface rewording P4 instruction 2 licenses, and 3 of 5 live V4
    # rejections were one `altered` atom. `missing` and `merged` are structural losses.
    "v4_gating_statuses": ("missing", "merged"),
    "v4_rule": "majority",  # of the panel must call an atom lost for it to count
    # -- plan structure (P3) --
    "n_claims": (14, 16),  # a dead copy of parse.N_CLAIMS_RANGE; keep them in step
    "n_relations": (8, 12),
    "n_non_relations": (4, 6),
    "validity_split": 0.55,  # target fraction valid
    "validity_tolerance": 0.15,  # per-family slack on that fraction
    # How many of P2's `[incorrect]` claims P3 must actually select. The response is required
    # to contain false content as well as true; without this, P3 selected none and the shipped
    # corpus was 100% `factual: true`. TWO, not four: P2 writes 4 incorrect of ~26 and P3
    # selects ~15, so ~2.3 are expected under indifferent sampling -- the quota sits just
    # below the natural rate. Demanding all 4 would force P3's hand on 4 of 15 slots and
    # collide with the five required senses and the selected-holding resolver, which is how
    # the next unsatisfiable gate gets built.
    "min_incorrect_atoms": 2,
    "window": 4,  # positions, or a shared named entity
    # -- perturbation (P5) --
    "length_drift": 0.15,
    # -- corpus level --
    "topic_floor": 3,
    "none_pool": 1500,
    # -- scoring margin (used by the metrics, not by admission) --
    "margin_sigmas": 2.0,
    "sigma_remines": 5,
}

# How much of V3's span lists to persist in a rejection record. Enough to diagnose what the
# auditor objected to without letting a pathological audit bloat the state file.
_SPAN_SAMPLE = 5
_SPAN_CHARS = 120


@dataclass
class GateResult:
    """The outcome of one gate.

    Attributes:
        gate: The validator or check, e.g. ``"V3"`` or ``"plan.rare_facets"``.
        passed: Whether it admitted the artefact.
        threshold: The threshold that applied, for the rejection record.
        observed: What was actually seen.
        detail: A human-readable reason, always set when ``passed`` is False.
    """

    gate: str
    passed: bool
    threshold: Any = None
    observed: Any = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "gate": self.gate,
            "passed": self.passed,
            "threshold": self.threshold,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass
class Verdict:
    """All gates for one artefact.

    Attributes:
        results: Every gate that ran, in order.
    """

    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every gate admitted."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[GateResult]:
        """The gates that rejected."""
        return [r for r in self.results if not r.passed]

    def reason(self) -> str:
        """A one-line summary of why this was rejected (empty if it passed)."""
        return "; ".join(f"{r.gate}: {r.detail}" for r in self.failures)

    def add(self, result: GateResult) -> Verdict:
        """Append a gate result and return self, for chaining."""
        self.results.append(result)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "passed": self.passed,
            "gates": [r.to_dict() for r in self.results],
            "reason": self.reason(),
        }


# ----------------------------------------------------------------------------
# Committee selection.
# ----------------------------------------------------------------------------


def committee_for(panel: list[Any], generator: str | None) -> list[Any]:
    """Return the voters for an item, excluding its own generator.

    Args:
        panel: The configured committee (objects with a ``name``, or plain strings).
        generator: The ``name`` of the model that ran P3/P4 for this item.

    Returns:
        The panel minus the generator, order preserved.
    """

    def name_of(m: Any) -> str:
        return m if isinstance(m, str) else getattr(m, "name", str(m))

    return [m for m in panel if name_of(m) != generator]


def majority(votes: list[Any]) -> tuple[Any, int, bool]:
    """Return the modal vote, its count, and whether it is a strict majority.

    Args:
        votes: The individual votes; may be empty.

    Returns:
        ``(winner, count, is_majority)``. ``winner`` is None for no votes. A tie yields
        the first-seen value with ``is_majority`` False, so the caller escalates rather
        than silently picking one.
    """
    if not votes:
        return None, 0, False
    counts: dict[Any, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    winners = [v for v in votes if counts[v] == top]
    return winners[0], top, top * 2 > len(votes)


def unanimous(votes: list[Any]) -> bool:
    """Whether every vote agrees (and there is at least one)."""
    return bool(votes) and len(set(votes)) == 1


# ----------------------------------------------------------------------------
# Plan gates (P3).
# ----------------------------------------------------------------------------


def gate_plan(
    plan: dict[str, Any], claims: list[dict[str, str]] | None = None
) -> Verdict:
    """Apply the P3 gates a parser cannot: rare facets, window, validity split, factuality.

    Structural checks (counts, legal values, sense/coupling agreement) already happened
    in :func:`fact_reasoner.locobench.parse.parse_plan`; these are the semantic ones,
    because they carry thresholds and thresholds live here.

    Args:
        plan: A parsed P3 plan.
        claims: P2's parsed claims, enabling the factuality gate. When omitted that gate is
            skipped, so callers that only have a plan keep working.

    Returns:
        The verdict.
    """
    v = Verdict()
    rels = plan.get("relations", [])
    senses = [r.get("sense") for r in rels]

    # Rare-facet floor: the categories with no prior gold data must be present, or the
    # corpus will not grade the couplings the benchmark exists for.
    missing = [s for s in REQUIRED_SENSES if s not in senses]
    if not any(s in senses for s in REQUIRED_EITHER):
        missing.append("/".join(REQUIRED_EITHER))
    v.add(
        GateResult(
            "plan.rare_facets",
            not missing,
            threshold=">= 1 each",
            observed=sorted(set(senses)),
            detail=f"missing required sense(s): {missing}" if missing else "",
        )
    )

    # Window: OBSERVED, NOT ENFORCED. Phase 1 R2 handles recoverability in three places
    # and assigns this one the weakest role: "P3 instruction 4 *biases* generation into the
    # admissible set". The authoritative check is `schema.annotate_window_admission`, which
    # runs the real candidate selector on the REALIZED text and records
    # window/gate/discourse_promoted/out_of_window per edge; the metrics then exclude
    # out-of-window gold and report the structural-miss rate separately.
    #
    # So rejecting a family here was over-enforcement, and expensive: it was the single
    # largest cause of plan-stage rejection on live runs, and the violations were mostly
    # end-of-plan edges referring back to the opening claim -- i.e. a conclusion tying to a
    # thesis, which is good discourse writing. Suppressing it would systematically strip
    # long-range links from the corpus. Note also that `_shares_entity` admits any two
    # claims sharing a leading capitalized word ("The ..."), so this check was already far
    # weaker than it looks.
    #
    # The count is still computed and reported, because the far-edge rate per generator is
    # a finding worth having.
    pos_of = {a["pos"]: a.get("text", "") for a in plan.get("atoms", [])}
    win = THRESHOLDS["window"]
    far = []
    for r in rels:
        d = abs(r["source_pos"] - r["target_pos"])
        if d <= win:
            continue
        if _shares_entity(
            pos_of.get(r["source_pos"], ""), pos_of.get(r["target_pos"], "")
        ):
            continue
        far.append((r["source_pos"], r["target_pos"], d))
    v.add(
        GateResult(
            "plan.window",
            True,  # observation only -- see the note above
            threshold=f"<= {win} positions or a shared entity (recorded, not enforced)",
            observed=far,
            detail=f"{len(far)} edge(s) beyond the window; recoverability is decided at "
            f"build time by the real candidate selector, not here: {far}"
            if far
            else "",
        )
    )

    # Validity split: the benchmark must be correct at the coherent end too, so the
    # corpus is balanced rather than skewed like LoReFact's.
    n = len(rels)
    n_valid = sum(1 for r in rels if r.get("validity") == "valid")
    frac = n_valid / n if n else 0.0
    target, tol = THRESHOLDS["validity_split"], THRESHOLDS["validity_tolerance"]
    ok = n == 0 or abs(frac - target) <= tol
    v.add(
        GateResult(
            "plan.validity_split",
            ok,
            threshold=f"{target:.2f} +/- {tol:.2f}",
            observed=round(frac, 3),
            detail="" if ok else f"valid fraction {frac:.2f} outside {target}+/-{tol}",
        )
    )

    # FACTUALITY: the response must contain false claims, not merely true ones. P2 generates
    # 4 `[incorrect]` claims of ~26, but P3 selects only 14-16 of them and nothing required it
    # to keep any -- so the shipped corpus had 170/170 atoms `factual: true` and the benchmark
    # had no false content at all. Checked here rather than at admission because `gate_plan`
    # is wired as a retryable check (`pipeline._plan_ok`), so a plan short on false claims is
    # re-planned with this reason attached instead of killing the family.
    if claims is not None:
        need = THRESHOLDS["min_incorrect_atoms"]
        incorrect = {
            " ".join((c.get("text") or "").split()).rstrip(".").lower()
            for c in claims
            if c.get("tag") == "incorrect"
        }
        selected = [
            " ".join((a.get("text") or "").split()).rstrip(".").lower()
            for a in plan.get("atoms", [])
        ]
        n_false = sum(1 for t in selected if t in incorrect)
        ok_f = n_false >= need
        v.add(
            GateResult(
                "plan.factuality",
                ok_f,
                threshold=f">= {need} incorrect claim(s) selected",
                observed={"n_incorrect_selected": n_false, "n_incorrect_available": len(incorrect)},
                detail=""
                if ok_f
                else (
                    f"only {n_false} of the {len(incorrect)} [incorrect] claim(s) were "
                    f"selected, need {need}. The response must assert false claims as well "
                    "as true ones; choose more of the [incorrect] claims."
                ),
            )
        )
    return v


def _shares_entity(a: str, b: str) -> bool:
    """Whether two claim texts share a capitalized multi-character token.

    A deliberately crude proxy for "shares a named entity": it is the escape hatch P3's
    instruction 4 offers for long-range pairs, and a crude test that errs toward
    admitting is preferable to rejecting a legitimate plan.
    """

    def ents(s: str) -> set[str]:
        return {
            w.strip(".,;:!?\"'")
            for w in s.split()
            if len(w) > 2 and w[0].isupper() and not w.isupper()
        }

    return bool(ents(a) & ents(b))


# ----------------------------------------------------------------------------
# Response gates (V1, V3, V4).
# ----------------------------------------------------------------------------


def _pair_key(source: Any, target: Any, sense: str = "") -> tuple[Any, ...]:
    """The IDENTITY of one relation: which two atoms it holds between, order-insensitive.

    Deliberately sense-independent. Deriving the key space from the sense -- which the
    previous version did, using ordered keys for directed senses and unordered for
    symmetric ones -- meant a planned and a recovered relation over the *same two atoms*
    could land in **different key spaces** whenever their senses disagreed in
    directedness, so the pair was unmatchable and scored 0 on sense *and* 0 on coupling.
    That is a category error: it counted "recovered with the wrong sense" as "not
    recovered", which is exactly the distinction V1 exists to measure. Measured live:
    gpt-oss-120b recovered a planned ``Concession(7, 12)`` as ``Contrast(7, 12)`` -- same
    endpoints, and both senses compile to ``contradiction`` -- and earned nothing, because
    ``Concession`` keys as ``("d", 7, 12)`` and ``Contrast`` as ``("u", "12", "7")``.

    Direction is not discarded, it moves: :func:`gate_recovery` compares the endpoint
    ORDER separately and withholds sense and coupling credit when a directed planned sense
    was recovered reversed. So ``wrong_direction`` (a first-class ``error_kind`` in P3
    instruction 8) still earns no credit -- it is now visible as a matched pair that failed
    on direction rather than as an absence, which is strictly more diagnosable.

    ``sense`` is accepted and ignored, so the existing call sites and any test that passes
    it keep working.

    Args:
        source: The relation's source position.
        target: The relation's target position.
        sense: Unused; retained for call-site compatibility.

    Returns:
        A hashable key identifying the unordered atom pair.
    """
    # Stringified before sorting because a planned position is an int while a recovered one
    # may be a numeric string, and mixed types are not orderable in Python 3.
    return tuple(sorted((str(source), str(target))))


def _is_reversed(planned: dict[str, Any], got: dict[str, Any]) -> bool:
    """Whether a matched pair was recovered with its endpoints the wrong way round.

    Only meaningful when the *planned* sense is directed: an undirected sense carries no
    order, so neither ordering is wrong. The planned sense is the reference because it is
    the ground truth being recovered against -- consulting the recovered sense instead
    would let a model escape the direction check by mislabelling the sense as symmetric.

    Args:
        planned: A plan relation (``source_pos``/``target_pos``/``sense``).
        got: The recovered relation matched to it (``source``/``target``).

    Returns:
        True if the planned sense is directed and the endpoints are swapped.
    """
    from fact_reasoner.locobench.taxonomy_bridge import is_directed

    try:
        if not is_directed(planned.get("sense", "")):
            return False
    except ValueError:
        # An unknown planned sense cannot be classified. Treat it as directed, matching
        # `_pair_key`'s historical conservative default, so a bogus sense cannot buy
        # direction-blind credit.
        pass
    return str(got.get("source")) != str(planned["source_pos"])


def _looks_zero_based(
    recovered: list[dict[str, Any]], n_atoms: int
) -> tuple[int, int] | None:
    """The observed endpoint range, when it looks 0-based against 1..n_atoms.

    Returns ``(lo, hi)`` if every endpoint is an int inside ``[0, n_atoms - 1]`` *and* at
    least one is 0 -- the signature of a model that indexed the atom list from zero while
    the plan numbers positions from one. Returns None otherwise.
    """
    if not recovered or n_atoms <= 0:
        return None
    vals: list[int] = []
    for r in recovered:
        for key in ("source", "target"):
            val = r.get(key)
            if not isinstance(val, int) or isinstance(val, bool):
                return None
            vals.append(val)
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if lo == 0 and hi <= n_atoms - 1:
        return (lo, hi)
    return None


def gate_recovery(
    planned: list[dict[str, Any]],
    recovered: list[dict[str, Any]],
    *,
    n_atoms: int | None = None,
) -> Verdict:
    """Apply the V1 gate: were the planned relations recoverable from the prose?

    Compares on ``(source, target)`` position pairs -- ordered for directed senses,
    unordered for the symmetric ones (see :func:`_pair_key`).

    Both sides are **1-based plan positions**. V1 is handed a mapping keyed 1..N precisely
    so it can return those keys without being shown the plan; when ``n_atoms`` is supplied
    a 0-based reply is detected and rejected with the evidence rather than silently
    shifted, because a silent shift would hide a prompt regression indefinitely. (Measured
    once for real: a 0-based reply scored 0.08/0.00, indistinguishable in the verdict from
    recovering nothing, while the same output re-indexed scored 0.50/0.50.)

    Relations planned ``validity: "invalid"`` are EXCLUDED from both rates. They are
    deliberate errors, so not recovering them as planned is the intended outcome; counting
    them made the gate unsatisfiable, because P3 mandates 4 invalid of 10 and the ceiling
    was therefore 6/10 = 0.60 against a 0.80 threshold. Whether each planted error was
    faithfully realized is reported separately as the non-gating ``V1.planted``. A relation
    with no ``validity`` key is graded, so older plans and call sites are unaffected.

    Args:
        planned: The plan's relations (``source_pos``/``target_pos``/``sense``, and
            optionally ``validity``/``error_kind``).
        recovered: V1's output (``source``/``target``/``sense``/``coupling``).
        n_atoms: How many atoms the plan selected. Enables the index-base check.

    Returns:
        The verdict. ``observed`` carries the two rates, the denominator they were taken
        over, *and* the pairs behind them, so a low rate is diagnosable without re-running
        the model.
    """
    v = Verdict()
    if not planned:
        return v.add(GateResult("V1", True, detail="no planned relations"))

    from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

    planned_pairs = [(p["source_pos"], p["target_pos"], p["sense"]) for p in planned]
    rec_pairs = [(r.get("source"), r.get("target"), r.get("sense")) for r in recovered]

    # Index-base check first: a base mismatch makes every rate meaningless, so report
    # that rather than a recall number computed against the wrong key space.
    if n_atoms is not None:
        zb = _looks_zero_based(recovered, n_atoms)
        if zb is not None:
            return v.add(
                GateResult(
                    "V1",
                    False,
                    threshold="1-based plan positions",
                    observed={
                        "index_range": list(zb),
                        "n_atoms": n_atoms,
                        "recovered_pairs": rec_pairs,
                        "planned_pairs": planned_pairs,
                    },
                    detail=(
                        f"indices appear 0-based (range {zb[0]}..{zb[1]}, "
                        f"n_atoms={n_atoms}); V1 must return the 1-based keys of the "
                        "atoms mapping it was given. Not auto-shifted on purpose: that "
                        "would mask a prompt regression."
                    ),
                )
            )

    # A LIST per pair, not one entry, and each recovered relation is consumed at most once.
    # A plan may legitimately carry two relations over the same two atoms -- P3 admits it
    # and a live plan did exactly that, pairing 10-11 as both `Alternative` and
    # `Concession` -- so a dict keyed by pair silently dropped all but the last, and the
    # survivor was then compared against *every* planned relation on that pair. Measured:
    # a plan whose two same-pair relations were BOTH recovered with the exactly correct
    # sense scored 0.5, not 1.0. (The previous sense-derived key masked this whenever the
    # two senses happened to differ in directedness, which is why it went unnoticed.)
    rec_by_pair: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in recovered:
        rec_by_pair.setdefault(_pair_key(r.get("source"), r.get("target")), []).append(r)

    def _quality(p: dict[str, Any], c: dict[str, Any]) -> tuple[bool, bool]:
        """How well a recovered relation explains a planned one: (sense, direction)."""
        return (c.get("sense") == p["sense"], not _is_reversed(p, c))

    # Assign in QUALITY order, not plan order, because assignment is competitive: several
    # planned relations can share one atom pair and contend for the same recovery. Greedy
    # plan-order assignment mis-awards it -- measured on a live plan holding both
    # `(1,2) Precedence` and `(2,1) Cause-Effect` against a single recovered
    # `(2,1) Evidence`: `Precedence` came first, claimed it, and was scored REVERSED, while
    # `Cause-Effect` -- which the recovery matched in direction and coupling class -- was
    # left with nothing and scored as absent. Two relations were penalized for one defect.
    # Best-first assignment gives the recovery to the relation it actually explains.
    order = sorted(
        range(len(planned)),
        key=lambda i: max(
            (
                _quality(planned[i], c)
                for c in rec_by_pair.get(
                    _pair_key(planned[i]["source_pos"], planned[i]["target_pos"]), []
                )
            ),
            default=(False, False),
        ),
        reverse=True,
    )

    def _is_planted(p: dict[str, Any]) -> bool:
        """Whether this relation was planned as a deliberate error.

        Defaults to NOT planted, so a relation carrying no ``validity`` key is graded --
        which keeps every pre-existing call site and any plan predating the field behaving
        exactly as before.
        """
        return p.get("validity", "valid") == "invalid"

    n_coupling = n_sense = 0
    n_planted_realized = 0
    by_kind: dict[str, dict[str, int]] = {}
    matched: list[tuple[Any, Any]] = []
    reversed_pairs: list[tuple[Any, Any, str]] = []
    for i in order:
        p = planned[i]
        planted = _is_planted(p)
        if planted:
            kind = str(p.get("error_kind") or "unspecified")
            slot = by_kind.setdefault(kind, {"n": 0, "recovered_as_planned": 0})
            slot["n"] += 1
        candidates = rec_by_pair.get(_pair_key(p["source_pos"], p["target_pos"]))
        if not candidates:
            continue
        got = max(candidates, key=lambda c: _quality(p, c))
        candidates.remove(got)
        matched.append((p["source_pos"], p["target_pos"]))
        # DIRECTION, graded here rather than baked into the key. The pair is identified by
        # its endpoints; whether the prose got the direction right is part of *how well* it
        # was recovered. A directed planned sense recovered with the endpoints swapped
        # earns neither sense nor coupling credit -- reversing Evidence inverts which claim
        # supports which, and reversing Precedence into Evidence turns a chronology into an
        # inference. Undirected senses carry no order, so nothing is checked for them.
        if _is_reversed(p, got):
            reversed_pairs.append(
                (p["source_pos"], p["target_pos"], p.get("sense", ""))
            )
            continue
        # Coupling recall COMPILES the recovered sense rather than reading the model's
        # own `coupling` field. COMPILE is the authority for that mapping everywhere else
        # -- `schema.py` asserts gold cannot contradict it and `parse.py` rejects a plan
        # that names a coupling disagreeing with its sense -- and the recovered field has
        # no other consumer in the harness. Grading a free-choice string against a derived
        # value made a *perfect* recovery fail: a model that returned the planned sense but
        # labelled its coupling non-canonically (Concession/"entailment" rather than
        # "contradiction") scored sense 1.00 and coupling 0.00. Compiling both sides is
        # what makes coupling the coarser, more robust label its higher threshold assumes.
        rec_sense = got.get("sense", "")
        try:
            rec_coupling = coupling_for_sense(rec_sense)
        except ValueError:
            # An unrecognized sense cannot be compiled, so it earns no coupling credit.
            # Guarded rather than propagated: `_pair_key` falls back to directed on a bad
            # sense, so such a relation can reach here and must not take down the gate.
            rec_coupling = None
        # PLANTED ERRORS ARE MEASURED, NOT GRADED. A relation planned `invalid` is broken
        # on purpose, so failing to recover it as planned is the intended outcome and must
        # not count against recall. Counting it was an unsatisfiable gate: P3 mandates 4
        # invalid of 10, so the ceiling was 6/10 = 0.60 against a 0.80 threshold, and no
        # writer and no reader could pass. Measured before the fix, five frontier models
        # independently scored one family at 0.80/0.80 over all 10 while scoring the 6
        # valid relations at 1.00/1.00 -- the shortfall was entirely this arithmetic.
        # Whether the error was faithfully realized is reported as `V1.planted` below.
        if planted:
            if rec_sense == p["sense"]:
                n_planted_realized += 1
                by_kind[kind]["recovered_as_planned"] += 1
            continue
        if rec_coupling is not None and rec_coupling == coupling_for_sense(p["sense"]):
            n_coupling += 1
        if rec_sense == p["sense"]:
            n_sense += 1

    n_planted = sum(1 for p in planned if _is_planted(p))
    total = len(planned) - n_planted
    if total <= 0:
        # Every planned relation was a deliberate error, so there is nothing to recover.
        # Passing is correct here: the gate measures recoverability of the sound relations
        # and this plan asserts none. Guarded explicitly rather than left to divide by zero.
        return v.add(
            GateResult(
                "V1",
                True,
                threshold={"coupling": THRESHOLDS["v1_coupling"],
                           "sense": THRESHOLDS["v1_sense"]},
                observed={"n_graded": 0, "n_planted": n_planted},
                detail="no valid planned relations to recover",
            )
        )
    r_coupling, r_sense = n_coupling / total, n_sense / total
    t_coupling, t_sense = THRESHOLDS["v1_coupling"], THRESHOLDS["v1_sense"]
    ok = r_coupling >= t_coupling and r_sense >= t_sense
    v.add(
        GateResult(
            "V1",
            ok,
            threshold={"coupling": t_coupling, "sense": t_sense},
            observed={
                "coupling": round(r_coupling, 3),
                "sense": round(r_sense, 3),
                # The denominator, recorded so a rate is never ambiguous about what it was
                # taken over. This is the field that makes the old unsatisfiable-gate bug
                # visible at a glance in a persisted record.
                "n_graded": total,
                "n_planted": n_planted,
                # The pairs, so "recovered nothing" and "matched the wrong key space" are
                # distinguishable from the persisted record alone.
                "matched_pairs": matched,
                # Matched on endpoints but recovered backwards. Called out separately
                # because it is a DIFFERENT defect from a missing relation and needs a
                # different fix (P4 direction wording, not more coverage), and because it
                # used to be indistinguishable from an absence.
                "reversed_pairs": reversed_pairs,
                "recovered_pairs": rec_pairs,
                "planned_pairs": planned_pairs,
            },
            detail=""
            if ok
            else f"recovery too low: coupling {r_coupling:.2f} (need {t_coupling}), "
            f"sense {r_sense:.2f} (need {t_sense}); matched "
            f"{len(matched)}/{total} planned pair(s)"
            + (
                f"; {len(reversed_pairs)} recovered with reversed endpoints: "
                f"{reversed_pairs[:4]}"
                if reversed_pairs
                else ""
            ),
        )
    )

    # Was the planted error faithfully realized? RECORDED, NEVER ENFORCED -- mirrors
    # `plan.window` and `V3.artifacts`. This is the readout the old denominator destroyed:
    # a planted error recovered as planned means P4 wrote the broken relation the plan
    # asked for, and one that vanished means the injection failed. Both are findings about
    # the corpus, and neither is grounds to reject a family. Without this the two cases are
    # indistinguishable, which is how a family whose error injection silently failed once
    # scored 0.90/0.90 and looked like the BEST result in the run.
    if n_planted:
        v.add(
            GateResult(
                "V1.planted",
                True,  # observation only
                threshold="recorded, not enforced",
                observed={
                    "n_invalid": n_planted,
                    "recovered_as_planned": n_planted_realized,
                    "by_error_kind": by_kind,
                },
                detail=(
                    f"{n_planted_realized}/{n_planted} planted error(s) realized as "
                    f"planned: { {k: v2['recovered_as_planned'] for k, v2 in by_kind.items()} }"
                ),
            )
        )
    return v


# Phrasings P4 REQUIRES and V3 must therefore never call leakage. A relation's coupling is
# a claim about two propositions' joint truth, and there is no way to express it in prose
# without one of these -- so flagging them punishes the writer for obeying instruction 3.
#
# V3's prompt already lists most of them as exempt, and live auditors flagged them anyway:
# measured on one response, auditors reported "at least one of these was true:" and "as
# indicated by the fact that" as leakage and "perhaps both" as hedging, all three named
# verbatim in the prompt as NOT reportable. So the exemption is enforced here as well as
# stated there. The five added beyond the prompt's own list are the P4-mandated phrasings it
# omitted: "one of these accounts must be wrong" (P4 instruction 3's Alternative marker, and
# the riskiest, since it talks about accounts being wrong), "one or both", and the three
# ordering markers past "before"/"after".
V3_EXEMPT_SPANS: tuple[str, ...] = (
    "either",
    "or",
    "at least one of",
    "at least one of these",
    "one or both",
    "perhaps both",
    "the two cannot both be true, and one of them must hold",
    "one of these accounts must be wrong",
    "although",
    "despite",
    "even though",
    "whereas",
    "by contrast",
    "on the other hand",
    "that is",
    "in other words",
    "equivalently",
    "for example",
    "specifically",
    "in one case",
    "before",
    "after",
    "subsequently",
    "earlier than",
    "which postdates",
    "which predates",
    "came after",
    "as indicated by",
    "as shown by",
    "which indicates",
    "confirms",
    "if",
    "provided that",
    "only when",
)


# Function words that carry no reference to the annotation machinery. Removed alongside the
# exempt phrases when deciding whether a span is *nothing but* mandated connective language,
# because an auditor quotes a clause rather than a bare connective: "at least one of these
# was true" and "either this or that" are the exempt marker plus glue, and the glue must not
# keep a false positive alive. Deliberately excludes every noun the leakage criterion names
# ("claim", "atom", "statement", "plan", "sense", "relation", ...), so a span mentioning one
# can never be filtered out this way.
_V3_FILLER = frozenset(
    """
    a an the this that these those it its they them their there here is are was were be been
    being of to in on at as by for from with and both one two each other others such same
    true false hold holds held case cases thing things above below fact facts
    """.split()
)


def _is_exempt_span(span: str) -> bool:
    """Whether a reported leakage span is nothing but P4-mandated connective language.

    A span is exempt when, after removing every exempt phrase and the function words that
    glue one into a clause, no content word survives -- i.e. the auditor quoted a connective
    rather than a reference to the plan. Deliberately NOT a substring test in the other
    direction: a real leakage span such as "the earlier statement that they were manufactured
    after 6500 BCE" *contains* "after", and dropping it for that reason would suppress a true
    positive. It survives here because "statement" and "manufactured" are content words.
    """
    low = " " + " ".join(str(span).lower().split()) + " "
    for phrase in sorted(V3_EXEMPT_SPANS, key=len, reverse=True):
        low = low.replace(f" {phrase} ", "  ")
    residue = [w.strip(".,;:!?\"'()") for w in low.split()]
    return not any(w and w.isalpha() and w not in _V3_FILLER for w in residue)


def _filter_spans(audit: dict[str, Any], response: str | None = None) -> dict[str, Any]:
    """Drop reported spans that V3's own prompt says are not reportable.

    Three rules, each enforcing a promise the prompt already makes:

    1. **The hedge list is CLOSED.** A hedging span must contain one of
       :data:`fact_reasoner.locobench.parse.HEDGE_WORDS`. This is not a second opinion about
       the prose: ``parse_response`` installs that same word-bounded check as the P4
       rejection-sampling predicate, so any hedge word still present has already been
       cleared, and a span without one was never a hedge. Live auditors flagged "perhaps
       both" (mandated by P4) and "One or both of these factors may hold" (no listed word at
       all) as hedging.
    2. **P4-mandated connectives are not leakage.** See :data:`V3_EXEMPT_SPANS`.
    3. **Spans must be quoted verbatim.** The prompt says "if you cannot copy it out of the
       text, it is not there and must not be reported"; a span absent from the response is a
       hallucinated quote and cannot evidence anything. Skipped when ``response`` is None.

    Args:
        audit: One auditor's parsed V3 output. Not mutated.
        response: The prose it judged, for the verbatim check. Optional.

    Returns:
        A shallow copy with the gated span lists filtered, plus ``_raw_counts`` recording
        what each list held before filtering, so suppression stays auditable rather than
        silently discarding a systematically-wrong auditor's evidence.
    """
    from fact_reasoner.locobench.parse import HEDGE_WORDS

    out = dict(audit)
    raw: dict[str, int] = {}
    hay = " ".join((response or "").lower().split())
    for kind in THRESHOLDS["v3_empty_spans"]:
        spans = list(audit.get(kind) or [])
        raw[kind] = len(spans)
        kept = []
        for s in spans:
            text = str(s)
            flat = " ".join(text.lower().split())
            if response is not None and flat and flat not in hay:
                continue  # rule 3: not quoted from the text
            if kind == "hedging":
                if not any(
                    re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)
                    for w in HEDGE_WORDS
                ):
                    continue  # rule 1: not on the closed list
            elif kind == "leakage" and _is_exempt_span(text):
                continue  # rule 2: mandated connective
            kept.append(s)
        out[kind] = kept
    out["_raw_counts"] = raw
    return out


def gate_audit(audit: dict[str, Any]) -> Verdict:
    """Apply the V3 gate: naturalness scores and the absence of leakage.

    Args:
        audit: V3's parsed output.

    Returns:
        The verdict.
    """
    v = Verdict()
    floor = THRESHOLDS["v3_min_score"]
    low = {
        k: audit.get(k)
        for k in ("fluency", "formality", "organization")
        if (audit.get(k) or 0) < floor
    }
    v.add(
        GateResult(
            "V3.scores",
            not low,
            threshold=f">= {floor}",
            observed={
                k: audit.get(k) for k in ("fluency", "formality", "organization")
            },
            detail=f"below floor: {low}" if low else "",
        )
    )
    dirty = {k: audit.get(k) for k in THRESHOLDS["v3_empty_spans"] if audit.get(k)}
    # Record the span TEXT, not just the counts. A bare count is undiagnosable: "leakage:
    # 16" reads as leaky prose when in fact every span was a connective P4 *mandates*,
    # which is only visible once you can see them. The V1 index bug hid behind exactly this
    # kind of count-only record. Bounded so a pathological audit cannot bloat the state
    # file; the full prose is already persisted alongside as `rejected_response`, so this
    # exposes nothing new.
    v.add(
        GateResult(
            "V3.spans",
            not dirty,
            threshold="empty",
            observed={
                **{k: len(audit.get(k, [])) for k in THRESHOLDS["v3_empty_spans"]},
                "spans": {
                    k: [
                        str(s)[:_SPAN_CHARS]
                        for s in (audit.get(k) or [])[:_SPAN_SAMPLE]
                    ]
                    for k in THRESHOLDS["v3_empty_spans"]
                    if audit.get(k)
                },
            },
            detail=f"non-empty span list(s): { {k: len(x) for k, x in dirty.items()} }"
            if dirty
            else "",
        )
    )
    # `artifacts` is judged but never gated -- enumerated structure and template phrasing
    # are quality signals, not admission criteria. It was previously dropped on the floor
    # entirely; recorded here so a reviewer can see what V3 objected to. Mirrors
    # `plan.window`: passed=True, observation only.
    arts = audit.get("artifacts") or []
    v.add(
        GateResult(
            "V3.artifacts",
            True,  # observation only
            threshold="recorded, not enforced",
            observed={
                "count": len(arts),
                "spans": [str(s)[:_SPAN_CHARS] for s in arts[:_SPAN_SAMPLE]],
            },
            detail=f"{len(arts)} artifact span(s) recorded" if arts else "",
        )
    )
    return v


def gate_audit_panel(
    audits: list[tuple[str, dict[str, Any]]], *, response: str | None = None
) -> Verdict:
    """Apply the V3 gate over several auditors, rejecting only on a majority.

    A single rater must not decide admission here, because V3's judgments diverge far more
    across models than the other validators' do. Measured on one response, identical prose
    and identical prompt: ``opus-5`` 0 leakage spans, ``sonnet-4-6`` **5**, ``opus-4-8`` 0,
    ``opus-4-7`` 0 -- and the harness had been picking whichever model happened to sit first
    in committee order. Three of four agreed; the arbitrary pick was the outlier, and it
    rejected the family. All four agreed on one hedging span, which is what a true positive
    looks like under this scheme.

    Each facet is voted separately rather than voting on the whole verdict, so a real
    leakage span still rejects even when the scores are unanimous and vice versa.

    Every auditor's spans pass through :func:`_filter_spans` FIRST, so the vote is taken over
    reportable spans only. Filtering before the vote rather than after is deliberate: two
    auditors independently flagging the same prompt-exempt connective would otherwise be a
    "majority" and would reject the family, which is exactly what happened live.

    Args:
        audits: ``(auditor name, parsed V3 output)`` pairs. A single entry degrades to the
            same behaviour as :func:`gate_audit`, so a one-model config still works.
        response: The prose judged, enabling the verbatim-quote filter. Optional so existing
            call sites keep working.

    Returns:
        The verdict, with every auditor's vote recorded in ``observed`` -- a lone dissenter
        is otherwise invisible, and distinguishing "everyone saw it" from "one model did"
        is the whole point of voting.
    """
    v = Verdict()
    if not audits:
        return v.add(GateResult("V3", False, detail="no audit output"))

    raw_counts = {
        name: {k: len(a.get(k) or []) for k in THRESHOLDS["v3_empty_spans"]}
        for name, a in audits
    }
    audits = [(name, _filter_spans(a, response)) for name, a in audits]

    floor = THRESHOLDS["v3_min_score"]
    keys = ("fluency", "formality", "organization")
    n = len(audits)
    needed = n // 2 + 1  # strict majority

    low_votes = {
        name: {k: a.get(k) for k in keys if (a.get(k) or 0) < floor}
        for name, a in audits
    }
    n_low = sum(1 for d in low_votes.values() if d)
    v.add(
        GateResult(
            "V3.scores",
            n_low < needed,
            threshold=f">= {floor} (majority of {n} auditor(s))",
            observed={
                "votes": {name: {k: a.get(k) for k in keys} for name, a in audits},
                "n_below_floor": n_low,
                "needed_to_reject": needed,
            },
            detail=(
                f"{n_low}/{n} auditor(s) scored below the floor: "
                f"{ {k: d for k, d in low_votes.items() if d} }"
            )
            if n_low >= needed
            else "",
        )
    )

    span_keys = THRESHOLDS["v3_empty_spans"]
    dirty_votes = {
        name: {k: len(a.get(k) or []) for k in span_keys if a.get(k)}
        for name, a in audits
    }
    # Vote PER SPAN KIND, not on "flagged anything". Leakage and hedging are independent
    # failure modes with different reliability: the measured panel agreed unanimously on one
    # hedge while splitting 1-3 on leakage, so pooling them would let the unanimous hedge
    # carry the disputed leakage into a rejection and hide the disagreement entirely.
    per_kind = {k: sum(1 for d in dirty_votes.values() if d.get(k)) for k in span_keys}
    rejecting = {k: c for k, c in per_kind.items() if c >= needed}
    v.add(
        GateResult(
            "V3.spans",
            not rejecting,
            threshold=f"empty (majority of {n} auditor(s), per span kind)",
            observed={
                "votes": dirty_votes,
                "n_flagging_per_kind": per_kind,
                "needed_to_reject": needed,
                # What each auditor reported BEFORE `_filter_spans`. Kept so the filtering is
                # auditable: a large gap between raw and voted counts means an auditor is
                # systematically reporting prompt-exempt spans, which is a finding about the
                # auditor, and silently discarding it would hide a broken panel member.
                "raw_counts": raw_counts,
                "spans": {
                    name: {
                        k: [
                            str(s)[:_SPAN_CHARS]
                            for s in (a.get(k) or [])[:_SPAN_SAMPLE]
                        ]
                        for k in span_keys
                        if a.get(k)
                    }
                    for name, a in audits
                    if any(a.get(k) for k in span_keys)
                },
            },
            detail=(
                f"a majority flagged { {k: f'{c}/{n}' for k, c in rejecting.items()} }: "
                f"{ {k: d for k, d in dirty_votes.items() if d} }"
            )
            if rejecting
            else "",
        )
    )

    arts = {name: (a.get("artifacts") or []) for name, a in audits}
    v.add(
        GateResult(
            "V3.artifacts",
            True,  # observation only, as in the single-auditor path
            threshold="recorded, not enforced",
            observed={
                "counts": {name: len(x) for name, x in arts.items()},
                "spans": {
                    name: [str(s)[:_SPAN_CHARS] for s in x[:_SPAN_SAMPLE]]
                    for name, x in arts.items()
                    if x
                },
            },
            detail="",
        )
    )
    return v


def gate_coverage(entries: list[dict[str, Any]], n_atoms: int) -> Verdict:
    """Apply the V4 gate: every planned atom present, and the window re-verified.

    Only the statuses in ``THRESHOLDS["v4_gating_statuses"]`` reject. ``altered`` is
    recorded but does not gate: it means "asserts something related but changes the
    content", which is the judgment most sensitive to the surface rewording P4 instruction 2
    explicitly licenses, whereas ``missing`` and ``merged`` are structural. Measured across
    11 live rejections, 5 were V4 and 3 of those were a SINGLE ``altered`` atom at coverage
    0.938-0.944 -- a family lost to one rater's wording judgment.

    Args:
        entries: V4's parsed output.
        n_atoms: How many atoms the plan selected.

    Returns:
        The verdict.
    """
    v = Verdict()
    gating = THRESHOLDS["v4_gating_statuses"]
    bad = [e for e in entries if e.get("status") in gating]
    altered = [e for e in entries if e.get("status") == "altered"]
    present = n_atoms - len(bad)
    frac = present / n_atoms if n_atoms else 0.0
    need = THRESHOLDS["v4_coverage"]
    ok = frac >= need
    v.add(
        GateResult(
            "V4",
            ok,
            threshold=need,
            observed={
                "coverage": round(frac, 3),
                "gating_statuses": list(gating),
                # Non-gating, but recorded: a rising `altered` rate is a P4 fidelity signal
                # worth having even though it must not cost a family.
                "altered": [(e.get("index"), e.get("span")) for e in altered[:6]],
                "n_altered": len(altered),
            },
            detail=""
            if ok
            else f"{len(bad)} atom(s) not present: "
            f"{[(e.get('index'), e.get('status')) for e in bad][:6]}",
        )
    )
    return v


def gate_coverage_panel(
    coverages: list[tuple[str, list[dict[str, Any]]]], n_atoms: int
) -> Verdict:
    """Apply the V4 gate over several raters, counting an atom lost only on a majority.

    V4 was the last gate decided by a single rater, and it is a conjunction over 14-16
    stochastic per-atom judgments, so its family-level false-negative rate compounds: at a
    2% per-atom error rate roughly a quarter of well-formed families fail, and the measured
    rate was nearer 5%. Voting per ATOM rather than on the whole verdict is what fixes that
    without weakening the requirement -- "every atom present" stays absolute, but one rater
    can no longer decide that an atom is absent.

    Measured on identical prose: six raters returned all 17 atoms asserted while one returned
    three defects and another one. Under the single-rater gate, drawing either of the latter
    two would have rejected a family the other six judged perfect.

    Args:
        coverages: ``(rater name, parsed V4 output)`` pairs. A single entry reduces to
            :func:`gate_coverage`'s behaviour.
        n_atoms: How many atoms the plan selected.

    Returns:
        The verdict, with every rater's per-atom dissent recorded.
    """
    v = Verdict()
    if not coverages:
        return v.add(GateResult("V4", False, detail="no coverage output"))

    gating = THRESHOLDS["v4_gating_statuses"]
    n = len(coverages)
    needed = n // 2 + 1  # strict majority, mirroring `gate_audit_panel`

    # Per atom index: who called it lost, and who called it merely altered.
    #
    # The index is CANONICALIZED to an int where it looks like one. Live raters disagree about
    # the type -- `parse_coverage` accepts both `3` and `"3"`, and on the first live run one
    # model returned ints while another returned numeric strings. Two consequences, both bad
    # and both silent: sorting the mixed keys raises TypeError, and, worse, `3` and `"3"` key
    # as DIFFERENT atoms, so two raters agreeing that one atom is missing would each be
    # counted once against a quorum of two and the majority would never form.
    def _idx(raw: Any) -> Any:
        if isinstance(raw, bool) or raw is None:
            return raw
        if isinstance(raw, int):
            return raw
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return str(raw)

    def _order(key: Any) -> tuple[int, float, str]:
        """Sort ints numerically, then everything else by text, with None last."""
        if key is None:
            return (2, 0.0, "")
        if isinstance(key, int) and not isinstance(key, bool):
            return (0, float(key), "")
        return (1, 0.0, str(key))

    lost_by: dict[Any, list[str]] = {}
    altered_by: dict[Any, list[str]] = {}
    for name, entries in coverages:
        for e in entries:
            idx = _idx(e.get("index"))
            status = e.get("status")
            if status in gating:
                lost_by.setdefault(idx, []).append(name)
            elif status == "altered":
                altered_by.setdefault(idx, []).append(name)

    lost = sorted(
        (idx for idx, voters in lost_by.items() if len(set(voters)) >= needed),
        key=_order,
    )
    present = n_atoms - len(lost)
    frac = present / n_atoms if n_atoms else 0.0
    need = THRESHOLDS["v4_coverage"]
    ok = frac >= need
    v.add(
        GateResult(
            "V4",
            ok,
            threshold=need,
            observed={
                "coverage": round(frac, 3),
                "gating_statuses": list(gating),
                "needed_to_reject": needed,
                # The full dissent map, so a lone rater's objection is visible rather than
                # silently outvoted -- the same reason `gate_audit_panel` records its votes.
                "lost_votes": {
                    str(k): v2
                    for k, v2 in sorted(lost_by.items(), key=lambda kv: _order(kv[0]))
                },
                "altered_votes": {
                    str(k): v2
                    for k, v2 in sorted(altered_by.items(), key=lambda kv: _order(kv[0]))
                },
                "n_raters": n,
            },
            detail=""
            if ok
            else f"{len(lost)} atom(s) a majority of {n} rater(s) found not present: "
            f"{[(i, lost_by[i]) for i in lost[:6]]}",
        )
    )
    return v


def gate_length_drift(base: str, perturbed: str, *, operator: str = "") -> GateResult:
    """Apply the P5 length-drift gate.

    Args:
        base: The parent response.
        perturbed: The perturbed response.
        operator: The call applied. Three operators legitimately add or remove a
            sentence, so they get a sentence's worth of extra slack.

    Returns:
        The gate result.
    """
    nb, np_ = len(base.split()), len(perturbed.split())
    limit = THRESHOLDS["length_drift"]
    if operator in ("inject_contradiction", "remove_resolution", "break_chain"):
        limit += 0.10
    drift = abs(np_ - nb) / nb if nb else 0.0
    ok = drift <= limit
    return GateResult(
        "P5.length_drift",
        ok,
        threshold=f"<= {limit:.0%}",
        observed=f"{drift:.1%}",
        detail="" if ok else f"length drifted {drift:.1%} (limit {limit:.0%})",
    )


def gate_text_changed(base: str, perturbed: str, *, operator: str = "") -> GateResult:
    """Apply the P5 text-edit gate: a perturbation must actually change the prose.

    Every one of :data:`perturb.ALL_CALLS` is a text edit -- even the two edge-invariant
    ones, since ``shuffle_order`` reorders sentences and ``ordering_only`` swaps a
    connective -- so a rung whose response is byte-identical to its parent's did not
    happen, whatever the labels say.

    This is the *text-side* counterpart of the adjacency gate, and neither subsumes the
    other. The adjacency gate compares edge SIGNATURES, so it passes a rung whose labels
    differ while its prose does not: measured on f013, where ``add_resolution`` set the
    resolution flag on the gold edge but returned the base prose unchanged. The pair then
    has gold that a gold-arm readout can separate and text that a mined arm cannot, so the
    ``c1`` strict-increase assertion over it is unfalsifiable from the response alone --
    exactly the class of defect the per-rung relation fix (Defect 2) was meant to end.

    Whitespace-insensitive: a reflowed but otherwise identical response is still no edit.

    Args:
        base: The parent response.
        perturbed: The perturbed response.
        operator: The call applied, for the message.

    Returns:
        The gate result.
    """
    same = " ".join(base.split()) == " ".join(perturbed.split())
    return GateResult(
        "P5.text_changed",
        not same,
        threshold="response differs from parent",
        observed="identical" if same else "differs",
        detail=""
        if not same
        else (
            f"{operator or 'the perturbation'} returned the parent response unchanged, so "
            "the rung is not a distinct item; its labels would assert a coherence "
            "difference no reader or scorer could see in the text"
        ),
    )


# ----------------------------------------------------------------------------
# Human-annotation sampling.
# ----------------------------------------------------------------------------


def stratified_sample(items: list[dict[str, Any]], n: int = 120) -> list[str]:
    """Choose the human-annotation subsample, stratified as Phase 1 requires.

    Priority order: every item carrying an ``exclusive`` or ``co_necessity`` gold edge,
    every resolved Concession, every ordering-only rung, then a deterministic remainder.
    Those are exactly the facets with no prior gold data and the highest expected model
    unreliability, so uniform sampling would spend the human budget where it is least
    needed.

    Args:
        items: The admitted corpus.
        n: Target sample size.

    Returns:
        Item ids, in priority order, at most ``n`` of them.
    """
    must: list[str] = []
    seen: set[str] = set()

    def take(item: dict[str, Any]) -> None:
        iid = item.get("id")
        if iid and iid not in seen:
            seen.add(iid)
            must.append(iid)

    for it in items:
        rels = it.get("relations", [])
        if any(r.get("level1_coupling") in ("exclusive", "co_necessity") for r in rels):
            take(it)
        elif any(r.get("is_resolved_concession") for r in rels):
            take(it)
        elif any(r.get("ordering_only") for r in rels):
            take(it)

    if len(must) < n:
        for it in items:  # deterministic remainder, by corpus order
            if len(must) >= n:
                break
            take(it)
    return must[:n]


__all__ = [
    "THRESHOLDS",
    "V3_EXEMPT_SPANS",
    "GateResult",
    "Verdict",
    "committee_for",
    "gate_audit",
    "gate_audit_panel",
    "gate_coverage",
    "gate_coverage_panel",
    "gate_length_drift",
    "gate_text_changed",
    "gate_plan",
    "gate_recovery",
    "majority",
    "stratified_sample",
    "unanimous",
]
