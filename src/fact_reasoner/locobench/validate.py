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

# The gates, the committee, and the agreement statistics.
#
# EVERY acceptance threshold in the harness lives in THRESHOLDS below and nowhere else.
# A run's behaviour is entirely determined by that dict, so it should be auditable in one
# screen -- and a reviewer should be able to diff it against Phase 1's table directly.
#
# Two rules here are not conveniences:
#
#   * GENERATOR EXCLUSION (R3). The model that wrote an item may not vote on it. A model
#     asked to recover relations from its own prose recovers its own lexical
#     fingerprints, which inflates every Target-A number.
#   * V2 UNANIMITY (R1). An `exclusive` gold label needs every voter to agree, because
#     exhaustiveness requires quantifying over possible worlds and is the label with no
#     prior art. A split vote still ships -- flagged `low_agreement` -- because an
#     unreliable facet is itself a result about the taxonomy.

from __future__ import annotations

import math
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
    "v1_coupling": 0.80,  # fraction recovered with the correct coupling
    "v1_sense": 0.70,  # ... and with the correct sense
    "v1_rule": "majority",  # of the committee must meet both rates
    "v2_exclusive": "unanimity",  # to assign an `exclusive` gold label
    "v3_min_score": 4,  # fluency / formality / organization, on 1..5
    "v3_empty_spans": ("leakage", "hedging"),  # must be empty; artifacts recorded only
    "v4_coverage": 1.00,  # every atom `asserted`
    # -- plan structure (P3) --
    "n_claims": (14, 18),
    "n_relations": (8, 12),
    "n_non_relations": (4, 6),
    "validity_split": 0.55,  # target fraction valid
    "validity_tolerance": 0.15,  # per-family slack on that fraction
    "window": 4,  # positions, or a shared named entity
    # -- perturbation (P5) --
    "length_drift": 0.15,
    # -- corpus level --
    "topic_floor": 3,
    "none_pool": 1500,
    # -- agreement (V5) --
    "kappa_coupling": 0.70,
    "kappa_sense": 0.60,
    "kappa_exhaustive": 0.55,
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


def gate_plan(plan: dict[str, Any]) -> Verdict:
    """Apply the P3 gates a parser cannot: rare facets, window, validity split.

    Structural checks (counts, legal values, sense/coupling agreement) already happened
    in :func:`fact_reasoner.locobench.parse.parse_plan`; these are the semantic ones,
    because they carry thresholds and thresholds live here.

    Args:
        plan: A parsed P3 plan.

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


def _pair_key(source: Any, target: Any, sense: str) -> tuple[Any, ...]:
    """The comparison key for one relation: directed unless the sense is symmetric.

    A ``frozenset`` for every sense -- which is what this used to be -- made the match
    blind to direction, so ``wrong_direction`` (a first-class ``error_kind`` in P3
    instruction 8) scored as a hit on the pair. It also collapsed self-loops and silently
    overwrote when V1 emitted both (i, j) and (j, i). Directed senses now compare in
    order; genuinely undirected ones (Alternative, Disjunction, Restatement, ...) keep the
    order-insensitive comparison they need.
    """
    from fact_reasoner.locobench.taxonomy_bridge import is_directed

    try:
        directed = is_directed(sense)
    except ValueError:  # an unknown sense cannot match a planned edge anyway
        directed = True
    if directed:
        return ("d", source, target)
    return ("u", *sorted((str(source), str(target))))


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

    Args:
        planned: The plan's relations (``source_pos``/``target_pos``/``sense``).
        recovered: V1's output (``source``/``target``/``sense``/``coupling``).
        n_atoms: How many atoms the plan selected. Enables the index-base check.

    Returns:
        The verdict. ``observed`` carries the two rates *and* the pairs behind them, so a
        low rate is diagnosable without re-running the model.
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

    rec_by_pair: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in recovered:
        rec_by_pair[_pair_key(r.get("source"), r.get("target"), r.get("sense", ""))] = r

    n_coupling = n_sense = 0
    matched: list[tuple[Any, Any]] = []
    for p in planned:
        got = rec_by_pair.get(_pair_key(p["source_pos"], p["target_pos"], p["sense"]))
        if not got:
            continue
        matched.append((p["source_pos"], p["target_pos"]))
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
        if rec_coupling is not None and rec_coupling == coupling_for_sense(p["sense"]):
            n_coupling += 1
        if rec_sense == p["sense"]:
            n_sense += 1

    total = len(planned)
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
                # The pairs, so "recovered nothing" and "matched the wrong key space" are
                # distinguishable from the persisted record alone.
                "matched_pairs": matched,
                "recovered_pairs": rec_pairs,
                "planned_pairs": planned_pairs,
            },
            detail=""
            if ok
            else f"recovery too low: coupling {r_coupling:.2f} (need {t_coupling}), "
            f"sense {r_sense:.2f} (need {t_sense}); matched "
            f"{len(matched)}/{total} planned pair(s)",
        )
    )
    return v


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


def gate_coverage(entries: list[dict[str, Any]], n_atoms: int) -> Verdict:
    """Apply the V4 gate: every planned atom asserted, and the window re-verified.

    Args:
        entries: V4's parsed output.
        n_atoms: How many atoms the plan selected.

    Returns:
        The verdict.
    """
    v = Verdict()
    asserted = sum(1 for e in entries if e.get("status") == "asserted")
    frac = asserted / n_atoms if n_atoms else 0.0
    need = THRESHOLDS["v4_coverage"]
    bad = [e for e in entries if e.get("status") != "asserted"]
    ok = frac >= need
    v.add(
        GateResult(
            "V4",
            ok,
            threshold=need,
            observed=round(frac, 3),
            detail=""
            if ok
            else f"{len(bad)} atom(s) not asserted: "
            f"{[(e.get('index'), e.get('status')) for e in bad][:6]}",
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


# ----------------------------------------------------------------------------
# Agreement statistics.
# ----------------------------------------------------------------------------


def fleiss_kappa(ratings: list[list[Any]]) -> float | None:
    """Fleiss' kappa over categorical ratings.

    Args:
        ratings: One list of rater labels per item. Items whose rater count differs are
            allowed; each is normalized independently.

    Returns:
        The kappa, or None when it is undefined (fewer than two items, or fewer than two
        raters on some item).
    """
    rows = [r for r in ratings if r and len(r) >= 2]
    if len(rows) < 2:
        return None
    cats = sorted({c for r in rows for c in r}, key=str)
    if len(cats) < 2:
        return 1.0  # everyone agreed on everything; kappa is degenerate but perfect

    p_i = []
    col_sums = dict.fromkeys(cats, 0.0)
    total = 0
    for r in rows:
        n = len(r)
        counts = {c: r.count(c) for c in cats}
        p_i.append((sum(v * v for v in counts.values()) - n) / (n * (n - 1)))
        for c in cats:
            col_sums[c] += counts[c]
        total += n
    p_bar = sum(p_i) / len(p_i)
    p_e = sum((col_sums[c] / total) ** 2 for c in cats)
    if math.isclose(p_e, 1.0):
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def cohen_kappa(a: list[Any], b: list[Any]) -> float | None:
    """Cohen's kappa between two raters.

    Args:
        a: Rater A's labels.
        b: Rater B's labels, same length and order.

    Returns:
        The kappa, or None if the inputs are unusable.
    """
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if math.isclose(expected, 1.0):
        return 1.0
    return (agree - expected) / (1.0 - expected)


def krippendorff_alpha_ordinal(
    ratings: list[list[Any]], levels: list[Any]
) -> float | None:
    """Krippendorff's alpha for ordinal data, over an explicit level order.

    Used for strength bands (weak < moderate < strong), where treating a
    weak-vs-strong disagreement as no worse than weak-vs-moderate would flatter the
    annotators.

    Args:
        ratings: One list of rater labels per item.
        levels: The ordered level values.

    Returns:
        The alpha, or None when undefined.
    """
    rank = {lv: i for i, lv in enumerate(levels)}
    rows = [[rank[x] for x in r if x in rank] for r in ratings]
    rows = [r for r in rows if len(r) >= 2]
    if len(rows) < 2:
        return None

    def sq(x: float) -> float:
        return x * x

    d_o_num = d_o_den = 0.0
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d_o_num += sq(r[i] - r[j])
                    d_o_den += 1
    if not d_o_den:
        return None
    d_o = d_o_num / d_o_den

    flat = [x for r in rows for x in r]
    d_e_num = d_e_den = 0.0
    for i, x in enumerate(flat):
        for j, y in enumerate(flat):
            if i != j:
                d_e_num += sq(x - y)
                d_e_den += 1
    if not d_e_den:
        return None
    d_e = d_e_num / d_e_den
    if math.isclose(d_e, 0.0):
        return 1.0
    return 1.0 - d_o / d_e


def agreement_report(
    coupling: list[list[str]],
    sense: list[list[str]],
    exhaustive: tuple[list[bool], list[bool]] | None = None,
    strength: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Compute the V5 agreement statistics and flag facets below their floors.

    A facet below its floor is not dropped: it ships flagged ``low_agreement`` and is
    excluded from headline metrics but reported, because an unreliable facet is itself a
    result about the taxonomy.

    Args:
        coupling: Per-edge rater coupling labels.
        sense: Per-edge rater sense labels.
        exhaustive: Two raters' binary exhaustiveness calls, for Cohen's kappa.
        strength: Per-edge rater strength bands, for ordinal alpha.

    Returns:
        The statistics, the floors, and a ``low_agreement`` list of facet names.
    """
    k_coupling = fleiss_kappa(coupling)
    k_sense = fleiss_kappa(sense)
    k_exh = cohen_kappa(*exhaustive) if exhaustive else None
    a_strength = (
        krippendorff_alpha_ordinal(strength, ["weak", "moderate", "strong"])
        if strength
        else None
    )

    low: list[str] = []
    for name, value, floor in (
        ("coupling", k_coupling, THRESHOLDS["kappa_coupling"]),
        ("sense", k_sense, THRESHOLDS["kappa_sense"]),
        ("exhaustiveness", k_exh, THRESHOLDS["kappa_exhaustive"]),
    ):
        if value is not None and value < floor:
            low.append(name)

    return {
        "kappa_coupling": k_coupling,
        "kappa_sense": k_sense,
        "kappa_exhaustive": k_exh,
        "alpha_strength": a_strength,
        "floors": {
            "coupling": THRESHOLDS["kappa_coupling"],
            "sense": THRESHOLDS["kappa_sense"],
            "exhaustiveness": THRESHOLDS["kappa_exhaustive"],
        },
        "low_agreement": low,
    }


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
    "GateResult",
    "Verdict",
    "agreement_report",
    "cohen_kappa",
    "committee_for",
    "fleiss_kappa",
    "gate_audit",
    "gate_coverage",
    "gate_length_drift",
    "gate_plan",
    "gate_recovery",
    "krippendorff_alpha_ordinal",
    "majority",
    "stratified_sample",
    "unanimous",
]
