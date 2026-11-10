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

# Rung realization: make a perturbed rung's prose carry the change its labels assert.
#
# Before this module, a rung's labels came from `perturb.apply_calls` and its text from
# P5, and the only checks tying them together were that the text CHANGED and that
# adjacent label sets DIFFERED. A blind side-by-side reading afterwards found the planted
# change realized in 5/39 dev and 132/268 release transitions; `drop_relation` 0/18 and
# 31/113, its conflict-carrying sentence surviving verbatim in 17/18 dev drops.
#
# Three pieces, used by `pipeline.generate_family` and by `rung_repair`:
#
#   * `screen_targets` -- edges a call cannot realize, found BEFORE targeting:
#       - drop_relation: a pair whose claims are incompatible ON CONTENT (V5c). A drop
#         keeps both atoms, so it can only remove a conflict the text constructs.
#       - add_resolution: a conflict the base text already presents as resolved (V5 on
#         base vs base). Resolving it again changes nothing a reader can see.
#   * `perturb_call` -- one P5 call gated by the deterministic span check and then V5,
#     retried with the gate's reason fed back.
#   * `realize_rung` -- a rung's calls applied in order to its parent's text, each call
#     gated against the text it was handed, with one re-target on exhaustion.
#
# Each call is gated against ITS OWN input text, not the rung's parent: a composed rung
# (CONFLICT rung 4 = resolve + drop + drop) is then a sequence of local, individually
# checkable edits, and the ladder's adjacent-rung differences follow from the
# parent-chain construction rather than needing a separate whole-rung reading.

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fact_reasoner.locobench import parse, validate
from fact_reasoner.locobench.perturb import apply_calls

# Calls whose target is an existing edge and whose change V5 can read.
EDGE_CALLS = ("drop_relation", "add_resolution", "break_chain", "remove_resolution")
# Untouched coupling-bearing edges sampled per call to guard against spill-over.
# Every untouched coupling-bearing edge is read, not a sample: pilot 4's 3-edge sample
# let a drop rewrite another rung's resolution sentence unnoticed. None = no cap.
N_UNTOUCHED: int | None = None
# P5 attempts per call before re-targeting, and re-targets per call before rejection.
P5_ATTEMPTS = 3
RETARGETS = 1
# A spurious link's pair is a free choice among the declared non-relations, so a hard
# pair should not cost the family: try up to two more.
SPURIOUS_RETARGETS = 2


@dataclass
class RaterPanel:
    """The V5 raters: `(name, caller)` pairs whose callers expose `ask` like `_Caller`.

    The generator must not be on the panel (R3); `pipeline.committee_for` already
    enforces that for the committee this is built from.
    """

    raters: list[tuple[str, Any]]
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    # Dry-run seam: `(expected_lo, expected_hi) -> reading` used INSTEAD of the raters.
    # Mock prose cannot be read, so an offline run is told what a faithful reader sees;
    # that exercises the gate's majority, order and retry logic. Never set live.
    oracle: Callable[[tuple, tuple], dict] | None = None
    # Set by `_v5` before each read so the oracle knows the gold states; ignored live.
    _expect: tuple | None = None

    def read(self, text_lo: str, text_hi: str, x: str, y: str) -> list[tuple[dict, bool]]:
        """One V5 reading per rater, each in an independently random text order."""
        if self.oracle is not None and self._expect is not None:
            return [(self.oracle(*self._expect), False) for _ in range(max(1, len(self.raters)))]
        out = []
        for _name, ac in self.raters:
            flipped = self.rng.random() < 0.5
            t1, t2 = (text_hi, text_lo) if flipped else (text_lo, text_hi)
            reading, err = ac.ask(
                "V5", parse.parse_realization, text_a=t1, text_b=t2, claim_x=x, claim_y=y
            )
            if reading is not None and err is None:
                out.append((reading, flipped))
        return out

    def resolution(self, text: str, x: str, y: str) -> list[dict[str, Any]]:
        """One V5r verdict per rater: does this ONE text already resolve X vs Y?"""
        if self.oracle is not None:
            return [{"resolved": False, "resolving_sentence": ""}] * max(1, len(self.raters))
        out = []
        for _name, ac in self.raters:
            v, err = ac.ask("V5r", parse.parse_resolution, text_a=text, claim_x=x, claim_y=y)
            if v is not None and err is None:
                out.append(v)
        return out

    def compatible(self, x: str, y: str) -> bool | None:
        """Majority content-only verdict (V5c), or None if no rater answered."""
        if self.oracle is not None:
            return True
        votes = []
        for _name, ac in self.raters:
            v, err = ac.ask("V5c", parse.parse_compatibility, claim_x=x, claim_y=y)
            if v is not None and err is None:
                votes.append(bool(v["compatible"]))
        if not votes:
            return None
        return sum(votes) * 2 > len(votes)


def _edge(relations: Sequence[Mapping[str, Any]], eid: str) -> dict[str, Any] | None:
    return next((dict(e) for e in relations if str(e.get("id")) == eid), None)


def _pair_state(
    relations: Sequence[Mapping[str, Any]], a: str, b: str
) -> dict[str, Any] | None:
    """The coupling-bearing edge over an unordered atom pair, if any."""
    for e in relations:
        if e.get("level1_coupling") in (None, "none"):
            continue
        if {str(e["source_id"]), str(e["target_id"])} == {a, b}:
            return dict(e)
    return None


def screen_targets(
    base_text: str,
    base_relations: Sequence[Mapping[str, Any]],
    atom_text: Mapping[str, str],
    panel: RaterPanel,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Edges each call may NOT target, with the evidence for each exclusion.

    Returns:
        ``(exclude, log)`` -- ``exclude`` feeds `perturb.plan_targets(exclude=...)`.
    """
    exclude: dict[str, set[str]] = {"drop_relation": set(), "add_resolution": set()}
    # Per-edge evidence, kept so an exclusion can be audited after the fact: the pilot
    # rejected f001 on two "already resolved" verdicts with no quote to check them by.
    log: dict[str, Any] = {"drop_incompatible": [], "already_resolved": [], "evidence": {}}
    for e in base_relations:
        if e.get("level1_coupling") not in ("contradiction", "exclusive"):
            continue
        eid = str(e["id"])
        x, y = atom_text[str(e["source_id"])], atom_text[str(e["target_id"])]
        ev: dict[str, Any] = {"x": x, "y": y}
        compat = panel.compatible(x, y)
        ev["compatible"] = compat
        if compat is False:
            exclude["drop_relation"].add(eid)
            log["drop_incompatible"].append(eid)
        if not e.get("is_resolved_concession") and e.get("validity") == "valid":
            votes = panel.resolution(base_text, x, y)
            resolved = [v["resolved"] for v in votes]
            ev["resolved_votes"] = resolved
            ev["resolved_evidence"] = [v["resolving_sentence"] for v in votes]
            if resolved and sum(resolved) * 2 > len(resolved):
                exclude["add_resolution"].add(eid)
                log["already_resolved"].append(eid)
        log["evidence"][eid] = ev
    return exclude, log


# Ladders whose rungs call `drop_relation`. CONFLICT left this set when it became
# resolution-only (2026-11-08); CHAIN's coherent rung still drops one conflict.
DROPPING_FAMILIES = ("CHAIN",)
# Ladders whose plans must leave VALID open conflicts for `add_resolution`.
RESOLVING_FAMILIES = ("CONFLICT",)
# Head-on pairs allowed among the conflicts a base response must leave open. Zero: in the
# 2026-11-09 run P4 adjudicated the one head-on reserved pair it was allowed (f008, "a
# single transition state" vs "at least one intermediate") on all 5 attempts, and a P3
# re-plan costs ~$0.05 against ~$1.50 for a family lost at P4.
MAX_HEAD_ON_RESERVED = 0


def plan_droppability(
    family: str,
    relations: Sequence[Mapping[str, Any]],
    atom_text: Mapping[str, str],
    panel: RaterPanel,
) -> tuple[str | None, dict[str, Any]]:
    """Plan-stage check: can this plan's drop rungs be realized at all?

    A drop keeps both atoms, so it can only remove a TEXT-drawn conflict. Pilot 2 found
    freshly generated plans full of HEAD-ON pairs ("no crater" vs "a crater") and both
    families died at `screen.no_target` after paying for P4. Checking at P3 lets the plan
    be retried with the offending pairs named, before any prose is written.

    Only the content screen (V5c) runs here: whether the TEXT resolves a conflict (V5r)
    cannot be asked before the text exists. Resolution targets are left to the
    post-response screen.

    Returns:
        ``(reason, log)`` -- ``reason`` is None when every drop rung has a target, else a
        retry note for P3.
    """
    from fact_reasoner.locobench.perturb import ladder_for, plan_targets

    log: dict[str, Any] = {"head_on": [], "checked": []}
    if family not in DROPPING_FAMILIES + RESOLVING_FAMILIES:
        return None, log
    if family not in DROPPING_FAMILIES:
        # Resolution-only: every resolution rung needs a target, and AT MOST ONE of the
        # pairs reserved for resolution may be head-on. P4 must leave each reserved pair
        # open, and the 2026-11-08 run showed it adjudicates flat contradictions ("the
        # vendor blamed an insider" vs "the victim blamed an external exploit") even when
        # told not to. Text-drawn tensions it can leave standing.
        targets = plan_targets(family, list(relations))
        calls = {r.index: r.calls for r in ladder_for(family).rungs if not r.is_base}
        log["targets"] = {str(k): v for k, v in targets.items()}
        missing = unfillable_calls(calls, targets)
        if not missing:
            reserved = keep_open_targets(family, log["targets"])
            by_id = {str(e["id"]): e for e in relations}
            head = []
            for eid in sorted(reserved):
                e = by_id[eid]
                log["checked"].append(eid)
                x, y = atom_text[str(e["source_id"])], atom_text[str(e["target_id"])]
                if panel.compatible(x, y) is False:
                    head.append((eid, x, y))
                    log["head_on"].append(eid)
            if len(head) <= MAX_HEAD_ON_RESERVED:
                return None, log
            return (
                f"{len(head)} of the {len(reserved)} conflicting pairs the answer must leave "
                "open are HEAD-ON contradictions, which a written answer cannot leave "
                "unadjudicated: " + "; ".join(f'"{x}" vs "{y}"' for _, x, y in head[:3])
                + (". None of the valid unresolved conflicts may be head-on"
                   if MAX_HEAD_ON_RESERVED == 0 else
                   f". At most {MAX_HEAD_ON_RESERVED} of the valid unresolved conflicts may be head-on")
                + "; make them TEXT-DRAWN tensions - two claims that could both be true "
                "taken alone, which the answer sets against each other.",
                log,
            )
        return (
            f"the ladder settles two conflicts in turn but the plan leaves only "
            f"{len([t for t in targets.get(4, []) if t])} VALID and UNRESOLVED conflicting "
            "relation(s) to settle; make at least two conflicting relations valid and "
            "unresolved.",
            log,
        )
    head_on: set[str] = set()
    for e in relations:
        if e.get("level1_coupling") not in ("contradiction", "exclusive"):
            continue
        eid = str(e["id"])
        log["checked"].append(eid)
        if panel.compatible(atom_text[str(e["source_id"])], atom_text[str(e["target_id"])]) is False:
            head_on.add(eid)
            log["head_on"].append(eid)
    targets = plan_targets(
        family, list(relations), exclude={"drop_relation": head_on}, prefer_resolve=head_on
    )
    calls = {r.index: r.calls for r in ladder_for(family).rungs if not r.is_base}
    # Every edge-targeting call, not only drops: pilot 3's f001 plan had three droppable
    # conflicts and no VALID open one to resolve, and died at `screen.no_target` after P4.
    missing = unfillable_calls(calls, targets)
    log["targets"] = {str(k): v for k, v in targets.items()}
    if not missing:
        return None, log
    if all(m.endswith(":add_resolution") for m in missing):
        return (
            "the plan has no conflicting relation that is VALID and UNRESOLVED to be "
            "settled later (planted-invalid and already-resolved conflicts cannot be). "
            "Make at least one conflicting relation valid and unresolved, in addition to "
            "the text-drawn conflicts.",
            log,
        )
    pairs = []
    for e in relations:
        if str(e["id"]) in head_on:
            pairs.append(
                f'"{atom_text[str(e["source_id"])]}" vs "{atom_text[str(e["target_id"])]}"'
            )
    return (
        f"{len(head_on)} of the plan's {len(log['checked'])} conflicting pairs are HEAD-ON "
        f"contradictions that no edit can drop while keeping both claims: "
        + "; ".join(pairs[:4])
        + ". Replace them with TEXT-DRAWN conflicts: two claims that could both be true "
        "taken alone, which the response sets against each other.",
        log,
    )


def keep_open_targets(family: str, targets: Mapping[str, Sequence[str]]) -> set[str]:
    """Edge ids the base response must leave unresolved: the ladder's resolution targets."""
    from fact_reasoner.locobench.perturb import ladder_for

    out: set[str] = set()
    for r in ladder_for(family).rungs:
        ts = list(targets.get(str(r.index), targets.get(r.index, [])) or [])
        for i, c in enumerate(r.calls):
            if c == "add_resolution" and i < len(ts) and ts[i]:
                out.add(ts[i])
    return out


def unfillable_calls(calls_by_rung: Mapping[int, Sequence[str]],
                     targets: Mapping[int, Sequence[str]]) -> list[str]:
    """Edge-targeting calls left with no eligible edge, as ``"rung<i>:<call>"``.

    Checked before any P5 call: a call with no target cannot be realized, and running it
    anyway spent 4 P5 calls and 12 minutes in the pilot before failing under a misleading
    reason ("changed no coupling-bearing label").
    """
    out = []
    for idx, calls in sorted(calls_by_rung.items()):
        ts = list(targets.get(idx, []))
        for i, c in enumerate(calls):
            if c in EDGE_CALLS and not (ts[i] if i < len(ts) else ""):
                out.append(f"rung{idx}:{c}")
    return out


def _untouched_sample(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    skip: set[tuple[str, str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Up to N coupling-bearing edges present, and identical, before and after."""
    cands = []
    for e in before:
        if e.get("level1_coupling") in (None, "none"):
            continue
        key = tuple(sorted((str(e["source_id"]), str(e["target_id"]))))
        if key in skip:
            continue
        twin = _pair_state(after, *key)
        if twin and validate.expected_reading(twin) == validate.expected_reading(dict(e)):
            cands.append(dict(e))
    rng.shuffle(cands)
    return cands if N_UNTOUCHED is None else cands[:N_UNTOUCHED]


@dataclass
class CallOutcome:
    text: str
    relations: list[dict[str, Any]]
    non_relations: list[dict[str, Any]]
    log: list[dict[str, Any]]
    gates: list[validate.GateResult]
    attempts: int


def perturb_call(
    caller: Any,
    panel: RaterPanel | None,
    *,
    call: str,
    target: str,
    operator: str,
    parent_text: str,
    parent_relations: list[dict[str, Any]],
    parent_non_relations: list[dict[str, Any]],
    plan_json: str,
    atom_text: Mapping[str, str],
    generator: str,
    length_base: str,
) -> tuple[CallOutcome | None, str | None, list[validate.GateResult]]:
    """One gated P5 call: span check, then V5, retried with the reason fed back.

    Labels for the call come from `apply_calls` on the PARENT'S labels, so the gate
    compares exactly the before/after states the shipped item will carry.

    Returns:
        ``(outcome, None, gates)`` on success or ``(None, reason, gates)`` after
        :data:`P5_ATTEMPTS` failures.
    """
    rels, nons, log = apply_calls(
        parent_relations,
        (call,),
        targets=[target],
        non_relations=parent_non_relations,
        generator=generator,
    )
    gates: list[validate.GateResult] = []
    reason: str | None = None
    for attempt in range(P5_ATTEMPTS):
        op = operator if reason is None else f"{operator}\nPREVIOUS ATTEMPT REJECTED: {reason}"
        out, err = caller.ask(
            "P5", parse.parse_perturbation, response=parent_text, plan=plan_json, operator=op
        )
        if not out:
            reason = err or "no output"
            continue
        text, diff = out
        g = validate.gate_text_changed(parent_text, text, operator=operator)
        gates.append(g)
        if not g.passed:
            reason = g.detail
            continue
        g = validate.gate_length_drift(length_base, text, operator=call)
        gates.append(g)
        if not g.passed:
            reason = g.detail
            continue
        g = validate.gate_spans(parent_text, text, diff, operator=call)
        gates.append(g)
        if not g.passed:
            reason = g.detail
            continue
        if panel is not None:
            g = _v5(panel, parent_text, text, parent_relations, rels, log, atom_text, call)
            gates.append(g)
            if not g.passed:
                reason = g.detail
                continue
        return CallOutcome(text, rels, nons, log, gates, attempt + 1), None, gates
    return None, reason, gates


def _changed_pairs(log: Sequence[Mapping[str, Any]], before, after) -> list[tuple[str, str]]:
    """Unordered atom pairs whose coupling-bearing state the call changed."""
    pairs = set()
    for e in list(before) + list(after):
        if e.get("level1_coupling") in (None, "none"):
            continue
        pairs.add(tuple(sorted((str(e["source_id"]), str(e["target_id"])))))
    return sorted(
        p for p in pairs
        if validate.expected_reading(_pair_state(before, *p))
        != validate.expected_reading(_pair_state(after, *p))
    )


def _v5(panel, text_lo, text_hi, rel_lo, rel_hi, log, atom_text, call) -> validate.GateResult:
    changed = _changed_pairs(log, rel_lo, rel_hi)
    targeted = []
    for a, b in changed:
        lo = validate.expected_reading(_pair_state(rel_lo, a, b))
        hi = validate.expected_reading(_pair_state(rel_hi, a, b))
        panel._expect = (lo, hi)
        targeted.append(
            {
                "edge": f"{a}-{b}",
                "lower": lo,
                "upper": hi,
                "readings": panel.read(text_lo, text_hi, atom_text[a], atom_text[b]),
            }
        )
    untouched = []
    for e in _untouched_sample(rel_lo, rel_hi, set(changed), panel.rng):
        a, b = str(e["source_id"]), str(e["target_id"])
        r = validate.expected_reading(e)
        panel._expect = (r, r)
        untouched.append(
            {
                "edge": f"{a}-{b}",
                "lower": r,
                "upper": r,
                "readings": panel.read(text_lo, text_hi, atom_text[a], atom_text[b]),
            }
        )
    panel._expect = None
    if not targeted and call in EDGE_CALLS + ("spurious_relation",):
        return validate.GateResult(
            f"V5.{call}", False, detail=f"{call} changed no coupling-bearing label"
        )
    return validate.gate_realization(targeted, untouched, rung=call)


@dataclass
class RungOutcome:
    text: str
    relations: list[dict[str, Any]]
    non_relations: list[dict[str, Any]]
    log: list[dict[str, Any]]
    targets: list[str]
    gates: list[validate.GateResult]
    p5_attempts: int


def realize_rung(
    caller: Any,
    panel: RaterPanel | None,
    *,
    calls: Sequence[str],
    targets: Sequence[str],
    alternates: Mapping[str, Sequence[str]],
    operator_for: Callable[[str, str], str],
    parent_text: str,
    parent_relations: list[dict[str, Any]],
    parent_non_relations: list[dict[str, Any]],
    plan_json: str,
    atom_text: Mapping[str, str],
    generator: str,
    already_used: Sequence[str] = (),
) -> tuple[RungOutcome | None, str | None, list[validate.GateResult]]:
    """Apply a rung's calls in order, each gated, with one re-target per call.

    Args:
        alternates: ``{call: [edge ids in preference order]}`` -- the eligible edges the
            call may fall back to when its planned target cannot be realized. Edges
            already used in this rung are skipped.
        operator_for: ``(call, target) -> "call(target)"`` rendering for P5.

    Returns:
        ``(outcome, None, gates)``, or ``(None, reason, gates)`` when some call could not
        be realized on any eligible target.
    """
    text = parent_text
    rels = [dict(r) for r in parent_relations]
    nons = [dict(n) for n in parent_non_relations]
    logs: list[dict[str, Any]] = []
    used: list[str] = []
    # Targets an incremental rung's PARENT already consumed: a fallback must never reach
    # them (re-resolving a resolved edge changes no label and wastes the rung -- seen on
    # f013, where rung 4's fallback picked rung 3's r005).
    taken = {t for t in already_used if t}
    all_gates: list[validate.GateResult] = []
    p5 = 0
    for i, call in enumerate(calls):
        planned = targets[i] if i < len(targets) else ""
        budget = SPURIOUS_RETARGETS if call == "spurious_relation" else RETARGETS
        tries = [planned] + [
            t for t in alternates.get(call, [])
            if t and t != planned and t not in used and t not in taken
        ][:budget]
        outcome, reason = None, None
        for tgt in tries:
            if tgt in used and tgt:
                continue
            outcome, reason, gates = perturb_call(
                caller,
                panel,
                call=call,
                target=tgt,
                operator=operator_for(call, tgt),
                parent_text=text,
                parent_relations=rels,
                parent_non_relations=nons,
                plan_json=plan_json,
                atom_text=atom_text,
                generator=generator,
                length_base=parent_text,
            )
            all_gates.extend(gates)
            p5 += outcome.attempts if outcome is not None else P5_ATTEMPTS
            if outcome is not None:
                used.append(tgt)
                break
        if outcome is None:
            return None, f"{call}({planned}): {reason}", all_gates
        text, rels, nons = outcome.text, outcome.relations, outcome.non_relations
        logs.extend(outcome.log)
    return RungOutcome(text, rels, nons, logs, used, all_gates, p5), None, all_gates


__all__ = [
    "EDGE_CALLS",
    "N_UNTOUCHED",
    "P5_ATTEMPTS",
    "RETARGETS",
    "CallOutcome",
    "RaterPanel",
    "RungOutcome",
    "perturb_call",
    "realize_rung",
    "screen_targets",
    "unfillable_calls",
    "plan_droppability",
    "keep_open_targets",
]
