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

"""The slot3 stage machine: plan -> template -> fillers -> splice -> admit (per ladder).

Stages and their failure policy:

* **P1, P2** -- unchanged from v1.
* **S3 plan** -- retried with the deterministic gate's complaint (`slots.check_plan`),
  which includes the exact gold-ladder check, so a plan that cannot carry its ladders
  never reaches prose.
* **S4 template** -- retried with the committee's complaint. V3 (fluency, majority), V4
  (coverage, majority per atom) and SV1 (pair-by-pair verification of the fixed relations,
  blind to the plan) run on the BASE splice inside the retry loop. An unconfirmed fixed
  relation is not a rejection: gold follows the committee's majority reading of that pair
  (dropped, or replaced by what the text draws), and the ladder check is re-run on it.
* **SV5 fillers** -- each filler is read in context by every rater, blind to its intended
  link. A failed filler is REGENERATED alone (SF) with the readings as feedback, then a
  spare slot stands in. Nothing here regenerates the family.
* **ORDER** -- rungs are paragraph permutations of the base splice, each re-verified with
  SV1; a permutation that loses a relation is replaced by the next candidate.
* **admit** -- per LADDER. A family is stored when at least one of its ladders survives.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from fact_reasoner.locobench import parse, prompts, slots as S, validate
from fact_reasoner.locobench.config import GenConfig
from fact_reasoner.locobench.pipeline import LLM, _Caller
from fact_reasoner.locobench.topics import domain_of
from fact_reasoner.locobench.validate import GateResult, Verdict

# SV1: the fraction of fixed relations the committee must confirm for the template to
# stand. Below it the prose does not realize the plan, and correcting gold pair by pair
# would describe a different answer than the one planned.
SV1_MIN_CONFIRMED = 0.75
# Words in the base splice; the prompt asks for 420-520.
WORDS = (380, 650)
MIN_PARAGRAPHS = {"ORDER": 4}
# Extra attempts per failing filler (SF), before a spare slot is tried.
FILLER_ATTEMPTS = 2
# Paragraph-order candidates tried per ORDER rung.
ORDER_CANDIDATES = 4

GLOSS = {
    "Cause-Effect": "the source brings the target about ('led to', 'caused', 'as a result')",
    "Evidence": "the source is evidence that shows the target ('shows', 'indicates', "
    "'confirms')",
    "Instantiation": "the target is a specific instance of the general source "
    "('for example', 'in one case')",
}


@dataclass
class Slot3Result:
    """The outcome of one family."""

    family_id: str
    canonical_topic: str
    primary: str
    items: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    verdict: Verdict = field(default_factory=Verdict)
    stage: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    # Every gate evaluation, in order, including the ones a retry repaired: the pilot's
    # per-gate pass rates are computed from this, not from the final verdict.
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return bool(self.manifest and self.manifest.get("ladders"))


def describe_filler(slot: dict[str, Any], kind: str) -> str:
    """What one filler kind of a slot must do, in words, for S4 and SF."""
    role = slot["role"]
    sense = S.slot_links(slot)[kind]
    if kind == "none":
        extra = " (both in the same confident register)" if role == "sign_tf" else ""
        return ("state the owned atom(s) with NO link between the source and the target"
                + extra)
    if role == "conflict_tt":
        return ("say that the source and the target CANNOT BOTH be true ('cannot be "
                "squared with', 'rules out'), asserting both")
    if role == "chain_link":
        return f"draw the link: {GLOSS[sense]}"
    if role == "sign_tf" and kind == "conflict":
        return ("present the target as following from the source, confidently, as the "
                f"answer's own inference: {GLOSS[sense]}; assert the target")
    if role == "sign_tf" and kind == "support":
        return ("use the source to REFUTE the target: given the source, the target cannot "
                "be true. Mention the target only to reject it; do not assert it")
    if role == "control":
        return (f"draw the link {GLOSS[sense]}; para1, para2 and para3 must word it in "
                "clearly different ways (connective, clause order)")
    raise ValueError(f"no description for {role}/{kind}")


def _plan_for_template(plan: dict[str, Any], base: dict[str, str]) -> dict[str, Any]:
    owns = S.slot_owns(plan)
    slot_of = {p: sid for sid, ps in owns.items() for p in ps}
    text = {a["pos"]: a["text"] for a in plan["atoms"]}
    return {
        "atoms": [
            {"pos": a["pos"], "text": a["text"], "factual": a["factual"],
             "slot": slot_of.get(a["pos"])}
            for a in sorted(plan["atoms"], key=lambda a: a["pos"])
        ],
        "relations": [
            {"source_pos": r["source_pos"], "target_pos": r["target_pos"], "sense": r["sense"]}
            for r in plan["relations"]
        ],
        "non_relations": plan["non_relations"],
        "slots": [
            {
                "id": s["id"],
                "source_pos": s["source_pos"],
                "target_pos": s["target_pos"],
                "source": text[s["source_pos"]],
                "target": text[s["target_pos"]],
                "owns": owns[s["id"]],
                "fillers": {k: describe_filler(s, k) for k in S.ROLES[s["role"]].kinds},
                "base": base[s["id"]],
            }
            for s in sorted(plan["slots"], key=lambda s: s["source_pos"])
        ],
    }


def _atoms_arg(plan: dict[str, Any]) -> str:
    atoms = sorted(plan["atoms"], key=lambda a: a["pos"])
    return json.dumps({str(a["pos"]): a["text"] for a in atoms})


def _flip_label(label: str) -> str:
    return {"i_supports_j": "j_supports_i", "j_supports_i": "i_supports_j",
            "precedes": "follows", "follows": "precedes"}.get(label, label)


class _Panel:
    """The committee: `(name, _Caller)` pairs, each asked in turn."""

    def __init__(self, raters: list[tuple[str, _Caller]], rng: random.Random):
        self.raters = raters
        self.rng = rng

    def ask_all(self, pid: str, parser, **values: str) -> list[tuple[str, Any]]:
        out = []
        for name, ac in self.raters:
            v, err = ac.ask(pid, parser, **values)
            if v is not None and err is None:
                out.append((name, v))
        return out

    @property
    def counts(self) -> dict[str, int]:
        agg: dict[str, int] = {}
        for _n, ac in self.raters:
            for k, v in ac.counts.items():
                agg[k] = agg.get(k, 0) + v
        return agg


def verify_pairs(panel: _Panel, text: str, plan: dict[str, Any],
                 pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """SV1 over listed pairs; the majority reading of each pair.

    Args:
        pairs: ``{"source_pos", "target_pos", "expected"}``; ``expected`` is the label
            the plan predicts with i = source.

    Returns:
        ``{"readings": [{..., "majority", "labels", "confirmed"}], "n_raters": k}``.
    """
    order = list(range(len(pairs)))
    panel.rng.shuffle(order)
    listed = [[pairs[i]["source_pos"], pairs[i]["target_pos"]] for i in order]
    got = panel.ask_all("SV1", parse.parse_link_list, response=text, atoms=_atoms_arg(plan),
                        pairs=json.dumps(listed))
    out = []
    for p in pairs:
        labels = []
        for _name, entries in got:
            lab = None
            for e in entries:
                if e["pair"] == [p["source_pos"], p["target_pos"]]:
                    lab = e["label"]
                elif e["pair"] == [p["target_pos"], p["source_pos"]]:
                    lab = _flip_label(e["label"])
                if lab:
                    break
            if lab:
                labels.append(lab)
        maj = S.majority(labels)
        out.append({**p, "labels": labels, "majority": maj,
                    "confirmed": maj is not None and S.label_matches(p["expected"], maj)})
    return {"readings": out, "n_raters": len(got)}


def corrected_fixed(plan: dict[str, Any], sv1: dict[str, Any]) -> tuple[list, list, list]:
    """Gold for the fixed part, following the committee where it disagrees with the plan.

    Returns:
        ``(relations, non_relations, corrections)`` in plan form.
    """
    rels, nons, log = [], [], []
    by_pair = {(r["source_pos"], r["target_pos"]): r for r in sv1["readings"]}
    for r in plan["relations"]:
        rd = by_pair.get((r["source_pos"], r["target_pos"]))
        if rd is None or rd["confirmed"]:
            rels.append({**r, "provenance": {"source": "plan", "confirmed": True}})
            continue
        maj = rd["majority"]
        if maj is not None and S.LABEL_COUPLING[maj] != "none":
            s_, t_ = r["source_pos"], r["target_pos"]
            if maj == "j_supports_i":
                s_, t_ = t_, s_
            new = {"source_pos": s_, "target_pos": t_, "sense": S.LABEL_SENSE[maj],
                   "strength_band": r.get("strength_band", "moderate"),
                   "provenance": {"source": "committee", "planned_sense": r["sense"]}}
            rels.append(new)
            log.append({"pair": [r["source_pos"], r["target_pos"]], "planned": r["sense"],
                        "action": "replaced", "reading": maj})
        else:
            log.append({"pair": [r["source_pos"], r["target_pos"]], "planned": r["sense"],
                        "action": "dropped", "reading": maj, "labels": rd["labels"]})
    for nr in plan["non_relations"]:
        rd = by_pair.get((nr["source_pos"], nr["target_pos"]))
        maj = rd["majority"] if rd else None
        if maj is not None and S.LABEL_COUPLING[maj] != "none":
            s_, t_ = nr["source_pos"], nr["target_pos"]
            if maj == "j_supports_i":
                s_, t_ = t_, s_
            rels.append({"source_pos": s_, "target_pos": t_, "sense": S.LABEL_SENSE[maj],
                         "strength_band": "moderate",
                         "provenance": {"source": "committee", "planned_sense": None}})
            log.append({"pair": [nr["source_pos"], nr["target_pos"]], "planned": None,
                        "action": "added", "reading": maj})
        else:
            nons.append(dict(nr))
    return rels, nons, log


def _sv1_pairs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = [{"source_pos": r["source_pos"], "target_pos": r["target_pos"],
              "kind": "relation", "expected": S.expected_label(r["sense"])}
             for r in plan["relations"]]
    pairs += [{"source_pos": n["source_pos"], "target_pos": n["target_pos"],
               "kind": "non_relation", "expected": "unlinked"}
              for n in plan["non_relations"]]
    return pairs


def judge_filler(readings: list[tuple[str, dict, bool]], slot: dict[str, Any], kind: str,
                 owns: list[int]) -> tuple[bool, dict[str, Any]]:
    """Majority verdict on one filler's SV5 readings.

    Args:
        readings: ``(rater, reading, flipped)``; ``flipped`` means X was the target.
    """
    expected = S.expected_label(S.slot_links(slot)[kind])
    want_status = {slot["source_pos"]: "asserted", slot["target_pos"]: "asserted"}
    if slot["role"] == "sign_tf" and kind == "support":
        want_status[slot["target_pos"]] = "denied"
    oks, others, detail = [], [], []
    for name, rd, flipped in readings:
        link = _flip_label(rd["link"]) if flipped else rd["link"]
        st = {slot["source_pos"]: rd["y_status" if flipped else "x_status"],
              slot["target_pos"]: rd["x_status" if flipped else "y_status"]}
        link_ok = S.label_matches(expected, link)
        status_ok = all(st[p] == want_status[p] for p in owns)
        oks.append(link_ok and status_ok)
        others.append(bool(rd["other_links"]))
        detail.append({"rater": name, "link": link, "status": st, "link_ok": link_ok,
                       "status_ok": status_ok, "other_links": rd["other_links"][:2],
                       "evidence": rd["evidence"][:200]})
    n = len(readings)
    passed = n >= 2 and sum(oks) * 2 > n and not (sum(others) * 2 > n)
    return passed, {"expected": expected, "readings": detail, "n": n}


def generate_family_slot3(
    family_id: str,
    canonical_topic: str,
    primary: str,
    cfg: GenConfig,
    *,
    llm: LLM,
    auditor_llms: list[tuple[str, LLM]],
    generator: str,
    rare: list[str],
    resume_from: dict[str, Any] | None = None,
    split: str = "pilot",
) -> Slot3Result:
    """Run one family through the slot3 stage machine.

    Args:
        family_id: The stable id, e.g. ``f003``.
        canonical_topic: One of the 36 topics.
        primary: The primary ladder type (CONFLICT, CHAIN, SIGN, ORDER); every family also
            carries a CONTROL ladder.
        cfg: The run config.
        llm: The generator.
        auditor_llms: The committee, generator excluded.
        generator: The generator's name, for provenance.
        rare: The rare senses this family's plan must include.
        resume_from: ``question`` / ``claims`` / ``plan`` from an earlier attempt.
        split: Recorded on every item.
    """
    if primary not in S.PRIMARY_TYPES:
        raise ValueError(f"primary must be one of {S.PRIMARY_TYPES}, got {primary!r}")
    res = Slot3Result(family_id, canonical_topic, primary)
    caller = _Caller(llm, attempts=cfg.max_attempts)
    panel = _Panel([(n, _Caller(f, attempts=cfg.max_attempts)) for n, f in auditor_llms],
                   random.Random(f"{cfg.seed}-{family_id}"))
    prior = dict(resume_from or {})
    t0 = time.perf_counter()

    def event(gate: str, passed: bool, **obs: Any) -> None:
        res.events.append({"gate": gate, "passed": passed, **obs})

    def finish(stage: str) -> Slot3Result:
        res.stage = stage
        res.calls = {**caller.counts, **{f"{k}@committee": v for k, v in panel.counts.items()}}
        res.timing["total"] = round(time.perf_counter() - t0, 1)
        return res

    # ---- P1, P2 -------------------------------------------------------------
    question = prior.get("question")
    if not question:
        question, err = caller.ask("P1", parse.parse_question, topic=canonical_topic)
        event("P1", question is not None and err is None, detail=err)
        if not question:
            res.verdict.add(GateResult("P1", False, detail=err or "no question"))
            return finish("plan")
    claims = prior.get("claims")
    if not claims:
        claims, err = caller.ask("P2", parse.parse_claims, question=question)
        event("P2", claims is not None and err is None, detail=err)
        if not claims:
            res.verdict.add(GateResult("P2", False, detail=err or "no claims"))
            res.artifacts = {"question": question}
            return finish("plan")
    res.artifacts = {"question": question, "claims": claims}

    # ---- S3 plan ------------------------------------------------------------
    plan = prior.get("plan")
    plan_report: dict[str, Any] = {}
    if not plan:
        def _plan_ok(cand: dict[str, Any]) -> str | None:
            reason, rep = S.check_plan(cand, claims, primary, rare)
            event("S3.gate", reason is None, detail=reason)
            if reason is None:
                plan_report.update(rep)
            return reason

        plan, err = caller.ask(
            "S3", parse.parse_slot_plan, check=_plan_ok, question=question,
            claims="\n".join(f"- {c['text']} [{c['tag']}]" for c in claims),
            slots=prompts.SLOT_REQUIREMENTS[primary], senses=", ".join(rare),
        )
        if err is not None:
            res.verdict.add(GateResult("S3", False, detail=err))
            if plan:
                res.artifacts["rejected_plan"] = plan
            return finish("plan")
    else:
        reason, plan_report = S.check_plan(plan, claims, primary, rare)
        if reason:
            res.verdict.add(GateResult("S3", False, detail=reason))
            return finish("plan")
    res.artifacts["plan"] = plan
    res.verdict.add(GateResult("S3", True, observed=plan_report.get("ladders")))
    res.timing["plan"] = round(time.perf_counter() - t0, 1)

    # ---- S4 template + base gates --------------------------------------------
    casting0 = S.default_casting(plan, primary)
    base0 = S.assignments(plan, primary, casting0)["base"]
    n_atoms = len(plan["atoms"])
    false_idx = {a["pos"] for a in plan["atoms"] if not a["factual"]}
    held: dict[str, Any] = {}

    def _base_gates(text: str, extra: list[str]) -> Verdict:
        v = Verdict()
        # V3 on the base splice AND on the coverage splices, so every filler that some rung
        # carries has been read by the auditors (the base alone left rung 0 of CONFLICT,
        # with both conflicts, unaudited on the first live run).
        for k, t in enumerate([text] + extra):
            audits = panel.ask_all("V3", parse.parse_audit, response=t)
            if not audits:
                return v.add(GateResult("V3", False, detail="no audit output"))
            got = validate.gate_audit_panel(audits, response=t)
            if k:
                for g in got.results:
                    g.gate = f"{g.gate}@cover"
            v.results.extend(got.results)
            if not got.passed:
                return v
        covs = panel.ask_all("V4", parse.parse_coverage, response=text,
                             atoms=_atoms_arg(plan))
        if not covs:
            return v.add(GateResult("V4", False, detail="no coverage output"))
        v.results.extend(validate.gate_coverage_panel(covs, n_atoms,
                                                      false_indices=false_idx).results)
        if not v.passed:
            return v
        sv1 = verify_pairs(panel, text, plan, _sv1_pairs(plan))
        held["sv1"] = sv1
        rel_rd = [r for r in sv1["readings"] if r["kind"] == "relation"]
        frac = sum(r["confirmed"] for r in rel_rd) / max(1, len(rel_rd))
        bad = [f"{r['source_pos']}->{r['target_pos']} planned {r['expected']}, read "
               f"{r['majority'] or 'no majority'} {r['labels']}"
               for r in rel_rd if not r["confirmed"]]
        v.add(GateResult("SV1", frac >= SV1_MIN_CONFIRMED and sv1["n_raters"] >= 2,
                         threshold=SV1_MIN_CONFIRMED,
                         observed={"confirmed": round(frac, 3), "n_raters": sv1["n_raters"]},
                         detail="" if frac >= SV1_MIN_CONFIRMED else
                         "these relations are not drawn as planned: " + "; ".join(bad)))
        return v

    def _template_ok(obj: dict[str, Any]) -> str | None:
        reason = S.check_template(obj["template"], obj["fillers"], plan)
        if reason is None:
            text = S.splice(obj["template"], obj["fillers"], base0)
            wc = S.word_count(text)
            npar = len(S.paragraphs(text))
            if not WORDS[0] <= wc <= WORDS[1]:
                reason = (f"the answer is {wc} words with its base fillers; write "
                          "420-520")
            elif npar < MIN_PARAGRAPHS.get(primary, 3):
                reason = (f"the answer has {npar} paragraph(s); write at least "
                          f"{MIN_PARAGRAPHS.get(primary, 3)}")
        if reason is not None:
            event("S4.structure", False, detail=reason)
            return reason
        event("S4.structure", True)
        extra = [S.splice(obj["template"], obj["fillers"], a)
                 for a in S.coverage_assignments(plan, base0)]
        v = _base_gates(text, extra)
        held["verdict"] = v
        for g in v.results:
            event(g.gate, g.passed, detail=g.detail if not g.passed else "",
                  observed=g.observed if g.gate in ("SV1", "V3.scores") else None)
        return None if v.passed else v.reason()

    t1 = time.perf_counter()
    tpl, err = caller.ask("S4", parse.parse_template, check=_template_ok, question=question,
                          plan=json.dumps(_plan_for_template(plan, base0)),
                          layout=prompts.LAYOUT_REQUIREMENTS[primary])
    if err is not None:
        if held.get("verdict") is not None:
            res.verdict.results.extend(held["verdict"].results)
        res.verdict.add(GateResult("S4", False, detail=err))
        if tpl:
            res.artifacts["rejected_template"] = tpl
        elif caller.last_raw.get("S4"):
            # The parser rejected every attempt: keep what the model actually wrote, or the
            # failure ("SamplingFailed") names no text and cannot be diagnosed.
            res.artifacts["rejected_template_raw"] = caller.last_raw["S4"][:20000]
        return finish("template")
    res.verdict.results.extend(held["verdict"].results)
    template, fillers = tpl["template"], {k: dict(v) for k, v in tpl["fillers"].items()}
    fixed, nons, corrections = corrected_fixed(plan, held["sv1"])
    res.artifacts.update({"template": template, "fillers_initial": tpl["fillers"],
                          "sv1": held["sv1"], "gold_corrections": corrections})
    res.timing["template"] = round(time.perf_counter() - t1, 1)

    # ---- SV5 fillers ----------------------------------------------------------
    t2 = time.perf_counter()
    owns = S.slot_owns(plan)
    by_id = {s["id"]: s for s in plan["slots"]}
    atom_text = {a["pos"]: a["text"] for a in plan["atoms"]}
    status: dict[str, dict[str, bool]] = {sid: {} for sid in by_id}
    filler_log: list[dict[str, Any]] = []

    def check(sid: str, kind: str) -> bool:
        if kind in status[sid]:
            return status[sid][kind]
        slot = by_id[sid]
        x, y = atom_text[slot["source_pos"]], atom_text[slot["target_pos"]]
        for attempt in range(1 + FILLER_ATTEMPTS):
            asg = dict(base0)
            asg[sid] = kind
            text = S.splice(template, fillers, asg)
            passage = fillers[sid][kind]
            readings = []
            for name, ac in panel.raters:
                flipped = panel.rng.random() < 0.5
                cx, cy = (y, x) if flipped else (x, y)
                rd, e = ac.ask("SV5", parse.parse_link_reading, text=text, passage=passage,
                               claim_x=cx, claim_y=cy)
                if rd is not None and e is None:
                    readings.append((name, rd, flipped))
            ok, det = judge_filler(readings, slot, kind, owns[sid])
            filler_log.append({"slot": sid, "kind": kind, "attempt": attempt, "passed": ok,
                               "filler": passage, **det})
            event("SV5", ok, slot=sid, kind=kind, attempt=attempt, role=slot["role"])
            if ok:
                status[sid][kind] = True
                return True
            if attempt == FILLER_ATTEMPTS:
                break
            ctx = S.splice(template, fillers, base0)
            ctx = ctx.replace(fillers[sid][base0[sid]].strip(), "[[HERE]]", 1)
            new, e = caller.ask(
                "SF", parse.parse_filler, context=ctx,
                slot=json.dumps({"source": x, "target": y,
                                 "owns": [atom_text[p] for p in owns[sid]],
                                 "link": describe_filler(slot, kind)}),
                siblings=json.dumps({k: v for k, v in fillers[sid].items() if k != kind}),
                failed=passage,
                readings=json.dumps([{"read_as": d["link"], "claims": d["status"],
                                      "evidence": d["evidence"],
                                      "other_links": d["other_links"]}
                                     for d in det["readings"]]),
            )
            event("SF", new is not None and e is None, slot=sid, kind=kind)
            if not new or S.MARKER_RE.search(new) or "\n\n" in new or S.META_RE.search(
                new
            ) or S.restated(new, [x, y]) or (kind == "none" and S.ANNOUNCING_RE.search(new)):
                break
            fillers[sid][kind] = new
        status[sid][kind] = False
        return False

    spec = S.LADDERS[primary]
    # Designated slots and the CONTROL slot need every kind; the rest need one kind.
    for sid in list(casting0.primary) + ([casting0.control] if casting0.control else []):
        role = S.ROLES[by_id[sid]["role"]]
        for kind in role.kinds:
            if not check(sid, kind):
                # This slot cannot play its role; do not spend calls on its other kinds,
                # except the one it would rest at, since it stays in the text regardless.
                if not status[sid].get(role.rest):
                    for k in [role.rest] + [k for k in role.kinds if k != role.rest]:
                        if k not in status[sid] and check(sid, k):
                            break
                break
    for sid, kind in casting0.rest.items():
        role = S.ROLES[by_id[sid]["role"]]
        for k in [kind] + [k for k in role.kinds if k != kind]:
            if check(sid, k):
                break

    def passing() -> dict[str, set[str]]:
        return {sid: {k for k, ok in st.items() if ok} for sid, st in status.items()}

    casting = S.default_casting(plan, primary, passing(), allow_partial=True)
    if casting is not None and len(casting.primary) < len(spec.roles):
        # A designated slot failed: validate the spares' remaining kinds and recast.
        for sid, s in by_id.items():
            if s["role"] in spec.roles and sid not in casting0.primary:
                for kind in S.ROLES[s["role"]].kinds:
                    if not check(sid, kind):
                        break
        casting = S.default_casting(plan, primary, passing(), allow_partial=True)
    res.artifacts.update({"fillers": fillers, "filler_log": filler_log})
    res.timing["fillers"] = round(time.perf_counter() - t2, 1)
    if casting is None:
        res.verdict.add(GateResult("SV5", False,
                                   detail="a slot has no filler the committee reads as "
                                          "intended, so no rung can be built"))
        return finish("fillers")

    # ---- ladders ---------------------------------------------------------------
    asg = S.assignments(plan, primary, casting)
    base = asg["base"]
    base_text = S.splice(template, fillers, base)
    reason, gold_rep = S.ladder_gold_check(plan, primary, casting, fixed)
    gold_reasons = {}
    if reason:
        # Attribute each reason to its ladder (the report keys mirror the reasons' prefix).
        for part in reason.split("; "):
            gold_reasons[part.split(" ", 1)[0]] = part
    ladders: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}

    # Primary.
    prim_texts: list[tuple[dict[str, str], list[int] | None]] | None = None
    if primary == "ORDER":
        pars = S.paragraphs(base_text)
        chosen: list[list[int]] = []
        base_conf = {(r["source_pos"], r["target_pos"]): r["confirmed"]
                     for r in held["sv1"]["readings"]}
        pairs = [p for p in _sv1_pairs(plan)
                 if base_conf.get((p["source_pos"], p["target_pos"]))]
        seed = int("".join(ch for ch in family_id if ch.isdigit()) or 0)
        for order in S.permutations_for(len(pars), seed, k=ORDER_CANDIDATES + 1):
            if len(chosen) == 2:
                break
            ptext = S.permute(base_text, order)
            sv1p = verify_pairs(panel, ptext, plan, pairs)
            lost = [r for r in sv1p["readings"] if not r["confirmed"]]
            event("ORDER.permutation", not lost, order=order, lost=len(lost))
            if not lost:
                chosen.append(order)
        if len(chosen) == 2:
            ident = list(range(len(pars)))
            prim_texts = [(base, chosen[0]), (base, ident), (base, chosen[1])]
        else:
            rejected["ORDER"] = ("fewer than two paragraph orders keep every confirmed "
                                 "relation readable")
    elif asg["primary"] is None:
        rejected[primary] = "no castable slots: a designated slot and its spare both failed SV5"
    elif primary in gold_reasons:
        rejected[primary] = gold_reasons[primary]
    else:
        prim_texts = [(a, None) for a in asg["primary"]]
    if prim_texts is not None:
        ladders.append({"type": primary, "rungs": prim_texts})
    # CONTROL.
    if asg["control"] is None:
        rejected["CONTROL"] = "the control slot failed SV5"
    elif "CONTROL" in gold_reasons:
        rejected["CONTROL"] = gold_reasons["CONTROL"]
    else:
        ident = None if primary != "ORDER" else list(range(len(S.paragraphs(base_text))))
        ladders.append({"type": "CONTROL", "rungs": [(a, ident) for a in asg["control"]]})
    for lt, why in rejected.items():
        event(f"ladder.{lt}", False, detail=why)
    for lad in ladders:
        event(f"ladder.{lad['type']}", True)
    if not ladders:
        res.verdict.add(GateResult("admit", False, detail="; ".join(
            f"{k}: {v}" for k, v in rejected.items())))
        return finish("ladders")

    # ---- items and manifest -----------------------------------------------------
    keyed: dict[tuple, dict[str, Any]] = {}
    manifest_ladders = []
    gold_nons = [{"source_id": S.atom_id(n["source_pos"]), "target_id": S.atom_id(n["target_pos"]),
                  "position_distance": abs(n["source_pos"] - n["target_pos"])} for n in nons]
    xs = S.xstar_of(plan)
    for lad in ladders:
        ltype = lad["type"]
        lid = f"{family_id}-{ltype}"
        ids = []
        for k, (a, order) in enumerate(lad["rungs"]):
            ident = order is None or order == sorted(order)
            key = (tuple(sorted(a.items())), None if ident else tuple(order))
            if key not in keyed:
                text = S.splice(template, fillers, a)
                if not ident:
                    text = S.permute(text, order)
                iid = f"{cfg.dataset_name}-{family_id}-t{len(keyed)}"
                keyed[key] = {
                    "id": iid,
                    "name": f"{canonical_topic} -- {ltype} r{k}",
                    "source": f"generated:S4/{generator}",
                    "response": text,
                    "num_atoms": n_atoms,
                    "atoms": S.build_atoms(plan, S.stance_for(plan, a)),
                    "relations": S.gold_relations(plan, a, fixed),
                    "non_relations": gold_nons,
                    "expected": {"design": S.DESIGN, "family_id": family_id,
                                 "primary": primary, "assignment": dict(a),
                                 "paragraph_order": None if ident else list(order),
                                 "ladders": []},
                    "meta": {"canonical_topic": canonical_topic,
                             "domain": domain_of(canonical_topic), "framing": question,
                             "split": split, "word_count": S.word_count(text),
                             "generator": generator, "taxonomy": S.TAXONOMY,
                             "polarity": "verdict", "priors": "flat"},
                }
            item = keyed[key]
            item["expected"]["ladders"].append({"ladder_id": lid, "type": ltype,
                                                "rung_index": k})
            ids.append(item["id"])
        graphs = [S.factor_rels(keyed_item["relations"]) for keyed_item in
                  (next(v for v in keyed.values() if v["id"] == i) for i in ids)]
        scores = [S.lambda_lcs(n_atoms, g, xs) for g in graphs]
        manifest_ladders.append({
            "ladder_id": lid,
            "type": ltype,
            "declared": S.LADDERS[ltype].declared,
            "item_ids": ids,
            "polarity_invariant": S.LADDERS[ltype].polarity_invariant,
            "gold": {"lambda": [round(s[0], 4) for s in scores],
                     "lcs": [round(s[1], 4) for s in scores]},
            "assertion": gold_rep.get("primary" if ltype == primary else "control", {}),
        })
    items = list(keyed.values())
    for it in items:
        S.validate_item(it)
    res.items = items
    res.manifest = {
        "family_id": family_id,
        "design": S.DESIGN,
        "dataset": cfg.dataset_name,
        "canonical_topic": canonical_topic,
        "domain": domain_of(canonical_topic),
        "framing": question,
        "generator": generator,
        "primary": primary,
        "rare_senses": list(rare),
        "ladders": manifest_ladders,
        "rejected_ladders": rejected,
        "gold_corrections": corrections,
    }
    res.verdict.add(GateResult("admit", True, observed=[l["type"] for l in ladders]))
    return finish("admitted")


__all__ = ["Slot3Result", "generate_family_slot3", "judge_filler", "verify_pairs",
           "corrected_fixed", "describe_filler"]
