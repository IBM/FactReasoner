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

"""The deterministic offline LLM for slot3 dry runs.

It answers every prompt the slot3 pipeline issues -- P1/P2/V3/V4 through the v1 mocks, and
S3/S4/SF/SV1/SV5 here -- so a dry run executes every gate, splice, casting and schema check
with no credentials. Its readers are FAITHFUL: SV1/SV5 return the planned reading, so a dry
run exercises the acceptance path. ``faults`` makes chosen fillers read wrongly, to drive
the regeneration and spare-slot paths.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fact_reasoner.locobench import mock as _mock
from fact_reasoner.locobench import slots as S
from fact_reasoner.locobench.pipeline import which_prompt

_PAD = (
    "Specialists have examined this question at length, and the record they assembled "
    "is detailed enough to support careful conclusions about each step of the account."
)


def _claims_from(rendered: str) -> list[tuple[str, str]]:
    out = []
    for line in rendered.split("ATOMIC CLAIMS:", 1)[-1].splitlines():
        m = re.match(r"\s*-\s*(.+?)\s*\[([a-z0-9-]+)\]\s*$", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _primary_from(rendered: str) -> str:
    rendered = rendered.split("Plan the SLOTS.", 1)[-1].split("Rules for every slot", 1)[0]
    if '"conflict_tt"' in rendered:
        return "CONFLICT"
    if '"chain_link"' in rendered:
        return "CHAIN"
    if '"sign_tf"' in rendered:
        return "SIGN"
    return "ORDER"


def mock_slot_plan(rendered: str) -> str:
    claims = _claims_from(rendered)
    T = [t for t, tag in claims if tag != "incorrect"]
    F = [t for t, tag in claims if tag == "incorrect"]
    primary = _primary_from(rendered)
    senses = rendered.split("each of these senses:", 1)[-1].split(".\n", 1)[0]
    if primary == "SIGN":
        order = [T[0], T[1], T[2], F[0], T[3], T[4], T[5], F[1], T[6], T[7], T[8], T[9]]
        slots = [
            {"id": "S1", "role": "sign_tf", "source_pos": 3, "target_pos": 4, "sense": "Cause-Effect"},
            {"id": "S2", "role": "sign_tf", "source_pos": 7, "target_pos": 8, "sense": "Evidence"},
            {"id": "S3", "role": "control", "source_pos": 10, "target_pos": 11, "sense": "Evidence"},
        ]
    else:
        order = [T[0], T[1], T[2], T[3], T[4], T[5], F[0], T[6], T[7], T[8], F[1], T[9]]
        if primary == "CONFLICT":
            slots = [
                {"id": "S1", "role": "conflict_tt", "source_pos": 1, "target_pos": 2},
                {"id": "S2", "role": "conflict_tt", "source_pos": 3, "target_pos": 4},
                {"id": "S3", "role": "conflict_tt", "source_pos": 5, "target_pos": 6},
                {"id": "S4", "role": "control", "source_pos": 8, "target_pos": 9, "sense": "Evidence"},
            ]
        elif primary == "CHAIN":
            slots = [
                {"id": "S1", "role": "chain_link", "source_pos": 1, "target_pos": 2, "sense": "Cause-Effect"},
                {"id": "S2", "role": "chain_link", "source_pos": 2, "target_pos": 3, "sense": "Evidence"},
                {"id": "S3", "role": "control", "source_pos": 8, "target_pos": 9, "sense": "Evidence"},
            ]
        else:
            slots = [{"id": "S1", "role": "control", "source_pos": 8, "target_pos": 9,
                      "sense": "Evidence"}]
    for s in slots:
        s["strength_band"] = "strong"
        s.setdefault("sense", "Refutation")
    rels = [
        {"source_pos": 1, "target_pos": 3, "sense": "Restatement", "strength_band": "strong"},
        {"source_pos": 4, "target_pos": 6, "sense": "Cause-Effect", "strength_band": "moderate"},
        {"source_pos": 6, "target_pos": 7, "sense": "Alternative", "strength_band": "strong"},
        {"source_pos": 9, "target_pos": 10, "sense": "Precedence", "strength_band": "weak"},
        {"source_pos": 10, "target_pos": 12, "sense": "Disjunction", "strength_band": "moderate"},
        {"source_pos": 11, "target_pos": 12, "sense": "Refutation", "strength_band": "moderate"},
    ]
    del senses
    plan = {
        "atoms": [{"pos": i + 1, "text": t, "factual": True} for i, t in enumerate(order)],
        "relations": rels,
        "slots": slots,
        "non_relations": [{"source_pos": 2, "target_pos": 5}, {"source_pos": 5, "target_pos": 8},
                          {"source_pos": 7, "target_pos": 9}],
    }
    return "```json\n" + json.dumps(plan, indent=1) + "\n```"


def _filler(slot: dict[str, Any], kind: str, owns: list[int], text: dict[int, str]) -> str:
    x, y = slot["source"], slot["target"]
    own = [text[p] for p in owns]
    if kind == "none":
        return " ".join(own)
    if slot.get("_role") == "conflict_tt" or (kind == "support" and slot.get("_role") == "sign_tf"):
        lead = own[0] if len(own) == 2 else ""
        return f"{lead} That cannot be squared with the further point that {y[0].lower()}{y[1:]}".strip()
    words = {"para1": "This shows that", "para2": "It follows, as the record indicates, that",
             "para3": "On that evidence, one can conclude that"}.get(kind, "As a result,")
    lead = own[0] if len(own) == 2 else ""
    return f"{lead} {words} {y[0].lower()}{y[1:]}".strip()


def mock_template(rendered: str, holder: dict[str, Any]) -> str:
    plan = json.loads(rendered.split("PLAN:", 1)[-1].rsplit("OUTPUT:", 1)[0].strip())
    text = {a["pos"]: a["text"] for a in plan["atoms"]}
    slot_at: dict[int, dict] = {}
    for s in plan["slots"]:
        slot_at[min(s["owns"])] = s
    order_family = "SELF-CONTAINED" in rendered
    parts: list[str] = []
    skip: set[int] = {p for s in plan["slots"] for p in s["owns"]}
    for a in sorted(plan["atoms"], key=lambda a: a["pos"]):
        p = a["pos"]
        if p in slot_at:
            parts.append("{{" + slot_at[p]["id"] + "}}")
        if p not in skip:
            parts.append(a["text"])
    n_par = 5 if order_family else 4
    per = max(1, -(-len(parts) // n_par))
    pars = [" ".join(parts[i:i + per] + [_PAD, _PAD, _PAD, _PAD]) for i in range(0, len(parts), per)]
    roles = holder.get("roles", {})
    fillers = {}
    for s in plan["slots"]:
        s["_role"] = roles.get(s["id"])
        fillers[s["id"]] = {k: _filler(s, k, s["owns"], text) for k in s["fillers"]}
        for k, f in fillers[s["id"]].items():
            holder.setdefault("filler_index", {})[f] = (s["id"], k)
    return "```json\n" + json.dumps({"template": "\n\n".join(pars), "fillers": fillers}) + "\n```"


def _between(s: str, a: str, b: str) -> str:
    i = s.rfind(a)
    if i == -1:
        return ""
    j = s.find(b, i + len(a))
    return s[i + len(a): j if j != -1 else None]


def make_slot3_mock_llm(seed: int = 0, holder: dict[str, Any] | None = None,
                        faults: dict[tuple[str, str], int] | None = None):
    """The dry-run LLM.

    Args:
        holder: Shared state: the pipeline's plan (``plan``), roles by slot id, and the
            filler index the S4 mock writes. Pass the SAME dict to generator and raters.
        faults: ``{(slot id, kind): n}`` -- the first ``n`` readings of that filler are
            wrong, which exercises SF regeneration (and, for large n, the spare path).
    """
    holder = holder if holder is not None else {}
    faults = dict(faults or {})
    seen: dict[tuple[str, str], int] = {}

    def llm(rendered: str, *, attempt: int = 0) -> str:
        pid = which_prompt(rendered)
        if pid == "P1":
            return _mock.mock_question(rendered.rsplit("TOPIC:", 1)[-1].strip().split("\n")[0], seed)
        if pid == "P2":
            return _mock.mock_claims(rendered, seed)
        if pid == "S3":
            out = mock_slot_plan(rendered)
            plan = json.loads(out.strip("`").removeprefix("json"))
            holder["plan"] = plan
            holder["roles"] = {s["id"]: s["role"] for s in plan["slots"]}
            return out
        if pid == "S4":
            return mock_template(rendered, holder)
        if pid == "SF":
            failed = _between(rendered, "FAILED: ", "\nREADERS:").strip()
            sid, kind = holder.get("filler_index", {}).get(failed, (None, None))
            new = failed + " Indeed it does."
            if sid:
                holder["filler_index"][new] = (sid, kind)
            return json.dumps({"filler": new})
        if pid == "V3":
            return _mock.mock_audit(rendered, seed)
        if pid == "V4":
            atoms = [a["text"] for a in sorted(holder.get("plan", {}).get("atoms", []),
                                               key=lambda a: a["pos"])]
            return _mock.mock_coverage(rendered, atoms, seed)
        if pid == "SV1":
            i = rendered.rfind("pairs=") + len("pairs=")
            pairs = json.loads(rendered[i: rendered.rfind(")")])
            plan = holder.get("plan", {})
            exp = {(r["source_pos"], r["target_pos"]): S.expected_label(r["sense"])
                   for r in plan.get("relations", [])}
            return json.dumps([{"pair": p, "label": exp.get(tuple(p), "unlinked"),
                                "evidence": ""} for p in pairs])
        if pid == "SV5":
            passage = _between(rendered, 'passage="', '", x="')
            x = _between(rendered, '", x="', '", y="')
            sid, kind = holder.get("filler_index", {}).get(passage, (None, None))
            if sid is None:
                return json.dumps({"x_status": "asserted", "y_status": "asserted",
                                   "link": "unlinked", "other_links": [], "evidence": ""})
            slot = next(s for s in holder["plan"]["slots"] if s["id"] == sid)
            label = S.expected_label(S.slot_links(slot)[kind])
            src_text = next(a["text"] for a in holder["plan"]["atoms"]
                            if a["pos"] == slot["source_pos"])
            flipped = x != src_text
            if flipped:
                label = {"i_supports_j": "j_supports_i"}.get(label, label)
            k = (sid, kind)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= faults.get(k, 0):
                label = "unlinked" if label != "unlinked" else "incompatible"
            denied = slot["role"] == "sign_tf" and kind == "support"
            ys = "denied" if denied else "asserted"
            xs = "asserted"
            if flipped:
                xs, ys = ys, xs
            return json.dumps({"x_status": xs, "y_status": ys, "link": label,
                               "other_links": [], "evidence": passage[:40]})
        return ""

    return llm


__all__ = ["make_slot3_mock_llm", "mock_slot_plan", "mock_template"]
