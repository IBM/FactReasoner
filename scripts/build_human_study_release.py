#!/usr/bin/env python
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

"""Build the Label Studio projects for the LoCoBench validation study (paper, sec-study).

Two projects, because the two session types have different interfaces:

* **core** -- one task per family. The annotator reads the base response once, then on
  the same screen checks 15 claim pairs (link? both true? both false?), checks whether
  each claim is asserted as written, and judges each edited version shown as a
  highlighted change against the base. 22 families, 3 annotators each, 7 annotators.
* **instruction** -- one task per whole-text ORDER pair (7 families x 3 shuffle rungs). The instruction varies by
  screen, balanced within each annotator; 3 annotators.

Output layout under ``--out-dir`` (default ``docs/human_study_release``)::

    core/labeling_config.xml        instruction/labeling_config.xml
    core/tasks_C1.json ... C7       instruction/tasks_I1.json ... I3
    core/practice.json              instruction/practice.json
    brief_core.md  brief_instruction.md  README.md  manifest.json

Every task carries ``annotator`` and ``task_id`` and nothing that reveals an answer.
Which rung each version is, which pairs are gold relations, and which claims are false
live in ``answer_key.jsonl`` -- written to ``--key-dir`` (default
``artifacts/human_study_release``, git-ignored), never into the release folder. The leak
check runs on the serialized tasks, not on the construction, and refuses to write if any
answer-revealing key or string is present.

Run::

    python scripts/build_human_study_release.py
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.locoeval.runner import load_families, load_items  # noqa: E402

# The stratified sample of sec-study. Drawn by `random.Random(SEED).sample` per ladder
# type in this order; the seed is the paper's.
SEED = 0
QUOTA_FIRST = (("CONFLICT", 11), ("CHAIN", 6), ("ORDER", 5), ("CONTROL", 5))
EXTRA_ORDER = 2

N_CORE, N_INSTR, RATERS = 7, 3, 3

# The practice family: one per session type, outside the sample, adjudicated in advance.
PRACTICE_CORE_TYPE, PRACTICE_INSTR_TYPE = "CONFLICT", "ORDER"

# The two instructions of the H4 test, verbatim from the paper.
INSTRUCTIONS = {
    "consistency": (
        "If you believed everything each response says, which would leave you believing "
        "something consistent?"
    ),
    "argument": "Which response hangs together better as an argument?",
}

FORBIDDEN_KEYS = (
    "rung_name", "rung_index", "perturbation", "notes", "factual", "role", "relations",
    "non_relations", "expected", "family_id", "validity", "error_kind", "level2_sense",
    "level1_coupling", "sense", "coupling", "is_base", "declared", "edge_id", "source",
)
FORBIDDEN_SUBSTRINGS = (
    "worse", "coherent\"", "concession_resolved", "fix_one_conflict", "break_",
    "shuffle_", "ordering_only", "direction_reversal", "locobench-opus", "spurious",
    "false_endpoint", "wrong_sense", "wrong_direction", "planted",
)


# -- sample and allocation ----------------------------------------------------------


def draw_sample(fams: dict[str, dict]) -> list[str]:
    """The 29 study families, exactly as sec-study specifies.

    Two draws, so that growing the instruction test cannot move the core sample:

    * The first is the original stratified draw, one shared RNG over QUOTA_FIRST in that
      order. Its 22 non-ORDER families are the core sample whose edge, claim and sense
      counts the paper reports, and its 5 ORDER families start the instruction test.
    * The second adds EXTRA_ORDER more ORDER families, from those not yet drawn, on its
      own seed. The instruction test needs 7 ORDER families for power 0.8 on H4; 5 gave
      0.65.
    """
    allf = sorted(fams.values(), key=lambda f: f["family_id"])
    rng = random.Random(SEED)
    out: list[str] = []
    for ladder, q in QUOTA_FIRST:
        pool = [f for f in allf if f["family"] == ladder]
        out += [f["family_id"] for f in rng.sample(pool, q)]
    rest = [f for f in allf if f["family"] == "ORDER" and f["family_id"] not in out]
    out += [f["family_id"] for f in random.Random(f"{SEED}-order-extra").sample(rest, EXTRA_ORDER)]
    return out


def pick_practice(fams: dict[str, dict], exclude: set[str], ladder: str) -> str:
    """A practice family of the given type, outside the sample, first by id."""
    return next(
        f for f in sorted(fams) if fams[f]["family"] == ladder and f not in exclude
    )


def allocate(families: list[str], fams: dict[str, dict], n: int, rng: random.Random):
    """Give each family to RATERS distinct annotators, balancing load and spreading topics.

    Two families on one topic have different questions and different claims, so sharing a
    topic leaks nothing; it only makes a session less varied. A hard "one family per topic
    per annotator" rule is infeasible here (Botany has 3 families, so 9 slots over 7
    annotators), so topic repeats are minimized rather than forbidden: each family goes to
    the RATERS annotators with the fewest families on its topic, then the lowest load.
    Asserted afterwards: every family has exactly RATERS annotators, and no annotator holds
    more than two families on any one topic.
    """
    load = [0.0] * n
    topics: list[Counter] = [Counter() for _ in range(n)]
    held: list[list[str]] = [[] for _ in range(n)]
    order = list(families)
    rng.shuffle(order)
    # Most-constrained topics first, CONTROL (lighter) last so it fills the gaps.
    tcount = Counter(fams[f]["canonical_topic"] for f in families)
    order.sort(key=lambda f: (fams[f]["family"] == "CONTROL", -tcount[fams[f]["canonical_topic"]]))
    for fid in order:
        topic = fams[fid]["canonical_topic"]
        weight = 13.0 if fams[fid]["family"] == "CONTROL" else 15.0
        cands = list(range(n))
        rng.shuffle(cands)
        cands.sort(key=lambda a: (topics[a][topic], load[a]))
        for a in cands[:RATERS]:
            load[a] += weight
            topics[a][topic] += 1
            held[a].append(fid)
    cover = Counter(f for h in held for f in h)
    assert all(cover[f] == RATERS for f in families), cover
    assert all(max(t.values()) <= 2 for t in topics), topics
    return held, load, [sorted(t for t, c in tc.items() if c > 1) for tc in topics]


# -- diffs ---------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")


def sentences(text: str) -> list[str]:
    """Split into sentences, keeping paragraph breaks as their own marker."""
    out: list[str] = []
    for i, para in enumerate(text.split("\n\n")):
        if i:
            out.append("\n\n")
        out += [s for s in _SENT.split(para.strip()) if s]
    return out


def highlight_change(base: str, edited: str) -> str:
    """The edited text as HTML, with removed text struck and added text marked.

    Word-level within changed sentences, sentence-level elsewhere, so an annotator sees
    exactly what moved and nothing else is noisy.
    """
    a, b = sentences(base), sentences(edited)
    parts: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            parts += [_plain(s) for s in a[i1:i2]]
            continue
        old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        wa, wb = old.split(" "), new.split(" ")
        for wop, x1, x2, y1, y2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
            if wop == "equal":
                parts.append(html.escape(" ".join(wa[x1:x2])))
            if wop in ("delete", "replace") and x2 > x1:
                parts.append(f'<del>{html.escape(" ".join(wa[x1:x2]))}</del>')
            if wop in ("insert", "replace") and y2 > y1:
                parts.append(f'<ins>{html.escape(" ".join(wb[y1:y2]))}</ins>')
    out = " ".join(parts)
    return out.replace(" \n\n ", "</p><p>").replace("\n\n", "</p><p>").join(("<p>", "</p>"))


def _plain(s: str) -> str:
    return "\n\n" if s == "\n\n" else html.escape(s)


def numbered_response(item: dict) -> str:
    """The response as HTML, followed by the numbered claim list the questions refer to."""
    body = "".join(f"<p>{html.escape(p)}</p>" for p in item["response"].split("\n\n"))
    claims = "".join(
        f"<li value=\"{i + 1}\">{html.escape(a['text'])}</li>"
        for i, a in enumerate(item["atoms"])
    )
    return f'<div class="resp">{body}</div><h4>Numbered claims</h4><ol class="claims">{claims}</ol>'


# -- tasks ---------------------------------------------------------------------------


def core_task(fid, fams, by_rung, rng, annotator, task_id):
    """One core task (one family) and its private key entry."""
    fam = fams[fid]
    base_idx = next(r["index"] for r in fam["rungs"] if r["is_base"])
    base = by_rung[(fid, base_idx)]
    num = {a["id"]: i + 1 for i, a in enumerate(base["atoms"])}

    # 15 claim pairs: the gold relations and declared non-relations, shuffled together.
    pairs = [("rel", e["source_id"], e["target_id"], e) for e in base["relations"]]
    pairs += [("non", n["source_id"], n["target_id"], n) for n in base["non_relations"]]
    rng.shuffle(pairs)
    pair_rows, pair_key = [], []
    for k, (kind, s, t, e) in enumerate(pairs):
        if rng.random() < 0.5:  # never let position encode direction
            s, t = t, s
        # One display string per row: Label Studio's Repeater substitutes only the first
        # {{idx}} in a value, so a value naming two fields renders the second as "undefined".
        pair_rows.append({"pair_id": f"p{k + 1:02d}", "claim_x": num[s], "claim_y": num[t],
                          "label": f"Claim {num[s]} and claim {num[t]}"})
        pair_key.append({
            "pair_id": f"p{k + 1:02d}", "kind": kind, "shown_as": [num[s], num[t]],
            **({k2: e.get(k2) for k2 in ("id", "source_id", "target_id", "level2_sense",
                                           "level1_coupling", "validity", "error_kind")}
               if kind == "rel" else {}),
        })

    # Every claim, asked "asserted as written?"; the false ones are among them, unmarked.
    claim_rows = [{"claim_no": i + 1, "text": a["text"], "label": f"{i + 1}. {a['text']}"}
                  for i, a in enumerate(base["atoms"])]
    claim_key = [{"claim_no": i + 1, "atom_id": a["id"], "factual": a["factual"]}
                 for i, a in enumerate(base["atoms"])]

    # Edited versions: distinct texts only (CONTROL's twin rungs share text by design).
    seen, versions, version_key = {}, [], []
    edited = [r for r in fam["rungs"] if not r["is_base"]]
    rng.shuffle(edited)
    for r in edited:
        txt = by_rung[(fid, r["index"])]["response"]
        if txt in seen:
            version_key[seen[txt]]["rungs"].append(r["index"])
            continue
        vid = f"v{len(versions) + 1}"
        seen[txt] = len(version_key)
        versions.append({"version_id": vid, "html": highlight_change(base["response"], txt)})
        version_key.append({"version_id": vid, "rungs": [r["index"]], "rung_names": [r["name"]],
                            "calls": r["calls"], "parent": r["parent"]})
    # Attention check: one version that restores the base word for word.
    task = {
        "task_id": task_id, "annotator": annotator, "kind": "core",
        "framing": fam["framing"], "base_html": numbered_response(base),
        "pairs": pair_rows, "claims": claim_rows, "versions": versions,
    }
    key = {
        "task_id": task_id, "annotator": annotator, "kind": "core", "family_id": fid,
        "ladder": fam["family"], "topic": fam["canonical_topic"], "base_rung": base_idx,
        "pairs": pair_key, "claims": claim_key, "versions": version_key,
    }
    return task, key


def add_attention_checks(task, key, base_item, rng):
    """Fold the two no-cost checks of sec-study into one core task."""
    # A version identical to the base: the right answer is "about as coherent, nothing else".
    vid = f"v{len(task['versions']) + 1}"
    task["versions"].insert(rng.randrange(len(task["versions"]) + 1),
                            {"version_id": vid, "html": highlight_change(base_item["response"], base_item["response"])})
    key["versions"].append({"version_id": vid, "attention_check": "identical_to_base"})
    # A claim pair that repeats one claim: the right answer is "linked, both true, not both false".
    c = rng.randrange(len(base_item["atoms"])) + 1
    pid = f"p{len(task['pairs']) + 1:02d}"
    task["pairs"].insert(rng.randrange(len(task["pairs"]) + 1),
                         {"pair_id": pid, "claim_x": c, "claim_y": c, "label": f"Claim {c} and claim {c}"})
    key["pairs"].append({"pair_id": pid, "kind": "attention_check", "shown_as": [c, c]})


def order_jobs(order_fams, fams):
    """Every (family, base rung, shuffle rung) the instruction test judges, in a fixed order."""
    jobs = []
    for fid in order_fams:
        fam = fams[fid]
        base_idx = next(r["index"] for r in fam["rungs"] if r["is_base"])
        for r in fam["rungs"]:
            if not r["is_base"] and r["name"].startswith("shuffle"):
                jobs.append((fid, base_idx, r))
    return jobs


def instruction_plan(jobs, n_annotators, rng):
    """Which instruction each annotator gets on each pair: a cyclic Latin assignment.

    Drawing the instruction independently per annotator left 9 of 21 pairs judged under
    only one instruction across all three annotators, so pair difficulty leaked into the
    H4 comparison. Here pair j goes to annotator a under instruction (j + a) mod 2, after
    shuffling the pairs, so with three annotators every pair is judged under BOTH
    instructions (twice under one, once under the other) and each annotator gets a near
    even split.
    """
    order = list(range(len(jobs)))
    rng.shuffle(order)
    names = ("consistency", "argument")
    return {a: {order[j]: names[(j + a) % 2] for j in range(len(jobs))} for a in range(n_annotators)}


def instruction_tasks(order_fams, fams, by_rung, rng, annotator, start, plan=None):
    """One instruction-test annotator's whole-text pairs.

    ``plan`` maps job index (into :func:`order_jobs`) to an instruction, from
    :func:`instruction_plan`; without it, as for the practice block, instructions are
    split evenly at random within this annotator.
    """
    base_jobs = order_jobs(order_fams, fams)
    if plan is None:
        names = ["consistency"] * (len(base_jobs) // 2)
        names += ["argument"] * (len(base_jobs) - len(names))
        rng.shuffle(names)
        plan = dict(enumerate(names))
    jobs = [(fid, base_idx, r, plan[i]) for i, (fid, base_idx, r) in enumerate(base_jobs)]
    rng.shuffle(jobs)
    # No two consecutive screens from one family.
    for _ in range(500):
        if all(jobs[i][0] != jobs[i + 1][0] for i in range(len(jobs) - 1)):
            break
        rng.shuffle(jobs)
    # The original's side is balanced exactly within each annotator, not coin-flipped:
    # at 21 screens independent flips left one annotator with the original on side A in
    # only 5, and side bias is a known confound in paired judgements.
    sides = [True] * (len(jobs) // 2) + [False] * (len(jobs) - len(jobs) // 2)
    rng.shuffle(sides)
    tasks, keys = [], []
    for k, ((fid, base_idx, r, ins), base_is_a) in enumerate(zip(jobs, sides)):
        base, shuf = by_rung[(fid, base_idx)], by_rung[(fid, r["index"])]
        a, b = (base, shuf) if base_is_a else (shuf, base)
        tid = f"{annotator}-{start + k:03d}"
        tasks.append({
            "task_id": tid, "annotator": annotator, "kind": "instruction",
            "framing": fam_framing(fams, fid), "instruction": INSTRUCTIONS[ins],
            "response_a": _paras(a["response"]), "response_b": _paras(b["response"]),
        })
        keys.append({"task_id": tid, "annotator": annotator, "kind": "instruction",
                     "family_id": fid, "rung": r["index"], "rung_name": r["name"],
                     "base_side": "A" if base_is_a else "B", "instruction": ins})
    return tasks, keys


def fam_framing(fams, fid):
    return fams[fid]["framing"]


def _paras(text: str) -> str:
    return "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n"))


# -- leak check ----------------------------------------------------------------------


def assert_no_leaks(tasks: list[dict]) -> None:
    """Refuse to write any task carrying an answer-revealing key or string.

    Checked on the serialized task. Strings are checked everywhere except the response
    and version bodies, which are real prose and may contain an innocent "spurious".
    """
    for t in tasks:
        blob = json.dumps(t)
        for bad in FORBIDDEN_KEYS:
            if f'"{bad}"' in blob:
                raise SystemExit(f"LEAK: key {bad!r} in task {t['task_id']}")
        meta = {k: v for k, v in t.items() if k not in ("base_html", "versions", "response_a", "response_b", "claims", "framing")}
        low = json.dumps(meta).lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in low:
                raise SystemExit(f"LEAK: {bad!r} in task {t['task_id']} metadata")


# -- main ----------------------------------------------------------------------------


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(repo, "data", "locobench-opus-5-5"))
    ap.add_argument("--out-dir", default=os.path.join(repo, "docs", "human_study_release"))
    ap.add_argument("--key-dir", default=os.path.join(repo, "artifacts", "human_study_release"))
    args = ap.parse_args()

    items = load_items(args.data_dir)
    fams = load_families(args.data_dir)
    by_rung = {(i["expected"]["family_id"], i["expected"]["rung_index"]): i for i in items}

    sample = draw_sample(fams)
    core_fams = [f for f in sample if fams[f]["family"] != "ORDER"]
    order_fams = [f for f in sample if fams[f]["family"] == "ORDER"]
    practice_core = pick_practice(fams, set(sample), PRACTICE_CORE_TYPE)
    practice_instr = pick_practice(fams, set(sample), PRACTICE_INSTR_TYPE)

    rng = random.Random(SEED + 1)
    held, load, repeats = allocate(core_fams, fams, N_CORE, rng)

    out_core, out_instr = os.path.join(args.out_dir, "core"), os.path.join(args.out_dir, "instruction")
    for d in (out_core, out_instr, args.key_dir):
        os.makedirs(d, exist_ok=True)

    keys: list[dict] = []
    all_tasks: list[dict] = []
    manifest = {"seed": SEED, "data_dir": os.path.relpath(args.data_dir, repo), "sample": sample,
                "practice": {"core": practice_core, "instruction": practice_instr},
                "core": {}, "instruction": {}}

    for a, fids in enumerate(held, 1):
        who = f"C{a}"
        arng = random.Random(f"{SEED}-{who}")
        # Increase families first, then CONTROL, then interleave by topic order
        order = sorted(fids, key=lambda f: (fams[f]["family"] == "CONTROL", arng.random()))
        tasks = []
        for k, fid in enumerate(order, 1):
            t, kk = core_task(fid, fams, by_rung, arng, who, f"{who}-{k:03d}")
            if k == 2:  # attention checks sit inside an early, real family
                base_idx = next(r["index"] for r in fams[fid]["rungs"] if r["is_base"])
                add_attention_checks(t, kk, by_rung[(fid, base_idx)], arng)
            tasks.append(t)
            keys.append(kk)
        manifest["core"][who] = {"families": order, "modelled_minutes": round(15 + 10 + load[a - 1]),
                                 "repeated_topics": repeats[a - 1]}
        all_tasks += tasks
        _write(os.path.join(out_core, f"tasks_{who}.json"), tasks)

    plan = instruction_plan(order_jobs(order_fams, fams), N_INSTR, random.Random(f"{SEED}-instruction-plan"))
    for a in range(1, N_INSTR + 1):
        who = f"I{a}"
        tasks, kk = instruction_tasks(order_fams, fams, by_rung, random.Random(f"{SEED}-{who}"), who, 1,
                                      plan=plan[a - 1])
        manifest["instruction"][who] = {"tasks": len(tasks),
                                        "per_instruction": dict(Counter(k["instruction"] for k in kk))}
        all_tasks += tasks
        keys += kk
        _write(os.path.join(out_instr, f"tasks_{who}.json"), tasks)

    # Practice tasks: the same shapes, outside the sample; their keys are kept too.
    pt, pk = core_task(practice_core, fams, by_rung, random.Random(f"{SEED}-practice"), "practice", "practice-core")
    _write(os.path.join(out_core, "practice.json"), [pt])
    ptasks, pkeys = instruction_tasks([practice_instr], fams, by_rung, random.Random(f"{SEED}-practice-i"), "practice", 1)
    _write(os.path.join(out_instr, "practice.json"), ptasks)  # its 3 shuffle rungs
    keys += [pk] + pkeys
    all_tasks += [pt] + ptasks

    assert_no_leaks(all_tasks)

    # The labeling configs are hand-written and live next to this script, so one run
    # regenerates the whole release folder.
    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_study_release_templates")
    for kind, dest in (("core", out_core), ("instruction", out_instr)):
        with open(os.path.join(tpl, f"{kind}_labeling_config.xml")) as f:
            cfg = f.read()
        with open(os.path.join(dest, "labeling_config.xml"), "w") as f:
            f.write(cfg)

    with open(os.path.join(args.key_dir, "answer_key.jsonl"), "w") as f:
        for k in keys:
            f.write(json.dumps(k) + "\n")
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"sample: {len(sample)} families; core {len(core_fams)} x {RATERS} raters over {N_CORE} "
          f"annotators; instruction {len(order_fams)} ORDER families x {N_INSTR} annotators")
    for who, m in manifest["core"].items():
        print(f"  {who}: {len(m['families'])} families, ~{m['modelled_minutes']} min"
              + (f", repeated topic {m['repeated_topics']}" if m["repeated_topics"] else ""))
    for who, m in manifest["instruction"].items():
        print(f"  {who}: {m['tasks']} pairs {m['per_instruction']}")
    print(f"practice: core {practice_core}, instruction {practice_instr}")
    print(f"tasks -> {args.out_dir}\nanswer key -> {args.key_dir}/answer_key.jsonl (git-ignored)")
    return 0


def _write(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
