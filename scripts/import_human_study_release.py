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

"""Turn Label Studio exports of the validation study into flat, keyed judgements.

The study (paper, sec-study) runs one Label Studio project per annotator, from the
task files in ``docs/human_study_release``. Export each project as **JSON** (not
JSON-MIN: that drops ``from_name``, without which the questions inside one task can't
be told apart) and pass every export here::

    python scripts/import_human_study_release.py \\
        --export exports/core_C1.json --export exports/core_C2.json ... \\
        --out artifacts/human_study_release/judgements.jsonl

Output is one JSON record per judgement, joined to the answer key, with a ``unit``
field naming the thing judged, so the same claim pair seen by three annotators shares
one unit. Units are keyed by CONTENT -- a pair by its two atoms, a claim by its atom, a
version by its rungs -- never by a task's own ids (``p01``, ``v1``), because each
annotator's copy of a family is shuffled independently and those ids differ between
annotators. Five record kinds:

* ``pair``      -- one claim pair: linked, both_true, both_false, plus the gold kind
* ``claim``     -- one claim as stated: asserted, plus whether it is planted-false
* ``version``   -- one edited version: coherence, collateral, plus its rungs and calls
* ``instruction`` -- one whole-text ORDER pair: choice and which side was the original
* attention checks keep their kind and carry ``attention_check`` set

WHY ANNOTATOR IDENTITY COMES FROM THE TASK, NOT FROM LABEL STUDIO. Each annotator has
their own project, and every project numbers its users from 1, so ``completed_by`` is
the same in every export. Every task carries the ``annotator`` it was built for, so the
importer reads that and checks that one export never mixes two.

HOW REPEATED QUESTIONS ARE NAMED. A ``Repeater`` block expands ``name="linked_{{idx}}"``
to ``linked_0``, ``linked_1``... in the order of the task's array. Label Studio's
expansion (``deepReplaceAttributes`` in ``src/core/Tree.tsx``) uses JavaScript's
``String.replace`` with a string pattern, which replaces only the FIRST placeholder in
an attribute; the configs therefore carry one per name, and this importer splits a
name at its last underscore.

Fails loudly, rather than guessing, on: an export with no annotations; a task whose
``task_id`` is not in the key; an answer value the config never offered; a task
annotated twice by one annotator; or a required question left unanswered.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

# The exact choice strings of the two labeling configs, mapped to analysis codes. A
# value outside these is a config/importer mismatch and stops the import.
CHOICES = {
    "linked": {"Linked": True, "Not linked": False},
    "both_true": {"Both could be true": True, "Not both true": False},
    "both_false": {"Both could be false": True, "Not both false": False},
    "asserted": {"Asserted as written": "asserted", "Changed or contradicted": "changed",
                 "Not stated": "not_stated"},
    "coherence": {"More coherent": "more", "About as coherent": "same", "Less coherent": "less"},
    "collateral": {"Nothing else changed": "none", "Minor other change": "minor",
                   "Material other change": "material"},
    "choice": {"Response A": "A", "Response B": "B", "About equal": "equal",
               "A": "A", "B": "B", "equal": "equal"},
    "confidence": {"low": "low", "medium": "medium", "high": "high"},
}
REQUIRED = {"pair": ("linked", "both_true", "both_false"), "claim": ("asserted",),
            "version": ("coherence", "collateral"), "instruction": ("choice", "confidence")}
TEXT_FIELDS = {"collateral_span", "why"}


class ImportError_(SystemExit):
    """A malformed or inconsistent export. Raised as SystemExit so the CLI stops."""


def split_name(from_name: str) -> tuple[str, int | None]:
    """``linked_3`` -> ``("linked", 3)``; ``choice`` -> ``("choice", None)``."""
    head, _, tail = from_name.rpartition("_")
    if head and tail.isdigit():
        return head, int(tail)
    return from_name, None


def _value(entry: dict) -> object:
    v = entry.get("value") or {}
    if "choices" in v:
        c = v["choices"]
        return c[0] if c else None
    if "text" in v:
        t = v["text"]
        return " ".join(t) if isinstance(t, list) else t
    return None


def _decode(control: str, raw: object, where: str) -> object:
    if control in TEXT_FIELDS:
        return (raw or "").strip() or None
    table = CHOICES.get(control)
    if table is None:
        raise ImportError_(f"{where}: unknown control {control!r}; the labeling config and "
                           "this importer disagree.")
    if raw not in table:
        raise ImportError_(f"{where}: {control}={raw!r} is not an option the config offers "
                           f"({sorted(table)}).")
    return table[raw]


def load_key(path: str) -> dict[str, dict]:
    with open(path) as f:
        return {k["task_id"]: k for k in (json.loads(l) for l in f if l.strip())}


def convert_task(task: dict, ann: dict, key: dict, export_name: str) -> list[dict]:
    """All judgements in one annotated task, joined to its key entry."""
    data = task["data"]
    tid, who = data["task_id"], data["annotator"]
    k = key[tid]
    where = f"{export_name}:{tid}"
    by_block: dict[tuple[str, int | None], dict] = collections.defaultdict(dict)
    for entry in ann.get("result") or []:
        control, idx = split_name(entry["from_name"])
        block = {"linked": "pair", "both_true": "pair", "both_false": "pair",
                 "asserted": "claim", "coherence": "version", "collateral": "version",
                 "collateral_span": "version", "choice": "instruction",
                 "confidence": "instruction", "why": "instruction"}.get(control)
        if block is None:
            raise ImportError_(f"{where}: unknown control {entry['from_name']!r}")
        by_block[(block, idx)][control] = _decode(control, _value(entry), where)

    base = {"annotator": who, "task_id": tid, "practice": who == "practice"}
    out: list[dict] = []
    if data["kind"] == "core":
        fam = {"family_id": k["family_id"], "ladder": k["ladder"], "topic": k["topic"]}
        pkey = {p["pair_id"]: p for p in k["pairs"]}
        for i, row in enumerate(data["pairs"]):
            got = _need(by_block, ("pair", i), "pair", where, f"pair {row['pair_id']}")
            pk = pkey[row["pair_id"]]
            # The unit is the pair's IDENTITY (its two atoms), never its pair_id: every
            # annotator sees the pairs in their own shuffled order, so p01 names a
            # different pair for each of them.
            rec = {**base, **fam, "kind": "pair", "unit": f"{k['family_id']}:{'-'.join(pk['atoms'])}",
                   "pair_id": row["pair_id"], "shown_as": pk["shown_as"], **got,
                   "attention_check": pk["kind"] == "attention_check", "gold_kind": pk["kind"]}
            if pk["kind"] == "rel":
                rec.update(gold_sense=pk["level2_sense"], gold_coupling=pk["level1_coupling"],
                           validity=pk["validity"], error_kind=pk["error_kind"])
            out.append(rec)
        ckey = {c["claim_no"]: c for c in k["claims"]}
        for i, row in enumerate(data["claims"]):
            got = _need(by_block, ("claim", i), "claim", where, f"claim {row['claim_no']}")
            ck = ckey[row["claim_no"]]
            out.append({**base, **fam, "kind": "claim", "unit": f"{k['family_id']}:{ck['atom_id']}",
                        "claim_no": row["claim_no"], "atom_id": ck["atom_id"], **got,
                        "factual": ck["factual"]})
        vkey = {v["version_id"]: v for v in k["versions"]}
        for i, row in enumerate(data["versions"]):
            got = _need(by_block, ("version", i), "version", where, f"version {row['version_id']}")
            got.setdefault("collateral_span", None)
            vk = vkey[row["version_id"]]
            attn = "attention_check" in vk
            out.append({**base, **fam, "kind": "version",
                        # Sorted: a CONTROL version stands for two twin rungs sharing one
                        # text, recorded in each annotator's shuffled order, so [3, 1] and
                        # [1, 3] are the same version and must be the same unit.
                        "unit": f"{k['family_id']}:{'attn' if attn else 'r' + '-'.join(map(str, sorted(vk['rungs'])))}",
                        "version_id": row["version_id"], **got, "attention_check": attn,
                        **({} if attn else {"rungs": vk["rungs"], "rung_names": vk["rung_names"],
                                            "calls": vk["calls"], "parent": vk["parent"]})})
    elif data["kind"] == "instruction":
        got = _need(by_block, ("instruction", None), "instruction", where, "the comparison")
        got.setdefault("why", None)
        out.append({**base, "kind": "instruction", "family_id": k["family_id"], "ladder": "ORDER",
                    "unit": f"{k['family_id']}:r{k['rung']}", "rung": k["rung"],
                    "rung_name": k["rung_name"], "instruction": k["instruction"],
                    "base_side": k["base_side"], **got})
    else:
        raise ImportError_(f"{where}: unknown task kind {data['kind']!r}")
    return out


def _need(by_block, slot, block, where, what) -> dict:
    got = dict(by_block.get(slot, {}))
    missing = [c for c in REQUIRED[block] if c not in got or got[c] is None]
    if missing:
        raise ImportError_(f"{where}: {what} has no answer for {missing}. Every question is "
                           "required in the config, so this export is incomplete or edited.")
    return got


def convert(exports: list[str], key: dict[str, dict], *, keep_practice: bool = False) -> list[dict]:
    """Convert several exports, enforcing one annotator per export and no repeats."""
    rows: list[dict] = []
    seen: dict[tuple[str, str], str] = {}
    for path in exports:
        name = os.path.basename(path)
        with open(path) as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            raise ImportError_(f"{name}: not a Label Studio JSON export (expected a list of tasks).")
        annotated = [t for t in tasks if t.get("annotations")]
        if not annotated:
            raise ImportError_(f"{name}: contains no completed annotations.")
        whos = {t["data"].get("annotator") for t in annotated} - {"practice"}
        if len(whos) > 1:
            raise ImportError_(f"{name}: tasks from several annotators {sorted(whos)}. Each "
                               "annotator needs their own project (see the README).")
        for t in annotated:
            data = t.get("data") or {}
            if "task_id" not in data:
                raise ImportError_(f"{name}: a task has no data.task_id; was it imported from "
                                   "docs/human_study_release?")
            tid = data["task_id"]
            if tid not in key:
                raise ImportError_(f"{name}: task {tid!r} is not in the answer key. The export and "
                                   "the key come from different builds.")
            live = [a for a in t["annotations"] if not a.get("was_cancelled")]
            if len(live) > 1:
                raise ImportError_(f"{name}: task {tid} was annotated {len(live)} times. Keep one.")
            if not live:
                continue
            if data["annotator"] == "practice" and not keep_practice:
                continue
            slot = (data["annotator"], tid)
            if slot in seen:
                raise ImportError_(f"task {tid} appears in both {seen[slot]} and {name}.")
            seen[slot] = name
            rows += convert_task(t, live[0], key, name)
    return rows


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="append", required=True, help="A Label Studio JSON export (repeat per project).")
    ap.add_argument("--key", default=os.path.join(repo, "artifacts", "human_study_release", "answer_key.jsonl"))
    ap.add_argument("--out", default=os.path.join(repo, "artifacts", "human_study_release", "judgements.jsonl"))
    ap.add_argument("--keep-practice", action="store_true", help="Include practice tasks (excluded from analysis by default).")
    args = ap.parse_args()

    rows = convert(sorted(args.export), load_key(args.key), keep_practice=args.keep_practice)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = collections.Counter((r["annotator"], r["kind"]) for r in rows)
    for who in sorted({a for a, _ in by}):
        print(f"  {who}: " + ", ".join(f"{n} {k}" for (a, k), n in sorted(by.items()) if a == who))
    print(f"{len(rows)} judgements from {len(args.export)} export(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
