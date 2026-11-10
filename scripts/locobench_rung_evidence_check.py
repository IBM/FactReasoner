#!/usr/bin/env python
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

"""Judge-free companion to the rung reading check.

For every transition in a reading-check run, take the span a judge quoted as
carrying the relation in the rung WHERE GOLD HAS THE EDGE, and test whether that
span survives verbatim in the rung where gold removed it (drops and spurious
removals), plus how many words the perturbation changed at all. A conflict sentence
that survives word for word in its "dropped" rung is unremoved by construction --
no judge opinion involved.

    python scripts/locobench_rung_evidence_check.py \\
        --data-dir data/locobench-opus-5-5 --out-dir results/locobench_opus_5_5_reading
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
from collections import Counter

KINDS = ("dropped", "spurious_removed", "resolved", "added")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--judge", default="gpt-6.1-sol", help="Whose quoted evidence to test.")
    args = ap.parse_args()

    items = {}
    with open(os.path.join(args.data_dir, "items.jsonl")) as f:
        for line in f:
            it = json.loads(line)
            items[it["id"]] = it
    with open(os.path.join(args.out_dir, "reading_check.json")) as f:
        rows = json.load(f)
    with open(os.path.join(args.out_dir, "reading_cache.json")) as f:
        cache = json.load(f)

    survive = Counter()
    words: dict[str, list[int]] = {k: [] for k in KINDS}
    for r in rows:
        lo, hi = (items[i] for i in r["items"])
        key = f"{args.judge}|{r['items'][0]}|{r['items'][1]}|{'-'.join(r['pair'])}|0"
        p = (cache.get(key) or {}).get("parsed") or {}
        # The rung where gold HAS the edge is the lower one for removals, the
        # higher one for additions; test survival in the other rung.
        # (Unflipped order: text_1 is the lower rung, text_2 the higher.)
        if r["kind"] == "added":
            has, other = "text_2", lo
        else:
            has, other = "text_1", hi
        ev = norm((p.get(has) or {}).get("evidence") or "")
        if not ev:
            survive[(r["kind"], "no_evidence")] += 1
        else:
            survive[(r["kind"], "verbatim" if ev in norm(other["response"]) else "gone")] += 1
        sm = difflib.SequenceMatcher(None, lo["response"].split(), hi["response"].split())
        words[r["kind"]].append(
            sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        )

    print(f"evidence = {args.judge}'s quote from the rung where gold has the edge")
    print(f"{'kind':18s} {'n':>4s} {'verbatim':>9s} {'gone':>6s} {'no-ev':>6s} | words changed: median [min-max]")
    for k in KINDS:
        n = sum(v for (kk, _), v in survive.items() if kk == k)
        if not n:
            continue
        w = words[k]
        print(f"{k:18s} {n:4d} {survive[(k, 'verbatim')]:9d} {survive[(k, 'gone')]:6d} "
              f"{survive[(k, 'no_evidence')]:6d} | {statistics.median(w):.0f} [{min(w)}-{max(w)}]")
    with open(os.path.join(args.out_dir, "evidence_check.json"), "w") as f:
        json.dump({"survival": {f"{k}|{v}": n for (k, v), n in survive.items()},
                   "words_changed": words}, f, indent=1)


if __name__ == "__main__":
    main()
