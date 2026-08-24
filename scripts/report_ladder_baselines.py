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

"""Score ladder-baseline results against the declared rung orderings.

Reads ``ladder_scores.jsonl`` from ``scripts/run_ladder_baselines.py`` and reports,
per baseline: the per-rung scores, which declared assertions it satisfies, and the
fraction satisfied. See that script's docstring for why only C1 and C3 apply to a
single-score baseline.

Run::

    python scripts/report_ladder_baselines.py --latex
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

#: Scores closer than this count as flat, so float noise cannot pass for a strict
#: increase.
TIE_TOLERANCE = 1e-6


def _assertions(family: dict) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for c in family.get("ordering_constraints") or []:
        if c.get("class") == "C1":
            for pair in sorted({tuple(p["pair"]) for p in c.get("pairs", [])}):
                out.append(("C1", pair[0], pair[1]))
        elif c.get("class") == "C3":
            for pair in c.get("required") or []:
                out.append(("C3", pair[0], pair[1]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--results",
        default=os.path.join(
            repo, "results", "ladder_baselines", "ladder_scores.jsonl"
        ),
    )
    parser.add_argument(
        "--data-dir", default=os.path.join(repo, "data", "locobench-claude-5-test")
    )
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    with open(args.results) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        print("No rows.", file=sys.stderr)
        return 1
    with open(os.path.join(args.data_dir, "families.json")) as f:
        families = {fam["family_id"]: fam for fam in json.load(f)["families"]}

    scores: dict[tuple[str, str, int], float | None] = {}
    for r in rows:
        scores[(r["name"], r["family_id"], r["rung"])] = r["score"]
    names = sorted({r["name"] for r in rows})
    fids = sorted({r["family_id"] for r in rows})

    # --- per-rung scores ---
    print("\nPer-rung scores (rungs 0..4, least to most coherent as declared)")
    print("-" * 78)
    for fid in fids:
        print(f"\n  {fid}")
        for name in names:
            vals = [scores.get((name, fid, r)) for r in range(5)]
            cells = " ".join(
                "  n/a  " if v is None else f"{v:7.4f}" for v in vals
            )
            print(f"    {name:<26}{cells}")

    # --- constraint satisfaction ---
    print("\n\nDeclared assertions satisfied (C1 consecutive increase, C3 endpoints)")
    print("-" * 78)
    header = (
        f"  {'baseline':<26}"
        + "".join(f"{f:>10}" for f in fids)
        + f"{'total':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    totals: dict[str, tuple[int, int]] = {}
    for name in names:
        per_family, tot_ok, tot_n = [], 0, 0
        for fid in fids:
            asserts = _assertions(families[fid])
            ok = n = 0
            for _cls, lo, hi in asserts:
                a, b = scores.get((name, fid, lo)), scores.get((name, fid, hi))
                if a is None or b is None:
                    continue
                n += 1
                if b - a > TIE_TOLERANCE:
                    ok += 1
            per_family.append(f"{ok}/{n}")
            tot_ok += ok
            tot_n += n
        totals[name] = (tot_ok, tot_n)
        pct = 100.0 * tot_ok / tot_n if tot_n else 0.0
        cells = "".join(f"{c:>10}" for c in per_family)
        print(f"  {name:<26}{cells}{f'{tot_ok}/{tot_n} ({pct:.1f}%)':>12}")

    if args.latex:
        print("\n% ---- LaTeX tabular body ----")
        for name in names:
            ok, n = totals[name]
            pct = 100.0 * ok / n if n else 0.0
            per = []
            for fid in fids:
                asserts = _assertions(families[fid])
                o = k = 0
                for _c, lo, hi in asserts:
                    a, b = scores.get((name, fid, lo)), scores.get((name, fid, hi))
                    if a is None or b is None:
                        continue
                    k += 1
                    if b - a > TIE_TOLERANCE:
                        o += 1
                per.append(f"${o}/{k}$")
            label = name.replace("_", r"\_")
            print(
                f"\\textsc{{{label}}} & " + " & ".join(per) + f" & ${pct:.1f}$" + r" \\"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
