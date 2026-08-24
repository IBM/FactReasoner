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

"""Summarize a coherence-baseline run into a table and a LaTeX fragment.

Reads the jsonl written by ``scripts/run_coherence_baselines.py`` and prints:

1. a per-fixture score matrix,
2. the declared-direction verdicts per baseline,
3. call-failure and abstention counts -- reported rather than hidden, because a
   throttled or unparseable call is a missing measurement and a reader has to know
   how many there were before trusting a column,
4. optionally, a LaTeX ``tabular`` body for pasting into the paper.

Run::

    python scripts/report_coherence_baselines.py \\
        --results results/coherence_baselines/baselines.jsonl --latex
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

#: Short labels for the paper table, matching the fixture shorthand already used in
#: the experiments section (e2-bio, e5-K, ...).
SHORT = {
    "aeroparts-recall": "aero",
    "example-1-damages": "e1-dmg",
    "example-2-biography": "e2-bio",
    "example-2-biography-contradicted": "e2-con",
    "example-3-narrative": "e3-nar",
    "example-4-summary": "e4-sum",
    "example-5-renda-K": "e5-K",
    "example-5-renda-S": "e5-S",
    "example-6-incident": "ex6",
}

#: Pairs whose relative order the fixtures declare, higher first.
DECLARED = (
    ("example-2-biography", "example-2-biography-contradicted"),
    ("example-5-renda-K", "example-5-renda-S"),
)


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--results",
        default=os.path.join(
            repo, "results", "coherence_baselines", "baselines.jsonl"
        ),
    )
    parser.add_argument(
        "--latex", action="store_true", help="Also emit a LaTeX tabular body."
    )
    args = parser.parse_args()

    rows = load(args.results)
    if not rows:
        print("No rows found.", file=sys.stderr)
        return 1

    by_key = {(r["item_id"], r["name"]): r for r in rows}
    items = sorted({r["item_id"] for r in rows})
    names = sorted({r["name"] for r in rows})

    # --- 1. score matrix ---
    print(f"\n{len(items)} fixtures x {len(names)} baselines\n")
    header = f"{'fixture':<34}{'atoms':>6}" + "".join(f"{n[:14]:>15}" for n in names)
    print(header)
    print("-" * len(header))
    for item in items:
        first = next(by_key[(item, n)] for n in names if (item, n) in by_key)
        line = f"{SHORT.get(item, item):<34}{first['num_atoms']:>6}"
        for name in names:
            r = by_key.get((item, name))
            if r is None:
                line += f"{'-':>15}"
            elif r["score"] is None:
                line += f"{'abstain':>15}"
            else:
                line += f"{r['score']:>15.4f}"
        print(line)

    # --- 2. declared-direction verdicts ---
    print("\nDeclared-direction verdicts (higher > lower)")
    print("-" * 78)
    for higher, lower in DECLARED:
        print(f"\n  {SHORT.get(higher, higher)} > {SHORT.get(lower, lower)}")
        for name in names:
            hi, lo = by_key.get((higher, name)), by_key.get((lower, name))
            if not hi or not lo or hi["score"] is None or lo["score"] is None:
                print(f"    {name:<28}n/a")
                continue
            # Compare on the raw quantity when one exists: an unbounded metric can
            # clamp two different values to 1.0 and fake a tie.
            h = hi["diagnostics"].get("raw", hi["score"])
            lo_v = lo["diagnostics"].get("raw", lo["score"])
            # Compare at reporting precision. Two scores that print identically are
            # a tie, not a separation: on the Renda pair roscoe_sc is 0.000000 for
            # both rungs, and a float difference in the 7th decimal of an
            # intermediate term would otherwise be reported as a correct ordering.
            if round(h, 6) == round(lo_v, 6):
                verdict = "FLAT"
            else:
                verdict = "OK" if h > lo_v else "WRONG"
            print(f"    {name:<28}{h:>10.4f} vs {lo_v:>10.4f}   {verdict}")

    # --- 3. failures and abstentions ---
    print("\nMeasurement health")
    print("-" * 78)
    for name in names:
        rs = [by_key[(i, name)] for i in items if (i, name) in by_key]
        abstained = sum(1 for r in rs if r["score"] is None)
        failures = sum(int(r["diagnostics"].get("call_failures", 0) or 0) for r in rs)
        pairs = sum(int(r.get("pairs_scored", 0) or 0) for r in rs)
        print(
            f"  {name:<28}items={len(rs):<4}abstained={abstained:<4}"
            f"pairs_scored={pairs:<7}call_failures={failures}"
        )

    # --- 4. LaTeX ---
    if args.latex:
        print("\n% ---- LaTeX tabular body ----")
        for item in items:
            first = next(by_key[(item, n)] for n in names if (item, n) in by_key)
            cells = []
            for name in names:
                r = by_key.get((item, name))
                if r is None or r["score"] is None:
                    cells.append("---")
                else:
                    v = r["diagnostics"].get("raw", r["score"])
                    cells.append(f"{v:.3f}")
            label = SHORT.get(item, item).replace("_", r"\_")
            print(
                f"\\texttt{{{label}}} & {first['num_atoms']} & "
                + " & ".join(cells)
                + r" \\"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
