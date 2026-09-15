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

"""Score the MINED ladder arms on the two axes the baseline tables use.

The paper's baseline-comparison tables (``tab:ladder-baselines``,
``tab:summary``) score the LCS on GOLD relations only, over two restricted
axes that a single-score baseline can also be held to:

* **Increase** -- the readout-independent assertions: the C1 consecutive-increase
  pairs plus the C3 endpoint separation, on the *increase-type* families
  (\\textsc{conflict}, \\textsc{chain}) only. C2 is excluded because it predicts the
  internal behaviour of a particular readout, which a baseline makes no claim about.
  That is 50 assertions over ten families.
* **Invariance** -- the rung pairs of the *invariance-type* families
  (\\textsc{order}, \\textsc{control}), satisfied when the score does not move.
  That is 20 pairs over four families.

This script computes the same two axes for the MINED arms, from the cached
``results.json``, so the mined graphs can be placed on exactly the axes the
baselines are judged on. It reads only cached scores: no LLM calls, no re-mining.

The gold rows it prints must reproduce the paper's published gold numbers
(37/50, 36/50, 34/50 increase; 20/20 invariance); that agreement is the check that
this reimplementation of the axes is faithful, so it is asserted rather than assumed.

Run::

    python scripts/report_mined_ladder_lcs.py
    python scripts/report_mined_ladder_lcs.py --latex
"""

from __future__ import annotations

import argparse
import json
import os
import sys

#: Scores closer than this count as unchanged, so float noise cannot read as a
#: strict increase (matches scripts/report_ladder_baselines.py).
TIE_TOLERANCE = 1e-6

#: The readouts the paper reports on these axes, in its own order.
READOUTS = ("mean_marginal", "log_partition", "consistency")

#: Ladder types asserting a monotone increase, versus those asserting invariance.
INCREASE_FAMILIES = ("CONFLICT", "CHAIN")
INVARIANCE_FAMILIES = ("ORDER", "CONTROL")

READOUT_TEX = {
    "mean_marginal": r"$\LCS_{\mm}$",
    "consistency": r"$\LCS_{\cons}$",
    "log_partition": r"$\LCS_{\lp}$",
    "reified": r"$\LCS_{\rei}$",
}


def _increase_pairs(arm: dict) -> list[tuple[int, int]]:
    """The readout-independent (C1, C3) rung pairs declared for one family.

    Taken from the arm's own ``checks`` so the pair set is exactly what the runner
    declared. Deduplicated across readouts: a pair is one assertion, not one per
    readout, which is what makes the axis readout-independent and therefore
    comparable to a single-score baseline.
    """
    pairs: set[tuple[int, int]] = set()
    for c in arm.get("checks") or []:
        if c.get("constraint_class") in ("C1", "C3"):
            p = c.get("pair")
            if p and len(p) == 2:
                pairs.add((int(p[0]), int(p[1])))
    return sorted(pairs)


def _all_pairs(arm: dict) -> list[tuple[int, int]]:
    """Every declared rung pair, for the invariance axis."""
    pairs: set[tuple[int, int]] = set()
    for c in arm.get("checks") or []:
        p = c.get("pair")
        if p and len(p) == 2:
            pairs.add((int(p[0]), int(p[1])))
    return sorted(pairs)


def score_arm(families: list[dict], arm_name: str, readout: str) -> dict:
    """Count satisfied assertions on both axes for one (arm, readout).

    Returns ``{increase_passed, increase_total, invariance_passed,
    invariance_total}``. A missing score fails rather than passes, matching the
    paper's protocol.
    """
    inc_p = inc_t = inv_p = inv_t = 0
    for f in families:
        arm = (f.get("arms") or {}).get(arm_name)
        if not arm:
            continue
        kind = str(f.get("family") or "").upper()
        by_rung = arm.get("scores_by_rung") or {}

        def value(rung: int):
            row = by_rung.get(str(rung)) or by_rung.get(rung) or {}
            return row.get(readout)

        if kind in INCREASE_FAMILIES:
            for lo_i, hi_i in _increase_pairs(arm):
                inc_t += 1
                lo, hi = value(lo_i), value(hi_i)
                if lo is None or hi is None:
                    continue  # a missing score fails
                if hi - lo > TIE_TOLERANCE:
                    inc_p += 1
        elif kind in INVARIANCE_FAMILIES:
            for lo_i, hi_i in _all_pairs(arm):
                inv_t += 1
                lo, hi = value(lo_i), value(hi_i)
                if lo is None or hi is None:
                    continue
                if abs(hi - lo) <= TIE_TOLERANCE:
                    inv_p += 1
    return {
        "increase_passed": inc_p, "increase_total": inc_t,
        "invariance_passed": inv_p, "invariance_total": inv_t,
    }


def mean_readout(families: list[dict], arm_name: str, readout: str) -> float | None:
    """Mean readout value over every rung of every family, for one arm."""
    vals = []
    for f in families:
        arm = (f.get("arms") or {}).get(arm_name)
        if not arm:
            continue
        for row in (arm.get("scores_by_rung") or {}).values():
            v = row.get(readout)
            if v is not None:
                vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def _pct(p: int, t: int) -> str:
    return f"{100.0 * p / t:.1f}" if t else "--"


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(
        repo, "results", "locobench_claude_5_v3_mined", "results.json"))
    ap.add_argument("--latex", action="store_true", help="Emit LaTeX table bodies.")
    ap.add_argument("--no-check", action="store_true",
                    help="Skip the gold-reproduction assertion.")
    args = ap.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)
    families = data["families"]
    arms = [a for a in (families[0].get("arms") or {})]
    mined = [a for a in arms if a.startswith("mined:")]

    # Validate the axes against the paper's published gold numbers before using
    # them for anything. If this drifts, every mined number below is suspect.
    expected_gold = {"mean_marginal": 37, "log_partition": 36, "consistency": 34}
    gold = {r: score_arm(families, "gold", r) for r in READOUTS}
    ok = all(gold[r]["increase_passed"] == expected_gold[r] for r in READOUTS)
    print("Gold reproduction check (must match the paper's 37/36/34 of 50):")
    for r in READOUTS:
        g = gold[r]
        print(f"  {r:14s} increase {g['increase_passed']}/{g['increase_total']}"
              f"  invariance {g['invariance_passed']}/{g['invariance_total']}")
    if not ok and not args.no_check:
        print("\nERROR: gold rows do not reproduce the published numbers; the axis "
              "definition here does not match the paper's. Refusing to report mined "
              "numbers computed on a different basis.", file=sys.stderr)
        return 1
    print("  -> axes reproduce the paper's gold numbers\n")

    rows = []
    for arm in ["gold", "gold_valid", *mined]:
        for r in READOUTS:
            s = score_arm(families, arm, r)
            rows.append((arm, r, s, mean_readout(families, arm, r)))

    print(f"{'arm':46s} {'readout':14s} {'increase':>14s} {'invariance':>13s} {'min%':>6s} {'mean':>7s}")
    for arm, r, s, mv in rows:
        ip, it = s["increase_passed"], s["increase_total"]
        vp, vt = s["invariance_passed"], s["invariance_total"]
        mn = min(100.0 * ip / it if it else 0.0, 100.0 * vp / vt if vt else 0.0)
        print(f"{arm:46s} {r:14s} {ip:5d}/{it:<3d} {_pct(ip,it):>5s} "
              f"{vp:4d}/{vt:<3d} {_pct(vp,vt):>5s} {mn:6.1f} "
              f"{(f'{mv:.3f}' if mv is not None else '--'):>7s}")

    if args.latex:
        _emit_latex(families, mined)
    return 0


def _emit_latex(families: list[dict], mined: list[str]) -> None:
    """Emit the two table bodies, mirroring tab:ladder-baselines / tab:summary."""
    def label(arm: str) -> str:
        if arm == "gold":
            return "gold relations"
        if arm == "gold_valid":
            return "gold, valid only"
        _, model, policy = arm.split(":")
        return f"\\texttt{{{model}}}, \\textsc{{{policy}}}"

    print("\n%% ---- generated by scripts/report_mined_ladder_lcs.py --latex ----")
    print("%% Body for the mined analogue of tab:ladder-baselines (increase axis).")
    for arm in ["gold", *mined]:
        for r in READOUTS:
            s = score_arm(families, arm, r)
            ip, it = s["increase_passed"], s["increase_total"]
            print(f"{READOUT_TEX[r]} ({label(arm)}) & ${ip}/{it}$ \\quad "
                  f"{_pct(ip, it)} \\\\")
    print("\n%% Body for the mined analogue of tab:summary (both axes).")
    for arm in ["gold", *mined]:
        for r in READOUTS:
            s = score_arm(families, arm, r)
            ip, it = s["increase_passed"], s["increase_total"]
            vp, vt = s["invariance_passed"], s["invariance_total"]
            mn = min(100.0 * ip / it if it else 0.0, 100.0 * vp / vt if vt else 0.0)
            print(f"{READOUT_TEX[r]} ({label(arm)}) & ${ip}$ \\quad {_pct(ip,it)} "
                  f"& ${vp}$ \\quad {_pct(vp,vt)} & {mn:.0f} \\\\")


if __name__ == "__main__":
    raise SystemExit(main())
