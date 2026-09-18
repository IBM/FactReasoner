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

Both axes of the paper's ``tab:summary-mined`` are computed here, on the same
definitions ``scripts/report_mined_ladder_lcs.py`` uses for the LCS arms, so the
baseline rows and the LCS rows of that table are commensurable:

* **Increase** -- the C1 consecutive-increase pairs plus the C3 endpoint separation
  over the *increase-type* families (\textsc{conflict}, \textsc{chain}), deduplicated
  across readouts. C2 is excluded: it predicts the internal behaviour of a particular
  readout, which a single-score baseline makes no claim about. That is 50 assertions.
* **Invariance** -- every rung pair of the *invariance-type* families
  (\textsc{order}, \textsc{control}), satisfied when the score does not move. That is
  20 pairs.

A baseline is run once per judge backend, so BOTH axes are per backend; pass
``--results`` once per backend directory. ``--check`` asserts the published cells for
whichever backend is supplied, so a drift in the axis definitions is caught rather
than silently republished.

Run::

    python scripts/report_ladder_baselines.py --latex
    python scripts/report_ladder_baselines.py \
        --results results/ladder_baselines_v2_gpt-oss-120b-a100/ladder_scores.jsonl
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
#: increase, and jitter below it counts as unchanged on the invariance axis.
TIE_TOLERANCE = 1e-6

#: Ladder types asserting a monotone increase, versus those asserting invariance.
#: Matches ``scripts/report_mined_ladder_lcs.py``.
INCREASE_FAMILIES = ("CONFLICT", "CHAIN")
INVARIANCE_FAMILIES = ("ORDER", "CONTROL")

#: The published per-backend cells, keyed by the backend substring identifying the
#: run. Asserted rather than assumed: these are the numbers the paper prints, and if
#: the axis definitions here drift, every cell reported below is suspect.
PUBLISHED = {
    "llama-3.3-70b": {
        "discourse_rc": (30, 16),
        "discourse_lc": (28, 16),
        "control_length": (20, 12),
        "discourse_entity_graph": (0, 20),
        "control_claim_count": (0, 20),
        "roscoe_sc_mean": (16, 9),
        "nli_contradiction": (12, 9),
        "judge_geval": (0, 13),
        "judge_direct": (4, 14),
        "roscoe_sc": (0, 20),
    },
    "gpt-oss-120b": {
        "discourse_rc": (30, 16),
        "discourse_lc": (28, 16),
        "control_length": (20, 12),
        "discourse_entity_graph": (0, 20),
        "control_claim_count": (0, 20),
        "roscoe_sc_mean": (17, 5),
        "nli_contradiction": (12, 8),
        "judge_geval": (21, 5),
        "judge_direct": (18, 5),
        "roscoe_sc": (0, 20),
    },
}


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


def _invariance_pairs(family: dict) -> list[tuple[int, int]]:
    """Every declared rung pair of an invariance-type family, deduplicated.

    An \textsc{order} or \textsc{control} family permutes prose while holding the
    claim set fixed, so each of its rung pairs asserts that the score does not move.
    The pairs come from the family's own constraints -- C2's adjacent pairs and C3's
    ``invariant`` endpoints -- rather than from a hardcoded range, so a family with a
    different rung count is still scored correctly. Deduplicated across readouts,
    since a rung pair is one assertion rather than one per readout.
    """
    pairs: set[tuple[int, int]] = set()
    for c in family.get("ordering_constraints") or []:
        for entry in c.get("pairs") or []:
            lo, hi = entry["pair"]
            pairs.add((lo, hi))
        for pair in c.get("invariant") or []:
            pairs.add((pair[0], pair[1]))
    return sorted(pairs)


def score_baseline(
    families: dict,
    scores: dict,
    name: str,
    fids: list[str],
) -> dict:
    """Count satisfied assertions on both axes for one baseline.

    Returns the passed/total on each axis plus the per-family increase breakdown,
    so the text table, the LaTeX body and the self-check all read one computation
    rather than each repeating the arithmetic.
    """
    per_family: list[str] = []
    inc_ok = inc_n = inv_ok = inv_n = 0
    for fid in fids:
        family = families[fid]
        ftype = family.get("family")
        if ftype in INCREASE_FAMILIES:
            ok = n = 0
            for _cls, lo, hi in _assertions(family):
                a, b = scores.get((name, fid, lo)), scores.get((name, fid, hi))
                if a is None or b is None:
                    continue
                n += 1
                if b - a > TIE_TOLERANCE:
                    ok += 1
            per_family.append(f"{ok}/{n}")
            inc_ok += ok
            inc_n += n
        elif ftype in INVARIANCE_FAMILIES:
            ok = n = 0
            for lo, hi in _invariance_pairs(family):
                a, b = scores.get((name, fid, lo)), scores.get((name, fid, hi))
                if a is None or b is None:
                    continue
                n += 1
                if abs(b - a) <= TIE_TOLERANCE:
                    ok += 1
            per_family.append(f"[{ok}/{n}]")
            inv_ok += ok
            inv_n += n
        else:
            per_family.append("--")
    return {
        "increase_passed": inc_ok,
        "increase_total": inc_n,
        "invariance_passed": inv_ok,
        "invariance_total": inv_n,
        "per_family": per_family,
    }


def _pct(ok: int, n: int) -> float:
    return 100.0 * ok / n if n else 0.0


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
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the published-cell assertion (use when scoring a new backend).",
    )
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
        ftype = families[fid].get("family")
        print(f"\n  {fid} ({ftype})")
        for name in names:
            vals = [scores.get((name, fid, r)) for r in range(5)]
            cells = " ".join(
                "  n/a  " if v is None else f"{v:7.4f}" for v in vals
            )
            print(f"    {name:<26}{cells}")

    scored = {n: score_baseline(families, scores, n, fids) for n in names}

    # --- both axes ---
    print("\n\nBoth axes. Increase (C1 consecutive + C3 endpoints) on the")
    print("CONFLICT/CHAIN families; [invariance] on the ORDER/CONTROL families.")
    print("-" * 78)
    header = (
        f"  {'baseline':<26}"
        + "".join(f"{f:>10}" for f in fids)
        + f"{'increase':>12}{'invariance':>13}{'min%':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in names:
        s_ = scored[name]
        ip, it = s_["increase_passed"], s_["increase_total"]
        vp, vt = s_["invariance_passed"], s_["invariance_total"]
        lo = min(_pct(ip, it), _pct(vp, vt))
        cells = "".join(f"{c:>10}" for c in s_["per_family"])
        print(
            f"  {name:<26}{cells}"
            f"{f'{ip}/{it} ({_pct(ip, it):.0f}%)':>12}"
            f"{f'{vp}/{vt} ({_pct(vp, vt):.0f}%)':>13}"
            f"{lo:>7.0f}"
        )

    # --- self-check against the published cells ---
    backend = next(
        (b for b in PUBLISHED if b in os.path.abspath(args.results)), None
    )
    if backend is None:
        print(
            "\nNo published cells on record for this results path; skipping the "
            "self-check.",
            file=sys.stderr,
        )
    else:
        expected = PUBLISHED[backend]
        bad = []
        for name, (einc, einv) in expected.items():
            got = scored.get(name)
            if got is None:
                bad.append(f"{name}: absent from results")
                continue
            if got["increase_passed"] != einc or got["invariance_passed"] != einv:
                bad.append(
                    f"{name}: got {got['increase_passed']}/{got['invariance_passed']}"
                    f", published {einc}/{einv}"
                )
        if bad:
            print(f"\nMISMATCH against the published {backend} cells:", file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            if not args.no_check:
                print(
                    "Refusing to report: the axis definitions here no longer "
                    "reproduce the paper's table.",
                    file=sys.stderr,
                )
                return 1
        else:
            print(f"\n  -> reproduces the published {backend} cells on both axes")

    if args.latex:
        print("\n% ---- LaTeX tabular body ----")
        print("% increase / invariance, one row per baseline, for this backend.")
        for name in names:
            s_ = scored[name]
            ip, it = s_["increase_passed"], s_["increase_total"]
            vp, vt = s_["invariance_passed"], s_["invariance_total"]
            label = name.replace("_", r"\_")
            print(
                f"\\textsc{{{label}}} & ${ip}$ ({_pct(ip, it):.0f}\\%) "
                f"& ${vp}$ ({_pct(vp, vt):.0f}\\%) & "
                f"{min(_pct(ip, it), _pct(vp, vt)):.0f}" + r" \\"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
