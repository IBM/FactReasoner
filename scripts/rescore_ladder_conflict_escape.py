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

"""Score the conflict-escape readout (e) on cached LoCoBench ladder records.

Each locoeval record stores the relation graph it was scored on (relations with
their final, concession-discounted weights) and the per-atom priors. This script
rebuilds each record's MRF from exactly those, scores every readout with the
exact brute-force oracle, and

1. **checks** that the rebuilt (a) mean marginal and (d) log-partition reproduce
   the cached values -- the guarantee that (e) is scored on the same networks the
   paper's other readouts were; and
2. **reports** (e) on the two readout-independent axes of ``tab:summary`` /
   ``tab:summary-mined`` (``scripts/report_mined_ladder_lcs.py``): Increase (C1 +
   C3 pairs on CONFLICT/CHAIN families) and Invariance (all declared pairs on
   ORDER/CONTROL families), next to (a), (d) and (b).

No LLM calls, no re-mining, no Merlin. Records with more atoms than the oracle
enumerates are skipped and counted (none in the shipped corpora).

Run::

    python scripts/rescore_ladder_conflict_escape.py
    python scripts/rescore_ladder_conflict_escape.py \
        --results results/locobench_opus_5_5_mined
    python scripts/rescore_ladder_conflict_escape.py --latex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from report_mined_ladder_lcs import (  # noqa: E402
    INCREASE_FAMILIES,
    INVARIANCE_FAMILIES,
    TIE_TOLERANCE,
    _all_pairs,
    _increase_pairs,
)

#: Readouts compared on the axes, (e) last.
READOUTS = (
    "mean_marginal", "log_partition", "consistency", "conflict_escape", "balanced",
)

#: Rebuilt readouts must match the cache to this tolerance by default. Right for a
#: cache scored by exact enumeration (the 70-item corpus). A cache scored by Merlin
#: needs ``--tol 2e-5``: Merlin prints log Z to 5-6 decimals, which on a
#: near-saturated graph moves the normalized log-partition by up to ~1e-5.
REPRODUCE_TOL = 1e-6

READOUT_TEX = {
    "mean_marginal": r"$\LCS_{\mm}$",
    "consistency": r"$\LCS_{\cons}$",
    "log_partition": r"$\LCS_{\lp}$",
    "conflict_escape": r"$\LCS_{\esc}$",
    "balanced": r"$\LCS_{\bal}$",
}


#: Penalty maps p -> w. Thm. thm-escape holds for any map that is non-decreasing in
#: p with values in [0, 1). ``linear`` is the scorer's default; ``threshold`` is the
#: earlier map, kept only to reproduce the ablation row of tab:escape-ladder.
PENALTY_MAPS = {
    "linear": None,  # w = p, as shipped in lcs_scorer._penalty_weight
    "threshold": lambda p: min(1.0 - 1e-6, max(0.0, 2.0 * p - 1.0)),  # max(0, 2p-1)
}


def _rescore(job: tuple[str, str]) -> dict:
    """Rebuild one record's MRF and score (a), (d), (e) with the exact oracle."""
    path, penalty = job
    from fact_reasoner.core.base import Atom
    from fact_reasoner.experiments.mock import (
        MAX_BRUTEFORCE_VARS,
        brute_force_run_merlin,
    )
    from fact_reasoner.fact_graph import Edge, FactGraph, Node
    from fact_reasoner.lcs import lcs_scorer as mod
    from fact_reasoner.lcs.relation_miner import MiningResult

    mod.run_merlin = brute_force_run_merlin
    mod.DEFAULT_MAX_NETWORK_VARS = MAX_BRUTEFORCE_VARS
    if PENALTY_MAPS[penalty] is not None:
        mod._penalty_weight = PENALTY_MAPS[penalty]

    with open(path) as fh:
        rec = json.load(fh)
    out = {
        "family_id": rec.get("family_id"),
        "arm": rec.get("arm"),
        "rung": rec.get("rung_index"),
        "cached": rec.get("lcs") or {},
    }
    priors = rec.get("node_priors") or {}
    atom_ids = sorted(priors, key=lambda a: int(a.lstrip("a")))
    if not atom_ids:
        out["skipped"] = "no atoms"
        return out
    if len(atom_ids) > MAX_BRUTEFORCE_VARS:
        out["skipped"] = f"{len(atom_ids)} atoms"
        return out

    fg = FactGraph()
    for aid in atom_ids:
        fg.add_node(Node(id=aid, type="atom", probability=float(priors[aid])))
    for r in rec.get("relations") or []:
        fg.add_edge(
            Edge(
                source=r["source"],
                target=r["target"],
                type=r["type"],
                probability=float(r["probability"]),
                link="atom_atom",
            )
        )
    result = MiningResult(
        atoms={aid: Atom(id=aid, text=aid) for aid in atom_ids},
        relations=[],
        fact_graph=fg,
        markov_network=None,
        config={"prior": 0.5},
    )
    scorer = mod.LCSScorer("/oracle")
    s = scorer.score_all(
        result,
        methods=("mean_marginal", "log_partition", "conflict_escape", "balanced"),
        node_priors=priors,
    )
    out["rescored"] = {
        k: s[k]
        for k in ("mean_marginal", "log_partition", "conflict_escape", "balanced")
    }
    return out


def _load(
    results_dir: str, workers: int, penalty: str
) -> tuple[list[dict], list[dict]]:
    with open(os.path.join(results_dir, "results.json")) as fh:
        families = json.load(fh)["families"]
    paths = sorted(glob.glob(os.path.join(results_dir, "records", "*.json")))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_rescore, [(p, penalty) for p in paths], chunksize=4))
    return families, rows


def _merge(
    families: list[dict], rows: list[dict], tol: float = REPRODUCE_TOL
) -> list[str]:
    """Write (e) into each arm's ``scores_by_rung``; return reproduction errors."""
    by_fam = {f["family_id"]: f for f in families}
    errors = []
    for row in rows:
        if "rescored" not in row:
            continue
        fam = by_fam.get(row["family_id"])
        arm = (fam or {}).get("arms", {}).get(row["arm"])
        if arm is None:
            continue
        cell = arm["scores_by_rung"].setdefault(str(row["rung"]), {})
        for key in ("mean_marginal", "log_partition"):
            cached, new = row["cached"].get(key), row["rescored"][key]
            if cached is None or new is None or abs(cached - new) > tol:
                errors.append(
                    f"{row['family_id']} {row['arm']} r{row['rung']} {key}: "
                    f"cached {cached} vs rescored {new}"
                )
        cell["conflict_escape"] = row["rescored"]["conflict_escape"]
        cell["balanced"] = row["rescored"]["balanced"]
    return errors


def _axes(families: list[dict], arm_name: str, readout: str) -> dict:
    """Increase / invariance counts, identical to report_mined_ladder_lcs.score_arm.

    Reimplemented only to read the merged ``scores_by_rung``; the pair sets come
    from the same ``_increase_pairs`` / ``_all_pairs``.
    """
    inc_p = inc_t = inv_p = inv_t = 0
    for f in families:
        arm = (f.get("arms") or {}).get(arm_name)
        if not arm:
            continue
        kind = str(f.get("family") or "").upper()
        by_rung = arm.get("scores_by_rung") or {}

        def value(r: int):
            return (by_rung.get(str(r)) or {}).get(readout)

        if kind in INCREASE_FAMILIES:
            for lo_i, hi_i in _increase_pairs(arm):
                inc_t += 1
                lo, hi = value(lo_i), value(hi_i)
                if lo is not None and hi is not None and hi - lo > TIE_TOLERANCE:
                    inc_p += 1
        elif kind in INVARIANCE_FAMILIES:
            for lo_i, hi_i in _all_pairs(arm):
                inv_t += 1
                lo, hi = value(lo_i), value(hi_i)
                if lo is not None and hi is not None and abs(hi - lo) <= TIE_TOLERANCE:
                    inv_p += 1
    return {"inc": (inc_p, inc_t), "inv": (inv_p, inv_t)}


def _non_decrease(families: list[dict], arm_name: str, readout: str) -> tuple[int, int]:
    """Increase-axis pairs where the readout does not fall (weak monotonicity)."""
    ok = tot = 0
    for f in families:
        arm = (f.get("arms") or {}).get(arm_name)
        if not arm or str(f.get("family") or "").upper() not in INCREASE_FAMILIES:
            continue
        by_rung = arm.get("scores_by_rung") or {}
        for lo_i, hi_i in _increase_pairs(arm):
            lo = (by_rung.get(str(lo_i)) or {}).get(readout)
            hi = (by_rung.get(str(hi_i)) or {}).get(readout)
            tot += 1
            if lo is not None and hi is not None and hi - lo >= -TIE_TOLERANCE:
                ok += 1
    return ok, tot


def _pct(p: int, t: int) -> str:
    return f"{100.0 * p / t:.1f}" if t else "--"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results",
        default=os.path.join(REPO, "results", "locobench_claude_5_v3_mined"),
        help="A locoeval results directory (results.json + records/).",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument(
        "--tol", type=float, default=REPRODUCE_TOL,
        help="Reproduction tolerance for (a) and (d) against the cache "
        f"(default {REPRODUCE_TOL:g}; use 2e-5 for a Merlin-scored cache).",
    )
    ap.add_argument(
        "--penalty", choices=sorted(PENALTY_MAPS), default="linear",
        help="Conflict penalty map for (e) and (f): linear w=p (the scorer's "
        "default) or threshold w=max(0,2p-1) (ablation). The threshold map also "
        "changes (f)'s penalties, since the two share _penalty_weight.",
    )
    ap.add_argument("--latex", action="store_true", help="Emit LaTeX table rows.")
    ap.add_argument(
        "--out", default=None, help="Write the merged per-rung (e) scores here."
    )
    args = ap.parse_args()

    families, rows = _load(args.results, args.workers, args.penalty)
    print(f"penalty map: {args.penalty}")
    skipped = [r for r in rows if "skipped" in r]
    errors = _merge(families, rows, args.tol)
    print(f"records: {len(rows)} rescored, {len(skipped)} skipped")
    if errors:
        print(f"ERROR: {len(errors)} cached readouts not reproduced, e.g.:",
              file=sys.stderr)
        for e in errors[:10]:
            print("  " + e, file=sys.stderr)
        return 1
    print(f"reproduction: (a) and (d) match the cache on every record "
          f"(tol {args.tol:g})\n")

    arms = list(families[0].get("arms") or {})
    print(f"{'arm':46s} {'readout':16s} {'increase':>13s} {'invariance':>12s} "
          f"{'no-fall':>11s}")
    table = {}
    for arm in arms:
        for r in READOUTS:
            ax = _axes(families, arm, r)
            nd = _non_decrease(families, arm, r)
            table[(arm, r)] = (ax, nd)
            (ip, it), (vp, vt) = ax["inc"], ax["inv"]
            print(f"{arm:46s} {r:16s} {ip:4d}/{it:<3d} {_pct(ip, it):>5s} "
                  f"{vp:3d}/{vt:<3d} {_pct(vp, vt):>5s} {nd[0]:4d}/{nd[1]:<3d}")
        print()

    if args.out:
        merged = {
            f["family_id"]: {
                arm: {
                    k: {"conflict_escape": v.get("conflict_escape"),
                        "balanced": v.get("balanced")}
                    for k, v in a["scores_by_rung"].items()
                }
                for arm, a in f["arms"].items()
            }
            for f in families
        }
        with open(args.out, "w") as fh:
            json.dump(merged, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")

    if args.latex:
        for arm in arms:
            for r in READOUTS:
                (ip, it), (vp, vt) = table[(arm, r)][0]["inc"], table[(arm, r)][0]["inv"]
                print(f"{READOUT_TEX[r]} ({arm}) & ${ip}/{it}$ \\quad {_pct(ip, it)} "
                      f"& ${vp}/{vt}$ \\quad {_pct(vp, vt)} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
