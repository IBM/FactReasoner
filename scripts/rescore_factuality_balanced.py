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

"""Score the headline readout (f) on the cached factuality-corpus graphs.

Only runs whose records STORE their mined relation graph can be rescored: the
coherence-only (uniform-prior) runs ``fc_*_nonepriors.jsonl``. The two-stage runs
``fc_*_direct.jsonl`` keep only scores, and their graphs were mined separately and
differ item by item from the uniform-prior ones (checked: 18-27 of 250 items per arm
even share a relation count), so they cannot be rescored and are not.

Each graph is rebuilt from its stored relations and priors and scored by Merlin
(responses run to 50 claims, beyond exact enumeration). Before any (f) value is
used, the rebuilt (a) mean marginal and (d) log-partition must reproduce the cached
values, so (f) is known to be scored on the same networks.

Reports, per miner x policy, the statistics of ``tab:fc-ablation``: ``d`` (distinct
values over the items, at four decimals), mean, sd, and Spearman rho against the
stage-one factuality score, which comes from the matching two-stage run.

Run::

    python scripts/rescore_factuality_balanced.py --merlin-path /path/to/merlin
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MINERS = ("llama-3.3-70b-instruct", "gpt-oss-120b-a100")
POLICIES = ("windowed", "bidirectional")

#: Merlin prints log Z to 5-6 decimals (see rescore_ladder_conflict_escape.py), and
#: these graphs are large and often saturated, so (a)/(d) are compared at 1e-4.
REPRODUCE_TOL = 1e-4


def _rescore(job: tuple[dict, str, str]) -> dict:
    from fact_reasoner.core.base import Atom
    from fact_reasoner.fact_graph import Edge, FactGraph, Node
    from fact_reasoner.lcs.lcs_scorer import LCSScorer
    from fact_reasoner.lcs.relation_miner import MiningResult

    rec, policy, merlin = job
    block = rec[f"lcs_{policy}"]
    priors = rec["priors"]
    ids = sorted(priors, key=lambda a: int(a.lstrip("a")))
    fg = FactGraph()
    for aid in ids:
        fg.add_node(Node(id=aid, type="atom", probability=float(priors[aid])))
    for r in block["relations"]:
        fg.add_edge(
            Edge(
                source=r["source_id"],
                target=r["target_id"],
                type=r["level1_type"],
                probability=float(r["probability"]),
                link="atom_atom",
            )
        )
    result = MiningResult(
        atoms={aid: Atom(id=aid, text=aid) for aid in ids},
        relations=[],
        fact_graph=fg,
        markov_network=None,
        config={"prior": 0.5},
    )
    s = LCSScorer(merlin).score_all(
        result,
        methods=("mean_marginal", "log_partition", "balanced"),
        node_priors=priors,
    )
    return {
        "item_id": rec["item_id"],
        "dataset": rec["dataset"],
        "policy": policy,
        "cached": block["scores"],
        "rescored": {k: s[k] for k in ("mean_marginal", "log_partition", "balanced")},
    }


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    return st.correlation(rx, ry)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--merlin-path", required=True)
    ap.add_argument(
        "--results", default=os.path.join(REPO, "results", "factuality_coherence")
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--tol", type=float, default=REPRODUCE_TOL)
    ap.add_argument("--out", default=None, help="Write per-item (f) values here.")
    args = ap.parse_args()

    out: dict = {}
    failed = False
    for miner in MINERS:
        path = os.path.join(args.results, f"fc_{miner}_direct_nonepriors.jsonl")
        with open(path) as fh:
            recs = [json.loads(line) for line in fh]
        with open(os.path.join(args.results, f"fc_{miner}_direct.jsonl")) as fh:
            fact = {
                (r := json.loads(line))["item_id"]: (r.get("factuality") or {}).get(
                    "factuality_score"
                )
                for line in fh
            }
        jobs = [(r, p, args.merlin_path) for r in recs for p in POLICIES
                if f"lcs_{p}" in r]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_rescore, jobs, chunksize=2))

        bad = []
        for row in rows:
            for key in ("mean_marginal", "log_partition"):
                c, n = row["cached"].get(key), row["rescored"][key]
                if c is None or n is None or abs(c - n) > args.tol:
                    bad.append(f"{row['item_id']} {row['policy']} {key}: {c} vs {n}")
        if bad:
            failed = True
            print(f"{miner}: {len(bad)} cached readouts NOT reproduced, e.g.",
                  file=sys.stderr)
            for b in bad[:5]:
                print("   " + b, file=sys.stderr)
            continue
        print(f"{miner}: (a) and (d) reproduce the cache on all {len(rows)} "
              f"item-policy cells (tol {args.tol:g})")

        for policy in POLICIES:
            cell = [r for r in rows if r["policy"] == policy]
            for readout in ("mean_marginal", "balanced"):
                v = [r["rescored"][readout] for r in cell]
                pairs = [(x, fact[r["item_id"]]) for x, r in zip(v, cell)
                         if fact.get(r["item_id"]) is not None]
                rho = _spearman([a for a, _ in pairs], [b for _, b in pairs])
                d = len({round(x, 4) for x in v})
                print(f"   {policy:13s} {readout:14s} d={d:3d}  mean={st.mean(v):.4f}  "
                      f"sd={st.stdev(v):.4f}  rho_fact={rho:+.3f}  (n={len(v)})")
            out.setdefault(miner, {})[policy] = {
                r["item_id"]: r["rescored"]["balanced"] for r in cell
            }

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
