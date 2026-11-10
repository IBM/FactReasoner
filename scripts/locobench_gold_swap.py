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

"""Gold-swap diagnostic: which class of mining error breaks the ladder checks?

Starting from each item's GOLD graph, inject ONE class of a mined arm's errors at a
time and re-score with Merlin. No LLM calls: the mined graph is read from the arm's
records. Edge weights are held at the gold band midpoint wherever a gold edge exists
(and the mined band otherwise), so weighting is not a variable here -- the
`-goldw` arms already showed it is not the bottleneck.

Variants (each applied to gold, alone):
  G      gold unchanged (reproduction check against the gold arm)
  +FP    add the mined edges whose unordered pair gold does not relate
  -FN    drop the gold edges whose unordered pair the miner did not relate
  lab    on matched pairs, take the mined sense/coupling/direction
  res    on matched pairs, take the mined `resolved` flag (concession discount)
  M      all four at once (= the mined edge set with gold weights)
The `+FP` edges are also split by coupling in the printed summary.

    python scripts/locobench_gold_swap.py \\
        --arm mined:claude-opus-5-5:graph:k5 \\
        --records results/locobench_claude_5_graph/arms/opus55/records
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from fact_reasoner.fact_graph import Edge, FactGraph, Node  # noqa: E402
from fact_reasoner.factors import build_markov_network  # noqa: E402
from fact_reasoner.lcs.lcs_scorer import LCSScorer  # noqa: E402
from fact_reasoner.lcs.relation_miner import MinedRelation, MiningResult, _atom_sort_key  # noqa: E402
from fact_reasoner.lcs.taxonomy import LEVEL1_CONFLICT_COUPLINGS, LEVEL1_NONE  # noqa: E402
from fact_reasoner.locoeval.gold_graph import (  # noqa: E402
    DEFAULT_CONCESSION_DISCOUNT,
    BAND_RANGES,
    atom_priors,
    band_probability,
    item_atoms,
)
from fact_reasoner.locoeval.runner import (  # noqa: E402
    evaluate_constraints,
    load_families,
    load_items,
)

VARIANTS = ("G", "+FP", "-FN", "lab", "res", "M")
READOUTS = ("mean_marginal", "consistency", "log_partition")
LAM = DEFAULT_CONCESSION_DISCOUNT


def upair(a, b):
    return tuple(sorted((a, b), key=_atom_sort_key))


def gold_edges(item):
    """Gold coupling-bearing edges as dicts keyed by unordered pair."""
    out = {}
    for r in item.get("relations", []) or []:
        if r.get("level1_coupling") == LEVEL1_NONE:
            continue
        out[upair(r["source_id"], r["target_id"])] = {
            "source": r["source_id"],
            "target": r["target_id"],
            "coupling": r["level1_coupling"],
            "sense": r["level2_sense"],
            "directed": bool(r.get("directed", True)),
            "resolved": bool(r.get("is_resolved_concession")),
            "w": band_probability(r),
        }
    return out


def mined_edges(rec):
    """Mined edges from a record, keyed by unordered pair, with the voted band."""
    votes = {upair(v["source_id"], v["target_id"]): v for v in rec.get("graph_vote") or []}
    bands = {k: v.get("band") for k, v in votes.items()}
    out = {}
    for r in rec.get("relations") or []:
        k = upair(r["source"], r["target"])
        lo, hi = BAND_RANGES.get(str(bands.get(k) or "moderate"), BAND_RANGES["moderate"])
        out[k] = {
            "source": r["source"],
            "target": r["target"],
            "coupling": r["type"],
            "sense": r["sense"],
            "directed": bool(r.get("directed", True)),
            # The raw voted flag, on any conflict coupling (not the record's
            # Concession-gated `concession_resolved`).
            "resolved": bool((votes.get(k) or {}).get("resolved"))
            and r["type"] in LEVEL1_CONFLICT_COUPLINGS,
            "w": (lo + hi) / 2.0,
        }
    return out


def variant_edges(g, m, variant):
    """Apply one error class to the gold edge set."""
    edges = {k: dict(v) for k, v in g.items()}
    matched = set(g) & set(m)
    if variant in ("+FP", "M"):
        for k in set(m) - set(g):
            edges[k] = dict(m[k])
    if variant in ("-FN", "M"):
        for k in set(g) - set(m):
            edges.pop(k, None)
    if variant in ("lab", "M"):
        for k in matched:
            for f in ("source", "target", "coupling", "sense", "directed"):
                edges[k][f] = m[k][f]
    if variant in ("res", "M"):
        for k in matched:
            edges[k]["resolved"] = m[k]["resolved"]
    return edges


def to_result(item, edges):
    atoms = item_atoms(item)
    priors = atom_priors(item)
    rels = []
    for e in edges.values():
        p = e["w"]
        if e["resolved"] and e["coupling"] in LEVEL1_CONFLICT_COUPLINGS:
            p = p * (1.0 - LAM)
        rels.append(
            MinedRelation(
                source_id=e["source"], target_id=e["target"], level2_sense=e["sense"],
                level1_type=e["coupling"], probability=max(1e-6, min(1 - 1e-6, p)),
                type_confidence=1.0, strength=e["w"], directed=e["directed"],
                concession_resolved=e["resolved"],
            )
        )
    fg = FactGraph()
    for aid in sorted(atoms, key=_atom_sort_key):
        fg.add_node(Node(id=aid, type="atom", probability=priors[aid]))
    for r in rels:
        fg.add_edge(Edge(source=r.source_id, target=r.target_id, type=r.level1_type,
                         probability=r.probability, link="atom_atom"))
    mn = build_markov_network(fg, use_priors=True, node_priors=priors)
    return MiningResult(atoms=atoms, relations=rels, fact_graph=fg, markov_network=mn,
                        coverage={}, config={"prior": 0.5, "node_priors": priors}), priors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/locobench-claude-5-test")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--merlin-path", default="/Users/radu/git/merlin/build_boost192/merlin")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = load_items(args.data_dir)
    families = load_families(args.data_dir)
    slug = args.arm
    for ch in (":", "/", ".", " "):
        slug = slug.replace(ch, "_")

    scorer = LCSScorer(args.merlin_path, ibound=6, verbose=False)
    scores = defaultdict(dict)  # variant -> item_id -> lcs
    err_stats = Counter()
    fp_by_coupling = Counter()
    for item in items:
        with open(os.path.join(args.records, f"{item['id']}__{slug}.json")) as f:
            rec = json.load(f)
        g, m = gold_edges(item), mined_edges(rec)
        matched = set(g) & set(m)
        err_stats["gold"] += len(g)
        err_stats["fp"] += len(set(m) - set(g))
        err_stats["fn"] += len(set(g) - set(m))
        err_stats["lab_diff"] += sum(
            1 for k in matched
            if (g[k]["coupling"], g[k]["sense"]) != (m[k]["coupling"], m[k]["sense"])
            or (g[k]["directed"] and (g[k]["source"], g[k]["target"]) != (m[k]["source"], m[k]["target"]))
        )
        err_stats["res_diff"] += sum(1 for k in matched if g[k]["resolved"] != m[k]["resolved"])
        err_stats["gold_resolved"] += sum(1 for k in g if g[k]["resolved"])
        for k in set(m) - set(g):
            fp_by_coupling[m[k]["coupling"]] += 1
        for v in VARIANTS:
            res, priors = to_result(item, variant_edges(g, m, v))
            s = scorer.score_all(res, methods=["mean_marginal", "consistency", "reified", "log_partition"],
                                 reified_prior=0.5, node_priors=priors)
            scores[v][item["id"]] = {k: s.get(k) for k in READOUTS}

    by_fam = defaultdict(dict)
    for item in items:
        e = item.get("expected") or {}
        by_fam[e["family_id"]][int(e["rung_index"])] = item["id"]

    table = {}
    for v in VARIANTS:
        tally = defaultdict(lambda: [0, 0])
        by_class = defaultdict(lambda: [0, 0])
        for fid, rungs in by_fam.items():
            fam = families.get(fid)
            if fam is None:
                continue
            sbr = {r: scores[v][iid] for r, iid in rungs.items()}
            for c in evaluate_constraints(fam, sbr):
                tally[c["readout"]][0] += c["passed"]
                tally[c["readout"]][1] += 1
                by_class[c["constraint_class"]][0] += c["passed"]
                by_class[c["constraint_class"]][1] += 1
        table[v] = {"readouts": dict(tally), "classes": dict(by_class)}

    print(f"arm: {args.arm}")
    print(f"edge errors vs gold ({err_stats['gold']} gold edges): FP={err_stats['fp']} "
          f"FN={err_stats['fn']} label-diff={err_stats['lab_diff']} "
          f"resolved-diff={err_stats['res_diff']} (gold resolved={err_stats['gold_resolved']})")
    print(f"FP edges by coupling: {dict(fp_by_coupling)}")
    classes = sorted({c for v in table.values() for c in v["classes"]})
    print(f"{'variant':8s} " + " ".join(f"{r[:13]:>13s}" for r in READOUTS) + " | "
          + " ".join(f"{c:>9s}" for c in classes))
    for v in VARIANTS:
        t = table[v]
        cells = [f"{t['readouts'][r][0]:>3d}/{t['readouts'][r][1]:<3d}={t['readouts'][r][0]/t['readouts'][r][1]:.2f}"
                 for r in READOUTS]
        cl = [f"{t['classes'][c][0]:>3d}/{t['classes'][c][1]:<3d}" for c in classes]
        print(f"{v:8s} " + " ".join(f"{x:>13s}" for x in cells) + " | " + " ".join(f"{x:>9s}" for x in cl))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"arm": args.arm, "errors": err_stats, "fp_by_coupling": fp_by_coupling,
                       "table": table, "scores": scores}, f, indent=2)


if __name__ == "__main__":
    main()
