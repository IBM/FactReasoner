"""Where the normalization Z matters: LCS against the signed weight sum (coherence paper,
review C1, Phase 3.1).

On LoCoBench's ladders the signed weight sum does as well as LCS, because both are monotone in
the coherence order and the ladders change one or two relations at a time. The two scores can
differ only on edits the order leaves unranked. This script builds three such edits on every
gold graph of the test split (verdict world, flat priors) and checks a principle each:

A. Presentation invariance (A2). The same constraint, stated as one relation or as its two
   halves, defines the same MRF and should get the same score:
     A1. an equivalence a<=>b versus the two entailments a=>b and b=>a (two true claims);
     A2. an exclusion versus a contradiction plus a co-necessity (a true and a false claim).
B. Repetition. Repeating one inference k=3 times should count for less than three distinct
   inferences (three new entailments between true claims that no relation joins).
C. Premise doubt. An inference from a premise that the argument itself contests (a conflict
   between the premise and another true claim) should count for less than the same inference
   from an uncontested premise.

The signed sum fails A and B by construction and is indifferent in C; the point is to measure
how far LCS separates the cases on realistic graphs, not to discover that it does.

    .venv/bin/python scripts/arr_z_stress.py --macros docs/arr2027/coherence/numbers_zstress.tex
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import statistics as st
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import arr_coherence_eval as A  # noqa: E402
import arr_slot3_eval as E  # noqa: E402

P = 0.925  # the strong band's midpoint, the weight of most gold relations
SEED = 20261127
EQ = 1e-9  # relative tolerance for "the same score"


def lam(n, rels, x):
    return A.confirmation(np.full(n, 0.5), rels, x)


def wsum(rels, x):
    return sum(p * (1 if A.is_support((s, t, ty, p), x) else -1) for s, t, ty, p in rels)


def same(a, b):
    return abs(a - b) <= EQ * max(1.0, abs(a), abs(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--macros", default=None)
    args = ap.parse_args()
    E.DATA = os.path.join(REPO, "data", "locobench-slot3-test")
    items, _f, _l = E.load()
    rng = random.Random(SEED)
    res = {k: [] for k in ("A1", "A2", "B", "C")}
    for iid in sorted(items):
        it = items[iid]
        atoms = sorted(it["atoms"], key=lambda a: a["pos"])
        n = len(atoms)
        x = np.array([1 if a["factual"] else 0 for a in atoms])
        g = [(E._idx(r["source"]), E._idx(r["target"]), r["type"], float(r["probability"]))
             for r in E.gold_relations(it)]
        touched = {frozenset((s, t)) for s, t, _ty, _p in g}
        T = [i for i in range(n) if x[i] == 1]
        F = [i for i in range(n) if x[i] == 0]
        free_tt = [pr for pr in itertools.combinations(T, 2) if frozenset(pr) not in touched]
        free_tf = [(s, t) for s in T for t in F if frozenset((s, t)) not in touched]
        rng.shuffle(free_tt)
        rng.shuffle(free_tf)
        base_l = lam(n, g, x)
        # A1: equivalence vs two entailments, between two true claims
        if free_tt:
            s, t = free_tt[0]
            g1 = g + [(s, t, "equivalence", P)]
            g2 = g + [(s, t, "entailment", P), (t, s, "entailment", P)]
            res["A1"].append((same(lam(n, g1, x), lam(n, g2, x)), same(wsum(g1, x), wsum(g2, x))))
        # A2: exclusion vs contradiction + co-necessity, between a true and a false claim
        if free_tf:
            s, t = free_tf[0]
            g1 = g + [(s, t, "exclusive", P)]
            g2 = g + [(s, t, "contradiction", P), (s, t, "co_necessity", P)]
            res["A2"].append((same(lam(n, g1, x), lam(n, g2, x)), same(wsum(g1, x), wsum(g2, x))))
        # B: one inference three times vs three distinct inferences (disjoint new pairs)
        disjoint = []
        for pr in free_tt:
            if all(not set(pr) & set(q) for q in disjoint):
                disjoint.append(pr)
            if len(disjoint) == 3:
                break
        if len(disjoint) == 3:
            (s, t) = disjoint[0]
            rep = g + [(s, t, "entailment", P)] * 3
            new = g + [(a, b, "entailment", P) for a, b in disjoint]
            lr, ln = lam(n, rep, x), lam(n, new, x)
            res["B"].append((ln > lr and not same(ln, lr), wsum(new, x) > wsum(rep, x) + 1e-12,
                             ln - base_l, lr - base_l))
        # C: an inference from a contested premise vs from an uncontested one
        for s in T:
            others = [u for u in T if u != s and frozenset((s, u)) not in touched]
            if len(others) < 2:
                continue
            u, w = others[0], others[1]
            if frozenset((u, w)) in touched:
                continue
            r = (s, u, "entailment", P)
            gain_plain = lam(n, g + [r], x) - base_l
            gd = g + [(s, w, "contradiction", P)]
            gain_doubt = lam(n, gd + [r], x) - lam(n, gd, x)
            res["C"].append((gain_doubt < gain_plain and not same(gain_doubt, gain_plain),
                             False, gain_plain, gain_doubt))
            break
    out = {}
    for k, R in res.items():
        out[k] = {"n": len(R), "lcs": sum(r[0] for r in R) / len(R),
                  "sum": sum(r[1] for r in R) / len(R)}
    out["B"]["gain_new"] = st.median(r[2] for r in res["B"])
    out["B"]["gain_rep"] = st.median(r[3] for r in res["B"])
    out["C"]["gain_plain"] = st.median(r[2] for r in res["C"])
    out["C"]["gain_doubt"] = st.median(r[3] for r in res["C"])
    for k, v in out.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    if args.macros:
        L = ["% Generated by scripts/arr_z_stress.py -- do not edit by hand."]
        m = lambda name, val: L.append(f"\\newcommand{{\\zs{name}}}{{{val}}}")  # noqa: E731
        for k, name in (("A1", "Eqv"), ("A2", "Exc"), ("B", "Rep"), ("C", "Doubt")):
            m(f"{name}N", f"{out[k]['n']:,}".replace(",", "{,}"))
            m(f"{name}Lcs", f"{100 * out[k]['lcs']:.0f}")
            m(f"{name}Sum", f"{100 * out[k]['sum']:.0f}")
        m("RepGainNew", f"{out['B']['gain_new']:.3f}")
        m("RepGainRep", f"{out['B']['gain_rep']:.3f}")
        m("DoubtGainPlain", f"{out['C']['gain_plain']:.3f}")
        m("DoubtGainDoubt", f"{out['C']['gain_doubt']:.3f}")
        m("P", f"{P}")
        open(args.macros, "w").write("\n".join(L) + "\n")
        print(f"wrote {args.macros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
