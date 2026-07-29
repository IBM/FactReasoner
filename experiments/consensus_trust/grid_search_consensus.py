"""Grid over {consensus target} x {NLI strength ceiling}, trust AND vanilla per cell.
Only the trust-vanilla DELTA is attributable to the trust layer; a gain that also
appears in vanilla is an NLI-calibration effect.

Usage (repo root): python3 experiments/consensus_trust/grid_search_consensus.py
"""
import sys, os, io, contextlib, json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, 'experiments/consensus_trust')
from collections import defaultdict
with contextlib.redirect_stdout(io.StringIO()):
    import replay_consensus_ablation as R
import consensus_variants as CV
from fact_reasoner.core.base import Relation

TARGETS = ['posterior', 'current', 'laplace', 'loo', 'loo_laplace', 'crh', 'loo_crh']
CEILINGS = [1.0, 0.99, 0.95, 0.90, 0.80]     # 1.0 = untouched (as-run)

def score_row(row, tuples, scorer, ceiling):
    A, C = R.build(row, scorer)
    rels = []
    for s, t, ty, p in tuples:
        if s in C and t in A:
            rels.append(Relation(source=C[s], target=A[t], type=ty,
                                 probability=min(float(p), ceiling), link='context_atom'))
    if not rels: return None, None, None
    live = {r.source.id for r in rels}
    Cf = {k: v for k, v in C.items() if k in live}
    for a in A.values():
        a.contexts = {c.id: c for c in a.get_contexts().values() if c.id in Cf}
    with contextlib.redirect_stdout(io.StringIO()):
        _, marg = R.make_pipeline(A, Cf, rels, row['ground_truth']).score()
    post = {m['variable']: m['probabilities'][1] for m in marg if m['variable'] in A}
    frac = sum(1 for p in post.values() if p > 0.5) / max(len(post), 1)
    return ('S' if frac > 0.5 else 'NS'), post, rels

def run(target, ceiling, arm):
    """arm: 'trust' (block 2, learned weights) or 'vanilla' (block 3, flat 0.9)."""
    sc = R.VariantScorer() if arm == 'trust' else None
    block = 2 if arm == 'trust' else 3
    preds, sat_hits, sat_tot = {}, 0, 0
    for idx, row in enumerate(R.fp_rows):
        tup = R.relations_for_row(idx, block)
        if not tup: continue
        pred, post, rels = score_row(row, tup, sc, ceiling)
        if pred is None: continue
        preds[idx] = pred
        for p in post.values():
            sat_tot += 1
            if p > 0.99 or p < 0.01: sat_hits += 1
        if arm != 'trust': continue
        pa = defaultdict(list)
        for r in rels:
            dom = sc.domain_of(getattr(r.source, 'link', ''))
            v = 1.0 if r.type == 'entailment' else (0.0 if r.type == 'contradiction' else None)
            if dom and v is not None:
                pa[r.target.id].append((dom, v, float(r.probability), sc.fused(dom)))
        for aid, votes in pa.items():
            CV.POSTERIOR["T"] = post.get(aid)        # for the 'posterior' target
            tmap = CV.targets_for_atom(votes, target)
            for dom, v, s, w in votes:
                T = tmap.get(dom)
                if T is None: continue
                err = s * (1.0 - T) if v == 1.0 else s * T
                sc.a[dom] = sc.a.get(dom, 0.0) + (1.0 - err)
                sc.b[dom] = sc.b.get(dom, 0.0) + err * err
                sc.total[dom] = sc.total.get(dom, 0) + 1
                if err < 0.5: sc.correct[dom] = sc.correct.get(dom, 0) + 1
    return preds, (sat_hits / max(sat_tot, 1))

gold = {i: r['ground_truth'] for i, r in enumerate(R.fp_rows)}
sub27 = R.sub27
def acc(preds, subset=None):
    ks = [k for k in preds if subset is None or k in subset]
    return sum(1 for k in ks if preds[k] == gold[k]), len(ks)

print(f"{'target':12s}{'ceil':>6s}{'trust40':>10s}{'van40':>9s}{'Δ40':>6s}"
      f"{'trust27':>10s}{'van27':>9s}{'Δ27':>6s}{'sat%':>7s}")
best = []
for ceil in CEILINGS:
    vp, vsat = run(None, ceil, 'vanilla')          # vanilla independent of target
    v40, v27 = acc(vp), acc(vp, sub27)
    for tgt in TARGETS:
        tp, tsat = run(tgt, ceil, 'trust')
        t40, t27 = acc(tp), acc(tp, sub27)
        d40, d27 = t40[0] - v40[0], t27[0] - v27[0]
        best.append((d27, d40, tgt, ceil, t27, t40, v27, v40))
        print(f"{tgt:12s}{ceil:>6.2f}{t40[0]:>7d}/{t40[1]:<3d}{v40[0]:>6d}/{v40[1]:<3d}"
              f"{d40:>+5d}{t27[0]:>7d}/{t27[1]:<3d}{v27[0]:>6d}/{v27[1]:<3d}{d27:>+5d}{tsat:>7.0%}")

print("\n=== ranked by trust-vanilla delta on the 27 (then 40) ===")
for d27, d40, tgt, ceil, t27, t40, v27, v40 in sorted(best, reverse=True)[:6]:
    print(f"  {tgt:12s} ceil={ceil:.2f}  27: {t27[0]}/{t27[1]} vs van {v27[0]} (Δ{d27:+d})   "
          f"40: {t40[0]}/{t40[1]} vs van {v40[0]} (Δ{d40:+d})")
print("\nHONESTY RULE: cells were not held out. Any winner here is SELECTED on the")
print("test set and must be re-validated on AVeriTeC-423 before it can be reported.")
print("A gain that also lifts vanilla is an NLI-calibration result, not a trust result.")
json.dump([{'target':t,'ceil':c,'trust27':list(a),'van27':list(b),'trust40':list(x),'van40':list(y)}
           for _,_,t,c,a,x,b,y in best], open('/u/samit/consensus_grid.json','w'), indent=1)
print("wrote /u/samit/consensus_grid.json")
