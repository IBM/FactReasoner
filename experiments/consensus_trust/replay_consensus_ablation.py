"""Prequential replay of the 40-row StateMedia eval under multiple consensus-target
variants, using FROZEN logged NLI relations + real Merlin inference. No API, no GPU.

Modeled on demo/export_full_dataset.py (same log parsing + build + make_pipeline),
but starts from an EMPTY trust state and evolves it row-by-row, so the update rule
is the only thing that differs between variants.

Usage (repo root):  python3 experiments/consensus_trust/replay_consensus_ablation.py
"""
import json, re, sys, os, csv, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'demo'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
import fact_reasoner.core.trust.credibility_scorer as _cs
import fact_reasoner.core.trust.credibility_scorer_v3 as _v3
_cs.score_url = _v3.score_url
from state_media_eval import make_pipeline
from fact_reasoner.core.base import Atom, Context, Relation
from consensus_variants import targets_for_atom

ROOT = os.path.join(HERE, '..', '..', 'data', 'trust_eval')
LOG      = os.path.join(ROOT, 'final_40v2.txt')
RESULTS  = os.path.join(ROOT, 'final_eval_40v2_results', 'results.json')
FP_JSONL = os.path.join(ROOT, 'eval_dataset_fp.jsonl')
CC27     = '/u/samit/eval_dataset_cc_27.jsonl'

BETA_CAP, BETA_HALF = 0.7, 2.0
CLIP_LO, CLIP_HI = 0.05, 0.97

# ───────── frozen relation parsing (identical to export_full_dataset) ─────────
log = open(LOG).read().splitlines()
start = next(i for i, l in enumerate(log) if 'WARMUP done' in l)
marks = [i for i, l in enumerate(log[start:], start) if re.match(r'\[ *\d+/', l)]
REL_RE = re.compile(r'\[NLI\] Found relation: \[(\S+) -> (\S+)\] : (\w+) : ([\d.]+)')
fp_rows = [json.loads(l) for l in open(FP_JSONL) if l.strip()]
d_results = json.load(open(RESULTS))

def relations_for_row(idx, block):
    seg_start = marks[idx-1] if idx > 0 else start
    seg = log[seg_start:marks[idx]] if idx < len(marks) else log[seg_start:]
    bs = [i for i, l in enumerate(seg) if 'Building atom-context relations' in l]
    if len(bs) != 4:
        return None
    bs.append(len(seg))
    return [REL_RE.search(l).groups() for l in seg[bs[block]:bs[block+1]] if 'Found relation' in l]

# ───────── variant-parameterised online trust state ─────────
class VariantScorer:
    """Mimics CredibilityTrustFusion.score(context) with our own evolving state."""
    def __init__(self):
        self.a, self.b, self.prior = {}, {}, {}
        self.correct, self.total = {}, {}
    @staticmethod
    def domain_of(link):
        from urllib.parse import urlparse
        try:
            net = urlparse(link or '').netloc.lower()
        except Exception:
            return ''
        if net.startswith('www.'): net = net[4:]
        return net
    def seed(self, dom, link):
        if dom and dom not in self.prior:
            try:    c = float(_v3.score_url(link))
            except Exception: c = 0.5
            self.prior[dom] = c; self.a[dom] = 0.0; self.b[dom] = 2.0/max(c, 1e-3)
    def reliability(self, dom):
        # MATCHES dynaTD.get_reliability: Laplace-smoothed success rate in (0,1)
        n = self.total.get(dom, 0)
        c = self.correct.get(dom, 0)
        return 0.5 if n == 0 else float((1.0 + c) / (2.0 + n))
    def fused(self, dom):
        c = self.prior.get(dom, 0.5)
        beta = min(self.a.get(dom, 0.0)/BETA_HALF, BETA_CAP)
        r = min(max(self.reliability(dom), 0.0), 1.0)
        return min(max((1-beta)*c + beta*r, CLIP_LO), CLIP_HI)
    def score(self, context):                      # interface used by build()
        dom = self.domain_of(getattr(context, 'link', ''))
        self.seed(dom, getattr(context, 'link', ''))
        return self.fused(dom) if dom else 0.5

def build(row, scorer):
    A = {a['id']: Atom(id=a['id'], text=a['text']) for a in row['atoms']}
    lut = {c['id']: c for c in row['contexts']}
    C = {}
    for a in row['atoms']:
        at = A[a['id']]; mine = []
        for cid in a['contexts'][:10]:
            if cid not in lut: continue
            c = lut[cid]
            x = Context(id=cid, atom=at, text=c['text'], title=c.get('title',''), link=c.get('link',''))
            x.set_probability(scorer.score(x) if scorer else 0.9)
            C[cid] = x; mine.append(x)
        at.add_contexts(mine)
    return A, C

def run_variant(variant, verbose=False):
    sc = VariantScorer(); preds = {}; skipped = []
    for idx, row in enumerate(fp_rows):
        tuples = relations_for_row(idx, 2)          # block 2 = fp_trust arm
        if tuples is None:
            skipped.append(idx); continue
        A, C = build(row, sc)
        rels = [Relation(source=C[s], target=A[t], type=ty, probability=float(p), link='context_atom')
                for s, t, ty, p in tuples if s in C and t in A]
        if not rels:
            skipped.append(idx); continue
        live = {r.source.id for r in rels}
        Cf = {k: v for k, v in C.items() if k in live}
        for a in A.values():
            a.contexts = {c.id: c for c in a.get_contexts().values() if c.id in Cf}
        _, marg = make_pipeline(A, Cf, rels, row['ground_truth']).score()
        post = {m['variable']: m['probabilities'][1] for m in marg if m['variable'] in A}
        frac = sum(1 for p in post.values() if p > 0.5)/max(len(post), 1)
        preds[idx] = 'S' if frac > 0.5 else 'NS'
        # ── consensus update, variant-specific ──
        per_atom = defaultdict(list)
        for r in rels:
            dom = sc.domain_of(getattr(r.source, 'link', ''))
            if not dom: continue
            v = 1.0 if r.type == 'entailment' else (0.0 if r.type == 'contradiction' else None)
            if v is None: continue
            per_atom[r.target.id].append((dom, v, float(r.probability), sc.fused(dom)))
        for aid, votes in per_atom.items():
            tmap = targets_for_atom(votes, variant)
            for dom, v, s, w in votes:
                T = tmap.get(dom)
                if T is None: continue              # LOO with no other voters -> skip
                err = s*(1.0-T) if v == 1.0 else s*T
                sc.a[dom] = sc.a.get(dom, 0.0) + (1.0-err)
                sc.b[dom] = sc.b.get(dom, 0.0) + err*err
                sc.total[dom] = sc.total.get(dom, 0) + 1
                if err < 0.5:
                    sc.correct[dom] = sc.correct.get(dom, 0) + 1
        if verbose:
            print(f"  row {idx:2d} gt={row['ground_truth']:2s} pred={preds[idx]:2s} rels={len(rels)}", flush=True)
    return preds, skipped

# ───────── subsets + scoring ─────────
gold = {i: r['ground_truth'] for i, r in enumerate(fp_rows)}
ids27 = set()
if os.path.exists(CC27):
    for l in open(CC27):
        if l.strip():
            o = json.loads(l); ids27.add((o.get('input') or '')[:60])
sub27 = {i for i, r in enumerate(fp_rows) if (r.get('input') or '')[:60] in ids27} if ids27 else None
print(f"[setup] fp rows={len(fp_rows)}  27-subset matched={len(sub27) if sub27 else 'N/A'}")

def acc(preds, subset=None):
    ks = [k for k in preds if subset is None or k in subset]
    return sum(1 for k in ks if preds[k] == gold[k]), len(ks)

print("\n[GATE] validating 'current' variant against the recorded run …")
base, sk = run_variant('current')
c40, n40 = acc(base); c27, n27 = acc(base, sub27)
print(f"[GATE] current: 40-set {c40}/{n40}   27-subset {c27}/{n27}   skipped rows={len(sk)}")
print( "[GATE] recorded reference: 22/27 on the hard-fact subset.")
if sub27 and (c27, n27) != (22, 27):
    print( "[GATE] *** MISMATCH — replay is not faithful. Do NOT trust variant deltas below.")
    print( "       Likely causes: prequential state differs from the recorded run's warmup,")
    print( "       or block index 2 is not the fp_trust arm. Inspect before interpreting.")
print()

VARIANTS = ['current', 'laplace', 'loo', 'loo_laplace', 'crh', 'loo_crh']
print(f"{'variant':13s}{'40-set':>12s}{'27-subset':>14s}   (skipped)")
results = {}
for v in VARIANTS:
    p, s = run_variant(v)
    a40, a27 = acc(p), acc(p, sub27)
    results[v] = {'40': a40, '27': a27, 'skipped': len(s)}
    print(f"{v:13s}{a40[0]:>4d}/{a40[1]:<4d}{a27[0]:>8d}/{a27[1]:<4d}  "
          f"({a40[0]/max(a40[1],1):5.1%} / {a27[0]/max(a27[1],1):5.1%})   {len(s)}")
json.dump(results, open('/u/samit/consensus_ablation_results.json', 'w'), indent=1)
print("\nwrote /u/samit/consensus_ablation_results.json")
print("Selection rule (pre-stated): prefer LOO family on principle; call it signal only")
print("at >= +2 rows over 'current' on the 40-set; winner must not regress AVeriTeC.")
