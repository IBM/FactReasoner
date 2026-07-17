"""Deterministic replay of the v2 run's BQ.1 row from its LOGGED relations.
No NLI calls. Rebuilds both arms (fp_trust, fp_vanilla) exactly as recorded,
runs Merlin (deterministic), exports the five demo CSVs.
"""
import os as _os, sys as _sys
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'experiments', 'consensus_trust'))
import json, glob, re, sys, os, csv
import fact_reasoner.core.trust.credibility_scorer as _cs
import fact_reasoner.core.trust.credibility_scorer_v3 as _v3
_cs.score_url = _v3.score_url
from state_media_eval import make_pipeline
from fact_reasoner.core.base import Atom, Context, Relation
from fact_reasoner.core.trust.credibility_fusion import CredibilityTrustFusion
from urllib.parse import urlparse

NEEDLE = 'New Omicron subvariants BQ.1 and BQ.1.1'
STATE  = 'data/trust_eval/dynaTD_state_40v2.json'
EXP    = 'data/trust_eval/demo_exports_v2replay'
os.makedirs(EXP, exist_ok=True)

# ---- 1. carve this row's four NLI blocks out of the v2 log -------------------
d = json.load(open(sorted(glob.glob('data/trust_eval/final_eval_40v2_*/results.json'))[-1]))
idx = next(i for i, r in enumerate(d['results']) if 'BQ.1' in r['input'])
log = open('data/trust_eval/final_40v2.txt').read().splitlines()
start = next(i for i, l in enumerate(log) if 'WARMUP done' in l)
marks = [i for i, l in enumerate(log[start:], start) if re.match(r'\[ *\d+/', l)]
this_m = next(i for i in marks if re.match(rf'\[ *{idx+1}/', log[i]))
prev_m = max([m for m in marks if m < this_m], default=start)
seg = log[prev_m:this_m]
block_starts = [i for i, l in enumerate(seg) if 'Building atom-context relations' in l]
assert len(block_starts) == 4, f"expected 4 blocks, got {len(block_starts)}"
block_starts.append(len(seg))
REL_RE = re.compile(r'\[NLI\] Found relation: \[(\S+) -> (\S+)\] : (\w+) : ([\d.]+)')
blocks = []
for b in range(4):
    rels = [REL_RE.search(l).groups() for l in seg[block_starts[b]:block_starts[b+1]]
            if 'Found relation' in l]
    blocks.append(rels)
# order per script: cc_trust, cc_vanilla, fp_trust, fp_vanilla
fp_trust_rels, fp_vanilla_rels = blocks[2], blocks[3]
print(f"recovered relations: fp_trust={len(fp_trust_rels)}, fp_vanilla={len(fp_vanilla_rels)}")

# ---- 2. rebuild the row exactly as the eval did ------------------------------
row = [json.loads(l) for l in open('data/trust_eval/eval_dataset_fp.jsonl') if NEEDLE in l][0]
gt = row['ground_truth']
sc = CredibilityTrustFusion(state_path=STATE)

def build(scorer):
    A = {a['id']: Atom(id=a['id'], text=a['text']) for a in row['atoms']}
    lut = {c['id']: c for c in row['contexts']}; C = {}
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

def replay(rel_tuples, scorer, tag):
    A, C = build(scorer)
    rels = []
    for src, tgt, typ, p in rel_tuples:
        if src in C and tgt in A:
            rels.append(Relation(source=C[src], target=A[tgt], type=typ, probability=float(p), link='context_atom'))
    live = {r_.source.id for r_ in rels}
    Cf = {k: v for k, v in C.items() if k in live}
    for a in A.values():
        a.contexts = {c.id: c for c in a.get_contexts().values() if c.id in Cf}
    _, marg = make_pipeline(A, Cf, rels, gt).score()
    post = {m['variable']: m['probabilities'][1] for m in marg if m['variable'] in A}
    prec = sum(1 for p in post.values() if p > 0.5) / max(len(post), 1)
    print(f"{tag:>8}: precision={prec:.4f} verdict={'S' if prec>0.5 else 'NS'} (gold {gt})")
    return A, Cf, rels, post, prec

A_t, C_t, R_t, post_t, prec_t = replay(fp_trust_rels, sc, 'trust')
A_v, C_v, R_v, post_v, prec_v = replay(fp_vanilla_rels, None, 'vanilla')
print("EXPECTED from v2 table: trust=S p=0.6667, vanilla=NS p=0.4286 -- must match above")

# ---- 3. exports ---------------------------------------------------------------
def dom(link): return urlparse(link or '').netloc.replace('www.', '')
lut = {c['id']: c for c in row['contexts']}
with open(f'{EXP}/contexts.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ctx_id','atom_id','domain','cred_v3_prior','dyna_learned_trust',
                'fused_prob_trust_arm','prob_vanilla_arm','text_snippet'])
    for a in row['atoms']:
        for cid in a['contexts'][:10]:
            if cid not in lut: continue
            link = lut[cid].get('link','')
            dk = sc.dynaTD._extract_domain(link)
            dyna = sc.dynaTD.get_reliability(dk) if dk else ''
            w.writerow([cid, a['id'], dom(link), round(_v3.score_url(link),3),
                        round(dyna,3) if dyna != '' else '',
                        round(C_t[cid].get_probability(),3) if cid in C_t else 'dropped',
                        0.9 if cid in C_v else 'dropped', lut[cid]['text'][:120]])
with open(f'{EXP}/relations.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['ctx_id','atom_id','nli_type','nli_strength','arm'])
    for tag, rl in (('trust', R_t), ('vanilla', R_v)):
        for r_ in rl: w.writerow([r_.source.id, r_.target.id, r_.type, round(r_.probability,4), tag])
with open(f'{EXP}/consensus.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['atom_id','consensus_target','n_voters'])
    num, den, cnt = {}, {}, {}
    for r_ in R_t:
        v = 1.0 if r_.type == 'entailment' else (0.0 if r_.type == 'contradiction' else None)
        if v is None: continue
        wgt = _v3.score_url(getattr(r_.source,'link','') or '') * r_.probability
        aid = r_.target.id
        num[aid]=num.get(aid,0)+wgt*v; den[aid]=den.get(aid,0)+wgt; cnt[aid]=cnt.get(aid,0)+1
    for aid in den: w.writerow([aid, round(num[aid]/den[aid],4), cnt[aid]])
atom_txt = {a['id']: a['text'] for a in row['atoms']}
with open(f'{EXP}/posteriors.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['atom_id','atom_text','P_true_trust','P_true_vanilla','verdict_trust','verdict_vanilla','flipped'])
    for aid in sorted(set(post_t)|set(post_v)):
        pt, pv = post_t.get(aid), post_v.get(aid)
        vt = '' if pt is None else ('S' if pt>0.5 else 'NS')
        vv = '' if pv is None else ('S' if pv>0.5 else 'NS')
        w.writerow([aid, atom_txt.get(aid,'')[:100],
                    '' if pt is None else round(pt,4), '' if pv is None else round(pv,4),
                    vt, vv, 'FLIP' if (vt and vv and vt!=vv) else ''])
with open(f'{EXP}/verdict.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['arm','precision','verdict','gold'])
    w.writerow(['trust', round(prec_t,4), 'S' if prec_t>0.5 else 'NS', gt])
    w.writerow(['vanilla', round(prec_v,4), 'S' if prec_v>0.5 else 'NS', gt])
print(f"EXPORTS -> {EXP}/")
