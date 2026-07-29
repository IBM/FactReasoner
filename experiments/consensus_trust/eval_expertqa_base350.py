"""AVeriTeC prequential eval: fp_trust (v3 prior, clean consensus) vs fp_vanilla.
Usage: python3 sweep_expertqa.py [N]   (N claims; default 5 = smoke test)"""
import os as _os, sys as _sys
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'experiments', 'consensus_trust'))
import asyncio, json, sys, os

# v3 prior everywhere the trust stack looks for a scorer — BEFORE fusion imports
import fact_reasoner.core.trust.credibility_scorer as _cs
import fact_reasoner.core.trust.credibility_scorer_v3 as _v3
_cs.score_url = _v3.score_url

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from state_media_eval import build_relations, make_pipeline
from granite_switch_vs_factreaser_demo import NLIFixed
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.trust.credibility_fusion import CredibilityTrustFusion
_sys.path.insert(0, _os.path.join(_REPO, 'experiments', 'consensus_trust', 'loaders'))
from expertqa_base350_loader import load

N   = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ST  = 'data/trust_eval/expertqa_base350_trust_state.json'
OUT = 'data/trust_eval/expertqa_base350_results.json'
if os.path.exists(ST): os.remove(ST)          # fresh state, this run only

llm = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS:512})
nli = NLIFixed(llm)

def to_row(r):
    ids = [f"c{i}" for i in range(len(r['contexts']))]
    return {
        'atoms': [{'id':'a0','text': r['claim'], 'contexts': ids}],
        'contexts': [{'id': ids[i], 'text': c['text'], 'title':'', 'link': c['link']}
                     for i, c in enumerate(r['contexts'])],
        'ground_truth': r['ground_truth']['a0'],
        '_meta': {'id': r['id'], 'date': r.get('claim_date'), 'label4': r.get('label4', r.get('raw_label'))},
    }

def date_key(r):
    d = r.get('claim_date') or '31-12-2099'
    p = d.split('-')
    return (p[2], p[1], p[0]) if len(p) == 3 else ('2099','12','31')

def build(d, sc):
    A = {a['id']: Atom(id=a['id'], text=a['text']) for a in d['atoms']}
    lut = {c['id']: c for c in d['contexts']}
    C = {}
    for a in d['atoms']:
        at = A[a['id']]; mine = []
        for cid in a['contexts']:
            c = lut[cid]
            x = Context(id=cid, atom=at, text=c['text'], title=c['title'], link=c['link'])
            x.set_probability(sc.score(x) if sc else 0.9)
            C[cid] = x; mine.append(x)
        at.add_contexts(mine)
    return A, C

async def fr(d, sc, tag, upd):
    gt = d['ground_truth']; A, C = build(d, sc)
    if not C: return None
    R = None
    for k in range(5):
        try:
            R = build_relations(atoms=A, contexts=C, nli_extractor=nli, rel_atom_context=True,
                                rel_context_context=False, use_summarized_contexts=False); break
        except Exception as e:
            print(f"  [{tag}] NLI {k+1}: {str(e)[:40]}", flush=True); await asyncio.sleep(5*(k+1))
    if not R: return None
    live = {r.source.id for r in R}; C = {k: v for k, v in C.items() if k in live}
    R = [r for r in R if r.source.id in C]
    if not C: return None
    for a in A.values(): a.contexts = {c.id: c for c in a.get_contexts().values() if c.id in C}
    _, M = make_pipeline(A, C, R, gt).score()
    P = {m['variable']: m['probabilities'][1] for m in M if m['variable'] in A}
    if not P: return None
    v = 'S' if list(P.values())[0] > 0.5 else 'NS'
    if sc and upd: sc.update_from_results(C, M, R)
    return {'verdict': v, 'p': round(list(P.values())[0], 4), 'correct': v == gt}

async def main():
    rows = sorted(load(), key=date_key)[:N]
    print(f"=== AVeriTeC prequential, N={len(rows)} (temporal order) ===", flush=True)
    sc = CredibilityTrustFusion(state_path=ST)
    res = {'trust': {'c':0,'t':0}, 'vanilla': {'c':0,'t':0}}; out = []
    for i, raw in enumerate(rows):
        d = to_row(raw); gt = d['ground_truth']
        o = {}
        o['vanilla'] = await fr(d, None, 'van', upd=False)
        o['trust']   = await fr(d, sc,  'tru', upd=True)     # predict-then-update
        for arm in res:
            if o[arm]: res[arm]['t'] += 1; res[arm]['c'] += o[arm]['correct']
        out.append({'meta': d['_meta'], 'gt': gt, **{k: v for k, v in o.items()}})
        print(f"row {i:3d} [{d['_meta']['date']}] gt={gt} "
              f"van={o['vanilla'] and o['vanilla']['verdict']} tru={o['trust'] and o['trust']['verdict']}", flush=True)
        if (i+1) % 10 == 0:
            json.dump(out, open(OUT, 'w'), indent=1)
            for a, s in res.items(): print(f"  [{i+1}] {a}: {s['c']}/{s['t']}", flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print("=" * 50)
    for a, s in res.items(): print(f"{a:>8}: {s['c']}/{s['t']} = {s['c']/max(s['t'],1):.1%}", flush=True)

asyncio.run(main())
