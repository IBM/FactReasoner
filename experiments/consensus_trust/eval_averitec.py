"""AVeriTeC prequential eval: fp_trust (v3 prior, clean consensus) vs fp_vanilla.
Usage: python3 sweep_averitec.py [N]   (N claims; default 5 = smoke test)"""
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

# --- ABLATION_FLAGS ---
if os.environ.get('NO_MBFC'):
    _flat = float(os.environ.get('NO_MBFC'))
    import fact_reasoner.core.trust.credibility_scorer as _csx
    import fact_reasoner.core.trust.credibility_fusion as _cfx
    _csx.score_url = lambda u, _f=_flat: _f
    _cfx.score_url = lambda u, _f=_flat: _f
    print("[ablation] NO_MBFC: flat prior %.2f for every source" % _flat, flush=True)
if os.environ.get('UTD_PRIOR'):
    from fact_reasoner.core.trust.url_trust import UTD, DEFAULT_MODEL_PATH
    _utd = UTD(model_path=DEFAULT_MODEL_PATH)
    def _utd_score(u):
        try: return float(_utd.score(u)) if u else 0.5
        except Exception: return 0.5
    import fact_reasoner.core.trust.credibility_scorer as _csu
    import fact_reasoner.core.trust.credibility_fusion as _cfu
    _csu.score_url = _utd_score
    _cfu.score_url = _utd_score
    print("[ablation] UTD_PRIOR: URL-feature model as prior (full URL)", flush=True)
if os.environ.get('MBFC_UTD_FALLBACK'):
    from urllib.parse import urlparse as _up
    from fact_reasoner.core.trust.url_trust import UTD as _UTDH, DEFAULT_MODEL_PATH as _DMP
    import fact_reasoner.core.trust.credibility_scorer_v3 as _v3h
    _utdh = _UTDH(model_path=_DMP)
    def _hybrid(u):
        if not u: return 0.5
        pm = _v3h.score_url(u)
        if pm != 0.5: return pm            # MBFC-rated or .gov rule
        d = _v3h._norm(_up(u if '://' in u else 'https://' + u).netloc)
        if not d: return 0.5
        try: return float(_utdh.score('https://' + d))   # bare domain, stable per source
        except Exception: return 0.5
    import fact_reasoner.core.trust.credibility_scorer as _csh
    import fact_reasoner.core.trust.credibility_fusion as _cfh
    _csh.score_url = _hybrid
    _cfh.score_url = _hybrid
    print("[ablation] MBFC_UTD_FALLBACK: MBFC primary, UTD(bare domain) for unrated", flush=True)
if os.environ.get('GRAPH_IRELI'):
    import json as _gj
    from urllib.parse import urlparse as _gu
    import fact_reasoner.core.trust.credibility_scorer_v3 as _v3i
    _G=_gj.load(open('/u/samit/reliability_scores.json'))
    def _ireli_score(u):
        pm=_v3i.score_url(u)
        if pm!=0.5: return pm
        d=_v3i._norm(_gu(u if '://' in u else 'https://'+u).netloc)
        v=_G.get(d)
        if v is None: return 0.5
        sc2=v['i-reliability']
        return 0.85 if sc2>=0.55 else (0.30 if sc2<=-0.45 else 0.50)
    import fact_reasoner.core.trust.credibility_scorer as _csi
    import fact_reasoner.core.trust.credibility_fusion as _cfi
    _csi.score_url=_ireli_score
    _cfi.score_url=_ireli_score
    print("[ablation] GRAPH_IRELI: MBFC primary, idiap I-reliability graph for unrated", flush=True)
if os.environ.get('GRAPH_PRIOR'):
    import json as _gj
    from urllib.parse import urlparse as _gu
    import fact_reasoner.core.trust.credibility_scorer_v3 as _v3g
    _G=_gj.load(open('/u/samit/reliability_scores.json'))
    def _graph_score(u):
        pm=_v3g.score_url(u)
        if pm!=0.5: return pm
        d=_v3g._norm(_gu(u if '://' in u else 'https://'+u).netloc)
        v=_G.get(d)
        if v is None: return 0.5
        sc2=v['p+fp-average']
        return 0.85 if sc2>=0.55 else (0.30 if sc2<=-0.45 else 0.50)
    import fact_reasoner.core.trust.credibility_scorer as _csg
    import fact_reasoner.core.trust.credibility_fusion as _cfg
    _csg.score_url=_graph_score
    _cfg.score_url=_graph_score
    print("[ablation] GRAPH_PRIOR: MBFC primary, idiap graph-tier for unrated", flush=True)
if os.environ.get('MODEL_FB'):
    import pickle, sys as _sy
    _sy.path.insert(0, '/u/samit')
    from train_bare_domain import feats as _bf
    from urllib.parse import urlparse as _up2
    import fact_reasoner.core.trust.credibility_scorer_v3 as _v3m
    _mdl = pickle.load(open('/u/samit/bare_domain_model.pkl','rb'))['clf']
    def _fb_score(u):
        pm = _v3m.score_url(u)
        if pm != 0.5: return pm
        d = _v3m._norm(_up2(u if '://' in u else 'https://' + u).netloc)
        if not d or '.' not in d: return 0.5
        try: return 0.05 + 0.9 * float(_mdl.predict_proba([_bf(d)])[0,1])
        except Exception: return 0.5
    import fact_reasoner.core.trust.credibility_scorer as _csm
    import fact_reasoner.core.trust.credibility_fusion as _cfm
    _csm.score_url = _fb_score
    _cfm.score_url = _fb_score
    print("[ablation] MODEL_FB: MBFC primary, bare-domain RF for unrated", flush=True)
if os.environ.get('PRIOR_ONLY'):
    CredibilityTrustFusion.update_from_results = lambda self, *a, **k: None
    print("[ablation] PRIOR_ONLY: consensus learning DISABLED", flush=True)
# --- end ABLATION_FLAGS ---
from averitec_loader import load

N   = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ST  = 'data/trust_eval/averitec_trust_state.json'
OUT = 'data/trust_eval/averitec_results.json'
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
        '_meta': {'id': r['id'], 'date': r.get('claim_date'), 'label4': r['label4']},
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
    R = None; _e = 0
    for k in range(6):
        try:
            R = build_relations(atoms=A, contexts=C, nli_extractor=nli, rel_atom_context=True,
                                rel_context_context=False, use_summarized_contexts=False)
            if R: break
            _e += 1
            if _e >= 3:
                print(f"  [{tag}] empty x3 -> skip", flush=True); return None
            print(f"  [{tag}] empty {_e}/3 -> fast retry", flush=True)
            await asyncio.sleep(1.5); continue
        except Exception as e:
            print(f"  [{tag}] NLI {k+1}: {str(e)[:40]}", flush=True)
        if k < 5: await asyncio.sleep(4*(k+1))
    if not R:
        print(f"  [{tag}] GAVE UP after 6 attempts -> row dropped", flush=True)
        return None
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
    _aug = os.environ.get('AVERITEC_AUG_JSONL')
    if _aug:
        rows = sorted([json.loads(_l) for _l in open(_aug) if _l.strip()], key=date_key)[:N]
        print("[aug] %d rows from %s" % (len(rows), _aug), flush=True)
    else:
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
