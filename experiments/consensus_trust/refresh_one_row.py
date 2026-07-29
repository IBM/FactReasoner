"""Re-retrieve evidence for ONE row with fresh Serper, re-run NLI + Merlin both arms,
compare per-atom posteriors old vs new. DIAGNOSTIC for temporal retrieval mismatch.

NOTE: searching today for a 2022 claim returns retrospective sources that did not exist
at original retrieval time. This confirms/refutes the recency bottleneck; it is NOT an
accuracy improvement and must not be folded into headline numbers.

Usage (repo root):
  python3 experiments/consensus_trust/refresh_one_row.py --match "BQ.1" --topk 6
"""
import os, sys, json, re, time, hashlib, argparse, asyncio
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'src')); sys.path.insert(0, _HERE)
import requests
from bs4 import BeautifulSoup
import fact_reasoner.core.trust.credibility_scorer as _cs
import fact_reasoner.core.trust.credibility_scorer_v3 as _v3
_cs.score_url = _v3.score_url
from state_media_eval import build_relations, make_pipeline
from granite_switch_vs_factreaser_demo import NLIFixed
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.trust.credibility_fusion import CredibilityTrustFusion
from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption

ap = argparse.ArgumentParser()
ap.add_argument('--match', default='BQ.1')
ap.add_argument('--topk', type=int, default=6)
ap.add_argument('--state', default='data/trust_eval/dynaTD_state_40v2.json')
args = ap.parse_args()
TAG = re.sub(r'\W+', '', args.match)[:12]

KEY = os.environ.get('SERPER_API_KEY')
if not KEY: raise SystemExit("SERPER_API_KEY not set")
CACHE = os.path.join(_REPO, 'data', 'trust_eval', f'serper_cache_refresh_{TAG}.json')
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

rows = [json.loads(l) for l in open(os.path.join(_REPO,'data','trust_eval','eval_dataset_fp.jsonl')) if l.strip()]
cand = [r for r in rows if args.match in (r.get('input') or '')]
if len(cand) != 1: raise SystemExit(f"--match {args.match!r} hit {len(cand)} rows; need exactly 1")
row = cand[0]
print(f"[row] gt={row['ground_truth']} atoms={len(row['atoms'])} old_ctx={len(row['contexts'])}")
print(f"[row] {row['input'][:140]}\n", flush=True)

def serper(q):
    k = hashlib.sha1(q.encode()).hexdigest()
    if k in cache: return cache[k]
    r = requests.post("https://google.serper.dev/search",
                      headers={"X-API-KEY": KEY, "Content-Type": "application/json"},
                      json={"q": q, "num": args.topk}, timeout=30)
    r.raise_for_status()
    hits = [{"title":h.get("title",""),"link":h.get("link",""),"snippet":h.get("snippet","")}
            for h in r.json().get("organic", [])[:args.topk]]
    cache[k] = hits; json.dump(cache, open(CACHE,'w')); time.sleep(0.5)
    return hits

def page_text(url, cap=1200):
    try:
        rr = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if rr.ok and 'text/html' in rr.headers.get('Content-Type',''):
            s = BeautifulSoup(rr.text,'html.parser')
            for t in s(['script','style','nav','header','footer']): t.decompose()
            return re.sub(r'\s+',' ', s.get_text(' ')).strip()[:cap]
    except Exception: pass
    return ""

new_ctx, amap = [], {}
for ai, a in enumerate(row['atoms']):
    ids = []
    for h in serper(a['text'][:150]):
        snip = (h.get('snippet') or '').strip()
        page = page_text(h['link'])
        body = (snip + ' ' + page).strip() if snip else page
        if not body: continue
        cid = f"a{ai}_n{len(ids)}"
        new_ctx.append({"id":cid,"text":body,"title":h['title'],"link":h['link']}); ids.append(cid)
    amap[a['id']] = ids
    print(f"  {a['id']}: {len(ids)} fresh contexts | {a['text'][:65]}", flush=True)

newrow = dict(row); newrow['contexts'] = new_ctx
newrow['atoms'] = [dict(a, contexts=amap[a['id']]) for a in row['atoms']]
open(os.path.join(_REPO,'data','trust_eval',f'refresh_{TAG}.jsonl'),'w').write(json.dumps(newrow)+"\n")
print(f"\n[frozen] data/trust_eval/refresh_{TAG}.jsonl\n", flush=True)

llm = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS:512})
nli = NLIFixed(llm)
scorer = CredibilityTrustFusion(state_path=args.state)

def build(data, sc=None):
    atoms = {a['id']: Atom(id=a['id'], text=a['text']) for a in data['atoms']}
    lut = {c['id']: c for c in data['contexts']}; ctxs = {}
    for a in data['atoms']:
        atom, mine = atoms[a['id']], []
        for cid in a['contexts'][:int(os.environ.get('CTX_CAP','10'))]:
            if cid not in lut: continue
            c = lut[cid]
            ctx = Context(id=cid, atom=atom, text=c['text'], title=c.get('title',''), link=c.get('link',''))
            ctx.set_probability(sc.score(ctx) if sc else 0.9)
            ctxs[cid] = ctx; mine.append(ctx)
        atom.add_contexts(mine)
    return atoms, ctxs

def arm(sc, label):
    atoms, ctxs = build(newrow, sc)
    rels = build_relations(atoms=atoms, contexts=ctxs, nli_extractor=nli,
                           rel_atom_context=True, rel_context_context=False,
                           use_summarized_contexts=False)
    if not rels:
        print(f"[{label}] NO RELATIONS"); return None, None
    live = {r.source.id for r in rels}
    ctxs = {k:v for k,v in ctxs.items() if k in live}
    rels = [r for r in rels if r.source.id in ctxs]
    for a in atoms.values():
        a.contexts = {c.id:c for c in a.get_contexts().values() if c.id in ctxs}
    _, marg = make_pipeline(atoms, ctxs, rels, row['ground_truth']).score()
    post = {m['variable']: m['probabilities'][1] for m in marg if m['variable'] in atoms}
    e = sum(1 for r in rels if r.type=='entailment'); c = sum(1 for r in rels if r.type=='contradiction')
    print(f"\n[{label}] relations={len(rels)} entail={e} contradict={c}")
    for r in sorted(rels, key=lambda x:(x.target.id, x.source.id)):
        print(f"    {r.source.id:9s} -> {r.target.id}  {r.type:13s} {float(r.probability):.4f}")
    return post, {'n':len(rels),'entail':e,'contradict':c}

pt, st = arm(scorer, "ConTrust")
pv, sv = arm(None,   "Vanilla ")
OLD = {'a0':(0.000,0.000),'a1':(0.998,1.000),'a2':(0.500,0.500),
       'a3':(1.000,1.000),'a4':(1.000,1.000),'a5':(0.572,0.093)}
atxt = {a['id']:a['text'] for a in row['atoms']}
print("\n" + "="*86)
print(f"{'atom':6s}{'OLDtrust':>10s}{'OLDvan':>9s}{'NEWtrust':>10s}{'NEWvan':>9s}   atom text")
for aid in sorted(pt or {}):
    o = OLD.get(aid,(None,None))
    print(f"{aid:6s}{o[0]:>10.3f}{o[1]:>9.3f}{pt[aid]:>10.3f}{pv[aid]:>9.3f}   {atxt[aid][:40]}")
if pt:
    ft = sum(1 for p in pt.values() if p>0.5)/len(pt); fv = sum(1 for p in pv.values() if p>0.5)/len(pv)
    print(f"\nrow verdict  ConTrust={'S' if ft>0.5 else 'NS'}  Vanilla={'S' if fv>0.5 else 'NS'}  gold={row['ground_truth']}")
    json.dump({'gt':row['ground_truth'],'new_trust':pt,'new_vanilla':pv,'stats_trust':st,
               'stats_vanilla':sv,'n_new_contexts':len(new_ctx)},
              open(os.path.join(_REPO,'data','trust_eval',f'refresh_{TAG}_report.json'),'w'), indent=1)
    print(f"[report] data/trust_eval/refresh_{TAG}_report.json")
