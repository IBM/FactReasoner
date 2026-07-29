"""Serper-augment ExpertQA/AVeriTeC rows -> FROZEN jsonl (multi-source condition).
Stratified: keeps ALL minority-class rows + fills with majority to N.
Writes BOTH <ds>_aug_<N>.jsonl (augmented) and <ds>_base_<N>.jsonl (same rows, untouched)
so aug-vs-base evals compare identical rows.
Usage: python3 experiments/consensus_trust/augment_with_serper.py expertqa 350 [--snippets-only]
"""
import json, os, re, sys, time, hashlib
import requests
from bs4 import BeautifulSoup

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'experiments', 'consensus_trust', 'loaders'))

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'expertqa'
N       = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SNIPPETS_ONLY = '--snippets-only' in sys.argv
TOP_K, FETCH_CHARS = int(os.environ.get('TOP_K','5')), int(os.environ.get('FETCH_CHARS','1500'))
KEY = os.environ.get('SERPER_API_KEY')
if not KEY: raise SystemExit("SERPER_API_KEY not set")

if DATASET == 'expertqa':
    from expertqa_loader import load
elif DATASET == 'averitec':
    from averitec_loader import load
elif DATASET == 'liar':
    from liar_loader import load
else:
    raise SystemExit(f"unknown dataset {DATASET!r}")

_SUF = '_blocked' if os.environ.get('BLOCK_FACTCHECK','0')=='1' else ''
CACHE_PATH = os.path.join(_REPO,'data','trust_eval',f'serper_cache_{DATASET}{_SUF}.json')
PAGE_CACHE_PATH = os.path.join(_REPO,'data','trust_eval',f'page_cache_{DATASET}.json')
OUT_AUG  = os.path.join(_REPO,'data','trust_eval',f'{DATASET}{_SUF}_aug_{N}.jsonl')
OUT_BASE = os.path.join(_REPO,'data','trust_eval',f'{DATASET}{_SUF}_base_{N}.jsonl')
cache  = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
pcache = json.load(open(PAGE_CACHE_PATH)) if os.path.exists(PAGE_CACHE_PATH) else {}

def gt_of(r):
    g = r['ground_truth']
    return next(iter(g.values())) if isinstance(g, dict) else g

def serper(q):
    k = hashlib.sha1(q.encode()).hexdigest()
    if k in cache: return cache[k]
    r = requests.post("https://google.serper.dev/search",
                      headers={"X-API-KEY": KEY, "Content-Type": "application/json"},
                      json={"q": q, "num": TOP_K}, timeout=30)
    r.raise_for_status()
    _blocked = os.environ.get('BLOCK_FACTCHECK','0') == '1'
    _BL = ('politifact.com','snopes.com','factcheck.org','fullfact.org','apnews.com/hub/ap-fact-check',
           'washingtonpost.com/news/fact-checker','truthorfiction.com','leadstories.com',
           'checkyourfact.com','factcheck.afp.com','reuters.com/fact-check','usatoday.com/story/news/factcheck')
    _org = r.json().get("organic", [])
    if _blocked:
        _org = [h for h in _org if not any(b in (h.get("link","") or "").lower() for b in _BL)]
    hits = [{"title":h.get("title",""),"link":h.get("link",""),"snippet":h.get("snippet","")}
            for h in _org[:TOP_K]]
    cache[k] = hits
    if len(cache) % 10 == 0: json.dump(cache, open(CACHE_PATH,'w'))
    time.sleep(0.6)
    return hits

def page_text(url):
    if url in pcache: return pcache[url]
    text = ""
    try:
        rr = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if rr.ok and 'text/html' in rr.headers.get('Content-Type',''):
            soup = BeautifulSoup(rr.text, 'html.parser')
            for t in soup(['script','style','nav','header','footer']): t.decompose()
            text = re.sub(r'\s+',' ', soup.get_text(' ')).strip()[:FETCH_CHARS]
    except Exception:
        text = ""
    pcache[url] = text
    if len(pcache) % 10 == 0: json.dump(pcache, open(PAGE_CACHE_PATH,'w'))
    return text

allrows = load()
ns_rows = [r for r in allrows if gt_of(r) == 'NS']
s_rows  = [r for r in allrows if gt_of(r) == 'S']
rows = (ns_rows + s_rows)[:max(N, len(ns_rows))]
print(f"[aug] stratified: {len([r for r in rows if gt_of(r)=='NS'])} NS + "
      f"{len([r for r in rows if gt_of(r)=='S'])} S = {len(rows)} rows", flush=True)

with open(OUT_AUG,'w') as fa, open(OUT_BASE,'w') as fb:
    for i, r in enumerate(rows):
        fb.write(json.dumps(r) + "\n")
        base_ctx = list(r['contexts']); next_id = 0
        new_ctx, new_ids = [], []
        for a_idx, atom in enumerate(r['atoms']):
            try: hits = serper(atom['text'][:120])
            except Exception as e:
                print(f"[{i+1}] serper FAIL: {str(e)[:60]}", flush=True); hits = []
            for h in hits:
                body = h['snippet'] if SNIPPETS_ONLY else (page_text(h['link']) or h['snippet'])
                if not body: continue
                cid = f"r{i}_web{next_id}"; next_id += 1
                new_ctx.append({"id":cid,"text":body,"title":h['title'],"link":h['link']})
                new_ids.append((a_idx, cid))
        out = dict(r); out['contexts'] = base_ctx + new_ctx
        atoms = [dict(a) for a in r['atoms']]
        for a_idx, a in enumerate(atoms):
            base_ids = a.get('contexts') or [c['id'] for c in base_ctx]
            a['contexts'] = list(base_ids) + [cid for j,cid in new_ids if j == a_idx]
        out['atoms'] = atoms; out['augmented'] = True
        fa.write(json.dumps(out) + "\n")
        if (i+1) % 25 == 0:
            print(f"[{i+1}/{len(rows)}] ctx {len(base_ctx)} -> {len(base_ctx)+len(new_ctx)}", flush=True)

json.dump(cache, open(CACHE_PATH,'w')); json.dump(pcache, open(PAGE_CACHE_PATH,'w'))
print(f"[aug] wrote {OUT_AUG}\n[aug] wrote {OUT_BASE}", flush=True)
