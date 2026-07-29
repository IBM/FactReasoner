"""Phase 1: fresh top-K retrieval for ALL 40 rows -> frozen jsonl. Resumable.
No LLM calls. Run before refresh_eval_40.py.
Usage:  python3 experiments/consensus_trust/refresh_all_40.py --topk 20
"""
import os, sys, json, re, time, hashlib, argparse
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup

_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.abspath(os.path.join(_HERE,'..','..'))
ap=argparse.ArgumentParser()
ap.add_argument('--topk',type=int,default=20)
ap.add_argument('--src',default='fp',choices=['fp','cc'])
ap.add_argument('--workers',type=int,default=8)
a=ap.parse_args()

KEY=os.environ.get('SERPER_API_KEY')
if not KEY: raise SystemExit("SERPER_API_KEY not set")
IN  = os.path.join(_REPO,'data','trust_eval',f'eval_dataset_{a.src}.jsonl')
OUT = os.path.join(_REPO,'data','trust_eval',f'eval_dataset_{a.src}_refresh{a.topk}.jsonl')
CACHE=os.path.join(_REPO,'data','trust_eval',f'serper_cache_refresh40_{a.topk}.json')
cache=json.load(open(CACHE)) if os.path.exists(CACHE) else {}
rows=[json.loads(l) for l in open(IN) if l.strip()]
print(f"[in] {len(rows)} rows from {IN}")

done={}
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            r=json.loads(l); done[(r.get('input') or '')[:60]]=r
    print(f"[resume] {len(done)} rows already retrieved")

def serper(q):
    k=hashlib.sha1((q+str(a.topk)).encode()).hexdigest()
    if k in cache: return cache[k]
    r=requests.post("https://google.serper.dev/search",
        headers={"X-API-KEY":KEY,"Content-Type":"application/json"},
        json={"q":q,"num":a.topk},timeout=30)
    r.raise_for_status()
    hits=[{"title":h.get("title",""),"link":h.get("link",""),"snippet":h.get("snippet","")}
          for h in r.json().get("organic",[])[:a.topk]]
    cache[k]=hits; json.dump(cache,open(CACHE,'w')); time.sleep(0.4)
    return hits

def page(url,cap=1200):
    try:
        rr=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        if rr.ok and 'text/html' in rr.headers.get('Content-Type',''):
            s=BeautifulSoup(rr.text,'html.parser')
            for t in s(['script','style','nav','header','footer']): t.decompose()
            return re.sub(r'\s+',' ',s.get_text(' ')).strip()[:cap]
    except Exception: pass
    return ""

fout=open(OUT,'a')
t0=time.time()
for i,row in enumerate(rows):
    key=(row.get('input') or '')[:60]
    if key in done:
        print(f"[{i+1}/{len(rows)}] skip (done)",flush=True); continue
    hits_per_atom=[serper(at['text'][:150]) for at in row['atoms']]
    flat=[h for hs in hits_per_atom for h in hs]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        bodies=list(ex.map(lambda h: page(h['link']), flat))
    it=iter(bodies); new_ctx=[]; amap={}
    for ai,(at,hs) in enumerate(zip(row['atoms'],hits_per_atom)):
        ids=[]
        for h in hs:
            b=next(it)
            snip=(h.get('snippet') or '').strip()
            body=(snip+' '+b).strip() if snip else b
            if not body: continue
            cid=f"a{ai}_r{len(ids)}"
            new_ctx.append({"id":cid,"text":body,"title":h['title'],"link":h['link']}); ids.append(cid)
        amap[at['id']]=ids
    nr=dict(row); nr['contexts']=new_ctx
    nr['atoms']=[dict(at,contexts=amap[at['id']]) for at in row['atoms']]
    fout.write(json.dumps(nr)+"\n"); fout.flush()
    print(f"[{i+1}/{len(rows)}] {len(new_ctx)} contexts  ({time.time()-t0:.0f}s elapsed)",flush=True)
fout.close()
print(f"\n[out] {OUT}")
