import json, os, glob
def load():
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..'))
    cands = glob.glob(os.path.join(repo,'data','trust_eval','expertqa_base_*.jsonl'))
    if not cands: raise SystemExit("no expertqa_base_*.jsonl — run augment_with_serper.py first")
    path = max(cands, key=os.path.getmtime)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"[expertqa_base loader] {os.path.basename(path)} rows={len(rows)}", flush=True)
    return rows
