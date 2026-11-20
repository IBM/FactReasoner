# Usage: python scripts/locobench_slot3_reset_families.py <store dir> <fid,fid,...>
"""Archive and remove families from a slot3 store so the CLI regenerates them from P1."""
import json, os, sys

D, fids = sys.argv[1], set(sys.argv[2].split(","))
sup = os.path.join(D, "rejected", "superseded"); os.makedirs(sup, exist_ok=True)

def atomic(path, text):
    open(path + ".tmp", "w").write(text); os.replace(path + ".tmp", path)

items = [json.loads(l) for l in open(os.path.join(D, "items.jsonl")) if l.strip()]
man = json.load(open(os.path.join(D, "families.json")))
state = json.load(open(os.path.join(D, "state.json")))
for fid in sorted(fids):
    arch = {"items": [it for it in items if it["expected"]["family_id"] == fid],
            "manifest": next((f for f in man["families"] if f["family_id"] == fid), None),
            "state": state["families"].get(fid)}
    json.dump(arch, open(os.path.join(sup, f"{fid}.lost-primary.json"), "w"), indent=1)
    print(fid, "archived", len(arch["items"]), "items")
atomic(os.path.join(D, "items.jsonl"),
       "".join(json.dumps(it) + "\n" for it in items if it["expected"]["family_id"] not in fids))
man["families"] = [f for f in man["families"] if f["family_id"] not in fids]
atomic(os.path.join(D, "families.json"), json.dumps(man, indent=2))
for fid in fids:
    state["families"].pop(fid, None)
atomic(os.path.join(D, "state.json"), json.dumps(state, indent=2))
