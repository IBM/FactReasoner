"""LIAR augmented rows. LIAR_AUG_JSONL selects the condition:
   .../liar_aug_400.jsonl          organic (fact-checkers included)
   .../liar_blocked_aug_400.jsonl  fact-check domains filtered out
"""
import json, os
def load():
    p = os.environ.get('LIAR_AUG_JSONL')
    if not p: raise SystemExit("set LIAR_AUG_JSONL")
    rows=[json.loads(l) for l in open(p) if l.strip()]
    print("[liar_aug loader] %s rows=%d" % (os.path.basename(p), len(rows)), flush=True)
    return rows
