"""A non-sequitur control for LoCoBench (coherence paper, second review, Phase C4).

For the rung-0 response of each increasing LoCoBench test family, pick two TRUE claims that no
gold relation joins, from different paragraphs and with the least word overlap, and append to the
paragraph of the second a sentence asserting that it follows from the first:

    "That this is so follows from the fact that <first claim>."

The claims are unchanged (the sentence restates a claim the response already makes), so the edit
adds one asserted inference and nothing else. Couplings are material, so if the miner extracts it
LCS must rise, although the inference is irrelevant. The control measures how often that happens,
and whether a judge given the verdicts prefers the response without it.

    python3 scripts/ext_build_nonseq.py
"""

from __future__ import annotations

import itertools
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "locobench-slot3-test")
OUT = os.path.join(REPO, "data", "ext", "loco-nonseq")
INC = ("CONFLICT", "CHAIN", "SIGN")


def words(t):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower())}


def main() -> int:
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "items.jsonl")) if l.strip()}
    fams = json.load(open(os.path.join(DATA, "families.json")))["families"]
    out = []
    for f in fams:
        if f["primary"] not in INC:
            continue
        lad = next(l for l in f["ladders"] if l["type"] == f["primary"])
        it = items[lad["item_ids"][0]]
        paras = [p for p in it["response"].split("\n\n") if p.strip()]
        atoms = sorted(it["atoms"], key=lambda a: a["pos"])
        rel = {frozenset((r["source_id"], r["target_id"])) for r in it["relations"]}

        def para_of(a):
            for k, p in enumerate(paras):
                if a["text"][:40] in p:
                    return k
            return None
        true = [a for a in atoms if a["factual"] and a.get("stance") != "denied" and para_of(a) is not None]
        best = None
        for a, b in itertools.permutations(true, 2):
            if frozenset((a["id"], b["id"])) in rel or para_of(a) == para_of(b):
                continue
            ov = len(words(a["text"]) & words(b["text"]))
            if best is None or ov < best[0]:
                best = (ov, a, b)
        if best is None:
            continue
        _ov, a, b = best
        premise = a["text"].rstrip(".")
        premise = premise[0].lower() + premise[1:] if premise[:2] != premise[:2].upper() else premise
        k = para_of(b)
        paras2 = list(paras)
        paras2[k] = paras2[k].rstrip() + f" That this is so follows from the fact that {premise}."
        y = json.loads(json.dumps(it))
        y["id"] = y["name"] = f"nsq-{it['id']}"
        y["response"] = "\n\n".join(paras2)
        y["relations"] = []
        y["expected"] = {"family_id": f["family_id"], "dataset": "locobench", "check": "nonseq",
                         "orig_item": it["id"], "premise_atom": a["id"], "conclusion_atom": b["id"],
                         "word_overlap": _ov, "primary": f["primary"]}
        out.append(y)
        # The unedited response, mined in the same run (separate runs can drift).
        z = json.loads(json.dumps(it))
        z["id"] = z["name"] = f"nsq0-{it['id']}"
        z["relations"] = []
        z["expected"] = dict(y["expected"], check="nonseq_orig")
        out.append(z)
    import random
    random.Random(20261128).shuffle(out)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "items.jsonl"), "w") as fh:
        for y in out:
            fh.write(json.dumps(y) + "\n")
    open(os.path.join(OUT, "README.md"), "w").write(
        "# data/ext/loco-nonseq\n\nCoherence paper, second review, Phase C4 (2026-11-28): a "
        "non-sequitur control. For each increasing LoCoBench test family's rung-0 response, one "
        "sentence asserting that a true claim follows from another, unrelated true claim (no gold "
        "relation, different paragraphs, least word overlap) is appended to the conclusion's "
        f"paragraph; claims are unchanged. The unedited responses are included (`nsq0-` ids) and mined in the same run. {len(out)} items. Built by `scripts/ext_build_nonseq.py`.\n")
    print(len(out), "items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
