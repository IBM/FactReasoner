"""A non-sequitur control for LoCoBench (coherence paper, second review Phase C4; fixed in the third
review, N16).

For the rung-0 response of each increasing LoCoBench test family, two TRUE claims that no gold
relation joins are chosen, and a sentence naming both is inserted directly after the conclusion's
sentence:

    "That <conclusion> follows from the fact that <premise>."

The first version of this control appended "That this is so follows from ..." to the end of the
conclusion's paragraph, where "this" often referred to another sentence (N16). Now the conclusion is
named, so the asserted inference is unambiguous. Two variants:

* ``nonseq``: the pair with the least word overlap, from different paragraphs (an irrelevant
  inference);
* ``valid`` (CHAIN families only): a positive control. The CHAIN ladder's top rung adds valid
  inferences between true claims that rung 0 does not draw; one of them is stated in rung 0 with the
  same template, so the two variants differ only in whether the asserted inference is valid.

The claims are unchanged, and the unedited response is included (``nsq0-``) so that all versions
are mined in one run.

    python3 scripts/ext_build_nonseq.py
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "locobench-slot3-test")
OUT = os.path.join(REPO, "data", "ext", "loco-nonseq")
INC = ("CONFLICT", "CHAIN", "SIGN")


def words(t):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower())}


def clause(text):
    t = text.strip().rstrip(".")
    return t[0].lower() + t[1:] if len(t) > 1 and not t[1].isupper() else t


def main() -> int:
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "items.jsonl")) if l.strip()}
    fams = json.load(open(os.path.join(DATA, "families.json")))["families"]
    out = []
    for f in fams:
        if f["primary"] not in INC:
            continue
        lad = next(l for l in f["ladders"] if l["type"] == f["primary"])
        it = items[lad["item_ids"][0]]
        resp = it["response"]
        atoms = sorted(it["atoms"], key=lambda a: a["pos"])
        rel = {frozenset((r["source_id"], r["target_id"])) for r in it["relations"]}
        paras = [p for p in resp.split("\n\n") if p.strip()]

        def para_of(a):
            return next((k for k, p in enumerate(paras) if a["text"][:40] in p), None)
        true = [a for a in atoms if a["factual"] and a.get("stance") != "denied"
                and para_of(a) is not None and a["text"] in resp]
        cand = [(len(words(a["text"]) & words(b["text"])), a, b)
                for a, b in itertools.permutations(true, 2) if frozenset((a["id"], b["id"])) not in rel]
        if not cand:
            continue
        far = [c for c in cand if para_of(c[1]) != para_of(c[2])] or cand
        picks = {"nonseq": min(far, key=lambda c: c[0])}
        if f["primary"] == "CHAIN":
            top = items[lad["item_ids"][2]]
            have = {frozenset((r["source_id"], r["target_id"])) for r in it["relations"]}
            byid = {a["id"]: a for a in atoms}
            for r in top["relations"]:
                if (r["level1_coupling"] == "entailment" and
                        frozenset((r["source_id"], r["target_id"])) not in have):
                    a, b = byid.get(r["source_id"]), byid.get(r["target_id"])
                    if a and b and a in true and b in true:
                        picks["valid"] = (len(words(a["text"]) & words(b["text"])), a, b)
                        break
        common = {"family_id": f["family_id"], "dataset": "locobench", "orig_item": it["id"],
                  "primary": f["primary"]}
        z = json.loads(json.dumps(it))
        z["id"] = z["name"] = f"nsq0-{it['id']}"
        z["relations"] = []
        z["expected"] = dict(common, check="nonseq_orig")
        out.append(z)
        for kind, (ov, a, b) in picks.items():
            sent = f" That {clause(b['text'])} follows from the fact that {clause(a['text'])}."
            pos = resp.index(b["text"]) + len(b["text"])
            y = json.loads(json.dumps(it))
            y["id"] = y["name"] = f"nsq-{kind}-{it['id']}"
            y["response"] = resp[:pos] + sent + resp[pos:]
            y["relations"] = []
            y["expected"] = dict(common, check=kind, premise_atom=a["id"], conclusion_atom=b["id"],
                                 word_overlap=ov)
            out.append(y)
    random.Random(20261128).shuffle(out)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "items.jsonl"), "w") as fh:
        for y in out:
            fh.write(json.dumps(y) + "\n")
    n = sum(y["expected"]["check"] == "nonseq" for y in out)
    print("valid:", sum(y["expected"]["check"] == "valid" for y in out))
    open(os.path.join(OUT, "README.md"), "w").write(
        "# data/ext/loco-nonseq\n\nCoherence paper, non-sequitur control (third review, N16; "
        "2026-11-28). For each increasing LoCoBench test family's rung-0 response, a sentence "
        "\"That <conclusion> follows from the fact that <premise>.\" is inserted directly after the "
        "conclusion's sentence, naming both claims. `nonseq`: the unlinked pair of true claims with "
        "the least word overlap, from different paragraphs. `valid` (CHAIN families): an entailment "
        "the ladder's top rung adds, stated in rung 0 with the same template (positive control). Claims are unchanged; the unedited responses (`nsq0-`) "
        f"are mined in the same run. {n} families, {len(out)} items. Built by "
        "`scripts/ext_build_nonseq.py`.\n")
    print(n, "families", len(out), "items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
