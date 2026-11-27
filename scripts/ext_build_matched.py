"""Matched order datasets for the coherence paper (second review, Phase C1-C3).

Two datasets, each item copied twice (ids suffixed ``__r1``/``__r2``) and shuffled, so that the
re-mining noise floor (two mines of the same text) and the permutation effect are measured in one
interleaved run, for the miner and for the judge alike.

``contradoc-sections``  ContraDoc's clean Wikipedia documents with at least three headed sections. The
    first review of the order check found that splitting at blank lines makes every section header
    a paragraph and a claim, so a permutation separated headers from their text. Here the title
    and the lead stay first, each header stays with its section, whole headed sections are permuted, and headers are
    not claims. Stories are excluded: their paragraphs are not self-contained.
``loco-matched``  The LoCoBench originals (rung 0 of the 180 non-ORDER test families) and their two
    paragraph permutations, from data/ext/loco-shuffle (its ``sent`` variant is not used).

    python3 scripts/ext_build_matched.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fact_reasoner.exteval import adapters as AD  # noqa: E402

SEED = 20261128
HEADER = re.compile(r"^=( =)* .+ =( =)*$")


def sections(text):
    """(title, [section, ...]) with each section a list of paragraphs, its header first."""
    paras = AD.paragraphs(text)
    title = paras[0] if paras and HEADER.match(paras[0]) else None
    body = paras[1:] if title else paras
    secs, cur = [], []
    for p in body:
        if HEADER.match(p) and cur:
            secs.append(cur)
            cur = []
        cur.append(p)
    if cur:
        secs.append(cur)
    return title, secs


def doc_item(iid, title, secs, **kw):
    paras = ([title] if title else []) + [p for s in secs for p in s]
    claims = [c for p in paras if not HEADER.match(p) for c in AD.sentences(p)]
    return AD.item(iid, "\n\n".join(paras), claims, **kw)


def contradoc_sections(rng):
    d = json.load(open(os.path.join(REPO, "data", "ext", "contradoc", "raw", "ContraDoc.json")))
    out = []
    for uid, x in sorted(d["neg"].items()):
        if x.get("doc_type") != "wiki":
            continue
        title, secs = sections(x["text"])
        # The lead (the paragraphs before the first header) stays with the title; only headed
        # sections move.
        lead = [] if secs and HEADER.match(secs[0][0]) else secs[:1]
        body = secs[len(lead):]
        if len(body) < 3:
            continue
        common = dict(dataset="contradoc", check="sections", group=f"sec-{uid}", doc_type="wiki")
        variants = [("orig", lead + body)]
        seen = [list(range(len(body)))]
        for k in range(2):
            p = AD.derangement(len(body), rng, seen)
            seen.append(p)
            variants.append((f"p{k + 1}", lead + [body[i] for i in p]))
        for v, ss in variants:
            for r in ("r1", "r2"):
                it = doc_item(f"cds-{uid}-{v}__{r}", title, ss, variant=v, repeat=r, **common)
                out.append(it)
    return out


def loco_matched():
    src = [json.loads(l) for l in open(os.path.join(REPO, "data", "ext", "loco-shuffle", "items.jsonl"))
           if l.strip()]
    out = []
    for x in src:
        if x["expected"]["variant"] not in ("orig", "p1", "p2"):
            continue
        for r in ("r1", "r2"):
            y = json.loads(json.dumps(x))
            y["id"] = y["name"] = f"{x['id']}__{r}"
            y["expected"]["repeat"] = r
            y["expected"]["matched_of"] = x["id"]
            out.append(y)
    return out


def main() -> int:
    rng = random.Random(SEED)
    for name, items, note in (
            ("contradoc-sections", contradoc_sections(rng),
             "ContraDoc clean Wikipedia documents with >= 3 headed sections; whole headed sections "
             "permuted, title and lead first, headers kept with their sections and not claims; stories "
             "excluded."),
            ("loco-matched", loco_matched(),
             "LoCoBench rung-0 originals and their two paragraph permutations from loco-shuffle.")):
        random.Random(SEED).shuffle(items)
        d = os.path.join(REPO, "data", "ext", name)
        AD.write(os.path.join(d, "items.jsonl"), items)
        docs = len({it["expected"]["family_id"] for it in items})
        open(os.path.join(d, "README.md"), "w").write(
            f"# data/ext/{name}\n\nCoherence paper, second review, Phase C (2026-11-28): {note} "
            "Each original and permutation is copied twice (ids suffixed `__r1`, `__r2`) and the "
            "items are shuffled, so repeats and permutations are mined, and judged, in one "
            f"interleaved run. {docs} documents, {len(items)} items. Built by "
            "`scripts/ext_build_matched.py`.\n")
        print(name, docs, "documents", len(items), "items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
