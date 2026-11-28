"""Adapters: external datasets -> LoCoBench-format items for the LCS pipeline.

Every adapter writes items that ``fact_reasoner.locoeval.cli`` can mine and that
``scripts/ext_eval.py`` scores. An item is

    {"id", "name", "source", "response", "num_atoms",
     "atoms": [{"id": "a<k>", "pos": k, "text", "factual": True, "stance": "asserted"}],
     "relations": [], "non_relations": [],
     "expected": {"family_id": <group>, "dataset", "check", ...},   # gold lives here
     "meta": {...}}

Gold relations, when a dataset has them, are stored under ``expected["gold"]`` as
``{"source": "a<i>", "target": "a<j>", "type": <coupling>, "sense": <label>}`` and NOT under
``relations``, so the mining run never mistakes them for its own gold arm.

Claims are sentences for ContraDoc, ChatProtect and SummEval (their labels are per sentence),
gold propositions for QT30 and argument spans for GDTB. Raw inputs are read from
``data/ext/<name>/raw/`` (gitignored); every random choice uses ``SEED``.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

SEED = 20261124


# ----------------------------------------------------------------------------- helpers

def sentences(text: str) -> list[str]:
    """Sentence-split one paragraph with NLTK punkt; fragments under 3 words join the previous one.
    A missing space after a sentence end ("ended.The next") is restored first."""
    import nltk

    text = re.sub(r"([a-z0-9][.!?][\"')\]]?)([A-Z][a-z])", r"\1 \2", text)
    out: list[str] = []
    for s in nltk.sent_tokenize(text):
        s = s.strip()
        if not s:
            continue
        if out and len(s.split()) < 3:
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _span_hits(response: str, atoms: list[dict], ev: str) -> list[dict]:
    """Claims inside the evidence: character-span overlap on whitespace-normalised text, with
    the evidence's trailing punctuation dropped; failing that, the single most similar claim
    (difflib ratio >= 0.6)."""
    import difflib

    response, ev = _norm(response), _norm(ev).rstrip(" .!?;:,")
    atoms = [{**a, "text": _norm(a["text"])} for a in atoms]
    hits = _span_hits_exact(response, atoms, ev)
    if not hits and atoms:
        best = max(atoms, key=lambda a: difflib.SequenceMatcher(None, a["text"], ev).ratio())
        if difflib.SequenceMatcher(None, best["text"], ev).ratio() >= 0.6:
            hits = [best]
    return hits


def _span_hits_exact(response: str, atoms: list[dict], ev: str) -> list[dict]:
    spans, start = [], 0
    while ev and (k := response.find(ev, start)) >= 0:
        spans.append((k, k + len(ev)))
        start = k + 1
    hits, cursor = [], 0
    for a in atoms:
        k = response.find(a["text"], cursor)
        if k < 0:
            continue
        lo, hi = k, k + len(a["text"])
        cursor = hi
        inside = sum(max(0, min(hi, e) - max(lo, s)) for s, e in spans)
        if inside >= 0.5 * (hi - lo):
            hits.append(a)
    return hits


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


def atoms_of(texts: list[str]) -> list[dict]:
    return [{"id": f"a{k}", "pos": k, "text": t, "factual": True, "stance": "asserted"}
            for k, t in enumerate(texts)]


def item(iid: str, response: str, claims: list[str], *, dataset: str, check: str, group: str,
         name: str = "", meta: dict | None = None, **expected) -> dict:
    return {"id": iid, "name": name or iid, "source": dataset, "response": response,
            "num_atoms": len(claims), "atoms": atoms_of(claims), "relations": [],
            "non_relations": [],
            "expected": {"family_id": group, "dataset": dataset, "check": check, **expected},
            "meta": meta or {}}


def sentence_item(iid, paras: list[str], **kw) -> dict:
    """An item whose claims are the sentences of ``paras``, response = paragraphs joined."""
    claims = [s for p in paras for s in sentences(p)]
    return item(iid, "\n\n".join(paras), claims, **kw)


def write(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def derangement(n: int, rng: random.Random, avoid: list[list[int]] = ()) -> list[int]:
    """A permutation of range(n) that moves at least half the blocks and differs from ``avoid``."""
    for _ in range(1000):
        p = list(range(n))
        rng.shuffle(p)
        if sum(i != v for i, v in enumerate(p)) >= max(2, n // 2) and p not in avoid:
            return p
    raise ValueError(f"no permutation of {n} blocks")


# ----------------------------------------------------------------------------- Check 1

def loco_shuffles(repo: str, n_perm: int = 2) -> list[dict]:
    """Check 1a/1b: the rung-0 item of every non-ORDER LoCoBench test family.

    Per family: ``orig`` (the unchanged text, re-mined to measure the noise floor), ``p1``..``pk``
    (paragraph permutations) and ``sent`` (sentences shuffled within each paragraph). Atoms, ids
    and gold relations are those of the original item; only the prose order changes.
    """
    rng = random.Random(SEED)
    src = [json.loads(l) for l in open(os.path.join(repo, "data", "locobench-slot3-test",
                                                     "items.jsonl"))]
    out = []
    for it in src:
        if not it["id"].endswith("-t0") or it["expected"].get("primary") == "ORDER":
            continue
        paras = [p for p in it["response"].split("\n\n") if p.strip()]
        fam = it["expected"]["family_id"]
        base = {k: it[k] for k in ("atoms", "num_atoms", "relations", "non_relations")}

        def variant(tag, text):
            return {**base, "id": f"loco-{fam}-{tag}", "name": f"{it['name']} [{tag}]",
                    "source": "locobench-slot3-test", "response": text,
                    "expected": {"family_id": fam, "dataset": "locobench", "check": "shuffle",
                                 "variant": tag, "orig_item": it["id"],
                                 "primary": it["expected"].get("primary")},
                    "meta": {"paragraphs": len(paras)}}

        out.append(variant("orig", it["response"]))
        seen = [list(range(len(paras)))]
        for k in range(n_perm):
            p = derangement(len(paras), rng, seen)
            seen.append(p)
            out.append(variant(f"p{k + 1}", "\n\n".join(paras[i] for i in p)))
        shuffled = []
        for para in paras:
            s = sentences(para)
            if len(s) > 1:
                s = [s[i] for i in derangement(len(s), rng)] if len(s) > 2 else s[::-1]
            shuffled.append(" ".join(s))
        out.append(variant("sent", "\n\n".join(shuffled)))
    return out


def contradoc(raw: str, n_shuffle: int = 150, n_perm: int = 2) -> list[dict]:
    """Checks 1c and 2. ``pos`` docs: Insert -> pair (a = evidence deleted, b = as released);
    Replace -> b only (recall). ``neg`` docs: all kept (false positives); ``n_shuffle`` of those
    with >= 3 paragraphs also get ``n_perm`` paragraph permutations (Check 1c)."""
    rng = random.Random(SEED + 1)
    d = json.load(open(os.path.join(raw, "ContraDoc.json")))
    out = []
    for uid, x in sorted(d["pos"].items()):
        paras = paragraphs(x["text"])
        ev = x["evidence"].strip()
        common = dict(dataset="contradoc", group=uid, doc_type=x["doc_type"],
                      contra_plug=x["contra_plug"], scope=x["scope"],
                      contra_type=x["contra_type"], evidence=ev)
        # The evidence (the inserted or replacing sentence) is verbatim in the text: cut it out
        # as its own claim and sentence-split the text around it, so it never merges with a
        # neighbour (ContraDoc insertions often lack the space or the full stop that would
        # separate them).
        claims, hit_ids = [], set()
        for para in paras:
            pieces = para.split(ev) if ev and ev in para else [para]
            for k, piece in enumerate(pieces):
                claims += sentences(piece) if piece.strip() else []
                if k < len(pieces) - 1:
                    hit_ids.add(f"a{len(claims)}")
                    claims.append(ev)
        b = item(f"cd-{uid}-b", "\n\n".join(paras), claims, check="contradoc", variant="b",
                 **common)
        if not hit_ids:  # not verbatim (never observed): fall back to span/similarity matching
            hit_ids = {a["id"] for a in _span_hits(b["response"], b["atoms"], ev)}
        hit = [a for a in b["atoms"] if a["id"] in hit_ids]
        b["expected"]["evidence_atoms"] = sorted(hit_ids, key=lambda i: int(i[1:]))
        out.append(b)
        if x["contra_plug"] == "Insert" and hit:
            assert ev in x["text"]
            # The clean version: the same claims minus the inserted ones, and the same prose
            # minus the inserted sentence, so the pair differs only by the insertion.
            keep = [a["text"] for a in b["atoms"] if a["id"] not in hit_ids]
            text = b["response"].replace(ev, "")
            text = re.sub(r"[ \t]{2,}", " ", re.sub(r" +([.,;:])", r"\1", text)).strip()
            out.append(item(f"cd-{uid}-a", text, keep, check="contradoc", variant="a", **common))
    negs = sorted(d["neg"].items())
    eligible = [uid for uid, x in negs if len(paragraphs(x["text"])) >= 3]
    shuf = set(rng.sample(eligible, min(n_shuffle, len(eligible))))
    for uid, x in negs:
        paras = paragraphs(x["text"])
        common = dict(dataset="contradoc", group=f"neg-{uid}", doc_type=x["doc_type"])
        out.append(sentence_item(f"cdn-{uid}-orig", paras, check="contradoc_neg",
                                 variant="orig", shuffled=uid in shuf, **common))
        if uid in shuf:
            seen = [list(range(len(paras)))]
            for k in range(n_perm):
                p = derangement(len(paras), rng, seen)
                seen.append(p)
                out.append(sentence_item(f"cdn-{uid}-p{k + 1}", [paras[i] for i in p],
                                         check="shuffle", variant=f"p{k + 1}", **common))
    return out


def summeval(raw: str) -> list[dict]:
    """Check 1d: the 1,600 SummEval summaries with their three expert ratings."""
    out = []
    for line in open(os.path.join(raw, "model_annotations.aligned.jsonl")):
        x = json.loads(line)
        ex = x["expert_annotations"]
        mean = {k: sum(a[k] for a in ex) / len(ex) for k in ("coherence", "consistency",
                                                             "fluency", "relevance")}
        text = x["decoded"].strip()
        out.append(sentence_item(f"se-{x['id']}-{x['model_id']}", [text], dataset="summeval",
                                 check="summeval", group=x["id"], model_id=x["model_id"],
                                 expert=mean))
    return out


# ----------------------------------------------------------------------------- Check 3

def chatprotect(raw: str) -> list[dict]:
    """Check 3: every annotated (s, s') pair -> d0 = prefix + s, d1 = prefix + s + s'.

    glm files carry the contradiction label (["strong"] = contradictory, [] = not);
    verifiable_sc files also record which side is non-factual (``nonfactual``).
    """
    out, seen = [], set()
    files = sorted(glob.glob(os.path.join(raw, "glm", "reconciled", "*.json")))
    files += sorted(glob.glob(os.path.join(raw, "verifiable_sc", "*.json")))
    for path in files:
        tag = os.path.basename(path).replace(".json", "")
        for x in json.load(open(path)):
            if "alt" in x:  # verifiable_sc
                s, s2 = x["text"].strip(), x["alt"].strip()
                label = x.get("gt_label")
                nonfactual = x.get("nonfactual")
            else:
                s, _, s2 = x["text"].partition("---;;;")
                s, s2 = s.strip(), s2.strip()
                label = (x.get("label") or [None])[0]
                nonfactual = None
            if not s or not s2:
                continue
            key = (x.get("prefix", "").strip(), s, s2)
            if key in seen:
                continue
            seen.add(key)
            pid = f"cp-{tag}-{x['id']}"
            prefix = x.get("prefix", "").strip()
            pre = sentences(prefix) if prefix else []
            common = dict(dataset="chatprotect", check="chatprotect", group=pid,
                          label=label, contradiction=label == "strong", nonfactual=nonfactual,
                          target=x.get("target"), file=tag)
            d0 = item(f"{pid}-d0", " ".join([prefix, s]).strip(), pre + [s], variant="d0",
                      s_atom=f"a{len(pre)}", **common)
            d1 = item(f"{pid}-d1", " ".join([prefix, s, s2]).strip(), pre + [s, s2],
                      variant="d1", s_atom=f"a{len(pre)}", s2_atom=f"a{len(pre) + 1}", **common)
            out += [d0, d1]
    return out


# ----------------------------------------------------------------------------- Check 4

QT30_COUPLING = {"RA": "entailment", "CA": "contradiction", "MA": "equivalence"}
QUESTION_ILLOCUTIONS = ("Questioning", "Pure Questioning", "Assertive Questioning",
                        "Rhetorical Questioning", "Challenging", "Pure Challenging",
                        "Assertive Challenging", "Rhetorical Challenging")


def qt30(raw: str, n_windows: int = 300, lo: int = 12, hi: int = 20) -> list[dict]:
    """Check 4a: windows of ``lo``..``hi`` consecutive propositions (I-nodes) of a QT30 nodeset.

    The response is the window's locutions (speaker-prefixed), the claims are the propositions,
    and the gold relations are RA (premise => conclusion), CA (contradiction) and MA (rephrase =
    equivalence) between propositions inside the window. Rephrases with a questioning endpoint
    are dropped (they link a question to its answer, not two equivalent claims).
    """
    rng = random.Random(SEED + 4)
    windows = []
    for path in sorted(glob.glob(os.path.join(raw, "nodesets", "*.json"))):
        g = json.load(open(path))
        nodes = {n["nodeID"]: n for n in g.get("nodes", [])}
        out_e, in_e = defaultdict(list), defaultdict(list)
        for e in g.get("edges", []):
            out_e[e["fromID"]].append(e["toID"])
            in_e[e["toID"]].append(e["fromID"])
        I = sorted((n for n in nodes.values() if n["type"] == "I"),
                   key=lambda n: int(re.sub(r"\D", "", str(n["nodeID"])) or 0))
        if len(I) < lo:
            continue
        # Each I-node's locution (via its YA illocution) and illocution label.
        loc, illoc = {}, {}
        for n in I:
            for ya in in_e[n["nodeID"]]:
                if nodes.get(ya, {}).get("type") == "YA":
                    illoc[n["nodeID"]] = nodes[ya].get("text", "")
                    for l in in_e[ya]:
                        if nodes.get(l, {}).get("type") == "L":
                            loc[n["nodeID"]] = nodes[l]["text"]
        rels = []
        for n in nodes.values():
            if n["type"] not in QT30_COUPLING:
                continue
            src = [i for i in in_e[n["nodeID"]] if nodes.get(i, {}).get("type") == "I"]
            dst = [i for i in out_e[n["nodeID"]] if nodes.get(i, {}).get("type") == "I"]
            for s in src:
                for t in dst:
                    if n["type"] == "MA" and any(any(q in illoc.get(x, "") for q in
                                                     QUESTION_ILLOCUTIONS) for x in (s, t)):
                        continue
                    rels.append((s, t, QT30_COUPLING[n["type"]], n.get("text", n["type"])))
        size = rng.randint(lo, hi)
        for start in range(0, len(I) - size + 1, size):
            win = I[start:start + size]
            ids = {n["nodeID"]: f"a{k}" for k, n in enumerate(win)}
            gold = [{"source": ids[s], "target": ids[t], "type": ty, "sense": sense}
                    for s, t, ty, sense in rels if s in ids and t in ids]
            if not gold:
                continue
            locs = []
            for n in win:
                l = loc.get(n["nodeID"])
                if l and l not in locs:
                    locs.append(l)
            windows.append((os.path.basename(path).replace(".json", ""), start, win, gold, locs))
    rng.shuffle(windows)
    out = []
    for ns, start, win, gold, locs in sorted(windows[:n_windows], key=lambda w: (w[0], w[1])):
        claims = [n["text"].strip() for n in win]
        out.append(item(f"qt30-{ns}-{start}", "\n".join(locs) if locs else " ".join(claims),
                        claims, dataset="qt30", check="qt30", group=ns, gold=gold,
                        meta={"nodeset": ns, "start": start}))
    return out


#: PDTB-3 sense prefix -> (our sense, coupling, direction). Direction "12" means the gold
#: entailment runs arg1 => arg2, "21" the reverse, None for symmetric or no coupling.
#: Senses outside our inventory (Conjunction, Level-of-detail, Similarity, Manner, Exception,
#: Substitution, Negative-condition, SpeechAct, Hypophora) are excluded from the test.
PDTB_MAP = [
    ("Contingency.Cause+Belief.Reason+Belief", "Evidence", "entailment", "21"),
    ("Contingency.Cause+Belief.Result+Belief", "Evidence", "entailment", "12"),
    ("Contingency.Cause.Reason", "Cause-Effect", "entailment", "21"),
    ("Contingency.Cause.Result", "Cause-Effect", "entailment", "12"),
    ("Contingency.Condition.Arg1-as-cond", "Condition", "entailment", "12"),
    ("Contingency.Condition.Arg2-as-cond", "Condition", "entailment", "21"),
    ("Expansion.Instantiation.Arg2-as-instance", "Instantiation", "entailment", "21"),
    ("Expansion.Instantiation.Arg1-as-instance", "Instantiation", "entailment", "12"),
    ("Expansion.Equivalence", "Restatement", "equivalence", None),
    ("Expansion.Disjunction", "Disjunction", "co_necessity", None),
    ("Comparison.Contrast", "Contrast", "none", None),
    ("Comparison.Concession", "Concession", "none", None),
    ("Temporal.Synchronous", "Temporal", "none", None),
    ("Temporal.Asynchronous", "Temporal", "none", None),
    ("EntRel", "EntRel/NoRel", "none", None),
    ("NoRel", "EntRel/NoRel", "none", None),
]


def map_pdtb(rtype: str, sense: str):
    key = rtype if rtype in ("EntRel", "NoRel") else sense
    for prefix, ours, coup, direction in PDTB_MAP:
        if key.startswith(prefix):
            return ours, coup, direction
    return None


def _span(tokens: list[str], spec: str) -> str:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        out += tokens[lo - 1:hi]
    return " ".join(out)


def gdtb(raw: str, per_category: int = 170) -> list[dict]:
    """Check 4b: PDTB-3 relations of GUM (dev + test, Reddit excluded: its text is withheld).

    Each relation becomes one item: the response is the sentences that contain the two
    arguments, and the two claims are Arg1 and Arg2. Stratified by our category, up to
    ``per_category`` relations each.
    """
    rng = random.Random(SEED + 5)
    pool = defaultdict(list)
    for split in ("dev", "test"):
        path = os.path.join(raw, f"en_gum-ud-{split}.conllu")
        doc, toks, sent_of, sents = None, [], [], []

        def flush():
            if doc is None or "reddit" in doc:
                return
            for rel in rels_doc:
                parts = rel.split(":")
                if len(parts) < 6:
                    continue
                rtype, sense, conn, _, a1, a2 = parts[:6]
                m = map_pdtb(rtype, sense)
                if m is None or not a1 or not a2 or a1 == "_" or a2 == "_":
                    continue
                t1, t2 = _span(toks, a1), _span(toks, a2)
                if not t1.strip() or not t2.strip() or "_" in (t1.strip(), t2.strip()):
                    continue
                idx = [int(x) for p in (a1 + "," + a2).split(",") for x in p.split("-") if x]
                s_lo, s_hi = sent_of[min(idx) - 1], sent_of[max(idx) - 1]
                text = " ".join(sents[s_lo:s_hi + 1])
                pool[m[0]].append({"doc": doc, "split": split, "type": rtype, "sense": sense,
                                   "conn": conn, "arg1": t1, "arg2": t2, "text": text,
                                   "ours": m[0], "coupling": m[1], "direction": m[2]})

        rels_doc = []
        for line in open(path, encoding="utf-8"):
            if line.startswith("# newdoc id"):
                flush()
                doc, toks, sent_of, sents, rels_doc = line.split("=", 1)[1].strip(), [], [], [], []
            elif line.startswith("# text ="):
                sents.append(line.split("=", 1)[1].strip())
            elif line.strip() and not line.startswith("#"):
                cols = line.rstrip("\n").split("\t")
                if "-" in cols[0] or "." in cols[0]:
                    continue
                toks.append(cols[1])
                sent_of.append(len(sents) - 1)
                for kv in cols[9].split("|"):
                    if kv.startswith("PDTB="):
                        rels_doc += kv[5:].split(";")
        flush()
    out = []
    for cat in sorted(pool):
        rows = pool[cat]
        rng.shuffle(rows)
        for k, r in enumerate(sorted(rows[:per_category], key=lambda r: (r["doc"], r["arg1"]))):
            gold = []
            if r["coupling"] != "none":
                s, t = ("a0", "a1") if r["direction"] in ("12", None) else ("a1", "a0")
                gold = [{"source": s, "target": t, "type": r["coupling"], "sense": r["ours"]}]
            out.append(item(f"gdtb-{cat.replace('/', '').replace('-', '')}-{k:03d}", r["text"],
                            [r["arg1"], r["arg2"]], dataset="gdtb", check="gdtb", group=r["doc"],
                            gold=gold, category=r["ours"], gold_coupling=r["coupling"],
                            pdtb_type=r["type"], pdtb_sense=r["sense"], connective=r["conn"],
                            meta={"doc": r["doc"], "split": r["split"]}))
    return out


# ----------------------------------------------------------------------------- ExpertQA

EQA_TRUE = ("Definitely correct", "Probably correct")
EQA_FALSE = ("Definitely incorrect", "Likely incorrect")
_CITE = re.compile(r"\s*\[\d+\](?:\s*\[\d+\])*")


def _eqa_clean(t: str) -> str:
    """Drop citation markers ``[3] [4]`` and markdown bold; keep paragraph breaks."""
    t = _CITE.sub("", t or "").replace("**", "")
    return "\n\n".join(re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", t) if p.strip())


def expertqa(raw: str, min_words: int = 150, min_claims: int = 6) -> list[dict]:
    """Human study, Block B: ExpertQA (Malaviya et al., NAACL 2024) long-form answers whose
    sentence-level claims carry expert correctness labels.

    Pool: answers of at least ``min_words`` words and ``min_claims`` claims with at least one
    claim labelled incorrect. A claim is TRUE (``factual``) when labelled definitely or probably
    correct; incorrect, unsure and unlabelled claims are not established and count as false (the
    paper's verdict polarity). Citation markers and markdown bold are removed from answers and
    claims; an answer is skipped if some cleaned claim cannot be located in its cleaned text.
    """
    out = []
    for qi, line in enumerate(open(os.path.join(raw, "r2_compiled_anon.jsonl"))):
        if not line.strip():
            continue
        r = json.loads(line)
        for system, a in sorted(r["answers"].items()):
            claims = a.get("claims") or []
            text = a.get("answer_string") or ""
            if (len(text.split()) < min_words or len(claims) < min_claims
                    or not any(c.get("correctness") in EQA_FALSE for c in claims)):
                continue
            resp = _eqa_clean(text)
            flat = re.sub(r"\s+", " ", resp)
            cl = [_eqa_clean(c["claim_string"]).replace("\n\n", " ") for c in claims]
            pos, ok = 0, True
            for c in cl:
                k = flat.find(c, pos)
                if k < 0:
                    ok = False
                    break
                pos = k + len(c)
            if not ok or any(not c for c in cl):
                continue
            iid = f"eqa-{qi:04d}-{system}"
            it = item(iid, resp, cl, dataset="expertqa", check="expertqa", group=iid,
                      question=r["question"], field=r["metadata"].get("field"), system=system,
                      words=len(resp.split()),
                      labels=[c.get("correctness") for c in claims],
                      meta={"question_index": qi, "annotator_id": r.get("annotator_id")})
            for at, c in zip(it["atoms"], claims):
                at["factual"] = c.get("correctness") in EQA_TRUE
            out.append(it)
    return out


def summary(items: list[dict]) -> dict:
    c = Counter((it["expected"]["check"], it["expected"].get("variant")) for it in items)
    n = [it["num_atoms"] for it in items]
    return {"items": len(items), "by_check_variant": {f"{k[0]}:{k[1]}": v for k, v in c.items()},
            "claims_min": min(n), "claims_max": max(n), "claims_median": sorted(n)[len(n) // 2]}
