"""Build the human study of the coherence paper: samples, assignment, Label Studio tasks, gold key.

Design: ``docs/arr2027/coherence/HUMAN_STUDY.md``. Five annotators, at most two hours each.

* **Block A, edit pairs** (Q1-Q3): 80 pairs of versions of a LoCoBench test response, each judged
  by 3 of the 5 annotators (dealt over the 10 triples). The screen shows the question, the truth
  values of the claims in the changed sentences, and the changed sentences of versions A and B.
  Item types: repetition, presentation, premise doubt, non-sequitur, valid inference (positive
  control), ``chain`` and ``sign`` rung pairs, ``control`` rewordings, attention checks.
* **Block B, natural answers** (Q4-Q5): 28 ExpertQA answers with the experts' claim labels. First
  every answer is rated (project B1), then 4 relations per answer are checked (project B2), so the
  checks cannot prime the ratings. 4 core answers go to all 5 annotators, 24 to 2 each.

Families of the companion LoCoBench study's Task 2 (whole responses and rung pairs) are excluded;
families its Task 1 showed only as single filler sentences are used when no unused family is left.

Outputs:

* ``docs/arr2027/coherence/study/bundles/annotator_<k>/``: what annotator k receives and nothing
  else (the configs are also kept in ``study/labelstudio/``, as for the LoCoBench study);
* ``data/ext/coh-blocka/items.jsonl``: every Block A version as an item, to be mined in one run
  (``zsh scripts/run_ext_mined.sh coh-blocka 8 graph_gpt6``);
* ``artifacts/coherence_study/key.json`` (git-ignored): the gold key; its SHA-256 goes to
  ``docs/arr2027/coherence/study/key.sha256``.

Every bundle is checked to carry no answer-revealing field (``check_no_leak``). Deterministic in
``--seed``.

    .venv/bin/python scripts/coherence_study_build.py [--seed 20261128]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import itertools
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import arr_coherence_eval as A  # noqa: E402
import ext_eval as X  # noqa: E402
import locobench_study_build as LS  # noqa: E402
from fact_reasoner.exteval import adapters as AD  # noqa: E402

TEST = os.path.join(REPO, "data", "locobench-slot3-test")
DEV = os.path.join(REPO, "data", "locobench-slot3-pilot")
COMPANION_KEY = os.path.join(REPO, "docs", "arr2027", "locobench", "study", "locobench-study",
                             "key", "key.json")
KEY_DIR = os.path.join(REPO, "artifacts", "coherence_study")
BLOCKA_ITEMS = os.path.join(REPO, "data", "ext", "coh-blocka", "items.jsonl")
DOCS = os.path.join(REPO, "docs", "arr2027", "coherence", "study")
OUT = DOCS  # bundles, key hash and (after annotation) exports live with the study documents
ANNOTATORS = LS.ANNOTATORS
TRIPLES = LS.TRIPLES                                    # 10
PAIRS = list(itertools.combinations(ANNOTATORS, 2))    # 10
P = 0.925  # the strong band's midpoint, the weight of the template relations

# (type, n, family primary, what A/B are, LCS predicts, signed sum predicts); sides randomized
TYPES = [
    ("repetition", 10, "CHAIN", "restated existing inference | new distinct inference", "new", "tie"),
    ("presentation", 6, "CONFLICT", "'X if and only if Y' | 'if X then Y, and if Y then X'", "tie",
     "halves"),
    ("premise_doubt", 10, "CONFLICT", "conflict on a premise | conflict on an idle claim", "idle",
     "tie"),
    ("nonseq", 10, "ORDER", "base | + inference between unrelated true claims", "edited if mined",
     "edited if mined"),
    ("valid", 4, "CHAIN", "base | + valid inference between true claims", "edited", "edited"),
    ("chain", 12, "CHAIN", "rung 0 | rung 2 (two inferences added)", "rung2", "rung2"),
    ("sign", 12, "SIGN", "rung 0 (derives a falsehood) | rung 2 (refutes it)", "rung2", "rung2"),
    ("control", 8, "ORDER", "two rewordings of one inference", "tie", "tie"),
    ("attention", 8, "ORDER", "base | + 'It is not true that X' for a true X", "base", "base"),
]
PRACTICE_A = [("attention", 2), ("control", 2)]        # from the dev split; uncontroversial
N_B, N_CORE, N_REL, N_REL_MINED = 28, 4, 4, 3

INFER = "That {c} follows from the fact that {p}."
IFF = "{P} if and only if {q}."
HALVES = "If {p}, then {q}; and if {q}, then {p}."
CONFLICT = "That {p} cannot be squared with the fact that {w}."
NEGATE = "It is not true that {x}."

PREFERENCE_OPTIONS = LS.PREFERENCE_OPTIONS
CONFIDENCE_OPTIONS = LS.CONFIDENCE_OPTIONS
RATING_OPTIONS = ["1", "2", "3", "4", "5", "6", "7"]
ASSERTED_OPTIONS = ["Yes", "Partly", "No"]
KIND_OPTIONS = ["Support: one is given as a reason or evidence for the other",
                "Conflict: they are presented as unable to hold together",
                "Neither: contrast, sequence, or some other connection"]
READING = {  # the miner's relation, as shown in Block B (the distractor is shown as support)
    "entailment": "Claim X is given as support for claim Y",
    "equivalence": "Claims X and Y are presented as saying the same thing",
    "contradiction": "Claims X and Y are presented as unable to both be true",
    "exclusive": "Exactly one of claims X and Y is presented as true",
    "co_necessity": "At least one of claims X and Y is presented as true",
}

#: Data fields a task may carry. Anything else in a bundle is a leak.
FIELDS = {
    "blockA_pairs.json": {"task_id", "question", "claims_html", "a_html", "b_html", "context_html"},
    "blockB1_ratings.json": {"task_id", "question", "answer_html"},
    "blockB2_relations.json": {"task_id", "question", "answer_html", "claim_x", "claim_y",
                               "reading"},
}
FIELDS["practice_blockA.json"] = FIELDS["blockA_pairs.json"]
FIELDS["practice_blockB1.json"] = FIELDS["blockB1_ratings.json"]
FIELDS["practice_blockB2.json"] = FIELDS["blockB2_relations.json"]
TASK_ID = {"blockA_pairs.json": "A", "blockB1_ratings.json": "R", "blockB2_relations.json": "K",
           "practice_blockA.json": "PA", "practice_blockB1.json": "PR",
           "practice_blockB2.json": "PK"}

STYLE = ("<style>.cs p{margin:0 0 .55em 0;line-height:1.45} .cs mark{background:#ffe08a;"
         "padding:0 .15em} .cs .ctx{color:#6b6963;font-size:.92em} .cs .none{color:#6b6963;"
         "font-style:italic} .cs .tv{font-size:.78em;font-weight:600;padding:0 .25em;"
         "border-radius:3px;margin-left:.2em} .cs .T{background:#dcefdc} .cs .F{background:#f6d6d2}"
         " .cs .U{background:#e6e4de} .cs .hx{background:#cfe3f7;padding:0 .1em}"
         " .cs .hy{background:#f7e0bf;padding:0 .1em} .cs ol{margin:.2em 0 .5em 1.2em;padding:0}"
         " .cs summary{cursor:pointer;color:#47463f}</style>")


# ---------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------
def esc(t):
    return html.escape(t, quote=False)


def words(t):
    return {w for w in re.findall(r"[a-z0-9]{4,}", t.lower())}


FUNCTION_WORDS = {"the", "a", "an", "this", "that", "these", "those", "its", "his", "her", "their",
                  "some", "several", "many", "most", "each", "every", "both", "no", "one", "two",
                  "in", "on", "at", "by", "for", "from", "during", "after", "before", "under",
                  "when", "while", "although", "because", "if", "it", "they", "he", "she", "there",
                  "such", "all", "any", "only", "early", "later", "modern", "measured"}


_LEX = None


def lexicon():
    """Common words: lowercase in the LoCoBench responses (test and dev) or in the system word
    list (whose capitalized entries are names)."""
    global _LEX
    if _LEX is None:
        _LEX = set()
        for d in (TEST, DEV):
            for line in open(os.path.join(d, "items.jsonl")):
                if line.strip():
                    _LEX |= set(re.findall(r"\b[a-z][a-z'-]+\b", json.loads(line)["response"]))
        if os.path.exists("/usr/share/dict/words"):
            _LEX |= {w.strip() for w in open("/usr/share/dict/words") if w[:1].islower()}
    return _LEX


def common(word):
    w = word.lower().split("-")[0]
    return any(v in lexicon() for v in (w, w[:-1] if w.endswith("s") else w,
                                        w[:-2] if w.endswith("es") else w))


def clause(text, ctx=""):
    """A claim as an embedded clause: drop the full stop and lowercase the first word, unless it
    begins a name. Evidence, in order: a function word is lowercased; a first word the response
    itself uses in lowercase (before a lowercase word) is lowercased; a capitalized next word
    ("Scale AI", "Peter Brown") or mid-sentence capitalization in the response marks a name;
    otherwise common English words are lowercased."""
    t = text.strip().rstrip(".")
    ws = t.split()
    if len(t) < 2 or t[1].isupper() or not ws:
        return t
    first = re.sub(r"[^\w'-]", "", ws[0])
    nxt = ws[1] if len(ws) > 1 else ""
    low = t[0].lower() + t[1:]
    if first.lower() in FUNCTION_WORDS:
        return low
    if any(c.isupper() for c in first[1:]) or nxt in ("v.", "vs."):
        return t  # "D-Wave", "Colorado v. Connelly"
    if re.search(r"\b" + re.escape(first.lower() + " " + nxt) + r"\b", ctx):
        return low
    if nxt[:1].isupper():
        return t
    if re.search(r"\b" + re.escape(first.lower()) + r"\b", ctx):
        return low
    if re.search(r"[a-z,;] " + re.escape(first) + r"\b", ctx):
        return t
    return low if common(first) else t


def cap(t):
    return t[0].upper() + t[1:]


def split(resp):
    """Response -> list of paragraphs, each a list of sentences."""
    return [AD.sentences(p) for p in AD.paragraphs(resp)]


def join(pars):
    return "\n\n".join(" ".join(p) for p in pars)


def overlap(atom, sent):
    w = words(atom)
    return len(w & words(sent)) / max(1, len(w))


def anchor(pars, atom_text, lo=0.5):
    """(paragraph, sentence) of the sentence that best matches a claim, or None."""
    best = max(((overlap(atom_text, s), i, j) for i, p in enumerate(pars) for j, s in enumerate(p)),
               default=(0, None, None))
    return (best[1], best[2]) if best[0] >= lo else None


def insert_after(resp, atom_text, sentence):
    pars = split(resp)
    a = anchor(pars, atom_text)
    if a is None:
        return None
    pars[a[0]].insert(a[1] + 1, sentence)
    return join(pars)


def flat(resp):
    out = []
    for k, p in enumerate(split(resp)):
        if k:
            out.append("¶")
        out += p
    return out


def regions(ra, rb):
    """Sentence-level diff: [(context sentence, A sentences, B sentences)], one per region."""
    sa, sb = flat(ra), flat(rb)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        prev = next((s for s in reversed(sa[:i1]) if s != "¶"), None)
        out.append((prev, [s for s in sa[i1:i2] if s != "¶"],
                    [s for s in sb[j1:j2] if s != "¶"]))
    return out


def short(s, n=110):
    return s if len(s) <= n else "…" + s[-n:]


def side_html(regs, which):
    out = []
    for k, (prev, a, b) in enumerate(regs, 1):
        txt = a if which == "a" else b
        ctx = (f"<p class='ctx'>[{k}] after “{esc(short(prev))}”</p>" if prev
               else f"<p class='ctx'>[{k}] at the start</p>")
        body = (f"<p><mark>{esc(' '.join(txt))}</mark></p>" if txt
                else "<p class='none'>(nothing here)</p>")
        out.append(ctx + body)
    return STYLE + "<div class='cs'>" + "".join(out) + "</div>"


def context_html(ra, rb):
    """The shared text with each differing region marked [k], folded."""
    sa, sb = flat(ra), flat(rb)
    buf, pars, k = [], [], 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
        if op == "equal":
            for s in sa[i1:i2]:
                if s == "¶":
                    pars.append(" ".join(buf))
                    buf = []
                else:
                    buf.append(esc(s))
        else:
            k += 1
            buf.append(f"<mark>[{k}]</mark>")
    pars.append(" ".join(buf))
    body = "".join(f"<p>{p}</p>" for p in pars if p.strip())
    return (STYLE + "<div class='cs'><details><summary>Show the rest of the response (the same in "
            "A and B; [k] marks where they differ)</summary>" + body + "</details></div>")


def tv(f):
    return {True: "<span class='tv T'>TRUE</span>", False: "<span class='tv F'>FALSE</span>",
            None: "<span class='tv U'>UNVERIFIED</span>"}[f]


def claims_html(atoms, regs):
    changed = " ".join(" ".join(a + b) for _, a, b in regs)
    sents = [s for _, a, b in regs for s in a + b]
    near = [x for x in atoms if any(overlap(x["text"], s) >= 0.6 for s in sents)]
    near = near or [x for x in atoms if overlap(x["text"], changed) >= 0.6]
    lis = "".join(f"<li>{esc(x['text'])} {tv(bool(x['factual']))}</li>" for x in near)
    allc = "".join(f"<li>{esc(x['text'])} {tv(bool(x['factual']))}</li>" for x in atoms)
    return (STYLE + "<div class='cs'><p><b>Truth values of the claims in the changed sentences"
            "</b></p><ol>" + (lis or "<li class='none'>(none)</li>") + "</ol><details><summary>"
            "All claims of this response</summary><ol>" + allc + "</ol></details></div>")


# ---------------------------------------------------------------------------------------------
# Block A
# ---------------------------------------------------------------------------------------------
def load(d):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(d, "items.jsonl"))
             if l.strip()}
    fams = json.load(open(os.path.join(d, "families.json")))["families"]
    return items, fams


def companion_families():
    k = json.load(open(COMPANION_KEY))
    t1 = {v["family_id"] for v in k["task1"].values() if v.get("family_id")}
    t2 = {v["family_id"] for v in k["task2"].values() if v.get("family_id")}
    return t1, t2


def ladder(f, typ):
    return next(l for l in f["ladders"] if l["type"] == typ)


def atoms_of(it):
    return sorted(it["atoms"], key=lambda a: a["pos"])


def gold(it):
    return [(int(r["source_id"][1:]), int(r["target_id"][1:]), r["level1_coupling"],
             sum(r["strength_range"]) / 2) for r in it["relations"]
            if r["level1_coupling"] in A.VIOLATED]


def scores(it, rels):
    at = atoms_of(it)
    n = len(at)
    x = np.array([1 if a["factual"] else 0 for a in at])
    lcs = A.lcs_flat(A.confirmation(np.full(n, 0.5), rels, x), n) if rels else A.lcs_flat(0.0, n)
    return {"lcs_gold": float(lcs), "wsum_gold": X.signed_sum(n, rels, list(x))}


def true_asserted(it):
    return [a for a in atoms_of(it) if a["factual"] and a.get("stance", "asserted") == "asserted"]


def related(it):
    return {frozenset((r["source_id"], r["target_id"])) for r in it["relations"]
            if r["level1_coupling"] != "none"}


def ix(a):
    return int(a["id"][1:])


def make_repetition(f, items, rng, valid_only=False):
    lad = ladder(f, "CHAIN")
    r0, r2 = items[lad["item_ids"][0]], items[lad["item_ids"][2]]
    tru = {a["id"]: a for a in true_asserted(r0)}
    have = {(r["source_id"], r["target_id"]) for r in r0["relations"]
            if r["level1_coupling"] == "entailment"}
    old = [r for r in r0["relations"] if r["level1_coupling"] == "entailment"
           and r["source_id"] in tru and r["target_id"] in tru]
    new = [r for r in r2["relations"] if r["level1_coupling"] == "entailment"
           and (r["source_id"], r["target_id"]) not in have
           and r["source_id"] in tru and r["target_id"] in tru]
    if not new or (not old and not valid_only):
        return None
    nr = rng.choice(new)
    p, c = tru[nr["source_id"]], tru[nr["target_id"]]
    vb = insert_after(r0["response"], c["text"], INFER.format(c=clause(c["text"], r0["response"]),
                                                              p=clause(p["text"], r0["response"])))
    if vb is None:
        return None
    g = gold(r0)
    newt = (ix(p), ix(c), "entailment", P)
    if valid_only:
        return {"base": r0, "versions": {"base": r0["response"], "edited": vb},
                "graphs": {"base": g, "edited": g + [newt]},
                "detail": {"premise": p["id"], "conclusion": c["id"]}}
    rng.shuffle(old)
    for orr in old:
        q, d = tru[orr["source_id"]], tru[orr["target_id"]]
        if {q["id"], d["id"]} == {p["id"], c["id"]}:
            continue
        va = insert_after(r0["response"], d["text"], INFER.format(c=clause(d["text"], r0["response"]),
                                                                  p=clause(q["text"], r0["response"])))
        if va is None:
            continue
        return {"base": r0, "versions": {"restated": va, "new": vb},
                "graphs": {"restated": g + [(ix(q), ix(d), "entailment", P)],
                           "new": g + [newt]},
                "detail": {"restated": [q["id"], d["id"]], "new": [p["id"], c["id"]]}}
    return None


def clean_base(f, items):
    """CONFLICT families: rung 2, which has neither of the ladder's two conflicts."""
    return items[ladder(f, "CONFLICT")["item_ids"][2]]


def make_premise_doubt(f, items, rng):
    b = clean_base(f, items)
    tru = {a["id"]: a for a in true_asserted(b)}
    rel = related(b)
    touched = {x for pr in rel for x in pr}
    ents = [r for r in b["relations"] if r["level1_coupling"] == "entailment"
            and r["source_id"] in tru and r["target_id"] in tru]
    rng.shuffle(ents)
    idle = [a for a in tru.values() if a["id"] not in touched]
    pars = split(b["response"])
    for e in ents:
        p, c = tru[e["source_id"]], tru[e["target_id"]]
        for q in rng.sample(idle, len(idle)):
            ws = [w for w in tru.values() if w["id"] not in (p["id"], c["id"], q["id"])
                  and frozenset((w["id"], p["id"])) not in rel
                  and frozenset((w["id"], q["id"])) not in rel
                  and anchor(pars, w["text"]) != anchor(pars, p["text"])]
            if not ws:
                continue
            w = rng.choice(ws)
            va = insert_after(b["response"], p["text"], CONFLICT.format(p=clause(p["text"], b["response"]),
                                                                        w=clause(w["text"], b["response"])))
            vb = insert_after(b["response"], q["text"], CONFLICT.format(p=clause(q["text"], b["response"]),
                                                                        w=clause(w["text"], b["response"])))
            if va is None or vb is None:
                continue
            g = gold(b)
            ga = g + [(ix(p), ix(w), "contradiction", P)]
            gb = g + [(ix(q), ix(w), "contradiction", P)]
            sa, sb = scores(b, ga), scores(b, gb)
            if not sb["lcs_gold"] > sa["lcs_gold"] + 1e-9:
                continue  # the principle must hold on the gold graph, else the item tests nothing
            return {"base": b, "versions": {"premise": va, "idle": vb},
                    "graphs": {"premise": ga, "idle": gb},
                    "detail": {"inference": [p["id"], c["id"]], "contested_premise": p["id"],
                               "contested_idle": q["id"], "other": w["id"]}}
    return None


def make_presentation(f, items, rng):
    b = clean_base(f, items)
    tru = {a["id"]: a for a in true_asserted(b)}
    # an equivalence the response already states, so that "if and only if" reads naturally
    cand = [r for r in b["relations"] if r["level1_coupling"] == "equivalence"
            and r["source_id"] in tru and r["target_id"] in tru]
    for r in cand:
        p, q = tru[r["source_id"]], tru[r["target_id"]]
        cp, cq = clause(p["text"], b["response"]), clause(q["text"], b["response"])
        va = insert_after(b["response"], q["text"], IFF.format(P=cap(cp), q=cq))
        vb = insert_after(b["response"], q["text"], HALVES.format(p=cp, q=cq))
        if va is None or vb is None:
            continue
        g = gold(b)
        return {"base": b, "versions": {"iff": va, "halves": vb},
                "graphs": {"iff": g + [(ix(p), ix(q), "equivalence", P)],
                           "halves": g + [(ix(p), ix(q), "entailment", P),
                                          (ix(q), ix(p), "entailment", P)]},
                "detail": {"pair": [p["id"], q["id"]], "existing": r["level1_coupling"]}}
    return None


def order_base(f, items):
    lad = ladder(f, "ORDER")
    for iid in lad["item_ids"]:
        it = items[iid]
        po = it["expected"].get("paragraph_order")
        if po is None or list(po) == sorted(po):
            return it
    return items[lad["item_ids"][0]]


def make_nonseq(f, items, rng):
    b = order_base(f, items)
    tru = true_asserted(b)
    rel = related(b)
    pars = split(b["response"])
    cand = []
    for a, c in itertools.permutations(tru, 2):
        if frozenset((a["id"], c["id"])) in rel:
            continue
        pa, pc = anchor(pars, a["text"]), anchor(pars, c["text"])
        if pa is None or pc is None or pa[0] == pc[0]:
            continue
        cand.append((len(words(a["text"]) & words(c["text"])), rng.random(), a, c))
    if not cand:
        return None
    _, _, a, c = min(cand, key=lambda t: (t[0], t[1]))
    v = insert_after(b["response"], c["text"], INFER.format(c=clause(c["text"], b["response"]), p=clause(a["text"], b["response"])))
    g = gold(b)
    return {"base": b, "versions": {"base": b["response"], "edited": v},
            "graphs": {"base": g, "edited": g + [(ix(a), ix(c), "entailment", P)]},
            "detail": {"premise": a["id"], "conclusion": c["id"]}}


def make_attention(f, items, rng):
    b = order_base(f, items)
    pars = split(b["response"])
    tru = [a for a in true_asserted(b) if anchor(pars, a["text"], lo=0.8)]
    if not tru:
        return None
    x = rng.choice(tru)
    v = insert_after(b["response"], x["text"], NEGATE.format(x=clause(x["text"], b["response"])))
    g = gold(b)
    return {"base": b, "versions": {"base": b["response"], "contradiction": v},
            "graphs": {"base": g, "contradiction": g}, "detail": {"negated": x["id"]}}


def make_rungs(f, items, rng, typ):
    lad = ladder(f, typ)
    r0, r2 = items[lad["item_ids"][0]], items[lad["item_ids"][2]]
    return {"base": r0, "versions": {"rung0": r0["response"], "rung2": r2["response"]},
            "graphs": {"rung0": gold(r0), "rung2": gold(r2)}, "items": {"rung0": r0, "rung2": r2},
            "detail": {"ladder": lad["ladder_id"]}}


def make_control(f, items, rng):
    lad = ladder(f, "CONTROL")
    r0, r2 = items[lad["item_ids"][0]], items[lad["item_ids"][2]]
    if r0["response"] == r2["response"]:
        return None
    return {"base": r0, "versions": {"wording1": r0["response"], "wording2": r2["response"]},
            "graphs": {"wording1": gold(r0), "wording2": gold(r2)},
            "items": {"wording1": r0, "wording2": r2}, "detail": {"ladder": lad["ladder_id"]}}


MAKERS = {
    "repetition": make_repetition,
    "valid": lambda f, it, rng: make_repetition(f, it, rng, valid_only=True),
    "presentation": make_presentation,
    "premise_doubt": make_premise_doubt,
    "nonseq": make_nonseq,
    "attention": make_attention,
    "chain": lambda f, it, rng: make_rungs(f, it, rng, "CHAIN"),
    "sign": lambda f, it, rng: make_rungs(f, it, rng, "SIGN"),
    "control": make_control,
}


def build_block_a(rng, data, specs, exclude=(), prefer_not=()):
    """Draw each type's items from distinct families: unused first, then ``prefer_not``."""
    items, fams = load(data)
    used, out = set(exclude), []
    for typ, n, prim, *_ in specs:
        pool = [f for f in fams if f["primary"] == prim and f["family_id"] not in used]
        rng.shuffle(pool)
        pool.sort(key=lambda f: f["family_id"] in prefer_not)   # stable: unused families first
        got = 0
        for f in pool:
            if got == n:
                break
            r = MAKERS[typ](f, items, rng)
            if r is None or any(v is None for v in r["versions"].values()):
                continue
            regs = regions(*r["versions"].values())
            if not regs:
                continue
            used.add(f["family_id"])
            r.update(type=typ, family_id=f["family_id"], question=f["framing"],
                     companion_task1=f["family_id"] in prefer_not)
            out.append(r)
            got += 1
        if got < n:
            raise SystemExit(f"{typ}: only {got} of {n} items")
    return out


# ---------------------------------------------------------------------------------------------
# Block B
# ---------------------------------------------------------------------------------------------
def block_b_pool():
    its = X.items("expertqa")
    recs, err = X.records("expertqa", "graph_gpt6")
    if err or len(recs) < len(its):
        raise SystemExit(f"expertqa mining incomplete: {len(recs)}/{len(its)} records, {err} errors")
    rows = []
    for iid, x in sorted(its.items()):
        n = x["num_atoms"]
        r = X.rels_of(recs[iid])
        v = [1 if a["factual"] else 0 for a in sorted(x["atoms"], key=lambda a: a["pos"])]
        lv, lamv = X.lcs_at(n, r, v)
        la, lama = X.lcs_at(n, r, None)
        rows.append({"id": iid, "item": x, "rels": r, "n": n, "words": x["expected"]["words"],
                     "lcs_v": lv, "lam_v": lamv, "lcs_a": la, "lam_a": lama,
                     "wsum": X.signed_sum(n, r, v), "n_rel": len(r),
                     "violated": sum((v[q[0]], v[q[1]]) in X.VIOLATED[q[2]] for q in r),
                     "field": x["expected"]["field"]})
    return rows


def rank(v):
    return np.argsort(np.argsort(v))


def rho(a, b):
    a, b = rank(np.asarray(a)), rank(np.asarray(b))
    return float(np.corrcoef(a, b)[0, 1])


def select_b(rows, rng, n=N_B, tries=20000):
    """Stratified by LCS-at-verdicts quartile (n/4 each); among such draws, the one that best
    balances violated-relation presence and length terciles within quartiles and keeps LCS
    least collinear with length and with the signed sum."""
    rows = [r for r in rows if r["n_rel"] >= N_REL_MINED and r["words"] <= 350 and r["n"] <= 14]
    lv = np.array([r["lcs_v"] for r in rows])
    q = np.digitize(lv, np.quantile(lv, [.25, .5, .75]))
    wl = np.array([r["words"] for r in rows])
    t = np.digitize(wl, np.quantile(wl, [1 / 3, 2 / 3]))
    for r, qq, tt in zip(rows, q, t):
        r["lcs_quartile"], r["length_tercile"] = int(qq), int(tt)
    byq = [[r for r in rows if r["lcs_quartile"] == k] for k in range(4)]
    best, bs = None, None
    for _ in range(tries):
        s = [r for k in range(4) for r in rng.sample(byq[k], n // 4)]
        if max(sum(r["field"] == f for r in s) for f in {r["field"] for r in s}) > 4:
            continue
        cost = abs(rho([r["lcs_v"] for r in s], [r["words"] for r in s]))
        cost += abs(rho([r["lcs_v"] for r in s], [r["wsum"] for r in s])) * 0.5
        for k in range(4):
            sk = [r for r in s if r["lcs_quartile"] == k]
            cost += abs(sum(r["violated"] > 0 for r in sk) - len(sk) / 2) * 0.2
            cost += sum(abs(sum(r["length_tercile"] == j for r in sk) - len(sk) / 3)
                        for j in range(3)) * 0.1
        if bs is None or cost < bs:
            best, bs = s, cost
    # the core: one per quartile, alternating violated-relation presence, middle length first
    core = []
    for k in range(4):
        sk = [r for r in best if r["lcs_quartile"] == k]
        want = k % 2 == 0
        sk.sort(key=lambda r: ((r["violated"] > 0) != want, abs(r["length_tercile"] - 1),
                               rng.random()))
        core.append(sk[0])
    return best, core


def answer_html(row, hx=None, hy=None):
    """The answer with every claim's expert label; claims X and Y highlighted if given."""
    x = row["item"]
    atoms = sorted(x["atoms"], key=lambda a: a["pos"])
    labels = x["expected"]["labels"]
    text = x["response"]
    out, pos = [], 0
    flat_ = text
    for a, lab in zip(atoms, labels):
        k = flat_.find(a["text"], pos)
        if k < 0:  # claims are located on whitespace-normalized text by the adapter
            continue
        out.append(esc(flat_[pos:k]))
        f = True if lab in AD.EQA_TRUE else False if lab in AD.EQA_FALSE else None
        span = esc(a["text"])
        if a["id"] == hx:
            span = f"<span class='hx'><b>[X]</b> {span}</span>"
        elif a["id"] == hy:
            span = f"<span class='hy'><b>[Y]</b> {span}</span>"
        out.append(span + tv(f))
        pos = k + len(a["text"])
    out.append(esc(flat_[pos:]))
    body = "".join(f"<p>{p.strip()}</p>" for p in "".join(out).split("\n\n") if p.strip())
    return STYLE + "<div class='cs'>" + body + "</div>"


def relation_checks(row, rng):
    """3 of the miner's relations (one non-support one first, if any) and 1 unrelated pair."""
    rels = sorted(row["rels"], key=lambda r: (r[0], r[1]))
    rng.shuffle(rels)
    rels.sort(key=lambda r: r[2] == "entailment")
    mined = rels[:N_REL_MINED]
    touched = {frozenset((r[0], r[1])) for r in row["rels"]}
    n = row["n"]
    free = [(i, j) for i in range(n) for j in range(i + 1, n) if frozenset((i, j)) not in touched
            and 1 <= j - i <= 3]
    free = free or [(i, j) for i in range(n) for j in range(i + 1, n)
                    if frozenset((i, j)) not in touched]
    out = [{"x": f"a{s}", "y": f"a{t}", "miner": ty, "probability": p, "distractor": False}
           for s, t, ty, p in mined]
    while len(out) < N_REL and free:
        i, j = free.pop(rng.randrange(len(free)))
        out.append({"x": f"a{i}", "y": f"a{j}", "miner": None, "probability": None,
                    "distractor": True})
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------------------------
# Tasks, configs, bundles
# ---------------------------------------------------------------------------------------------
def pair_task(tid, r, flip):
    names = list(r["versions"])
    a, b = (names[1], names[0]) if flip else (names[0], names[1])
    ra, rb = r["versions"][a], r["versions"][b]
    regs = regions(ra, rb)
    atoms = atoms_of(r["base"])
    return {"data": {"task_id": tid, "question": r["question"],
                     "claims_html": claims_html(atoms, regs), "a_html": side_html(regs, "a"),
                     "b_html": side_html(regs, "b"), "context_html": context_html(ra, rb)}}, (a, b)


def write_configs(d):
    os.makedirs(d, exist_ok=True)
    ca = "\n".join([
        "<View>",
        '  <Header value="$question" size="5"/>',
        '  <HyperText name="claims" value="$claims_html" inline="true"/>',
        '  <View style="display:flex;gap:1.2em;align-items:flex-start">',
        '    <View style="flex:1;min-width:0">',
        '      <Header value="Version A" size="5"/>',
        '      <HyperText name="a" value="$a_html" inline="true"/>',
        "    </View>",
        '    <View style="flex:1;min-width:0">',
        '      <Header value="Version B" size="5"/>',
        '      <HyperText name="b" value="$b_html" inline="true"/>',
        "    </View>",
        "  </View>",
        '  <HyperText name="ctx" value="$context_html" inline="true"/>',
        '  <Header value="Which version\'s argument is more logically coherent: do the relations it'
        ' draws between its claims fit their truth values?" size="5"/>',
        LS._choices("preference", "a", PREFERENCE_OPTIONS, inline=True),
        '  <Header value="How sure are you?" size="5"/>',
        LS._choices("confidence", "a", CONFIDENCE_OPTIONS, inline=True),
        "</View>"])
    cb1 = "\n".join([
        "<View>",
        '  <Header value="$question" size="5"/>',
        '  <HyperText name="ans" value="$answer_html" inline="true" granularity="word"/>',
        '  <Header value="How logically coherent is this answer\'s argument: do the relations it'
        ' draws between its claims fit their truth values? (1 = the connections it draws'
        ' repeatedly clash with the truth values or with each other; 4 = some fit, some clash;'
        ' 7 = every connection it draws fits the truth values)" size="5"/>',
        LS._choices("rating", "ans", RATING_OPTIONS, inline=True),
        '  <Header value="How sure are you?" size="5"/>',
        LS._choices("confidence", "ans", CONFIDENCE_OPTIONS, inline=True),
        '  <Header value="Optional: select up to two places where the argument goes wrong"'
        ' size="6"/>',
        '  <Labels name="wrong" toName="ans" maxUsages="2">',
        '    <Label value="Argument goes wrong here" background="#f6d6d2"/>',
        "  </Labels>",
        "</View>"])
    cb2 = "\n".join([
        "<View>",
        '  <Header value="$question" size="5"/>',
        '  <View style="background:#f4f4f2;padding:0.5em 0.8em;margin:0.4em 0">',
        '    <Header value="Claim X" size="6"/>',
        '    <Text name="x" value="$claim_x"/>',
        '    <Header value="Claim Y" size="6"/>',
        '    <Text name="y" value="$claim_y"/>',
        '    <Header value="Proposed reading" size="6"/>',
        '    <Text name="reading" value="$reading"/>',
        "  </View>",
        '  <HyperText name="ans" value="$answer_html" inline="true"/>',
        '  <Header value="1. Does the answer draw a connection between X and Y like the proposed'
        ' reading?" size="5"/>',
        LS._choices("asserted", "ans", ASSERTED_OPTIONS, inline=True),
        '  <View visibleWhen="choice-selected" whenTagName="asserted"'
        ' whenChoiceValue="Yes,Partly">',
        '    <Header value="2. Which connection does the answer draw between X and Y?" size="5"/>',
        LS._choices("kind", "ans", KIND_OPTIONS, required=False),
        "  </View>",
        '  <Header value="How sure are you?" size="5"/>',
        LS._choices("confidence", "ans", CONFIDENCE_OPTIONS, inline=True),
        "</View>"])
    for name, c in (("edit_pairs.xml", ca), ("answer_rating.xml", cb1),
                    ("relation_check.xml", cb2)):
        open(os.path.join(d, name), "w").write(c + "\n")


PROJECTS = [  # (title, config, tasks); Label Studio caps titles at 50 characters
    ["Coherence practice A: pairs", "edit_pairs.xml", "practice_blockA.json"],
    ["Coherence block A: pairs", "edit_pairs.xml", "blockA_pairs.json"],
    ["Coherence practice B1: ratings", "answer_rating.xml", "practice_blockB1.json"],
    ["Coherence block B1: ratings", "answer_rating.xml", "blockB1_ratings.json"],
    ["Coherence practice B2: relations", "relation_check.xml", "practice_blockB2.json"],
    ["Coherence block B2: relations", "relation_check.xml", "blockB2_relations.json"],
]


def check_no_leak(d):
    """A bundle carries task text only: no type, version, family, item or miner field."""
    for name, allowed in FIELDS.items():
        tasks = json.load(open(os.path.join(d, name)))
        for t in tasks:
            assert set(t) == {"data"}, (name, set(t))
            assert set(t["data"]) <= allowed, (name, set(t["data"]) - allowed)
            blob = json.dumps(t["data"])
            for bad in (r"\bf\d{3}\b", r"locobench-slot3", r"eqa-\d{4}", r"\brung ?[0-2]\b",
                        r"premise_doubt|\bnonseq\b|\bwording[12]\b|\bdistractor\b|\bcompanion_",
                        r"graph_gpt6|\blcs\b|wsum|lam_[av]", r"\{\{S\d+\}\}", r"data-[a-z]+=",
                        r"<!--"):
                assert not re.search(bad, blob, re.I), (name, bad, t["data"]["task_id"])
            internal = {t_[0] for t_ in TYPES} | {"restated", "new", "iff", "halves", "premise",
                                                   "idle", "base", "edited", "rung0", "rung2",
                                                   "contradiction", "wording1", "wording2"}
            assert not internal & {str(v).strip().lower() for v in t["data"].values()}, name
            assert re.fullmatch(TASK_ID[name] + r"-\d{4}", t["data"]["task_id"]), t["data"]["task_id"]


def deal_triples(types, rng, tries=4000):
    """Item i -> one of the 10 triples, each triple used len/10 times (80 items: 8 each), with
    every item type spread as evenly as possible over the five annotators."""
    n = len(types)
    best, bc = None, None
    for _ in range(tries):
        order = list(range(n))
        rng.shuffle(order)
        order.sort(key=lambda i: types[i])          # items grouped by type, random within
        tri = TRIPLES[:]
        rng.shuffle(tri)
        deal = [None] * n
        for k, i in enumerate(order):
            deal[i] = tri[k % len(tri)]
        cost = 0
        for t in set(types):
            c = [sum(a in deal[i] for i in range(n) if types[i] == t) for a in ANNOTATORS]
            cost += max(c) - min(c)
        if bc is None or cost < bc:
            best, bc = deal, cost
    return best


def deal_pairs(n, rng):
    """Item i -> an annotator pair, keeping per-annotator loads within one of each other."""
    load_ = {a: 0 for a in ANNOTATORS}
    use = {p: 0 for p in PAIRS}
    out = []
    for _ in range(n):
        p = min(PAIRS, key=lambda p: (load_[p[0]] + load_[p[1]], use[p], rng.random()))
        out.append(p)
        use[p] += 1
        for a in p:
            load_[a] += 1
    return out


def practice_answers(key):
    """Feedback on the practice-A screens. Only uncontroversial types are practised (an inserted
    self-contradiction, two rewordings), so the feedback cannot teach the hypotheses."""
    L = ["# Practice A: answers", "",
         "Read this only after you have submitted the four practice-A screens.", ""]
    for tid, v in sorted(key["practice"].items()):
        if not tid.startswith("PA-"):
            continue
        if v["type"] == "attention":
            side = "A" if v["A"] == "base" else "B"
            other = "B" if side == "A" else "A"
            L.append(f"- **{tid}: {side} is more coherent.** Version {other} adds \"It is not true "
                     "that ...\" about a claim marked TRUE, which it also asserts: it contradicts "
                     "itself and a true claim.")
        else:
            L.append(f"- **{tid}: about equal.** The two versions word the same relation "
                     "differently; the relations they draw, and their fit with the truth values, "
                     "are the same.")
    L += ["", "Practice B1 and B2 have no fixed answers: they show how the screens work. Use the "
          "rubric in the brief.", ""]
    return "\n".join(L)


TEMPLATED = ("repetition", "presentation", "premise_doubt", "nonseq", "valid", "attention")


def write_edit_reviews(key, mined, rng):
    """Design step 2: every template edit is checked by one author who does not annotate it (one
    of the item's two non-annotators, balanced). One sheet per author, in the key directory,
    since a sheet says which version is which."""
    load_ = Counter()
    sheets = defaultdict(list)
    for tid, v in sorted(key["blockA"].items()):
        if v["type"] not in TEMPLATED:
            continue
        free = [a for a in ANNOTATORS if a not in v["annotators"]]
        a = min(free, key=lambda a: (load_[a], rng.random()))
        load_[a] += 1
        v["edit_checker"] = a
        regs = regions(mined[v["mined_items"]["A"]]["response"], mined[v["mined_items"]["B"]]["response"])
        L = [f"### {tid} ({v['type']}; A = {v['A']}, B = {v['B']})", ""]
        for k, (prev, ta, tb) in enumerate(regs, 1):
            L += [f"[{k}] after: {prev}", "", f"- A: {' '.join(ta) or '(nothing)'}",
                  f"- B: {' '.join(tb) or '(nothing)'}", ""]
        L += ["Check: [ ] grammatical  [ ] claims named correctly  [ ] placed sensibly  "
              "[ ] no other change  Notes:", ""]
        sheets[a] += L
    for a, L in sheets.items():
        open(os.path.join(KEY_DIR, f"edit_review_author_{a}.md"), "w").write(
            f"# Template edits to check (author {a})\n\nYou do not annotate these items. Check "
            "each edit and note problems; do not discuss them with the annotators.\n\n"
            + "\n".join(L))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(seed):
    rng = random.Random(seed)
    t1, t2 = companion_families()
    # ---- Block A
    pairs = build_block_a(rng, TEST, TYPES, exclude=t2, prefer_not=t1)
    sides = {}
    for typ, *_ in TYPES:
        idx = [k for k, r in enumerate(pairs) if r["type"] == typ]
        flips = [i % 2 == 1 for i in range(len(idx))]
        rng.shuffle(flips)
        sides.update(dict(zip(idx, flips)))
    order = list(range(len(pairs)))
    rng.shuffle(order)
    triples_by_item = deal_triples([r["type"] for r in pairs], rng)
    triples = [triples_by_item[k] for k in order]
    key = {"seed": seed, "design": "docs/arr2027/coherence/HUMAN_STUDY.md",
           "types": {t[0]: {"n": t[1], "family_primary": t[2], "versions": t[3],
                            "lcs_predicts": t[4], "sum_predicts": t[5]} for t in TYPES},
           "blockA": {}, "blockB1": {}, "blockB2": {}, "practice": {}, "assignment": {}}
    bundles = {a: {n: [] for n in FIELDS} for a in ANNOTATORS}
    mine = []
    for pos, k in enumerate(order):
        r = pairs[k]
        tid = f"A-{pos + 1:04d}"
        task, (va, vb) = pair_task(tid, r, sides[k])
        sc = {}
        for v in (va, vb):
            it = r.get("items", {}).get(v, r["base"])
            sc[v] = scores(it, r["graphs"][v])
            mid = f"cba-{pos + 1:04d}-{v}"
            y = json.loads(json.dumps(it))
            y["id"] = y["name"] = mid
            y["response"] = r["versions"][v]
            y["relations"] = []
            y["expected"] = {"family_id": r["family_id"], "dataset": "coh-blocka",
                             "check": r["type"], "task_id": tid, "version": v}
            mine.append(y)
        key["blockA"][tid] = {"type": r["type"], "family_id": r["family_id"],
                              "base_item": r["base"]["id"], "A": va, "B": vb,
                              "annotators": list(triples[pos]), "detail": r["detail"],
                              "gold_scores": {"A": sc[va], "B": sc[vb]},
                              "companion_task1_family": r["companion_task1"],
                              "mined_items": {"A": f"cba-{pos + 1:04d}-{va}",
                                              "B": f"cba-{pos + 1:04d}-{vb}"}}
        for a in triples[pos]:
            bundles[a]["blockA_pairs.json"].append(task)
    # ---- Block A practice (dev split)
    prac = build_block_a(random.Random(seed + 1), DEV,
                         [(t, n, "ORDER", "", "", "") for t, n in PRACTICE_A])
    for k, r in enumerate(prac):
        tid = f"PA-{k + 1:04d}"
        task, (va, vb) = pair_task(tid, r, k % 2 == 1)
        key["practice"][tid] = {"type": r["type"], "A": va, "B": vb}
        for a in ANNOTATORS:
            bundles[a]["practice_blockA.json"].append(task)
    # ---- Block B
    rows = block_b_pool()
    sample, core = select_b(rows, rng)
    core_ids = {r["id"] for r in core}
    rest = [r for r in sample if r["id"] not in core_ids]
    rng.shuffle(rest)
    dealt = dict(zip([r["id"] for r in rest], deal_pairs(len(rest), rng)))
    b_order = list(core) + rest
    rng.shuffle(b_order)
    rc = 0
    for pos, r in enumerate(b_order):
        tid = f"R-{pos + 1:04d}"
        who = list(ANNOTATORS) if r["id"] in core_ids else list(dealt[r["id"]])
        q = r["item"]["expected"]["question"]
        key["blockB1"][tid] = {k: r[k] for k in ("id", "n", "words", "lcs_v", "lam_v", "lcs_a",
                                                 "lam_a", "wsum", "n_rel", "violated", "field",
                                                 "lcs_quartile", "length_tercile")}
        key["blockB1"][tid].update(core=r["id"] in core_ids, annotators=who)
        rtask = {"data": {"task_id": tid, "question": q, "answer_html": answer_html(r)}}
        for a in who:
            bundles[a]["blockB1_ratings.json"].append(rtask)
        atoms = {x["id"]: x for x in r["item"]["atoms"]}
        for chk in relation_checks(r, rng):
            rc += 1
            kid = f"K-{rc:04d}"
            key["blockB2"][kid] = dict(chk, answer=r["id"], rating_task=tid, core=r["id"] in core_ids,
                                       annotators=who)
            ktask = {"data": {"task_id": kid, "question": q,
                              "answer_html": answer_html(r, chk["x"], chk["y"]),
                              "claim_x": atoms[chk["x"]]["text"], "claim_y": atoms[chk["y"]]["text"],
                              "reading": READING[chk["miner"] or "entailment"]}}
            for a in who:
                bundles[a]["blockB2_relations.json"].append(ktask)
    # Block B practice: one answer outside the sample, one rating and one relation check
    used = {r["id"] for r in sample}
    pr = next(r for r in sorted(rows, key=lambda r: r["id"]) if r["id"] not in used
              and r["words"] <= 220 and any(t == "entailment" for *_, t, _ in r["rels"]))
    key["practice"]["PR-0001"] = {"id": pr["id"]}
    s, t_, ty, p = next(x for x in pr["rels"] if x[2] == "entailment")
    key["practice"]["PK-0001"] = {"id": pr["id"], "x": f"a{s}", "y": f"a{t_}", "miner": ty}
    pat = {x["id"]: x for x in pr["item"]["atoms"]}
    for a in ANNOTATORS:
        bundles[a]["practice_blockB1.json"].append(
            {"data": {"task_id": "PR-0001", "question": pr["item"]["expected"]["question"],
                      "answer_html": answer_html(pr)}})
        bundles[a]["practice_blockB2.json"].append(
            {"data": {"task_id": "PK-0001", "question": pr["item"]["expected"]["question"],
                      "answer_html": answer_html(pr, f"a{s}", f"a{t_}"),
                      "claim_x": pat[f"a{s}"]["text"], "claim_y": pat[f"a{t_}"]["text"],
                      "reading": READING[ty]}})
    # per-annotator order: Block A shuffled per annotator; Block B ratings shuffled, and the
    # relation checks follow the order of that annotator's ratings
    for a in ANNOTATORS:
        ar = random.Random(seed * 10 + a)
        ar.shuffle(bundles[a]["blockA_pairs.json"])
        ar.shuffle(bundles[a]["blockB1_ratings.json"])
        rank_ = {t["data"]["task_id"]: i for i, t in enumerate(bundles[a]["blockB1_ratings.json"])}
        bundles[a]["blockB2_relations.json"].sort(
            key=lambda t: (rank_[key["blockB2"][t["data"]["task_id"]]["rating_task"]],
                           t["data"]["task_id"]))
        key["assignment"][str(a)] = {n: [t["data"]["task_id"] for t in v]
                                     for n, v in bundles[a].items()}
    # ---- write
    write_configs(os.path.join(DOCS, "labelstudio"))
    if os.path.isdir(os.path.join(OUT, "bundles")):
        shutil.rmtree(os.path.join(OUT, "bundles"))
    for a in ANNOTATORS:
        d = os.path.join(OUT, "bundles", f"annotator_{a}")
        os.makedirs(d)
        for n, v in bundles[a].items():
            json.dump(v, open(os.path.join(d, n), "w"), indent=1)
        for f in ("edit_pairs.xml", "answer_rating.xml", "relation_check.xml"):
            shutil.copy(os.path.join(DOCS, "labelstudio", f), d)
        json.dump(PROJECTS, open(os.path.join(d, "projects.json"), "w"), indent=1)
        shutil.copy(os.path.join(REPO, "scripts", "locobench_study_projects.py"), d)
        for doc in ("brief.md", "setup.md"):
            shutil.copy(os.path.join(DOCS, doc), d)
        open(os.path.join(d, "practice_answers.md"), "w").write(practice_answers(key))
        check_no_leak(d)
    os.makedirs(KEY_DIR, exist_ok=True)
    write_edit_reviews(key, {y["id"]: y for y in mine}, random.Random(seed + 2))
    kp = os.path.join(KEY_DIR, "key.json")
    json.dump(key, open(kp, "w"), indent=1, sort_keys=True)
    digest = sha(kp)
    open(os.path.join(OUT, "key.sha256"), "w").write(f"{digest}  artifacts/coherence_study/key.json\n")
    os.makedirs(os.path.dirname(BLOCKA_ITEMS), exist_ok=True)
    AD.write(BLOCKA_ITEMS, mine)
    open(os.path.join(os.path.dirname(BLOCKA_ITEMS), "README.md"), "w").write(
        "# data/ext/coh-blocka\n\nCoherence paper, human study Block A: every version of every "
        "edit pair, as an item to be mined in one run (`zsh scripts/run_ext_mined.sh coh-blocka 8 "
        "graph_gpt6`). Ids are `cba-<pair>-<version>`; which version is which is in the answer "
        f"key. {len(mine)} items. Built by `scripts/coherence_study_build.py --seed {seed}`.\n")
    return key, digest, bundles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261128)
    args = ap.parse_args()
    key, digest, bundles = build(args.seed)
    from collections import Counter
    print("block A:", dict(Counter(v["type"] for v in key["blockA"].values())),
          "companion task-1 families:", sum(v["companion_task1_family"] for v in key["blockA"].values()))
    print("block B:", len(key["blockB1"]), "answers,", len(key["blockB2"]), "relation checks")
    for a in ANNOTATORS:
        print(f"  annotator {a}: " + ", ".join(f"{n.split('.')[0]} {len(v)}" for n, v in bundles[a].items()))
    print("key sha256", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
