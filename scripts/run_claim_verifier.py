"""Verdicts from an error-prone factuality pipeline on LoCoBench (second review, Phase C5).

LoCoBench's verdicts are the generator's factuality labels, an error-free pipeline. Here each
distinct claim of each family is checked closed-book by an LLM (TRUE / FALSE / UNSURE, from the
model's own knowledge, as a FActScore-style verifier without retrieval would), with the family's
question as context. Under the paper's verdict polarity an unverifiable claim counts as false, so
UNSURE maps to 0. Then the ladders are read at these verdicts, on gold graphs and on the GPT-6
mined graphs, with the declared orders of the error-free labels.

    LOCOBENCH_NO_CACHE=1 .venv/bin/python scripts/run_claim_verifier.py verify --model gpt-6-sol
    .venv/bin/python scripts/run_claim_verifier.py eval --macros docs/arr2027/coherence/numbers_verifier.tex
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import arr_coherence_eval as A  # noqa: E402
import arr_slot3_eval as E  # noqa: E402
import run_fair_judges as F  # noqa: E402

OUT = os.path.join(REPO, "results", "arr2027_slot3_test", "verifier")
PROMPT = """You are a fact-checker. Using only your own knowledge, decide whether the claim below is
true. The claim comes from an answer to this question: {question}

Claim: {claim}

Answer TRUE if the claim is factually correct, FALSE if it is factually incorrect, and UNSURE if you
cannot tell. Reason briefly, then give your answer in square brackets: [TRUE], [FALSE] or [UNSURE]."""
_ANS = re.compile(r"\[\s*(TRUE|FALSE|UNSURE)\s*\]", re.I)


def load():
    E.DATA = os.path.join(REPO, "data", "locobench-slot3-test")
    E.ROOT = os.path.join(REPO, "results", "arr2027_slot3_test")
    items, fams, ladders = E.load()
    q = {f["family_id"]: f.get("framing") or f.get("canonical_topic") or "" for f in fams}
    return items, fams, ladders, q


def cmd_verify(args):
    items, _fams, _l, q = load()
    claims = sorted({(it["expected"]["family_id"], a["text"]) for it in items.values() for a in it["atoms"]})
    gen = F.make_generate(args.model)
    jobs = [{"key": [fam, text], "fam": fam, "text": text} for fam, text in claims]

    def worker(job):
        out = gen(PROMPT.format(question=q[job["fam"]], claim=job["text"]))
        m = _ANS.findall(out)
        a = m[-1].upper() if m else None
        return {"key": job["key"], "ok": a is not None, "answer": a, "tail": out[-200:]}

    F.run_jobs(jobs, os.path.join(OUT, f"verdicts__{args.model}.jsonl"), worker, args.workers)
    return 0


def cmd_eval(args):
    items, _fams, ladders, _q = load()
    res = {}
    mined = E.mined_relations("graph_gpt6")
    for path in sorted(p for p in os.listdir(OUT) if p.startswith("verdicts__")):
        model = path[len("verdicts__"):-6]
        ans = {tuple(r["key"]): r["answer"] for r in F.jsonl(os.path.join(OUT, path)) if r.get("ok")}
        # accuracy against the error-free labels
        tp = fp = tn = fn = unsure = 0
        seen = set()
        for it in items.values():
            fam = it["expected"]["family_id"]
            for a in it["atoms"]:
                k = (fam, a["text"])
                if k in seen or k not in ans:
                    continue
                seen.add(k)
                pred, gold = ans[k] == "TRUE", bool(a["factual"])
                unsure += ans[k] == "UNSURE"
                tp += pred and gold
                tn += (not pred) and (not gold)
                fp += pred and not gold
                fn += (not pred) and gold
        n = tp + tn + fp + fn
        r = {"claims": n, "accuracy": (tp + tn) / n, "unsure": unsure / n,
             "false_recall": tn / max(1, tn + fp), "true_recall": tp / max(1, tp + fn)}
        # ladders at the verifier's verdicts
        def world(it):
            fam = it["expected"]["family_id"]
            atoms = sorted(it["atoms"], key=lambda a: a["pos"])
            return np.array([1 if ans.get((fam, a["text"])) == "TRUE" else 0 for a in atoms])
        for src, graphs in (("gold", {iid: E.gold_relations(it) for iid, it in items.items()}),
                            ("gpt6", mined)):
            c = defaultdict(lambda: [0, 0])
            fam = defaultdict(lambda: defaultdict(list))
            for lad in ladders:
                if lad["type"] not in E.INCREASE or any(i not in graphs for i in lad["item_ids"]):
                    continue
                vals = {"lcs": [], "wsum": [], "indep": []}
                for iid in lad["item_ids"]:
                    it = items[iid]
                    n_ = len(it["atoms"])
                    x = world(it)
                    tup = [(E._idx(q["source"]), E._idx(q["target"]), q["type"], float(q["probability"]))
                           for q in graphs[iid]]
                    vals["lcs"].append(A.lcs_flat(A.confirmation(np.full(n_, 0.5), tup, x), n_))
                    vals["wsum"].append(sum(p * (1 if A.is_support((s, t, ty, p), x) else -1)
                                            for s, t, ty, p in tup))
                    vals["indep"].append(A.independent_evidence(np.full(n_, 0.5), tup, x))
                for i, j in itertools.combinations(range(3), 2):
                    okl = E.pair_ok(lad["type"], vals["lcs"][i], vals["lcs"][j], E.REL_TIE, True)
                    okw = E.pair_ok(lad["type"], vals["wsum"][i], vals["wsum"][j], E.REL_TIE, True)
                    for t in ("inc", lad["type"]):
                        fam[t][lad["family_id"]].append((float(okl), float(okw)))
                    for key, v in vals.items():
                        ok = E.pair_ok(lad["type"], v[i], v[j], E.REL_TIE, True)
                        for t in ("inc", lad["type"]):
                            c[(key, t)][0] += ok
                            c[(key, t)][1] += 1
            r[src] = {f"{k[0]}:{k[1]}": v[0] / v[1] for k, v in c.items()}
            import random
            rng = random.Random(20261128)
            for t, Fm in fam.items():
                Fl = list(Fm.values())
                vv = []
                for _ in range(2000):
                    smp = [rng.choice(Fl) for _ in Fl]
                    vv.append(sum(a - b for f in smp for a, b in f) / sum(len(f) for f in smp))
                vv.sort()
                r[src][f"difflo:{t}"] = vv[50]
                r[src][f"diffhi:{t}"] = vv[1949]
        res[model] = r
        print(model, json.dumps(r, indent=1)[:1500])
    if args.macros:
        L = ["% Generated by scripts/run_claim_verifier.py eval -- do not edit by hand."]
        tag = {"gpt-6-sol": "Gpt", "gpt-oss-120b": "Oss"}
        for model, r in res.items():
            t = tag.get(model, model)
            L.append(f"\\newcommand{{\\vf{t}Acc}}{{{100 * r['accuracy']:.0f}}}")
            L.append(f"\\newcommand{{\\vf{t}Unsure}}{{{100 * r['unsure']:.0f}}}")
            L.append(f"\\newcommand{{\\vf{t}FalseRecall}}{{{100 * r['false_recall']:.0f}}}")
            L.append(f"\\newcommand{{\\vf{t}TrueRecall}}{{{100 * r['true_recall']:.0f}}}")
            L.append(f"\\newcommand{{\\vf{t}Claims}}{{{r['claims']:,}}}".replace(",", "{,}"))
            for src, st in (("gold", "Gold"), ("gpt6", "Mined")):
                for k, v in r[src].items():
                    key, ty = k.split(":")
                    val = f"{100 * v:+.1f}" if key.startswith("diff") else f"{100 * v:.0f}"
                    L.append(f"\\newcommand{{\\vf{t}{st}{key.title()}{ty.title()}}}{{{val}}}")
        open(args.macros, "w").write("\n".join(L) + "\n")
        print(f"wrote {args.macros}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--model", default="gpt-6-sol")
    v.add_argument("--workers", type=int, default=10)
    e = sub.add_parser("eval")
    e.add_argument("--macros", default=None)
    args = ap.parse_args()
    return {"verify": cmd_verify, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
