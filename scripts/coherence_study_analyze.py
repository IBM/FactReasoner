"""Analysis of the coherence paper's human study, fixed before annotation.

Design: ``docs/arr2027/coherence/HUMAN_STUDY.md`` (section 5). Reads the answer key
(``artifacts/coherence_study/key.json``, checked against ``docs/arr2027/coherence/study/key.sha256``) and
the Label Studio JSON exports ``<exports>/annotator_<k>/{blockA,blockB1,blockB2}.json``.

``judge``     The comparison judges: the pairwise GPT-6 judge with verdicts on every Block A pair
              (both orders), and the absolute verdict-aware judge (3 runs) on the Block B answers.
              G-Eval on ExpertQA is run by ``scripts/run_expertqa_geval.sh``.
``simulate``  Synthetic exports (annotators who follow LCS on the gold graph with noise), to test
              this script end to end before the real data exist. Never used for results.
``analyze``   Block A: per type, the item-level majority, a one-sided sign test of the predicted
              preference, pooled judgments with an item-cluster bootstrap, Krippendorff's alpha,
              and how often each automatic method agrees with the human majority. Block B: Spearman
              and partial (after word count) correlations of the mean rating with LCS at the
              verdict and asserted worlds, the signed sum, G-Eval, the verdict-aware judge and
              word count, with bootstrap intervals over answers; alpha; relation precision, type
              accuracy, distractor rate and sign agreement.

    LOCOBENCH_NO_CACHE=1 .venv/bin/python scripts/coherence_study_analyze.py judge
    .venv/bin/python scripts/coherence_study_analyze.py analyze \\
        --exports docs/arr2027/coherence/study/exports --macros docs/arr2027/coherence/numbers_human.tex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import arr_coherence_eval as A  # noqa: E402
import asap_lcs as S  # noqa: E402
import coherence_study_build as B  # noqa: E402
import ext_eval as X  # noqa: E402

KEY = os.path.join(B.KEY_DIR, "key.json")
JUDGE_A = os.path.join(REPO, "results", "ext", "coh-blocka", "judge_pairwise_verdicts__gpt-6-sol.jsonl")
JUDGE_B = os.path.join(REPO, "results", "ext", "expertqa", "absolute_verdicts__gpt-6-sol.jsonl")
BOOT, SEED = 2000, 20261128
#: The version people are predicted to prefer, per type ("tie": "About equal" is predicted).
HUMAN_PREDICTS = {"repetition": "new", "presentation": "tie", "premise_doubt": "idle",
                  "nonseq": "base", "valid": "edited", "chain": "rung2", "sign": "rung2",
                  "control": "tie", "attention": "base"}
HYP = {"repetition": "H1", "presentation": "H2", "premise_doubt": "H3", "sign": "H4",
       "chain": "H5", "nonseq": "H6"}
KIND = {B.KIND_OPTIONS[0]: "support", B.KIND_OPTIONS[1]: "conflict", B.KIND_OPTIONS[2]: "neither"}
MINER_KIND = {"entailment": "support", "equivalence": "support", "contradiction": "conflict",
              "exclusive": "conflict", "co_necessity": "neither"}


def load_key():
    digest = hashlib.sha256(open(KEY, "rb").read()).hexdigest()
    want = open(os.path.join(B.OUT, "key.sha256")).read().split()[0]
    if digest != want:
        raise SystemExit(f"key {KEY} does not match docs/arr2027/coherence/study/key.sha256")
    return json.load(open(KEY))


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------
def alpha(units, metric="nominal"):
    """Krippendorff's alpha. ``units``: list of lists of values (one list per unit, the values
    of the raters who coded it). Units with fewer than two values are not pairable."""
    units = [u for u in units if len(u) >= 2]
    vals = sorted({v for u in units for v in u})
    if len(vals) < 2:
        return float("nan")
    ix = {v: k for k, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[ix[a], ix[b]] += 1 / (m - 1)
    nc = o.sum(1)
    n = nc.sum()
    if metric == "nominal":
        d = 1 - np.eye(len(vals))
    elif metric == "interval":
        v = np.array(vals, float)
        d = (v[:, None] - v[None, :]) ** 2
    else:  # ordinal
        d = np.zeros((len(vals), len(vals)))
        for a in range(len(vals)):
            for b in range(len(vals)):
                lo, hi = min(a, b), max(a, b)
                d[a, b] = (nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2
    do = (o * d).sum() / n
    de = (np.outer(nc, nc) * d).sum() / (n * (n - 1))
    return float(1 - do / de) if de else float("nan")


def sign_p(k, n):
    """One-sided exact binomial P(X >= k | n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else float("nan")


def ci(vals):
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return (float("nan"), float("nan"))
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


# ---------------------------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------------------------
def read_export(path):
    """Label Studio JSON export -> {task_id: {from_name: value, ...}} (last submitted annotation)."""
    out = {}
    if not os.path.exists(path):
        return out
    for t in json.load(open(path)):
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        res = {"lead_time": anns[-1].get("lead_time")}
        for r in anns[-1].get("result", []):
            v = r.get("value", {})
            if "choices" in v and v["choices"]:
                res[r["from_name"]] = v["choices"][0]
            elif "hypertextlabels" in v or "labels" in v:
                res.setdefault(r["from_name"], []).append(v.get("text", ""))
        out[t["data"]["task_id"]] = res
    return out


def read_all(exports):
    E = {}
    for a in B.ANNOTATORS:
        d = os.path.join(exports, f"annotator_{a}")
        E[a] = {blk: read_export(os.path.join(d, f"{blk}.json"))
                for blk in ("blockA", "blockB1", "blockB2")}
    return E


def side_to_version(pref, v):
    p = {B.PREFERENCE_OPTIONS[0]: "A", B.PREFERENCE_OPTIONS[1]: "=",
         B.PREFERENCE_OPTIONS[2]: "B"}[pref]
    return v["A"] if p == "A" else v["B"] if p == "B" else "tie"


# ---------------------------------------------------------------------------------------------
# Automatic methods on Block A
# ---------------------------------------------------------------------------------------------
def mined_scores():
    """{mined item id: {"lcs": ..., "wsum": ..., "dsum": ...}} on the GPT-6 graph mines."""
    its = X.items("coh-blocka")
    recs, _ = X.records("coh-blocka", "graph_gpt6")
    out = {}
    for iid, it in its.items():
        if iid not in recs:
            continue
        n = it["num_atoms"]
        r = X.rels_of(recs[iid])
        x = [1 if a["factual"] else 0 for a in sorted(it["atoms"], key=lambda a: a["pos"])]
        out[iid] = {"lcs": X.lcs_at(n, r, x)[0], "wsum": X.signed_sum(n, r, x)}
    return out


def judge_votes():
    """{task_id: net preference of the pairwise verdict judge for version A of the key (+1/-1/0)}."""
    votes = defaultdict(list)
    for r in X.jsonl(JUDGE_A) if os.path.exists(JUDGE_A) else []:
        if not r.get("ok"):
            continue
        tid, order = r["key"]
        first = "A" if order == "ab" else "B"
        v = 0 if r["choice"] == "=" else (1 if (r["choice"] == "A") == (first == "A") else -1)
        votes[tid].append(v)
    return {t: sum(v) for t, v in votes.items() if len(v) == 2}


def method_pref(sa, sb, v, tol=X.TIE):
    return v["A"] if sa > sb + tol else v["B"] if sb > sa + tol else "tie"


# ---------------------------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------------------------
def cmd_judge(args):
    import run_fair_judges as F
    key = load_key()
    gen = F.make_generate(args.model)
    its = X.items("coh-blocka")
    jobs = []
    for tid, v in sorted(key["blockA"].items()):
        a, b = its[v["mined_items"]["A"]], its[v["mined_items"]["B"]]
        for order in ("ab", "ba"):
            jobs.append({"key": [tid, order], "a": a, "b": b, "order": order})

    def worker(job):
        x, y = (job["a"], job["b"]) if job["order"] == "ab" else (job["b"], job["a"])
        text = gen(F.PAIRWISE_V.format(construct=F.CONSTRUCT, claims=F.claims_block(job["a"]),
                                       a=x["response"].strip(), b=y["response"].strip()))
        m = F._CHOICE.findall(text)
        return {"key": job["key"], "ok": bool(m), "choice": m[-1] if m else None,
                "tail": text[-300:]}

    F.run_jobs(jobs, JUDGE_A, worker, args.workers)
    eq = X.items("expertqa")
    jobs = [{"key": [v["id"], s], "iid": v["id"]} for v in key["blockB1"].values() for s in range(3)]

    def worker_b(job):
        it = eq[job["iid"]]
        text = gen(F.ABSOLUTE.format(construct=F.CONSTRUCT, claims=F.claims_block(it),
                                     response=it["response"].strip()))
        m = F._DIGIT.findall(text)
        r = int(m[-1]) if m and int(m[-1]) <= 5 else None
        return {"key": job["key"], "ok": r is not None, "rating": r, "tail": text[-300:]}

    F.run_jobs(jobs, JUDGE_B, worker_b, args.workers)
    return 0


# ---------------------------------------------------------------------------------------------
# simulate (tests only)
# ---------------------------------------------------------------------------------------------
def cmd_simulate(args):
    key = load_key()
    rng = random.Random(1)
    for a in B.ANNOTATORS:
        d = os.path.join(args.out, f"annotator_{a}")
        os.makedirs(d, exist_ok=True)
        asg = key["assignment"][str(a)]
        ta = []
        for tid in asg["blockA_pairs.json"]:
            v = key["blockA"][tid]
            g = v["gold_scores"]
            diff = g["A"]["lcs_gold"] - g["B"]["lcs_gold"]
            if v["type"] in ("attention",):
                diff = 1 if v["A"] == "base" else -1
            z = diff * 20 + rng.gauss(0, 0.6)
            pref = B.PREFERENCE_OPTIONS[0 if z > 0.5 else 2 if z < -0.5 else 1]
            ta.append({"data": {"task_id": tid}, "annotations": [{"result": [
                {"from_name": "preference", "value": {"choices": [pref]}},
                {"from_name": "confidence", "value": {"choices": ["2"]}}], "lead_time": 40}]})
        tb = []
        for tid in asg["blockB1_ratings.json"]:
            v = key["blockB1"][tid]
            r = int(min(7, max(1, round(1 + 6 * v["lcs_v"] + rng.gauss(0, 1.2)))))
            tb.append({"data": {"task_id": tid}, "annotations": [{"result": [
                {"from_name": "rating", "value": {"choices": [str(r)]}}], "lead_time": 120}]})
        tk = []
        for tid in asg["blockB2_relations.json"]:
            v = key["blockB2"][tid]
            yes = rng.random() < (0.15 if v["distractor"] else 0.7)
            res = [{"from_name": "asserted", "value": {"choices": ["Yes" if yes else "No"]}}]
            if yes:
                res.append({"from_name": "kind", "value": {"choices": [B.KIND_OPTIONS[0]]}})
            tk.append({"data": {"task_id": tid}, "annotations": [{"result": res, "lead_time": 30}]})
        for name, t in (("blockA", ta), ("blockB1", tb), ("blockB2", tk)):
            json.dump(t, open(os.path.join(d, f"{name}.json"), "w"))
    print(f"wrote synthetic exports to {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------
def majority(vals):
    c = Counter(vals).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return None
    return c[0][0]


def block_a(key, E, excluded, rng):
    jud = defaultdict(dict)  # tid -> annotator -> version
    for a in B.ANNOTATORS:
        if a in excluded:
            continue
        for tid, res in E[a]["blockA"].items():
            if tid in key["blockA"] and "preference" in res:
                jud[tid][a] = side_to_version(res["preference"], key["blockA"][tid])
    mined = mined_scores()
    jv = judge_votes()
    out = {}
    for typ in B.TYPES:
        typ = typ[0]
        tids = [t for t, v in key["blockA"].items() if v["type"] == typ and jud.get(t)]
        if not tids:
            continue
        pred = HUMAN_PREDICTS[typ]
        maj = {t: majority(list(jud[t].values())) for t in tids}
        r = {"items": len(tids), "judgments": sum(len(jud[t]) for t in tids), "predicted": pred}
        r["maj_pred"] = sum(maj[t] == pred for t in tids)
        r["maj_tie"] = sum(maj[t] == "tie" for t in tids)
        r["maj_none"] = sum(maj[t] is None for t in tids)
        if pred != "tie":
            other = {t: next(x for x in (key["blockA"][t]["A"], key["blockA"][t]["B"]) if x != pred)
                     for t in tids}
            r["maj_other"] = sum(maj[t] == other[t] for t in tids)
            n = r["maj_pred"] + r["maj_other"]
            r["sign_p"] = sign_p(r["maj_pred"], n)
            pooled = lambda ts: (sum(v == pred for t in ts for v in jud[t].values())  # noqa: E731
                                 - sum(v == other[t] for t in ts for v in jud[t].values())) / max(
                1, sum(len(jud[t]) for t in ts))
            r["pooled_diff"] = pooled(tids)
            r["pooled_ci"] = ci([pooled([rng.choice(tids) for _ in tids]) for _ in range(BOOT)])
        r["share_pred_judgments"] = sum(v == pred for t in tids for v in jud[t].values()) / r["judgments"]
        r["alpha"] = alpha([list(jud[t].values()) for t in tids])
        # agreement of automatic methods with the human majority (items with a majority)
        agree = defaultdict(lambda: [0, 0])
        for t in tids:
            # attention items check the annotators; a claim asserted and then denied is not a
            # relation between two claims, so the gold graphs cannot score them
            if maj[t] is None or typ == "attention":
                continue
            v = key["blockA"][t]
            g = v["gold_scores"]
            prefs = {"lcs_gold": method_pref(g["A"]["lcs_gold"], g["B"]["lcs_gold"], v),
                     "wsum_gold": method_pref(g["A"]["wsum_gold"], g["B"]["wsum_gold"], v)}
            ma, mb = mined.get(v["mined_items"]["A"]), mined.get(v["mined_items"]["B"])
            if ma and mb:
                prefs["lcs_mined"] = method_pref(ma["lcs"], mb["lcs"], v)
                prefs["wsum_mined"] = method_pref(ma["wsum"], mb["wsum"], v)
            if t in jv:
                prefs["judge"] = method_pref(jv[t], -jv[t], v, tol=0.5)
            for m, p in prefs.items():
                agree[m][0] += p == maj[t]
                agree[m][1] += 1
        r["agree"] = {m: c[0] / c[1] for m, c in agree.items() if c[1]}
        out[typ] = r
    allu = [list(jud[t].values()) for t in jud]
    return out, alpha(allu), jud


def attention_fails(key, E):
    fails = {}
    for a in B.ANNOTATORS:
        f = n = 0
        for tid, res in E[a]["blockA"].items():
            v = key["blockA"].get(tid)
            if v and v["type"] == "attention" and "preference" in res:
                n += 1
                f += side_to_version(res["preference"], v) != "base"
        fails[a] = (f, n)
    return fails


def block_b(key, E, excluded, rng):
    ratings = defaultdict(dict)
    for a in B.ANNOTATORS:
        if a in excluded:
            continue
        for tid, res in E[a]["blockB1"].items():
            if tid in key["blockB1"] and "rating" in res:
                ratings[tid][a] = int(res["rating"])
    tids = sorted(t for t in ratings if ratings[t])
    if len(tids) < 5:
        return {}
    base = X.baselines("expertqa").get("judge_geval@gpt-6-sol", {})
    absj = defaultdict(list)
    for r in X.jsonl(JUDGE_B) if os.path.exists(JUDGE_B) else []:
        if r.get("ok"):
            absj[r["key"][0]].append(r["rating"])
    rows = []
    for t in tids:
        v = key["blockB1"][t]
        rows.append({"human": float(np.mean(list(ratings[t].values()))), "lcs_v": v["lcs_v"],
                     "lcs_a": v["lcs_a"], "wsum": v["wsum"], "words": v["words"],
                     "n_rel": v["n_rel"], "geval": base.get(v["id"]),
                     "judge": float(np.mean(absj[v["id"]])) if absj.get(v["id"]) else None})
    preds = ["lcs_v", "lcs_a", "wsum", "n_rel", "geval", "judge", "words"]
    out = {"answers": len(rows)}

    def stat(rs, p, partial):
        rs = [r for r in rs if r[p] is not None]
        if len(rs) < 5:
            return float("nan")
        x, y = [r[p] for r in rs], [r["human"] for r in rs]
        if partial and p != "words":
            return S.partial_spearman(x, y, [r["words"] for r in rs])
        return S.spearman(x, y)

    boots = [[rng.choice(rows) for _ in rows] for _ in range(BOOT)]
    for p in preds:
        for partial in (False, True):
            k = f"{p}{'_partial' if partial else ''}"
            out[k] = stat(rows, p, partial)
            out[k + "_ci"] = ci([stat(b, p, partial) for b in boots])
    for p in ("lcs_a", "wsum", "n_rel", "geval", "judge"):
        d = [stat(b, "lcs_v", True) - stat(b, p, True) for b in boots]
        out[f"diff_{p}"] = out["lcs_v_partial"] - out[f"{p}_partial"]
        out[f"diff_{p}_ci"] = ci(d)
    core = [t for t in tids if key["blockB1"][t]["core"]]
    out["alpha_core"] = alpha([list(ratings[t].values()) for t in core], "ordinal")
    out["alpha_all"] = alpha([list(ratings[t].values()) for t in tids], "ordinal")
    return out


def block_b2(key, E, excluded):
    """Relation checks: precision of the miner's relations, type accuracy, distractor rate,
    and agreement of the miner's sign at the expert verdicts with the annotators' reading."""
    eq = X.items("expertqa")
    J = defaultdict(dict)
    for a in B.ANNOTATORS:
        if a in excluded:
            continue
        for tid, res in E[a]["blockB2"].items():
            if tid in key["blockB2"] and "asserted" in res:
                J[tid][a] = (res["asserted"], KIND.get(res.get("kind")))
    if not J:
        return {}
    mined = [t for t in J if not key["blockB2"][t]["distractor"]]
    dis = [t for t in J if key["blockB2"][t]["distractor"]]
    js = lambda ts: [j for t in ts for j in J[t].values()]  # noqa: E731
    out = {"mined_checks": len(mined), "distractor_checks": len(dis)}
    out["precision_yes"] = np.mean([j[0] == "Yes" for j in js(mined)])
    out["precision_yes_partly"] = np.mean([j[0] in ("Yes", "Partly") for j in js(mined)])
    out["distractor_yes_partly"] = np.mean([j[0] in ("Yes", "Partly") for j in js(dis)]) if dis else float("nan")
    typed = [(t, j) for t in mined for j in J[t].values() if j[0] in ("Yes", "Partly") and j[1]]
    out["type_correct"] = np.mean([MINER_KIND[key["blockB2"][t]["miner"]] == j[1] for t, j in typed]) if typed else float("nan")
    # sign at the verdicts: the miner's relation vs the annotator's reading of the same pair
    agree = []
    for t, j in typed:
        v = key["blockB2"][t]
        it = eq[v["answer"]]
        x = [1 if a["factual"] else 0 for a in sorted(it["atoms"], key=lambda a: a["pos"])]
        s, tt = int(v["x"][1:]), int(v["y"][1:])
        miner_sign = A.is_support((s, tt, v["miner"], 1.0), x)
        if j[1] == "neither":
            continue
        ty = {"support": "entailment" if v["miner"] == "entailment" else "equivalence",
              "conflict": "contradiction"}[j[1]]
        agree.append(miner_sign == A.is_support((s, tt, ty, 1.0), x))
    out["sign_agree"] = float(np.mean(agree)) if agree else float("nan")
    out["sign_n"] = len(agree)
    out["alpha_asserted"] = alpha([[j[0] for j in J[t].values()] for t in J])
    core = [t for t in J if key["blockB2"][t]["core"]]
    out["alpha_asserted_core"] = alpha([[j[0] for j in J[t].values()] for t in core])
    return out


def camel(s):
    return "".join(w.title() for w in s.replace("-", "_").split("_"))


def fmt(v, pct=False, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if pct:
        return f"{100 * v:.0f}"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def cmd_analyze(args):
    key = load_key()
    E = read_all(args.exports)
    rng = random.Random(SEED)
    fails = attention_fails(key, E)
    excluded = {a for a, (f, n) in fails.items() if f > 2}
    a_res, a_alpha, _ = block_a(key, E, excluded, rng)
    b_res = block_b(key, E, excluded, rng)
    k_res = block_b2(key, E, excluded)
    res = {"attention_fails": fails, "excluded": sorted(excluded), "blockA": a_res,
           "blockA_alpha": a_alpha, "blockB": b_res, "blockB2": k_res}
    print(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
    if args.macros:
        L = ["% Generated by scripts/coherence_study_analyze.py analyze -- do not edit by hand."]
        m = lambda k, v: L.append(f"\\newcommand{{\\hs{k}}}{{{v}}}")  # noqa: E731
        m("Excluded", len(excluded))
        m("AlphaA", fmt(a_alpha))
        for typ, r in a_res.items():
            T = camel(typ)
            m(f"{T}Items", r["items"])
            m(f"{T}MajPred", r["maj_pred"])
            m(f"{T}MajTie", r["maj_tie"])
            m(f"{T}SharePred", fmt(r["share_pred_judgments"], pct=True))
            m(f"{T}Alpha", fmt(r["alpha"]))
            if "sign_p" in r:
                m(f"{T}MajOther", r["maj_other"])
                m(f"{T}P", f"{r['sign_p']:.3f}" if r["sign_p"] >= 1e-3 else "<0.001")
                m(f"{T}Pooled", fmt(r["pooled_diff"], signed=True))
                m(f"{T}PooledLo", fmt(r["pooled_ci"][0], signed=True))
                m(f"{T}PooledHi", fmt(r["pooled_ci"][1], signed=True))
            for meth, v in r["agree"].items():
                m(f"{T}Agree{camel(meth)}", fmt(v, pct=True))
        for k, v in b_res.items():
            if k.endswith("_ci"):
                m(f"B{camel(k[:-3])}Lo", fmt(v[0], signed=True))
                m(f"B{camel(k[:-3])}Hi", fmt(v[1], signed=True))
            elif isinstance(v, float):
                m(f"B{camel(k)}", fmt(v, signed=not k.startswith("alpha")))
            else:
                m(f"B{camel(k)}", v)
        for k, v in k_res.items():
            pct = k.startswith(("precision", "distractor", "type", "sign_agree"))
            m(f"K{camel(k)}", fmt(v, pct=pct) if isinstance(v, float) else v)
        open(args.macros, "w").write("\n".join(L) + "\n")
        print(f"wrote {args.macros}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--model", default="gpt-6-sol")
    j.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("simulate")
    s.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--exports", default=os.path.join(B.OUT, "exports"))
    a.add_argument("--macros", default=None)
    args = ap.parse_args()
    return {"judge": cmd_judge, "simulate": cmd_simulate, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
