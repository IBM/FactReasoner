"""Evaluate the external checks of LCS (Part A of .claude/plans/external-datasets-eval.md).

Every prediction tested here is fixed in results/ext/PREREG.md. Records come from
scripts/run_ext_mined.sh (results/ext/<dataset>/arms/<arm>/records), baselines from
scripts/run_slot3_baselines.py (results/ext/<dataset>/baselines).

    .venv/bin/python scripts/ext_eval.py spotcheck          # committee: are ContraDoc insertions false?
    .venv/bin/python scripts/ext_eval.py check1|check2|check3|check4|all [--macros PATH]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import arr_coherence_eval as A  # noqa: E402
from fact_reasoner.exteval.exact import VIOLATED, lambda_exact  # noqa: E402

ARMS = {"graph_gpt6": "gpt-6-sol", "graph_gptoss": "gpt-oss-120b"}
TOL = 0.01
TIE = 1e-9
BOOT = 2000
SEED = 20261124
SYMMETRIC = {"contradiction", "equivalence", "co_necessity", "exclusive"}
CONFLICTISH = {"contradiction", "exclusive"}
BASELINES = {"discourse_rc": "DiscoScore RC", "discourse_lc": "DiscoScore LC",
             "discourse_entity_graph": "entity graph", "judge_geval@gpt-6-sol": "G-Eval (GPT-6)"}


# ----------------------------------------------------------------------------- loading

def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def items(ds):
    return {it["id"]: it for it in jsonl(os.path.join(REPO, "data", "ext", ds, "items.jsonl"))}


def records(ds, arm, root=None):
    root = root or os.path.join(REPO, "results", "ext", ds)
    out, errors = {}, 0
    for p in glob.glob(os.path.join(root, "arms", arm, "records", "*.json")):
        r = json.load(open(p))
        if r.get("error"):
            errors += 1
            continue
        out[r["item_id"]] = r
    return out, errors


def baselines(ds):
    out = defaultdict(dict)
    for p in glob.glob(os.path.join(REPO, "results", "ext", ds, "baselines", "*.jsonl")):
        for r in jsonl(p):
            if r.get("score") is not None:
                out[r["name"]][r["item_id"]] = r["score"]
    return out


def rels_of(rec):
    return [(int(x["source"][1:]), int(x["target"][1:]), x["type"], float(x["probability"]))
            for x in rec.get("relations") or [] if x["type"] in VIOLATED]


def lcs_at(n, rels, x=None):
    lam = lambda_exact(n, rels, x)["lambda"] if rels else 0.0
    return A.lcs_flat(lam, n), lam


def gold_rels(item):
    """LoCoBench gold relations (band-midpoint weights), as in scripts/arr_slot3_eval.py."""
    out = []
    for r in item.get("relations") or []:
        if r["level1_coupling"] in VIOLATED:
            lo, hi = r["strength_range"]
            out.append((int(r["source_id"][1:]), int(r["target_id"][1:]),
                        r["level1_coupling"], (lo + hi) / 2))
    return out


# ----------------------------------------------------------------------------- statistics

def sign_test(dec, inc):
    """Two-sided exact binomial test of dec vs inc (ties excluded)."""
    n = dec + inc
    if n == 0:
        return 1.0
    k = max(dec, inc)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    a, b = _rank(x), _rank(y)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


def boot(fn, groups, n=BOOT, seed=SEED):
    """Percentile bootstrap over groups (a list of lists of rows)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        rows = [r for g in (rng.choice(groups) for _ in groups) for r in g]
        v = fn(rows)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if len(vals) < n // 2:
        return (float("nan"), float("nan"))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def share(rows, f):
    return sum(1 for r in rows if f(r)) / len(rows) if rows else float("nan")


def grouped(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return list(g.values())


# ----------------------------------------------------------------------------- Check 1

def invariance_block(rows, metric_names):
    """rows: {group, variant, metric -> value, metric_orig -> value}. Per metric: invariance
    share within TOL, share of decreases, with group-bootstrap intervals."""
    out = {}
    groups = grouped(rows, "group")
    for m in metric_names:
        R = [r for r in rows if r.get(m) is not None and r.get(m + "_orig") is not None]
        if not R:
            continue
        inv = lambda S, m=m: share(S, lambda r: abs(r[m] - r[m + "_orig"]) <= TOL)  # noqa: E731
        dec = lambda S, m=m: share(S, lambda r: r[m] < r[m + "_orig"] - TIE)  # noqa: E731
        G = [[r for r in g if r in R] for g in groups]
        G = [g for g in G if g]
        out[m] = {"n": len(R), "invariant": inv(R), "invariant_ci": boot(inv, G),
                  "decrease": dec(R), "decrease_ci": boot(dec, G),
                  "mean_abs": st.mean(abs(r[m] - r[m + "_orig"]) for r in R)}
    return out


def check1(args):
    res = {}
    # 1a / 1b: LoCoBench
    it = items("loco-shuffle")
    base = baselines("loco-shuffle")
    test_items = {r["id"]: r for r in jsonl(os.path.join(REPO, "data", "locobench-slot3-test",
                                                         "items.jsonl"))}
    for arm, model in ARMS.items():
        recs, err = records("loco-shuffle", arm)
        test_recs, _ = records("", arm, root=os.path.join(REPO, "results", "arr2027_slot3_test"))
        by_fam = defaultdict(dict)
        for iid, x in it.items():
            by_fam[x["expected"]["family_id"]][x["expected"]["variant"]] = x
        rows, noise = [], []
        for fam, V in by_fam.items():
            o = V["orig"]
            if o["id"] not in recs:
                continue
            n = o["num_atoms"]
            lo = lcs_at(n, rels_of(recs[o["id"]]))[0]
            g = lcs_at(n, gold_rels(test_items[o["expected"]["orig_item"]]))[0]
            tr = test_recs.get(o["expected"]["orig_item"])
            if tr is not None:
                noise.append({"group": fam, "lcs": lo, "lcs_orig": lcs_at(n, rels_of(tr))[0]})
            for v in ("p1", "p2", "sent"):
                x = V.get(v)
                if x is None or x["id"] not in recs:
                    continue
                row = {"group": fam, "variant": v, "lcs": lcs_at(n, rels_of(recs[x["id"]]))[0],
                       "lcs_orig": lo, "gold": g, "gold_orig": g}
                for b in BASELINES:
                    if x["id"] in base.get(b, {}) and o["id"] in base.get(b, {}):
                        row[b], row[b + "_orig"] = base[b][x["id"]], base[b][o["id"]]
                rows.append(row)
        perm = [r for r in rows if r["variant"] in ("p1", "p2")]
        sent = [r for r in rows if r["variant"] == "sent"]
        names = ["lcs", "gold"] + list(BASELINES)
        res[f"1a:{arm}"] = {"errors": err, "perm": invariance_block(perm, names),
                            "sent": invariance_block(sent, names),
                            "noise_floor": invariance_block(noise, ["lcs"])}
    # 1c: ContraDoc clean documents
    it = items("contradoc")
    base = baselines("contradoc")
    for arm in ARMS:
        recs, err = records("contradoc", arm)
        by = defaultdict(dict)
        for iid, x in it.items():
            if x["expected"]["check"] in ("shuffle", "contradoc_neg") and \
                    x["expected"]["family_id"].startswith("neg-"):
                by[x["expected"]["family_id"]][x["expected"]["variant"]] = x
        rows = []
        for fam, V in by.items():
            o = V.get("orig")
            if not o or o["id"] not in recs or not o["expected"].get("shuffled"):
                continue
            lo = lcs_at(o["num_atoms"], rels_of(recs[o["id"]]))[0]
            for v in ("p1", "p2"):
                x = V.get(v)
                if x is None or x["id"] not in recs:
                    continue
                row = {"group": fam, "variant": v,
                       "lcs": lcs_at(x["num_atoms"], rels_of(recs[x["id"]]))[0], "lcs_orig": lo}
                for b in BASELINES:
                    if x["id"] in base.get(b, {}) and o["id"] in base.get(b, {}):
                        row[b], row[b + "_orig"] = base[b][x["id"]], base[b][o["id"]]
                rows.append(row)
        res[f"1c:{arm}"] = {"errors": err, "perm": invariance_block(rows, ["lcs"] + list(BASELINES))}
    # 1d: SummEval
    it = items("summeval")
    base = baselines("summeval")
    for arm in ARMS:
        recs, err = records("summeval", arm)
        rows = []
        for iid, x in it.items():
            if iid not in recs:
                continue
            r = rels_of(recs[iid])
            rows.append({"group": x["expected"]["family_id"], "lcs": lcs_at(x["num_atoms"], r)[0],
                         "empty": not r, **x["expected"]["expert"],
                         "judge": base.get("judge_geval@gpt-6-sol", {}).get(iid)})
        groups = grouped(rows, "group")
        out = {"errors": err, "n": len(rows), "empty_share": share(rows, lambda r: r["empty"])}
        for dim in ("coherence", "consistency", "fluency", "relevance"):
            f = lambda S, d=dim: spearman([r["lcs"] for r in S], [r[d] for r in S])  # noqa: E731
            out[f"rho_{dim}"] = f(rows)
            out[f"rho_{dim}_ci"] = boot(f, groups)
        J = [r for r in rows if r["judge"] is not None]
        if J:
            out["judge_rho_coherence"] = spearman([r["judge"] for r in J], [r["coherence"] for r in J])
        out["H1d_pass"] = abs(out["rho_coherence"]) < 0.1
        res[f"1d:{arm}"] = out
    # Pre-registered verdicts for H1a / H1c
    for key in [k for k in res if k[:2] in ("1a", "1c")]:
        P = res[key]["perm"]
        if "lcs" in P:
            others = [b for b in BASELINES if b in P]
            res[key]["H_pass"] = all(P["lcs"]["invariant"] > P[b]["invariant"] for b in others) \
                if others else None
    return res


# ----------------------------------------------------------------------------- Check 2

def committee_spotcheck(args):
    from fact_reasoner.exteval.committee import COMMITTEE, majority, run_jobs

    it = items("contradoc")
    labels = ("contradicted", "supported", "not_addressed")
    prompt = """Below is a DOCUMENT and a SENTENCE that was inserted into it. Judge the SENTENCE only
against the rest of the DOCUMENT, using no outside knowledge:
  "contradicted"   the DOCUMENT shows the SENTENCE to be false (they cannot both be true)
  "supported"      the DOCUMENT states or directly implies the SENTENCE
  "not_addressed"  the DOCUMENT neither supports nor contradicts it
Return ONLY a JSON list with one object: [{{"id": "s", "label": "<one label>"}}]

DOCUMENT:
\"\"\"{doc}\"\"\"

SENTENCE: {sent}
"""
    jobs = []
    for iid, x in it.items():
        if x["expected"].get("variant") == "a":
            ev = x["expected"]["evidence"]
            jobs += [(iid, m, prompt.format(doc=x["response"], sent=ev)) for m in COMMITTEE]
    cache = run_jobs(jobs, os.path.join(REPO, "results", "ext", "contradoc", "spotcheck.jsonl"))
    out = {}
    for iid in {j[0] for j in jobs}:
        labs = [(cache.get((iid, m)) or [{}])[0].get("label") for m in COMMITTEE]
        out[iid] = {"labels": labs, "majority": majority(labs, labels)}
    json.dump(out, open(os.path.join(REPO, "results", "ext", "contradoc", "spotcheck.json"), "w"),
              indent=1)
    c = Counter(v["majority"] for v in out.values())
    print("spot check majority:", dict(c))
    return 0


def check2(args):
    it = items("contradoc")
    base = baselines("contradoc")
    spot = {}
    sp = os.path.join(REPO, "results", "ext", "contradoc", "spotcheck.json")
    if os.path.exists(sp):
        spot = json.load(open(sp))
    res = {}
    for arm in ARMS:
        recs, err = records("contradoc", arm)
        pairs = []
        by = defaultdict(dict)
        for iid, x in it.items():
            if x["expected"]["check"] == "contradoc":
                by[x["expected"]["family_id"]][x["expected"]["variant"]] = x
        for fam, V in by.items():
            a, b = V.get("a"), V.get("b")
            if not a or not b or a["id"] not in recs or b["id"] not in recs:
                continue
            ev = b["expected"]["evidence_atoms"]
            evi = {int(e[1:]) for e in ev}
            rb, ra = rels_of(recs[b["id"]]), rels_of(recs[a["id"]])
            xb = [0 if k in evi else 1 for k in range(b["num_atoms"])]
            on_ev = [r for r in rb if r[2] in CONFLICTISH and (r[0] in evi or r[1] in evi)]
            row = {"group": fam, "scope": b["expected"]["scope"], "doc_type": b["expected"]["doc_type"],
                   "asserted_a": lcs_at(a["num_atoms"], ra)[0], "asserted_b": lcs_at(b["num_atoms"], rb)[0],
                   "verdict_a": lcs_at(a["num_atoms"], ra)[0],  # a has no evidence claim: x* = 1
                   "verdict_b": lcs_at(b["num_atoms"], rb, xb)[0], "edge_on_evidence": bool(on_ev),
                   "spot": (spot.get(a["id"]) or {}).get("majority")}
            for bl in BASELINES:
                if a["id"] in base.get(bl, {}) and b["id"] in base.get(bl, {}):
                    row[bl + "_a"], row[bl + "_b"] = base[bl][a["id"]], base[bl][b["id"]]
            pairs.append(row)

        def counts(R, w):
            dec = sum(r[f"{w}_b"] < r[f"{w}_a"] - TIE for r in R)
            inc = sum(r[f"{w}_b"] > r[f"{w}_a"] + TIE for r in R)
            return {"n": len(R), "decrease": dec, "increase": inc, "tie": len(R) - dec - inc,
                    "p_sign": sign_test(dec, inc)}

        edge = [r for r in pairs if r["edge_on_evidence"]]
        out = {"errors": err, "pairs": len(pairs),
               "asserted": counts(pairs, "asserted"), "verdict_all": counts(pairs, "verdict"),
               "verdict_edge": counts(edge, "verdict"), "asserted_edge": counts(edge, "asserted")}
        out["H2a_asserted_pass"] = out["asserted"]["decrease"] > out["asserted"]["increase"] and \
            out["asserted"]["p_sign"] < 0.05
        out["H2a_verdict_pass"] = out["verdict_edge"]["increase"] > out["verdict_edge"]["decrease"] and \
            out["verdict_edge"]["p_sign"] < 0.05
        for bl in BASELINES:
            R = [r for r in pairs if bl + "_a" in r]
            if R:
                out[f"baseline_{bl}"] = counts(R, bl)
        if spot:
            out["spot"] = dict(Counter(r["spot"] for r in pairs))
        # 2b recall on all contradicted documents
        rec_rows = []
        for fam, V in by.items():
            b = V.get("b")
            if not b or b["id"] not in recs:
                continue
            evi = {int(e[1:]) for e in b["expected"]["evidence_atoms"]}
            rb = rels_of(recs[b["id"]])
            hit = any(r[2] in CONFLICTISH and (r[0] in evi or r[1] in evi) for r in rb)
            scope = b["expected"]["scope"]
            scope = scope.strip("[]'\" ").split("'")[0] if isinstance(scope, str) else scope[0]
            rec_rows.append({"group": fam, "hit": hit, "scope": scope,
                             "plug": b["expected"]["contra_plug"], "doc_type": b["expected"]["doc_type"],
                             "other_conflicts": sum(r[2] in CONFLICTISH and not (r[0] in evi or r[1] in evi)
                                                    for r in rb), "claims": b["num_atoms"]})
        rc = {}
        for key in ("scope", "plug", "doc_type"):
            for val in sorted({r[key] for r in rec_rows}):
                R = [r for r in rec_rows if r[key] == val]
                f = lambda S: share(S, lambda r: r["hit"])  # noqa: E731
                rc[f"{key}={val}"] = {"n": len(R), "recall": f(R), "ci": boot(f, [[r] for r in R])}
        out["recall_all"] = share(rec_rows, lambda r: r["hit"])
        out["recall"] = rc
        sc = {k.split("=")[1]: v["recall"] for k, v in rc.items() if k.startswith("scope=")}
        if {"intra", "local", "global"} <= set(sc):
            out["H2b_pass"] = min(sc["intra"], sc["local"]) > sc["global"]
        # 2c false positives on clean documents
        neg = [x for x in it.values() if x["expected"]["check"] == "contradoc_neg" and x["id"] in recs]
        fp = [any(r[2] in CONFLICTISH for r in rels_of(recs[x["id"]])) for x in neg]
        per = [sum(r[2] in CONFLICTISH for r in rels_of(recs[x["id"]])) /
               max(1, x["num_atoms"] * (x["num_atoms"] - 1) / 2) for x in neg]
        out["fp_docs"] = {"n": len(neg), "share_with_conflict": st.mean(fp) if fp else float("nan"),
                          "conflicts_per_pair": st.mean(per) if per else float("nan")}
        res[f"2:{arm}"] = out
    return res


# ----------------------------------------------------------------------------- Check 3

def delta_r(n, rels, k, x):
    full = lambda_exact(n, rels, x)["lambda"]
    rest = [r for i, r in enumerate(rels) if i != k]
    return full - (lambda_exact(n, rest, x)["lambda"] if rest else 0.0)


def check3(args):
    it = items("chatprotect")
    res = {}
    for arm in ARMS:
        recs, err = records("chatprotect", arm)
        sweep_ok = sweep_n = 0
        rows, real = [], Counter()
        by = defaultdict(dict)
        for iid, x in it.items():
            by[x["expected"]["family_id"]][x["expected"]["variant"]] = x
        for pid, V in by.items():
            d0, d1 = V.get("d0"), V.get("d1")
            if not d1 or d1["id"] not in recs:
                continue
            s, s2 = int(d1["expected"]["s_atom"][1:]), int(d1["expected"]["s2_atom"][1:])
            R = rels_of(recs[d1["id"]])
            between = [k for k, r in enumerate(R) if {r[0], r[1]} == {s, s2}]
            for k in between:
                r = R[k]
                for xs in (0, 1):
                    for xs2 in (0, 1):
                        x = [1] * d1["num_atoms"]
                        x[s], x[s2] = xs, xs2
                        d = delta_r(d1["num_atoms"], R, k, x)
                        violated = (x[r[0]], x[r[1]]) in VIOLATED[r[2]]
                        sweep_n += 1
                        sweep_ok += (d < -TIE) if violated else (d > TIE or (abs(d) <= TIE and r[3] <= 0))
            conflict = any(R[k][2] in CONFLICTISH for k in between)
            row = {"group": pid, "label": d1["expected"]["label"], "conflict": conflict,
                   "nonfactual": d1["expected"].get("nonfactual")}
            if d0 and d0["id"] in recs:
                row["d1"] = lcs_at(d1["num_atoms"], R)[0]
                row["d0"] = lcs_at(d0["num_atoms"], rels_of(recs[d0["id"]]))[0]
            rows.append(row)
            nf = d1["expected"].get("nonfactual")
            if nf in ("none", "1", "2", "both") and conflict:
                xs, xs2 = {"none": (1, 1), "1": (0, 1), "2": (1, 0), "both": (0, 0)}[nf]
                for k in between:
                    if R[k][2] in CONFLICTISH:
                        x = [1] * d1["num_atoms"]
                        x[s], x[s2] = xs, xs2
                        real["support" if (x[R[k][0]], x[R[k][1]]) not in VIOLATED[R[k][2]]
                             else "conflict"] += 1
        strong = [r for r in rows if r["label"] == "strong"]
        none = [r for r in rows if r["label"] != "strong"]
        out = {"errors": err, "pairs": len(rows), "sweep_cases": sweep_n,
               "sweep_agree": sweep_ok / sweep_n if sweep_n else float("nan"),
               "p_conflict_strong": share(strong, lambda r: r["conflict"]),
               "p_conflict_none": share(none, lambda r: r["conflict"]),
               "real_verdict_signs": dict(real)}
        out["H3a_pass"] = sweep_n > 0 and sweep_ok == sweep_n
        out["H3_miner_pass"] = out["p_conflict_strong"] > out["p_conflict_none"]
        S = [r for r in strong if "d0" in r]
        dec = sum(r["d1"] < r["d0"] - TIE for r in S)
        inc = sum(r["d1"] > r["d0"] + TIE for r in S)
        out["asserted"] = {"n": len(S), "decrease": dec, "increase": inc, "p_sign": sign_test(dec, inc)}
        out["H3c_pass"] = dec > inc and out["asserted"]["p_sign"] < 0.05
        res[f"3:{arm}"] = out
    return res


# ----------------------------------------------------------------------------- Check 4

def check4(args):
    res = {}
    q = items("qt30")
    for arm in ARMS:
        recs, err = records("qt30", arm)
        tp_p = fp_p = fn_p = tp_c = fp_c = fn_c = 0
        per = defaultdict(lambda: [0, 0])  # coupling -> [found, total]
        dir_ok = dir_n = 0
        conf = Counter()
        for iid, x in q.items():
            if iid not in recs:
                continue
            gold = x["expected"]["gold"]
            mined = [r for r in recs[iid].get("relations") or [] if r["type"] in VIOLATED]

            def key(r, typed):
                s, t = r["source"], r["target"]
                if r["type"] in SYMMETRIC or not typed:
                    s, t = sorted((s, t))
                return (s, t, r["type"]) if typed else (s, t)

            G, M = {key(g, False) for g in gold}, {key(m, False) for m in mined}
            tp_p += len(G & M); fp_p += len(M - G); fn_p += len(G - M)
            G2, M2 = {key(g, True) for g in gold}, {key(m, True) for m in mined}
            tp_c += len(G2 & M2); fp_c += len(M2 - G2); fn_c += len(G2 - M2)
            by_pair = defaultdict(set)
            for m in mined:
                by_pair[key(m, False)].add(m["type"])
            for g in gold:
                kp = key(g, False)
                found = g["type"] in by_pair.get(kp, set())
                per[g["type"]][0] += found
                per[g["type"]][1] += 1
                conf[(g["type"], "/".join(sorted(by_pair.get(kp, {"none"}))))] += 1
                if g["type"] == "entailment" and found:
                    dir_n += 1
                    dir_ok += any(m["type"] == "entailment" and m["source"] == g["source"]
                                  for m in mined if key(m, False) == kp)

        def prf(tp, fp, fn):
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}

        res[f"4a:{arm}"] = {"errors": err, "items": sum(1 for i in q if i in recs),
                            "pair": prf(tp_p, fp_p, fn_p), "coupling": prf(tp_c, fp_c, fn_c),
                            "recall_by_coupling": {k: v[0] / v[1] for k, v in per.items()},
                            "n_by_coupling": {k: v[1] for k, v in per.items()},
                            "entailment_direction": dir_ok / dir_n if dir_n else float("nan"),
                            "confusion": {f"{a}->{b}": c for (a, b), c in conf.most_common(20)}}
    g = items("gdtb")
    for arm in ARMS:
        recs, err = records("gdtb", arm)
        rows = []
        for iid, x in g.items():
            if iid not in recs:
                continue
            mined = {r["type"] for r in recs[iid].get("relations") or [] if r["type"] in VIOLATED}
            gc = x["expected"]["gold_coupling"]
            rows.append({"group": x["expected"]["family_id"], "category": x["expected"]["category"],
                         "gold": gc, "any": bool(mined), "correct": gc in mined if gc != "none" else not mined,
                         "explicit": x["expected"]["pdtb_type"] == "Explicit"})
        none = [r for r in rows if r["gold"] == "none"]
        coup = [r for r in rows if r["gold"] != "none"]
        f_none = lambda S: share([r for r in S if r["gold"] == "none"], lambda r: r["any"])  # noqa: E731
        f_coup = lambda S: share([r for r in S if r["gold"] != "none"], lambda r: r["correct"])  # noqa: E731
        diff = lambda S: f_coup(S) - f_none(S)  # noqa: E731
        groups = grouped(rows, "group")
        out = {"errors": err, "items": len(rows),
               "none_any_coupling": f_none(rows), "coupling_correct": f_coup(rows),
               "diff": diff(rows), "diff_ci": boot(diff, groups),
               "by_category": {c: {"n": len([r for r in rows if r["category"] == c]),
                                   "any": share([r for r in rows if r["category"] == c], lambda r: r["any"]),
                                   "correct": share([r for r in rows if r["category"] == c], lambda r: r["correct"])}
                               for c in sorted({r["category"] for r in rows})},
               "explicit_recall": share([r for r in coup if r["explicit"]], lambda r: r["correct"]),
               "implicit_recall": share([r for r in coup if not r["explicit"]], lambda r: r["correct"]),
               "n_explicit": sum(r["explicit"] for r in coup), "n_implicit": sum(not r["explicit"] for r in coup)}
        out["H4b_none_pass"] = out["diff_ci"][0] > 0
        out["H4b_explicit_pass"] = out["explicit_recall"] > out["implicit_recall"]
        res[f"4b:{arm}"] = out
    return res


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spotcheck", "check1", "check2", "check3", "check4", "all"])
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ext", "eval.json"))
    args = ap.parse_args()
    if args.cmd == "spotcheck":
        return committee_spotcheck(args)
    fns = {"check1": check1, "check2": check2, "check3": check3, "check4": check4}
    run = list(fns) if args.cmd == "all" else [args.cmd]
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in run:
        out[name] = fns[name](args)
    json.dump(out, open(args.out, "w"), indent=1, default=list)
    for name in run:
        print(f"== {name}")
        print(json.dumps(out[name], indent=1, default=list)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
