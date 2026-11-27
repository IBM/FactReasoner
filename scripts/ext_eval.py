"""Evaluate the external checks of LCS (Part A of .claude/plans/external-datasets-eval.md).

The analyses and their thresholds are fixed in this script. Records come from
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


def signed_sum(n, rels, x=None, weighted=True):
    """Review C1 baseline: +p (or +1) for each relation the world satisfies, -p (or -1) for
    each it violates. Monotone under elementary improvements like LCS, but reads no MRF."""
    x = x if x is not None else [1] * n
    return float(sum((r[3] if weighted else 1.0) * (-1 if (x[r[0]], x[r[1]]) in VIOLATED[r[2]] else 1)
                     for r in rels))


def mean_abs_diff_ci(A_, B_, n=BOOT, seed=SEED):
    """Group bootstrap of mean|d|(A) - mean|d|(B), with A and B lists of groups of |d|."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        a = [v for g in (rng.choice(A_) for _ in A_) for v in g]
        b = [v for g in (rng.choice(B_) for _ in B_) for v in g]
        vals.append(st.mean(a) - st.mean(b))
    vals.sort()
    return vals[int(0.05 * n)], vals[int(0.95 * n) - 1], vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def mcnemar(rows, a, b):
    """Paired comparison of two binary outcomes on the same rows (exact two-sided McNemar):
    rows where only `a` holds against rows where only `b` holds."""
    only_a = sum(1 for r in rows if r[a] and not r[b])
    only_b = sum(1 for r in rows if r[b] and not r[a])
    return {"only_a": only_a, "only_b": only_b, "n": len(rows), "p": sign_test(only_a, only_b)}


def order_equivalence(perm, noise):
    """Is the permutation effect on LCS equivalent to re-mining noise? TOST at alpha = 0.05 with
    margin TOL on the difference of mean |change| (90% group-bootstrap interval inside
    (-TOL, TOL)), plus the direction of the changes beyond TOL in each set."""
    def groups(R):
        G = defaultdict(list)
        for r in R:
            G[r["group"]].append(abs(r["lcs"] - r["lcs_orig"]))
        return list(G.values())
    lo90, hi90, lo95, hi95 = mean_abs_diff_ci(groups(perm), groups(noise))
    up = lambda R: sum(r["lcs"] > r["lcs_orig"] + TOL for r in R)  # noqa: E731
    dn = lambda R: sum(r["lcs"] < r["lcs_orig"] - TOL for r in R)  # noqa: E731
    return {"diff": st.mean(abs(r["lcs"] - r["lcs_orig"]) for r in perm)
            - st.mean(abs(r["lcs"] - r["lcs_orig"]) for r in noise),
            "ci90": (lo90, hi90), "ci95": (lo95, hi95), "margin": TOL,
            "equivalent": -TOL < lo90 and hi90 < TOL,
            "perm_up": up(perm), "perm_down": dn(perm), "perm_p": sign_test(dn(perm), up(perm)),
            "noise_up": up(noise), "noise_down": dn(noise), "noise_p": sign_test(dn(noise), up(noise)),
            "n_perm": len(perm), "n_noise": len(noise)}


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
                  "down_tol": sum(r[m] < r[m + "_orig"] - TOL for r in R),
                  "up_tol": sum(r[m] > r[m + "_orig"] + TOL for r in R),
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
                            "noise_floor": invariance_block(noise, ["lcs"]),
                            "equivalence": order_equivalence(perm, noise) if noise else None}
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
        # Noise floor: the 150 orig documents re-mined with fresh samples.
        rr, _ = records("contradoc-rerun", arm)
        noise = []
        for fam, V in by.items():
            o = V.get("orig")
            if o and o.get("expected", {}).get("shuffled") and o["id"] in recs and o["id"] in rr:
                noise.append({"group": fam,
                              "lcs": lcs_at(o["num_atoms"], rels_of(rr[o["id"]]))[0],
                              "lcs_orig": lcs_at(o["num_atoms"], rels_of(recs[o["id"]]))[0]})
        res[f"1c:{arm}"] = {"errors": err, "perm": invariance_block(rows, ["lcs"] + list(BASELINES)),
                            "noise_floor": invariance_block(noise, ["lcs"]),
                            "equivalence": order_equivalence(rows, noise) if noise else None}
    # 1e (review Phase 3.4): matched design. Each original and permutation was mined twice in
    # one interleaved run, so noise = two mines of the same text, and the permutation effect
    # compares a permutation with the original from the other repeat (independent samples).
    mi = items("contradoc-matched") if os.path.exists(os.path.join(REPO, "data", "ext",
                                                                     "contradoc-matched")) else {}
    for arm in ARMS:
        if not mi:
            break
        recs, err = records("contradoc-matched", arm)
        if not recs:
            continue
        L = defaultdict(dict)
        for iid, x in mi.items():
            if iid in recs:
                fam, v, r = x["expected"]["family_id"], x["expected"]["variant"], x["expected"]["repeat"]
                L[fam][(v, r)] = lcs_at(x["num_atoms"], rels_of(recs[iid]))[0]
        noise, perm = [], []
        for fam, V in L.items():
            for v in ("orig", "p1", "p2"):
                if (v, "r1") in V and (v, "r2") in V:
                    noise.append({"group": fam, "lcs": V[(v, "r2")], "lcs_orig": V[(v, "r1")]})
            for v in ("p1", "p2"):
                for rp, ro in (("r1", "r2"), ("r2", "r1")):
                    if (v, rp) in V and ("orig", ro) in V:
                        perm.append({"group": fam, "lcs": V[(v, rp)], "lcs_orig": V[("orig", ro)]})
        res[f"1e:{arm}"] = {"errors": err, "records": len(recs),
                            "perm": invariance_block(perm, ["lcs"]),
                            "noise_floor": invariance_block(noise, ["lcs"]),
                            "equivalence": order_equivalence(perm, noise) if noise and perm else None}
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
            kinds = Counter(t for _, _, t, _ in r)
            rows.append({"group": x["expected"]["family_id"], "lcs": lcs_at(x["num_atoms"], r)[0],
                         "wsum": signed_sum(x["num_atoms"], r),
                         "lcs_norestate": lcs_at(x["num_atoms"],
                                                 [q for q in r if q[2] != "equivalence"])[0],
                         "empty": not r, "n_ent": kinds["entailment"], "n_eqv": kinds["equivalence"],
                         "n_con": sum(kinds[k] for k in CONFLICTISH), "n_rel": len(r),
                         **x["expected"]["expert"],
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
        # Summaries with at least one relation: on the rest LCS is exactly 1/2, so this checks that
        # the null correlation is not an artefact of the score being constant there.
        NE = [r for r in rows if not r["empty"]]
        f = lambda S: spearman([r["lcs"] for r in S], [r["coherence"] for r in S])  # noqa: E731
        out["nonempty_n"] = len(NE)
        out["nonempty_rho_coherence"] = f(NE)
        out["nonempty_rho_coherence_ci"] = boot(f, grouped(NE, "group"))
        # Restatements (equivalences) are supports, so they raise LCS, while raters penalize the
        # redundancy they signal; inferences alone, and the two relation counts, separate the effects.
        INF = [r for r in NE if r["n_eqv"] == 0]
        EQV = [r for r in NE if r["n_eqv"] > 0]
        for key, S in (("inference_only", INF), ("restatement", EQV)):
            out[f"{key}_n"] = len(S)
            out[f"{key}_rho_coherence"] = f(S)
            out[f"{key}_rho_coherence_ci"] = boot(f, grouped(S, "group"))
        for key, col in (("restatements", "n_eqv"), ("inferences", "n_ent")):
            g = lambda S, c=col: spearman([r[c] for r in S], [r["coherence"] for r in S])  # noqa: E731
            out[f"count_{key}_rho_coherence"] = g(NE)
            out[f"count_{key}_rho_coherence_ci"] = boot(g, grouped(NE, "group"))
        for key, col in (("wsum", "wsum"), ("norestate", "lcs_norestate")):
            g = lambda S, c=col: spearman([r[c] for r in S], [r["coherence"] for r in S])  # noqa: E731
            out[f"{key}_rho_coherence"] = g(rows)
            out[f"{key}_rho_coherence_ci"] = boot(g, groups)
            out[f"{key}_nonempty_rho_coherence"] = g(NE)
            out[f"{key}_nonempty_rho_coherence_ci"] = boot(g, grouped(NE, "group"))
        out["relations_by_kind"] = {"inference": sum(r["n_ent"] for r in NE),
                                    "restatement": sum(r["n_eqv"] for r in NE),
                                    "conflict": sum(r["n_con"] for r in NE),
                                    "all": sum(r["n_rel"] for r in NE)}
        JNE = [r for r in NE if r["judge"] is not None]
        if JNE:
            out["nonempty_judge_rho_coherence"] = spearman([r["judge"] for r in JNE],
                                                           [r["coherence"] for r in JNE])
        out["H1d_pass"] = abs(out["rho_coherence"]) < 0.1
        res[f"1d:{arm}"] = out
    # Order checks H1a / H1c: LCS kept within TOL more often than every baseline
    for key in [k for k in res if k[:2] in ("1a", "1c")]:
        P = res[key]["perm"]
        if "lcs" in P:
            others = [b for b in BASELINES if b in P]
            res[key]["H_pass"] = all(P["lcs"]["invariant"] > P[b]["invariant"] for b in others) \
                if others else None
            # D1: RC and LC are order-invariant by definition (bag-of-sentences noun counts).
            dyn = [b for b in others if b not in ("discourse_rc", "discourse_lc")]
            res[key]["H_pass_excl_order_free"] = all(
                P["lcs"]["invariant"] > P[b]["invariant"] for b in dyn) if dyn else None
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
            touch = [r for r in rb if r[0] in evi or r[1] in evi]
            row = {"group": fam, "scope": b["expected"]["scope"], "doc_type": b["expected"]["doc_type"],
                   "asserted_a": lcs_at(a["num_atoms"], ra)[0], "asserted_b": lcs_at(b["num_atoms"], rb)[0],
                   "verdict_a": lcs_at(a["num_atoms"], ra)[0],  # a has no evidence claim: x* = 1
                   "verdict_b": lcs_at(b["num_atoms"], rb, xb)[0], "edge_on_evidence": bool(on_ev),
                   "wsum_a": signed_sum(a["num_atoms"], ra), "wsum_b": signed_sum(b["num_atoms"], rb),
                   "spot": (spot.get(a["id"]) or {}).get("majority"),
                   # Exploratory: sign, at the verdict world, of every relation on the evidence.
                   "ev_supports": sum((xb[r[0]], xb[r[1]]) not in VIOLATED[r[2]] for r in touch),
                   "ev_conflicts": sum((xb[r[0]], xb[r[1]]) in VIOLATED[r[2]] for r in touch)}
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
        ev_sup = sum(r["ev_supports"] for r in edge)
        ev_con = sum(r["ev_conflicts"] for r in edge)
        falls = [r for r in edge if r["verdict_b"] < r["verdict_a"] - TIE]
        out = {"errors": err, "pairs": len(pairs),
               "exploratory_evidence_relations": {
                   "supports": ev_sup, "conflicts": ev_con,
                   "verdict_falls": len(falls),
                   "verdict_falls_without_evidence_conflict": sum(r["ev_conflicts"] == 0 for r in falls)},
               "asserted": counts(pairs, "asserted"), "verdict_all": counts(pairs, "verdict"),
               "verdict_edge": counts(edge, "verdict"), "asserted_edge": counts(edge, "asserted"),
               "wsum": counts(pairs, "wsum"), "wsum_edge": counts(edge, "wsum"),
               "mcnemar_falls": mcnemar([dict(r, lf=r["asserted_b"] < r["asserted_a"] - TIE,
                                              wf=r["wsum_b"] < r["wsum_a"] - TIE) for r in pairs], "lf", "wf"),
               "mcnemar_falls_edge": mcnemar([dict(r, lf=r["asserted_b"] < r["asserted_a"] - TIE,
                                                   wf=r["wsum_b"] < r["wsum_a"] - TIE) for r in edge], "lf", "wf"),
               "asserted_notfound": counts([r for r in pairs if not r["edge_on_evidence"]], "asserted"),
               "effect": {k: {"mean": st.mean(r["asserted_b"] - r["asserted_a"] for r in R),
                              "median": st.median(r["asserted_b"] - r["asserted_a"] for r in R),
                              "n": len(R)}
                          for k, R in (("found", edge),
                                       ("notfound", [r for r in pairs if not r["edge_on_evidence"]]))
                          if R}}
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
            nfv = d1["expected"].get("nonfactual")
            if d0 and d0["id"] in recs and nfv in ("1", "2", "both"):
                # Phase 3.6: the real verdicts, where the annotators recorded the false side. A
                # contradiction that involves a false sentence is a support there.
                xs_, xs2_ = {"1": (0, 1), "2": (1, 0), "both": (0, 0)}[nfv]
                x1 = [1] * d1["num_atoms"]
                x1[s], x1[s2] = xs_, xs2_
                x0 = [1] * d0["num_atoms"]
                s0 = int(d0["expected"]["s_atom"][1:])
                x0[s0] = xs_
                row["nfv"] = nfv
                row["v1"] = lcs_at(d1["num_atoms"], R, x1)[0]
                row["v0"] = lcs_at(d0["num_atoms"], rels_of(recs[d0["id"]]), x0)[0]
                row["a1"] = lcs_at(d1["num_atoms"], R)[0]
                row["a0"] = lcs_at(d0["num_atoms"], rels_of(recs[d0["id"]]))[0]
            if d0 and d0["id"] in recs:
                R0 = rels_of(recs[d0["id"]])
                row["d1"] = lcs_at(d1["num_atoms"], R)[0]
                row["d0"] = lcs_at(d0["num_atoms"], R0)[0]
                row["w1"], row["w0"] = signed_sum(d1["num_atoms"], R), signed_sum(d0["num_atoms"], R0)
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
        # Exploratory: the asserted-world change split by whether the contradiction was found.
        out["asserted_by_found"] = {
            "found_fall": sum(r["conflict"] and r["d1"] < r["d0"] - TIE for r in S),
            "found_rise": sum(r["conflict"] and r["d1"] > r["d0"] + TIE for r in S),
            "notfound_fall": sum(not r["conflict"] and r["d1"] < r["d0"] - TIE for r in S),
            "notfound_rise": sum(not r["conflict"] and r["d1"] > r["d0"] + TIE for r in S)}
        out["asserted"] = {"n": len(S), "decrease": dec, "increase": inc, "p_sign": sign_test(dec, inc)}

        def split(R, a="d0", b="d1"):
            dn = sum(r[b] < r[a] - TIE for r in R)
            up = sum(r[b] > r[a] + TIE for r in R)
            return {"n": len(R), "decrease": dn, "increase": up, "tie": len(R) - dn - up,
                    "p_sign": sign_test(dn, up)}
        # Specificity control: the same edit (adding the second sentence) on pairs that are NOT
        # contradictory. LCS should fall less often there than on contradictory pairs.
        C = [r for r in none if "d0" in r]
        out["control_noncontradictory"] = split(C)
        out["asserted_found"] = split([r for r in S if r["conflict"]])
        out["asserted_notfound"] = split([r for r in S if not r["conflict"]])
        out["wsum"] = split(S, "w0", "w1")
        out["wsum_found"] = split([r for r in S if r["conflict"]], "w0", "w1")
        fl = lambda R: [dict(r, lf=r["d1"] < r["d0"] - TIE, wf=r["w1"] < r["w0"] - TIE) for r in R]  # noqa: E731
        out["mcnemar_falls"] = mcnemar(fl(S), "lf", "wf")
        out["mcnemar_falls_found"] = mcnemar(fl([r for r in S if r["conflict"]]), "lf", "wf")
        out["mcnemar_falls_notfound"] = mcnemar(fl([r for r in S if not r["conflict"]]), "lf", "wf")
        out["mcnemar_falls_control"] = mcnemar(fl(C), "lf", "wf")
        VW = [r for r in rows if "v1" in r]
        out["verdict_world"] = {"all": split(VW, "v0", "v1"),
                                "found": split([r for r in VW if r["conflict"]], "v0", "v1"),
                                "found_asserted": split([r for r in VW if r["conflict"]], "a0", "a1"),
                                "found_both_false": sum(1 for r in VW if r["conflict"] and r["nfv"] == "both")}
        out["wsum_control"] = split(C, "w0", "w1")
        out["effect"] = {k: {"mean": st.mean(r["d1"] - r["d0"] for r in R),
                             "median": st.median(r["d1"] - r["d0"] for r in R), "n": len(R)}
                         for k, R in (("found", [r for r in S if r["conflict"]]),
                                      ("notfound", [r for r in S if not r["conflict"]]),
                                      ("control", C)) if R}
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


# ----------------------------------------------------------------------------- macros

def write_macros(ev, path):
    """LaTeX macros (prefix \\ext; no digits in names) for the coherence paper's external checks."""
    digits = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    L = ["% Generated by scripts/ext_eval.py macros -- do not edit by hand."]

    def m(name, val):
        name = "".join(digits[int(c)] if c.isdigit() else c for c in name)
        L.append(f"\\newcommand{{\\ext{name}}}{{{val}}}")

    def pct(x):
        return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}"

    def f2(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "--"
        return f"\\ensuremath{{{x:+.2f}}}"

    def pval(p):
        if p < 1e-3:  # an upper bound, a power of ten (read as "p < ...")
            e = int(math.floor(math.log10(p)))
            return f"\\ensuremath{{10^{{{e + 1}}}}}"
        return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"

    def held(b):
        return "held" if b else ("failed" if b is not None else "--")

    c1, c2, c3, c4 = (ev.get(k, {}) for k in ("check1", "check2", "check3", "check4"))
    for arm, tag in (("graph_gpt6", "Gpt"), ("graph_gptoss", "Oss")):
        a = c1.get(f"1a:{arm}")
        if a:
            P = a["perm"]
            m(f"LocoPermInv{tag}", pct(P["lcs"]["invariant"]))
            m(f"LocoPermDec{tag}", pct(P["lcs"]["decrease"]))
            m(f"LocoPermDownTol{tag}", P["lcs"]["down_tol"])
            m(f"LocoPermUpTol{tag}", P["lcs"]["up_tol"])
            m(f"LocoPermN{tag}", P["lcs"]["n"])
            m(f"LocoPermMean{tag}", f"{P['lcs']['mean_abs']:.3f}")
            m(f"LocoNoiseInv{tag}", pct(a["noise_floor"]["lcs"]["invariant"]))
            m(f"LocoSentInv{tag}", pct(a["sent"]["lcs"]["invariant"]))
            m(f"HOneA{tag}", held(a["H_pass"]))
            m(f"HOneAExcl{tag}", held(a["H_pass_excl_order_free"]))
            if tag == "Gpt":
                for b, bt in (("judge_geval@gpt-6-sol", "Judge"), ("discourse_entity_graph", "Entity"),
                              ("discourse_rc", "Rc"), ("discourse_lc", "Lc")):
                    if b in P:
                        m(f"LocoPermInv{bt}", pct(P[b]["invariant"]))
                        m(f"LocoPermDec{bt}", pct(P[b]["decrease"]))
                    if b in a["sent"]:
                        m(f"LocoSentInv{bt}", pct(a["sent"][b]["invariant"]))
                        m(f"LocoSentDec{bt}", pct(a["sent"][b]["decrease"]))
        c = c1.get(f"1c:{arm}")
        if c:
            P = c["perm"]
            m(f"CdPermInv{tag}", pct(P["lcs"]["invariant"]))
            m(f"CdPermN{tag}", P["lcs"]["n"])
            m(f"CdPermMean{tag}", f"{P['lcs']['mean_abs']:.3f}")
            if c.get("noise_floor", {}).get("lcs"):
                m(f"CdNoiseInv{tag}", pct(c["noise_floor"]["lcs"]["invariant"]))
                m(f"CdNoiseMean{tag}", f"{c['noise_floor']['lcs']['mean_abs']:.3f}")
            m(f"HOneC{tag}", held(c["H_pass"]))
            m(f"HOneCExcl{tag}", held(c["H_pass_excl_order_free"]))
            if tag == "Gpt":
                for b, bt in (("judge_geval@gpt-6-sol", "Judge"), ("discourse_entity_graph", "Entity"),
                              ("discourse_rc", "Rc"), ("discourse_lc", "Lc")):
                    if b in P:
                        m(f"CdPermInv{bt}", pct(P[b]["invariant"]))
                        m(f"CdPermDec{bt}", pct(P[b]["decrease"]))
        d = c1.get(f"1d:{arm}")
        if d:
            if tag == "Gpt":
                m("SeN", d["n"])
            m(f"SeRhoCoh{tag}", f2(d["rho_coherence"]))
            m(f"SeRhoCoh{tag}Lo", f2(d["rho_coherence_ci"][0]))
            m(f"SeRhoCoh{tag}Hi", f2(d["rho_coherence_ci"][1]))
            for k in ("consistency", "fluency", "relevance"):
                m(f"SeRho{k.title()}{tag}", f2(d[f"rho_{k}"]))
            m(f"SeEmpty{tag}", pct(d["empty_share"]))
            if "nonempty_rho_coherence" in d:
                m(f"SeNonEmptyN{tag}", f"{d['nonempty_n']:,}".replace(",", "{,}"))
                m(f"SeNonEmptyRhoCoh{tag}", f2(d["nonempty_rho_coherence"]))
                m(f"SeNonEmptyRhoCoh{tag}Lo", f2(d["nonempty_rho_coherence_ci"][0]))
                m(f"SeNonEmptyRhoCoh{tag}Hi", f2(d["nonempty_rho_coherence_ci"][1]))
                if "nonempty_judge_rho_coherence" in d:
                    m(f"SeNonEmptyJudgeRhoCoh{tag}", f2(d["nonempty_judge_rho_coherence"]))
            for key, kt in (("inference_only", "InfOnly"), ("restatement", "Restate")):
                if f"{key}_n" in d:
                    m(f"Se{kt}N{tag}", d[f"{key}_n"])
                    m(f"Se{kt}RhoCoh{tag}", f2(d[f"{key}_rho_coherence"]))
                    m(f"Se{kt}RhoCoh{tag}Lo", f2(d[f"{key}_rho_coherence_ci"][0]))
                    m(f"Se{kt}RhoCoh{tag}Hi", f2(d[f"{key}_rho_coherence_ci"][1]))
            for key, kt in (("restatements", "CountRestate"), ("inferences", "CountInf")):
                if f"count_{key}_rho_coherence" in d:
                    m(f"Se{kt}RhoCoh{tag}", f2(d[f"count_{key}_rho_coherence"]))
                    m(f"Se{kt}RhoCoh{tag}Lo", f2(d[f"count_{key}_rho_coherence_ci"][0]))
                    m(f"Se{kt}RhoCoh{tag}Hi", f2(d[f"count_{key}_rho_coherence_ci"][1]))
            for k_, v_ in d.get("relations_by_kind", {}).items():
                m(f"SeRel{k_.title()}{tag}", f"{v_:,}".replace(",", "{,}"))
            m(f"HOneD{tag}", held(d["H1d_pass"]))
            if tag == "Gpt" and "judge_rho_coherence" in d:
                m("SeJudgeRhoCoh", f2(d["judge_rho_coherence"]))
        x = c2.get(f"2:{arm}")
        if x:
            if tag == "Gpt":
                m("CdPairs", x["pairs"])
            ex_ = x.get("exploratory_evidence_relations", {})
            for k_, kt in (("supports", "EvSup"), ("conflicts", "EvCon"), ("verdict_falls", "VerdFalls"),
                           ("verdict_falls_without_evidence_conflict", "VerdFallsNoCon")):
                if k_ in ex_:
                    m(f"Cd{kt}{tag}", ex_[k_])
            for w, wt in (("asserted", "Assert"), ("verdict_all", "VerdAll"), ("verdict_edge", "VerdEdge"),
                          ("asserted_edge", "AssertEdge")):
                m(f"Cd{wt}Dec{tag}", x[w]["decrease"])
                m(f"Cd{wt}Inc{tag}", x[w]["increase"])
                m(f"Cd{wt}N{tag}", x[w]["n"])
                m(f"Cd{wt}P{tag}", pval(x[w]["p_sign"]))
            m(f"HTwoAAssert{tag}", held(x["H2a_asserted_pass"]))
            m(f"HTwoAVerd{tag}", held(x["H2a_verdict_pass"]))
            m(f"CdRecall{tag}", pct(x["recall_all"]))
            for k, v in x["recall"].items():
                key, val = k.split("=")
                if val in ("global", "local", "intra", "Insert", "Replace", "news", "story", "wiki"):
                    m(f"CdRecall{val.title()}{tag}", pct(v["recall"]))
                    if val in ("global", "local", "intra") and v.get("ci"):
                        m(f"CdRecall{val.title()}{tag}Lo", pct(v["ci"][0]))
                        m(f"CdRecall{val.title()}{tag}Hi", pct(v["ci"][1]))
            m(f"HTwoB{tag}", held(x.get("H2b_pass")))
            m(f"CdFp{tag}", pct(x["fp_docs"]["share_with_conflict"]))
            cpp = x["fp_docs"]["conflicts_per_pair"]
            ex = int(math.floor(math.log10(cpp)))
            m(f"CdFpPair{tag}", f"\\ensuremath{{{cpp / 10 ** ex:.1f}\\times10^{{{ex}}}}}")
            if tag == "Gpt":
                if "spot" in x:
                    m("CdSpotContradicted", x["spot"].get("contradicted", 0))
                for b, bt in (("judge_geval@gpt-6-sol", "Judge"), ("discourse_rc", "Rc"),
                              ("discourse_lc", "Lc"), ("discourse_entity_graph", "Entity")):
                    if f"baseline_{b}" in x:
                        B = x[f"baseline_{b}"]
                        m(f"Cd{bt}Dec", B["decrease"])
                        m(f"Cd{bt}Inc", B["increase"])
                        m(f"Cd{bt}N", B["n"])
        y = c3.get(f"3:{arm}")
        if y:
            if tag == "Gpt":
                m("CpPairs", y["pairs"])
            m(f"CpSweepN{tag}", f"{y['sweep_cases']:,}".replace(",", "{,}"))
            m(f"CpSweepAgree{tag}", pct(y["sweep_agree"]))
            m(f"CpConfStrong{tag}", pct(y["p_conflict_strong"]))
            m(f"CpConfNone{tag}", pct(y["p_conflict_none"]))
            m(f"CpAssertDec{tag}", y["asserted"]["decrease"])
            m(f"CpAssertInc{tag}", y["asserted"]["increase"])
            m(f"CpAssertN{tag}", y["asserted"]["n"])
            m(f"CpAssertP{tag}", pval(y["asserted"]["p_sign"]))
            for k_, kt in (("found_fall", "FoundFall"), ("found_rise", "FoundRise"),
                           ("notfound_fall", "NotFoundFall"), ("notfound_rise", "NotFoundRise")):
                m(f"Cp{kt}{tag}", y.get("asserted_by_found", {}).get(k_, "--"))
            m(f"CpRealSupport{tag}", y["real_verdict_signs"].get("support", 0))
            m(f"CpRealConflict{tag}", y["real_verdict_signs"].get("conflict", 0))
            for h, ht in (("H3a_pass", "HThreeA"), ("H3_miner_pass", "HThreeMiner"), ("H3c_pass", "HThreeC")):
                m(f"{ht}{tag}", held(y[h]))
        # Phase 2 of the 2026-11-27 review: signed baselines, controls, effect sizes, equivalence.
        def f3(v):
            return "--" if v is None else f"\\ensuremath{{{v:+.3f}}}"
        for key, kt in (("1a", "Loco"), ("1c", "Cd"), ("1e", "Mt")):
            e = (c1.get(f"{key}:{arm}") or {}).get("equivalence")
            if e:
                m(f"{kt}EqDiff{tag}", f3(e["diff"]))
                m(f"{kt}EqLo{tag}", f3(e["ci90"][0]))
                m(f"{kt}EqHi{tag}", f3(e["ci90"][1]))
                m(f"{kt}EqHeld{tag}", "yes" if e["equivalent"] else "no")
                for k_ in ("perm_up", "perm_down", "noise_up", "noise_down", "n_perm", "n_noise"):
                    m(f"{kt}Eq{k_.title().replace('_', '')}{tag}", e[k_])
                m(f"{kt}EqPermP{tag}", pval(e["perm_p"]))
                m(f"{kt}EqNoiseP{tag}", pval(e["noise_p"]))
        mt = c1.get(f"1e:{arm}")
        if mt:
            m(f"MtRecords{tag}", mt["records"])
            m(f"MtPermInv{tag}", pct(mt["perm"]["lcs"]["invariant"]))
            m(f"MtNoiseInv{tag}", pct(mt["noise_floor"]["lcs"]["invariant"]))
            m(f"MtPermMean{tag}", f"{mt['perm']['lcs']['mean_abs']:.3f}")
            m(f"MtNoiseMean{tag}", f"{mt['noise_floor']['lcs']['mean_abs']:.3f}")
        d = c1.get(f"1d:{arm}")
        if d and "wsum_rho_coherence" in d:
            for key, kt in (("wsum", "Wsum"), ("norestate", "NoRestate")):
                m(f"Se{kt}RhoCoh{tag}", f2(d[f"{key}_rho_coherence"]))
                m(f"Se{kt}NonEmptyRhoCoh{tag}", f2(d[f"{key}_nonempty_rho_coherence"]))
                m(f"Se{kt}NonEmptyRhoCoh{tag}Lo", f2(d[f"{key}_nonempty_rho_coherence_ci"][0]))
                m(f"Se{kt}NonEmptyRhoCoh{tag}Hi", f2(d[f"{key}_nonempty_rho_coherence_ci"][1]))
        x = c2.get(f"2:{arm}")
        if x and "wsum" in x:
            for k_, kt in (("wsum", "Wsum"), ("wsum_edge", "WsumEdge"), ("asserted_notfound", "AssertNotFound")):
                m(f"Cd{kt}Dec{tag}", x[k_]["decrease"])
                m(f"Cd{kt}Inc{tag}", x[k_]["increase"])
                m(f"Cd{kt}Tie{tag}", x[k_]["tie"])
                m(f"Cd{kt}N{tag}", x[k_]["n"])
                m(f"Cd{kt}P{tag}", pval(x[k_]["p_sign"]))
            m(f"CdAssertTie{tag}", x["asserted"]["tie"])
            for k_, kt in (("mcnemar_falls", "McAll"), ("mcnemar_falls_edge", "McFound")):
                if k_ in x:
                    m(f"Cd{kt}OnlyLcs{tag}", x[k_]["only_a"])
                    m(f"Cd{kt}OnlySum{tag}", x[k_]["only_b"])
                    m(f"Cd{kt}P{tag}", pval(x[k_]["p"]))
            for k_, kt in (("found", "Found"), ("notfound", "NotFound")):
                if k_ in x["effect"]:
                    m(f"CdEff{kt}Mean{tag}", f2(x["effect"][k_]["mean"]))
                    m(f"CdEff{kt}Median{tag}", f2(x["effect"][k_]["median"]))
        y = c3.get(f"3:{arm}")
        if y and "control_noncontradictory" in y:
            for k_, kt in (("control_noncontradictory", "Ctl"), ("asserted_found", "Found"),
                           ("asserted_notfound", "NotFound"), ("wsum", "Wsum"), ("wsum_control", "WsumCtl"),
                           ("wsum_found", "WsumFound")):
                m(f"Cp{kt}Dec{tag}", y[k_]["decrease"])
                m(f"Cp{kt}Inc{tag}", y[k_]["increase"])
                m(f"Cp{kt}Tie{tag}", y[k_]["tie"])
                m(f"Cp{kt}N{tag}", y[k_]["n"])
                m(f"Cp{kt}P{tag}", pval(y[k_]["p_sign"]))
            m(f"CpAssertTie{tag}", y["asserted"]["n"] - y["asserted"]["decrease"] - y["asserted"]["increase"])
            for k_, kt in (("mcnemar_falls", "McAll"), ("mcnemar_falls_found", "McFound"),
                           ("mcnemar_falls_notfound", "McNotFound"), ("mcnemar_falls_control", "McCtl")):
                if k_ in y:
                    m(f"Cp{kt}OnlyLcs{tag}", y[k_]["only_a"])
                    m(f"Cp{kt}OnlySum{tag}", y[k_]["only_b"])
                    m(f"Cp{kt}P{tag}", pval(y[k_]["p"]))
            if "found_both_false" in y.get("verdict_world", {}):
                m(f"CpVwFoundBothFalse{tag}", y["verdict_world"]["found_both_false"])
            for k_, kt in (("all", "VwAll"), ("found", "VwFound"), ("found_asserted", "VwFoundAssert")):
                v_ = y.get("verdict_world", {}).get(k_)
                if v_:
                    m(f"Cp{kt}Dec{tag}", v_["decrease"])
                    m(f"Cp{kt}Inc{tag}", v_["increase"])
                    m(f"Cp{kt}Tie{tag}", v_["tie"])
                    m(f"Cp{kt}N{tag}", v_["n"])
                    m(f"Cp{kt}P{tag}", pval(v_["p_sign"]))
            for k_, kt in (("found", "Found"), ("notfound", "NotFound"), ("control", "Ctl")):
                if k_ in y["effect"]:
                    m(f"CpEff{kt}Mean{tag}", f2(y["effect"][k_]["mean"]))
                    m(f"CpEff{kt}Median{tag}", f2(y["effect"][k_]["median"]))
        q = c4.get(f"4a:{arm}")
        if q:
            m(f"QtPairF{tag}", f"{q['pair']['f1']:.2f}")
            m(f"QtPairP{tag}", f"{q['pair']['p']:.2f}")
            m(f"QtPairR{tag}", f"{q['pair']['r']:.2f}")
            m(f"QtCoupF{tag}", f"{q['coupling']['f1']:.2f}")
            m(f"QtCoupP{tag}", f"{q['coupling']['p']:.2f}")
            m(f"QtCoupR{tag}", f"{q['coupling']['r']:.2f}")
            for k, v in q["recall_by_coupling"].items():
                m(f"QtRec{k.title().replace('_', '')}{tag}", pct(v))
            m(f"QtDir{tag}", pct(q["entailment_direction"]))
            if tag == "Gpt":
                m("QtWindows", q["items"])
                for k, v in q["n_by_coupling"].items():
                    m(f"QtN{k.title().replace('_', '')}", f"{v:,}".replace(",", "{,}"))
                eqv = q["confusion"].get("equivalence->entailment", 0)
                m("QtEqvAsEnt", eqv)
        g = c4.get(f"4b:{arm}")
        if g:
            if tag == "Gpt":
                m("GdItems", f"{g['items']:,}".replace(",", "{,}"))
            m(f"GdNone{tag}", pct(g["none_any_coupling"]))
            m(f"GdCorrect{tag}", pct(g["coupling_correct"]))
            m(f"GdExplicit{tag}", pct(g["explicit_recall"]))
            m(f"GdImplicit{tag}", pct(g["implicit_recall"]))
            m(f"HFourBNone{tag}", held(g["H4b_none_pass"]))
            m(f"HFourBExplicit{tag}", held(g["H4b_explicit_pass"]))
            for cat, v in g["by_category"].items():
                ct = cat.replace("-", "").replace("/", "")
                m(f"Gd{ct}Any{tag}", pct(v["any"]))
                m(f"Gd{ct}Correct{tag}", pct(v["correct"]))
                if tag == "Gpt":
                    m(f"Gd{ct}N", v["n"])
    # Held counts per arm. The paper groups the advance predictions into six about the score
    # (E1-E6: H1d; H1a and H1c together; H2a; H3c; H3a; H2a') and four about the miner alone
    # (H4b-none, which is E7, H2b, H3-miner, H4b-explicit).
    WORDS = "zero one two three four five six seven eight nine ten".split()  # counts, as prose
    vals = {l.split("}{")[0].split("\\ext", 1)[1]: l.split("}{", 1)[1][:-1] for l in L[1:]}
    for tag in ("Gpt", "Oss"):
        ok = lambda k: vals.get(f"{k}{tag}") == "held"  # noqa: E731
        score = [ok("HOneD"), ok("HOneA") and ok("HOneC"), ok("HTwoAAssert"), ok("HThreeC"),
                 ok("HThreeA"), ok("HTwoAVerd")]
        miner = [ok("HFourBNone"), ok("HTwoB"), ok("HThreeMiner"), ok("HFourBExplicit")]
        m(f"ScoreHeld{tag}", WORDS[sum(score)])
        m(f"MinerHeld{tag}", WORDS[sum(miner)])
    m("ScoreN", WORDS[6])
    m("MinerN", WORDS[4])
    names = [l.split("}{")[0] for l in L[1:]]
    dup = [k for k, n in Counter(names).items() if n > 1]
    assert not dup, f"duplicate macro names: {dup}"
    # Every \ext macro the paper uses gets a definition, so the paper builds while runs are
    # still in progress; a pending value prints as "--" and is listed here.
    import re as _re
    defined = {n.split("\\ext", 1)[1] for n in names}
    used = set()
    for tex in ("sec_external.tex", "appendix_external.tex", "sec_locobench.tex", "sec_intro.tex",
                "sec_end.tex", "main.tex"):
        tp = os.path.join(os.path.dirname(path), tex)
        if os.path.exists(tp):
            used |= set(_re.findall(r"\\ext([A-Za-z]+)", open(tp).read()))
    pending = sorted(used - defined)
    if pending:
        L.append("% Pending (data not yet complete): " + ", ".join(pending))
        L += [f"\\newcommand{{\\ext{n}}}{{--}}" for n in pending]
        print(f"pending macros ({len(pending)}): {', '.join(pending)}")
    open(path, "w").write("\n".join(L) + "\n")
    print(f"wrote {path} ({len(L) - 1} macros)")


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spotcheck", "check1", "check2", "check3", "check4", "all",
                                    "macros"])
    ap.add_argument("--macros", default=os.path.join(REPO, "docs", "arr2027", "coherence",
                                                     "numbers_ext.tex"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ext", "eval.json"))
    args = ap.parse_args()
    if args.cmd == "spotcheck":
        return committee_spotcheck(args)
    if args.cmd == "macros":
        write_macros(json.load(open(args.out)), args.macros)
        return 0
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
