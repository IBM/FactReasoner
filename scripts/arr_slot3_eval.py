"""Evaluate the ARR coherence score on the slot3 LoCoBench pilot (3-rung ladders).

Arms: ``gold`` (the items' labelled relations, band-midpoint weights) and every mined arm
under ``results/arr2027_slot3/arms/<arm>/records``. Readouts, all exact by enumeration:

* ``lambda``        Lambda at verdict polarity (x* = the claims' factuality labels), flat
                    priors -- the paper's score (LCS orders identically within a ladder);
* ``lambda_assert`` Lambda at assertion polarity, x* = each item's own STANCE on each claim
                    (1 asserted, 0 denied): SIGN's top rung denies the false claim it
                    refutes, so its committed world differs from the lower rungs';
* ``mean_marginal`` the mean posterior marginal over the row-stochastic tables, with priors
                    0.9 / 0.1 from the labels (the non-monotone reference of the paper).

Baselines are read from ``results/arr2027_slot3/baselines/*.jsonl`` (one score per item).

Every ladder contributes its three rung pairs. Increase ladders (CONFLICT, CHAIN, SIGN)
need a strict increase (> 1e-9 for exact readouts, > 1e-6 for baselines); invariance
ladders (ORDER, CONTROL) need equality within the same tolerance.

Run::

    python scripts/arr_slot3_eval.py --macros docs/arr2027/coherence/numbers_slot3.tex \\
        --json results/arr2027_slot3/eval.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import arr_coherence_eval as A  # noqa: E402

# Defaults: the pilot (development split). `--data` / `--root` select another split, and
# `--prefix` its macro prefix, so the test and pilot results can both feed the paper.
DATA = os.path.join(REPO, "data", "locobench-slot3-pilot")
ROOT = os.path.join(REPO, "results", "arr2027_slot3")
PREFIX = "sThree"
INCREASE = ("CONFLICT", "CHAIN", "SIGN")
INVARIANCE = ("ORDER", "CONTROL")
TYPES = INCREASE + INVARIANCE
TIE = 1e-9
TIE_BASELINE = 1e-6
# Invariance within a tolerance, on each score's own [0, 1] scale (LCS for the MRF arms,
# the mean marginal, each baseline's score). Exact equality is unattainable for a mined
# graph: a pairwise miner's weights are token log-probabilities, which move by ~1e-8 when
# any other sentence of the response changes, and a sampled miner re-draws its graph.
# Fixed at one point on the unit scale before the full pilot results were read.
TOL = 0.01

#: Mined arms: directory -> (macro tag, display name, miner, model).
MINED = {
    "graph_gpt6": ("GraphGpt", "gpt-6-sol", "graph", "gpt-6-sol"),
    "graph_gptoss": ("GraphOss", "gpt-oss-120b", "graph", "gpt-oss-120b"),
    "pair_gptoss": ("PairOss", "gpt-oss-120b", "pairwise", "gpt-oss-120b"),
    "pair_llama": ("PairLlama", "llama-3.3-70b", "pairwise", "llama-3.3-70b"),
}
#: Baselines: name in the jsonl -> (macro tag, display name).
BASELINES = {
    "control_claim_count": ("CtlClaims", "claim count"),
    "control_length": ("CtlLength", "response length"),
    "control_edit_distance": ("CtlEdit", "edit distance"),
    "discourse_rc": ("DiscRC", "DiscoScore RC"),
    "discourse_lc": ("DiscLC", "DiscoScore LC"),
    "discourse_entity_graph": ("DiscEG", "entity graph"),
    "nli_contradiction": ("NliContra", "NLI contradiction"),
    "roscoe_sc": ("RoscoeMax", "ROSCOE SC (max)"),
    "roscoe_sc_mean": ("RoscoeMean", "ROSCOE SC (mean)"),
    "judge_geval@gpt-oss-120b": ("GevalOss", "G-Eval, gpt-oss-120b"),
    "judge_direct@gpt-oss-120b": ("DirectOss", "direct rating, gpt-oss-120b"),
    "judge_geval@gpt-6-sol": ("GevalGpt", "G-Eval, gpt-6-sol"),
    "judge_direct@gpt-6-sol": ("DirectGpt", "direct rating, gpt-6-sol"),
}


def _idx(aid: str) -> int:
    return int(aid.lstrip("a"))


def load():
    items = {}
    for line in open(os.path.join(DATA, "items.jsonl")):
        if line.strip():
            it = json.loads(line)
            items[it["id"]] = it
    fams = json.load(open(os.path.join(DATA, "families.json")))["families"]
    ladders = [dict(l, family_id=f["family_id"], primary=f["primary"])
               for f in fams for l in f["ladders"]]
    return items, fams, ladders


def gold_relations(item):
    out = []
    for r in item["relations"]:
        if r["level1_coupling"] not in A.VIOLATED:
            continue
        lo, hi = r["strength_range"]
        out.append({"source": r["source_id"], "target": r["target_id"],
                    "type": r["level1_coupling"], "probability": (lo + hi) / 2})
    return out


def mined_relations(arm):
    out = {}
    for path in glob.glob(os.path.join(ROOT, "arms", arm, "records", "*.json")):
        rec = json.load(open(path))
        if rec.get("error"):
            continue
        out[rec["item_id"]] = [r for r in rec.get("relations") or []
                               if r["type"] in A.VIOLATED]
    return out


def score_item(item, rels):
    atoms = sorted(item["atoms"], key=lambda a: a["pos"])
    n = len(atoms)
    ids = [a["id"] for a in atoms]
    xv = np.array([1 if a["factual"] else 0 for a in atoms])
    xa = np.array([0 if a.get("stance") == "denied" else 1 for a in atoms])
    tup = [(_idx(r["source"]), _idx(r["target"]), r["type"], float(r["probability"]))
           for r in rels]
    pu = np.full(n, 0.5)
    lam = A.confirmation(pu, tup, xv)
    lam_a = A.confirmation(pu, tup, xa)
    pri = {aid: (0.9 if a["factual"] else 0.1) for aid, a in zip(ids, atoms)}
    mm = A.old_readouts(ids, pri, rels)["mean_marginal"]
    return {"lambda": lam, "lcs": A.lcs_flat(lam, n), "lambda_assert": lam_a,
            "lcs_assert": A.lcs_flat(lam_a, n),
            "mean_marginal": mm, "n": n, "graph": tup, "xv": xv, "xa": xa}


def pair_ok(kind, a, b, tie):
    if kind in INCREASE:
        return b - a > tie
    return abs(b - a) <= tie


def assess(ladders, value, tie, tol_value=None):
    """value(item_id) -> float or None. Returns counts per ladder type and overall.

    ``tol_value`` (default ``value``) is the [0, 1]-scale score the tolerant invariance
    check (``inv_tol``, |difference| <= TOL) is read on.
    """
    tol_value = tol_value or value
    c = defaultdict(lambda: [0, 0])
    for lad in ladders:
        kind = lad["type"]
        vals = [value(i) for i in lad["item_ids"]]
        if any(v is None for v in vals):
            continue
        oks = [pair_ok(kind, vals[i], vals[j], tie)
               for i, j in itertools.combinations(range(3), 2)]
        group = "inc" if kind in INCREASE else "inv"
        for ok in oks:
            c[f"{group}"][0] += ok
            c[f"{group}"][1] += 1
            c[f"pair_{kind}"][0] += ok
            c[f"pair_{kind}"][1] += 1
        c[f"lad_{group}"][0] += all(oks)
        c[f"lad_{group}"][1] += 1
        tv = [tol_value(i) for i in lad["item_ids"]]
        if kind in INCREASE:
            # A RESOLVED increase: the [0, 1]-scale score moves up by more than TOL, the
            # same resolution the tolerant invariance test grants.
            oks_t = [tv[j] - tv[i] > TOL for i, j in itertools.combinations(range(3), 2)]
            for ok_t in oks_t:
                c["inc_tol"][0] += ok_t
                c["inc_tol"][1] += 1
                c[f"tol_{kind}"][0] += ok_t
                c[f"tol_{kind}"][1] += 1
        if kind in INVARIANCE:
            for i, j in itertools.combinations(range(3), 2):
                ok_t = abs(tv[j] - tv[i]) <= TOL
                c["inv_tol"][0] += ok_t
                c["inv_tol"][1] += 1
                c[f"tol_{kind}"][0] += ok_t
                c[f"tol_{kind}"][1] += 1
        c[f"lad_{kind}"][0] += all(oks)
        c[f"lad_{kind}"][1] += 1
    return {k: tuple(v) for k, v in c.items()}


def pair_rows(ladders, value, tie, tol_value=None):
    """Every rung pair's outcome, for family-level bootstrap intervals and the figures."""
    tol_value = tol_value or value
    rows = []
    for lad in ladders:
        kind = lad["type"]
        vals = [value(i) for i in lad["item_ids"]]
        tv = [tol_value(i) for i in lad["item_ids"]]
        if any(v is None for v in vals):
            continue
        for i, j in itertools.combinations(range(3), 2):
            d, dt = vals[j] - vals[i], tv[j] - tv[i]
            rows.append({
                "family": lad["family_id"], "ladder": lad["ladder_id"], "type": kind,
                "pair": [i, j],
                "ok": bool(pair_ok(kind, vals[i], vals[j], tie)),
                "ok_tol": bool(dt > TOL) if kind in INCREASE else bool(abs(dt) <= TOL),
                "tie": bool(abs(d) <= tie),
            })
    return rows


def dominance_rows(ladders, scores):
    rows = []
    for lad in ladders:
        if lad["type"] not in INCREASE or any(i not in scores for i in lad["item_ids"]):
            continue
        ids = lad["item_ids"]
        for i, j in itertools.combinations(range(3), 2):
            lo, hi = scores[ids[i]], scores[ids[j]]
            g = A.dominates(hi["graph"], lo["graph"], hi["xv"])
            rev = A.dominates(lo["graph"], hi["graph"], hi["xv"])
            kind = ("forward" if g == "strict" else "equal" if g == "equal" else
                    "backward" if rev == "strict" else "incomparable")
            dl = hi["lambda"] - lo["lambda"]
            rows.append({"family": lad["family_id"], "type": lad["type"], "kind": kind,
                         "rises": bool(dl > TIE), "falls": bool(dl < -TIE), "dl": float(dl)})
    return rows


def dominance(ladders, scores):
    """Increase pairs by how the two graphs compare in the coherence order at x*."""
    # rises (> TIE), n, falls (< -TIE), ties (|dl| <= TIE), rises exactly (dl > 0)
    d = defaultdict(lambda: [0, 0, 0, 0, 0])
    for lad in ladders:
        if lad["type"] not in INCREASE:
            continue
        ids = lad["item_ids"]
        if any(i not in scores for i in ids):
            continue
        for i, j in itertools.combinations(range(3), 2):
            lo, hi = scores[ids[i]], scores[ids[j]]
            g = A.dominates(hi["graph"], lo["graph"], hi["xv"])
            rev = A.dominates(lo["graph"], hi["graph"], hi["xv"])
            kind = ("forward" if g == "strict" else "equal" if g == "equal" else
                    "backward" if rev == "strict" else "incomparable")
            dl = hi["lambda"] - lo["lambda"]
            d[kind][0] += dl > TIE
            d[kind][1] += 1
            d[kind][2] += dl < -TIE
            d[kind][3] += abs(dl) <= TIE
            d[kind][4] += dl > 0
    return {k: tuple(v) for k, v in d.items()}


def extraction(items, mined):
    """Pair- and coupling-level P/R/F1 of a mined arm against gold (factor relations)."""
    tp_p = fp_p = fn_p = tp_c = fp_c = fn_c = 0
    for iid, rels in mined.items():
        gold = gold_relations(items[iid])

        def keyset(rs, typed):
            out = set()
            for r in rs:
                s, t = r["source"], r["target"]
                if r["type"] in A.SYMMETRIC or not typed:
                    s, t = sorted((s, t))
                out.add((s, t, r["type"]) if typed else (s, t))
            return out

        G, M = keyset(gold, False), keyset(rels, False)
        tp_p += len(G & M); fp_p += len(M - G); fn_p += len(G - M)
        G, M = keyset(gold, True), keyset(rels, True)
        tp_c += len(G & M); fp_c += len(M - G); fn_c += len(G - M)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    return {"pair": prf(tp_p, fp_p, fn_p), "coupling": prf(tp_c, fp_c, fn_c),
            "n_items": len(mined)}


def spread(ladders, scores):
    inv, inc = [], []
    for lad in ladders:
        ids = lad["item_ids"]
        if any(i not in scores for i in ids):
            continue
        for i, j in itertools.combinations(range(3), 2):
            dl = scores[ids[j]]["lcs"] - scores[ids[i]]["lcs"]
            (inc if lad["type"] in INCREASE else inv).append(dl)
    return {"inv_abs_median": float(np.median(np.abs(inv))) if inv else None,
            "inc_median": float(np.median(inc)) if inc else None}


def load_baselines():
    out = defaultdict(dict)
    for path in glob.glob(os.path.join(ROOT, "baselines", "*.jsonl")):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r.get("score") is not None:
                    out[r["name"]][r["item_id"]] = r
    return out


def judge_tol_invariance(ladders, rows):
    """Invariance pairs whose difference is within the larger of the two seed s.d.s."""
    ok = n = 0
    for lad in ladders:
        if lad["type"] not in INVARIANCE:
            continue
        ids = lad["item_ids"]
        if any(i not in rows for i in ids):
            continue
        for i, j in itertools.combinations(range(3), 2):
            a, b = rows[ids[i]], rows[ids[j]]
            sd = max(float((a.get("diagnostics") or {}).get("sd") or 0),
                     float((b.get("diagnostics") or {}).get("sd") or 0))
            ok += abs(b["score"] - a["score"]) <= sd + TIE_BASELINE
            n += 1
    return ok, n


def main() -> int:
    global DATA, ROOT, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--macros", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--data", default=DATA, help="Dataset directory (items.jsonl, families.json).")
    ap.add_argument("--root", default=ROOT, help="Results root (arms/, baselines/).")
    ap.add_argument("--prefix", default=PREFIX, help="LaTeX macro prefix.")
    args = ap.parse_args()
    DATA, ROOT, PREFIX = args.data, args.root, args.prefix

    items, fams, ladders = load()
    arms = {"gold": {iid: gold_relations(it) for iid, it in items.items()}}
    for arm in MINED:
        m = mined_relations(arm)
        if m:
            arms[arm] = m
    scores = {arm: {iid: score_item(items[iid], rels) for iid, rels in g.items()}
              for arm, g in arms.items()}
    res = {}
    for arm, sc in scores.items():
        unit = {"lambda": "lcs", "lambda_assert": "lcs_assert", "mean_marginal":
                "mean_marginal"}
        res[arm] = {
            ro: assess(ladders, lambda i, ro=ro, sc=sc: sc[i][ro] if i in sc else None, TIE,
                       lambda i, ro=ro, sc=sc: sc[i][unit[ro]] if i in sc else None)
            for ro in ("lambda", "lambda_assert", "mean_marginal")
        }
        res[arm]["dominance"] = dominance(ladders, sc)
        # Saturation: near-certain weights pin LCS near 0, compressing Lambda's steps.
        w = [r[3] for v in sc.values() for r in v["graph"]]
        steps = [sc[l["item_ids"][j]]["lambda"] - sc[l["item_ids"][i]]["lambda"]
                 for l in ladders if l["type"] in INCREASE
                 and all(x in sc for x in l["item_ids"])
                 for i, j in itertools.combinations(range(3), 2)]
        res[arm]["saturation"] = {
            "weights_ge_099": float(np.mean(np.array(w) >= 0.99)) if w else 0.0,
            "lcs_lt_001": float(np.mean([v["lcs"] < 0.01 for v in sc.values()])),
            "median_lambda_step": float(np.median(steps)) if steps else None,
        }
        res[arm]["spread"] = spread(ladders, sc)
        res[arm]["n_items"] = len(sc)
        if arm != "gold":
            res[arm]["extraction"] = extraction(items, arms[arm])
            res[arm]["mean_relations"] = float(np.mean([len(v) for v in arms[arm].values()]))
    res["gold"]["mean_relations"] = float(np.mean([len(v) for v in arms["gold"].values()]))
    base = load_baselines()
    bres = {}
    for name, rows in base.items():
        bres[name] = assess(ladders, lambda i, rows=rows: rows[i]["score"] if i in rows else None,
                            TIE_BASELINE)
        bres[name]["n_items"] = len(rows)
        if name.startswith("judge_"):
            bres[name]["inv_seed_tol"] = judge_tol_invariance(ladders, rows)
    # Gold ladders: mean Lambda / LCS per rung and per type.
    gold_shape = {}
    for t in TYPES:
        ls = [l for l in ladders if l["type"] == t]
        gold_shape[t] = {
            "n_ladders": len(ls),
            "lcs": [float(np.mean([scores["gold"][l["item_ids"][k]]["lcs"] for l in ls]))
                    for k in range(3)],
            "lambda": [float(np.mean([scores["gold"][l["item_ids"][k]]["lambda"] for l in ls]))
                       for k in range(3)],
        }
    unit = {"lambda": "lcs", "lambda_assert": "lcs_assert", "mean_marginal": "mean_marginal"}
    pairs = {arm: {ro: pair_rows(ladders, lambda i, ro=ro, sc=sc: sc[i][ro] if i in sc else None,
                                 TIE, lambda i, ro=ro, sc=sc: sc[i][unit[ro]] if i in sc else None)
                   for ro in unit} for arm, sc in scores.items()}
    pairs_base = {name: pair_rows(ladders, lambda i, rows=rows: rows[i]["score"] if i in rows
                                  else None, TIE_BASELINE)
                  for name, rows in base.items()}
    dom_rows = {arm: dominance_rows(ladders, sc) for arm, sc in scores.items() if arm != "gold"}
    # The SIGN ladder rung by rung on gold graphs, under each readout (Figure: polarity).
    sign = {ro: [[scores["gold"][l["item_ids"][k]][ro] for k in range(3)]
                 for l in ladders if l["type"] == "SIGN"]
            for ro in ("lcs", "lcs_assert", "mean_marginal")}
    # Lambda vs LCS: the same score on two scales (LCS is increasing in Lambda for a fixed claim
    # set and prior). Count rung pairs whose direction differs, and pairs whose "moved by more
    # than TIE" decision would change if it were taken on LCS instead of Lambda.
    lam_lcs = {}
    for arm, sc in scores.items():
        for ro, lo in (("lambda", "lcs"), ("lambda_assert", "lcs_assert")):
            n = dis = rev = 0
            for lad in ladders:
                ids = lad["item_ids"]
                if any(i not in sc for i in ids):
                    continue
                for i, j in itertools.combinations(range(3), 2):
                    dl = sc[ids[j]][ro] - sc[ids[i]][ro]
                    dc = sc[ids[j]][lo] - sc[ids[i]][lo]
                    n += 1
                    dis += (abs(dl) > TIE) != (abs(dc) > TIE)
                    rev += (dl > TIE and dc < 0) or (dl < -TIE and dc > 0)
            lam_lcs[f"{arm}:{ro}"] = (n, dis, rev)
    out = {"arms": res, "baselines": bres, "gold_shape": gold_shape, "lambda_vs_lcs": lam_lcs,
           "pairs": pairs, "pairs_baseline": pairs_base, "dominance_rows": dom_rows,
           "sign_rungs": sign,
           "n_families": len(fams), "n_ladders": len(ladders), "n_items": len(items),
           "ladders_by_type": {t: sum(l["type"] == t for l in ladders) for t in TYPES}}
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=str)
    if args.macros:
        write_macros(args.macros, out)
    report(out)
    return 0


def _frac(t):
    return f"{t[0]}/{t[1]}" if t and t[1] else "---"


def write_macros(path, out):
    L = ["% Generated by scripts/arr_slot3_eval.py -- do not edit by hand."]

    def m(name, val):
        name = PREFIX + name[len("sThree"):] if name.startswith("sThree") else name
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
        # Every count "a/b" also gets a percentage twin, \<name>Pct (no % sign).
        mt = re.fullmatch(r"(\d+)/(\d+)", str(val))
        if mt and int(mt.group(2)):
            L.append(f"\\newcommand{{\\{name}Pct}}{{{100 * int(mt.group(1)) / int(mt.group(2)):.0f}}}")
        elif val == "---":
            L.append(f"\\newcommand{{\\{name}Pct}}{{---}}")

    m("sThreeFamilies", out["n_families"])
    m("sThreeTol", f"{TOL:g}")
    m("sThreeLadders", out["n_ladders"])
    m("sThreeItems", f"{out['n_items']:,}".replace(",", "{,}"))
    g = out["arms"]["gold"]["lambda"]
    m("sThreeNIncPairs", g["inc"][1])
    m("sThreeNInvPairs", g["inv"][1])
    for t, k in out["ladders_by_type"].items():
        m(f"sThreeN{t.title()}", k)
    tags = {"gold": "Gold", **{a: v[0] for a, v in MINED.items()}}
    ro_tag = {"lambda": "Lam", "lambda_assert": "LamA", "mean_marginal": "MM"}
    for arm, r in out["arms"].items():
        tg = tags[arm]
        for ro, rt in ro_tag.items():
            c = r[ro]
            for key in ("inc", "inc_tol", "inv", "inv_tol", "lad_inc", "lad_inv") + tuple(
                f"pair_{t}" for t in TYPES
            ) + tuple(f"tol_{t}" for t in TYPES):
                kk = {"inc": "Inc", "inc_tol": "IncTol", "inv": "Inv", "inv_tol": "InvTol",
                      "lad_inc": "LadInc",
                      "lad_inv": "LadInv"}.get(key, key.replace("pair_", "").replace(
                          "tol_", "Tol").title())
                m(f"sThree{tg}{rt}{kk}", _frac(c.get(key)))
        for kind in ("forward", "incomparable", "backward", "equal"):
            d = tuple(r["dominance"].get(kind, (0, 0, 0, 0)))
            m(f"sThree{tg}Dom{kind.title()}", d[1])
            m(f"sThree{tg}Dom{kind.title()}Ok", d[0])
            m(f"sThree{tg}Dom{kind.title()}Falls", d[2] if len(d) > 2 else 0)
            m(f"sThree{tg}Dom{kind.title()}Ties", d[3] if len(d) > 3 else 0)
            m(f"sThree{tg}Dom{kind.title()}Pos", d[4] if len(d) > 4 else 0)
        sp = r["spread"]
        if sp["inv_abs_median"] is not None:
            m(f"sThree{tg}SprInv", f"{sp['inv_abs_median']:.3f}")
            m(f"sThree{tg}SprInc", f"{sp['inc_median']:.3f}")
        m(f"sThree{tg}Rels", f"{r['mean_relations']:.1f}")
        m(f"sThree{tg}Items", r["n_items"])
        m(f"sThree{tg}NLadders", r["lambda"].get("lad_inc", (0, 0))[1]
          + r["lambda"].get("lad_inv", (0, 0))[1])
        sat = r["saturation"]
        m(f"sThree{tg}WHard", f"{100 * sat['weights_ge_099']:.0f}\\%")
        m(f"sThree{tg}Sat", f"{100 * sat['lcs_lt_001']:.0f}\\%")
        if sat["median_lambda_step"] is not None:
            m(f"sThree{tg}Step", f"{sat['median_lambda_step']:.2f}")
        if "extraction" in r:
            e = r["extraction"]
            m(f"sThree{tg}PairP", f"{e['pair'][0]:.2f}")
            m(f"sThree{tg}PairR", f"{e['pair'][1]:.2f}")
            m(f"sThree{tg}PairF", f"{e['pair'][2]:.2f}")
            m(f"sThree{tg}CoupF", f"{e['coupling'][2]:.2f}")
    ll = out["lambda_vs_lcs"]
    m("sThreeLamLcsSignDis", sum(v[2] for v in ll.values()))
    m("sThreeLamLcsPairs", f"{sum(v[0] for v in ll.values()):,}".replace(",", "{,}"))
    for arm in out["arms"]:
        n, dis, _sg = ll[f"{arm}:lambda"]
        m(f"sThree{tags[arm]}LamLcsDis", f"{dis}/{n}")
    for name, (tg, _disp) in BASELINES.items():
        c = out["baselines"].get(name)
        if not c:
            continue
        m(f"sThreeB{tg}Inc", _frac(c.get("inc")))
        m(f"sThreeB{tg}Inv", _frac(c.get("inv")))
        m(f"sThreeB{tg}InvTolU", _frac(c.get("inv_tol")))
        m(f"sThreeB{tg}IncTol", _frac(c.get("inc_tol")))
        m(f"sThreeB{tg}LadInc", _frac(c.get("lad_inc")))
        for t in TYPES:
            m(f"sThreeB{tg}{t.title()}", _frac(c.get(f"pair_{t}")))
        if "inv_seed_tol" in c:
            m(f"sThreeB{tg}InvTol", _frac(c["inv_seed_tol"]))
        prs = [r for r in out["pairs_baseline"].get(name, []) if r["type"] in INCREASE]
        if prs:
            m(f"sThreeB{tg}IncTies", f"{sum(r['tie'] for r in prs)}/{len(prs)}")
            m(f"sThreeB{tg}IncFalls",
              f"{sum(not r['ok'] and not r['tie'] for r in prs)}/{len(prs)}")
            for t in INCREASE:
                pt = [r for r in prs if r["type"] == t]
                m(f"sThreeB{tg}{t.title()}Falls",
                  f"{sum(not r['ok'] and not r['tie'] for r in pt)}/{len(pt)}")
    for t, g in out["gold_shape"].items():
        for k in range(3):
            m(f"sThreeGold{t.title()}Lcs{'ABC'[k]}", f"{g['lcs'][k]:.3f}")
    open(path, "w").write("\n".join(L) + "\n")


def report(out):
    print(f"families {out['n_families']}  ladders {out['n_ladders']} {out['ladders_by_type']}"
          f"  items {out['n_items']}")
    for arm, r in out["arms"].items():
        print(f"\n[{arm}] items={r['n_items']} rels/item={r['mean_relations']:.1f}")
        for ro in ("lambda", "lambda_assert", "mean_marginal"):
            c = r[ro]
            print(f"  {ro:14s} inc {_frac(c.get('inc')):7s} incTol {_frac(c.get('inc_tol')):7s} "
                  f"inv {_frac(c.get('inv')):7s} "
                  f"invTol {_frac(c.get('inv_tol')):7s} "
                  f"ladders inc {_frac(c.get('lad_inc')):6s} inv {_frac(c.get('lad_inv')):6s} | "
                  + " ".join(f"{t}:{_frac(c.get('pair_'+t))}" for t in TYPES))
        print(f"  dominance {r['dominance']}  spread {r['spread']}")
        if "extraction" in r:
            e = r["extraction"]
            print(f"  extraction pair P/R/F {tuple(round(x, 2) for x in e['pair'])} "
                  f"coupling F1 {e['coupling'][2]:.2f}")
    print("\n[baselines]")
    for name, c in sorted(out["baselines"].items()):
        print(f"  {name:28s} n={c['n_items']:3d} inc {_frac(c.get('inc')):7s} "
              f"incTol {_frac(c.get('inc_tol')):7s} "
              f"inv {_frac(c.get('inv')):7s} invTol {_frac(c.get('inv_tol')):7s} "
              f"lad_inc {_frac(c.get('lad_inc')):6s} "
              + " ".join(f"{t}:{_frac(c.get('pair_'+t))}" for t in TYPES)
              + (f" invTol {_frac(c['inv_seed_tol'])}" if "inv_seed_tol" in c else ""))
    print("\n[gold shape]", json.dumps(out["gold_shape"]))


if __name__ == "__main__":
    raise SystemExit(main())
