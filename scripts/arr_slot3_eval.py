"""Evaluate the ARR coherence score on the slot3 LoCoBench pilot (3-rung ladders).

Arms: ``gold`` (the items' labelled relations, band-midpoint weights) and every mined arm
under ``results/arr2027_slot3/arms/<arm>/records``. Readouts, all exact by enumeration:

* ``lambda``        the paper's score, LCS, at verdict polarity (x* = the claims' factuality
                    labels), flat priors. The key keeps its historical name (and the macro tag
                    ``Lam``); every decision is taken on LCS (see REL_TIE);
* ``lambda_assert`` the same score at assertion polarity, x* = each item's own STANCE on each claim
                    (1 asserted, 0 denied): SIGN's top rung denies the false claim it
                    refutes, so its committed world differs from the lower rungs';
* ``mean_marginal`` the mean posterior marginal over the row-stochastic tables, with priors
                    0.9 / 0.1 from the labels (the non-monotone reference of the paper).

Baselines are read from ``results/arr2027_slot3/baselines/*.jsonl`` (one score per item).

Every ladder contributes its three rung pairs. Increase ladders (CONFLICT, CHAIN, SIGN)
need a strict increase and invariance ladders (ORDER, CONTROL) equality, up to one tie
tolerance for every score: relative REL_TIE.

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
# Ties on LCS are relative: two values are tied when |a - b| <= REL_TIE * max(|a|, |b|), i.e.
# when they agree to ten significant digits. An absolute tolerance fails where LCS is pinned
# near 0 (near-certain mined weights), since a real step there moves LCS by far less than any
# fixed resolution. On the test split this rule takes the same decision on every rung pair as
# deciding on Lambda at TIE (checked and reported in ``lcs_rule``).
REL_TIE = 1e-10
# One tie rule for every score -- LCS, the mean marginal and every baseline: relative REL_TIE.
# (The baselines used an absolute 1e-6 before; only the gpt-oss-120b judges, whose ratings are
# probability-weighted over the rating token's logprobs and drift by 1e-13..1e-5, were affected.)
DECIDE = {"lambda": (REL_TIE, True), "lambda_assert": (REL_TIE, True),
          "mean_marginal": (REL_TIE, True),
          "wsum": (REL_TIE, True), "count": (REL_TIE, True), "dsum": (REL_TIE, True),
          "wsum_assert": (REL_TIE, True), "count_assert": (REL_TIE, True),
          "indep": (REL_TIE, True)}
#: The signed baselines over the same graphs, by readout key -> (macro tag, display name).
SIGNED = {"wsum": ("Wsum", "signed weight sum"), "count": ("Count", "signed count"),
          "dsum": ("Dsum", "deduplicated signed sum"),
          "indep": ("Indep", "independent evidence (sum of lambda_r)"),
          "wsum_assert": ("WsumA", "signed weight sum, asserted"),
          "count_assert": ("CountA", "signed count, asserted")}
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
    # Review Phase 3.5: graph-level miner from a vendor outside the generator and committee.
    "graph_llama": ("GraphLlama", "llama-3.3-70b", "graph", "llama-3.3-70b"),
}
#: Baselines: name in the jsonl -> (macro tag, display name).
BASELINES = {
    "control_claim_count": ("CtlClaims", "claim count"),
    "control_length": ("CtlLength", "response length"),
    "control_edit_distance": ("CtlEdit", "edit distance"),
    "control_markers": ("CtlMarkers", "marker count"),
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
    # Non-probabilistic baselines on the same graph (review C1): each relation counts +p (or +1)
    # if the committed world satisfies it and -p (or -1) if it violates it. Both are monotone
    # under every elementary improvement, like LCS, but read no MRF (no normalization Z).
    def signed(x, weighted):
        return float(sum((r[3] if weighted else 1.0) * (1 if A.is_support(r, x) else -1)
                         for r in tup))
    def dedup(x):
        # Review N1: one support and one conflict term per unordered pair of claims (the largest
        # weight of each), so a constraint stated as two halves or repeated counts once.
        best = {}
        for r in tup:
            key = (frozenset((r[0], r[1])), A.is_support(r, x))
            best[key] = max(best.get(key, 0.0), r[3])
        return float(sum(p if sup else -p for (_pair, sup), p in best.items()))
    return {"lambda": lam, "lcs": A.lcs_flat(lam, n), "lambda_assert": lam_a,
            "dsum": dedup(xv),
            # Lambda without its interaction term: each relation's evidence on its own (the
            # decomposition Lambda = sum_r lambda_r - I_R of the coherence paper, Sec. 2.4).
            "indep": A.independent_evidence(pu, tup, xv),
            "lcs_assert": A.lcs_flat(lam_a, n),
            "wsum": signed(xv, True), "count": signed(xv, False),
            "wsum_assert": signed(xa, True), "count_assert": signed(xa, False),
            "mean_marginal": mm, "n": n, "graph": tup, "xv": xv, "xa": xa}


def _thr(a, b, tie, rel):
    """The tie threshold for a pair: absolute ``tie``, or relative to the larger value."""
    return tie * max(abs(a), abs(b)) if rel else tie


def pair_ok(kind, a, b, tie, rel=False):
    t = _thr(a, b, tie, rel)
    if kind in INCREASE:
        return b - a > t
    return abs(b - a) <= t


def assess(ladders, value, tie, tol_value=None, rel=False):
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
        oks = [pair_ok(kind, vals[i], vals[j], tie, rel)
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


def pair_rows(ladders, value, tie, tol_value=None, rel=False):
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
                "ok": bool(pair_ok(kind, vals[i], vals[j], tie, rel)),
                "ok_tol": bool(dt > TOL) if kind in INCREASE else bool(abs(dt) <= TOL),
                "tie": bool(abs(d) <= _thr(vals[i], vals[j], tie, rel)),
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
            dl = hi["lambda"] - lo["lambda"]  # descriptive: the step in nats
            dc, t = hi["lcs"] - lo["lcs"], _thr(hi["lcs"], lo["lcs"], REL_TIE, True)
            rows.append({"family": lad["family_id"], "type": lad["type"], "kind": kind,
                         "rises": bool(dc > t), "falls": bool(dc < -t), "dl": float(dl),
                         "dlcs": float(dc)})
    return rows


def dominance(ladders, scores, key="lcs", tol=1e-12):
    """Increase pairs by how the two graphs compare in the coherence order at x*.

    ``key`` is the score read on each pair (LCS, or a signed baseline over the same graphs);
    ``tol`` is the weight tolerance of the comparison (exact by default; 1e-3 absorbs the
    ~1e-8 drift of the pairwise miners' token-probability weights)."""
    # On LCS: rises (> tie), n, falls (< -tie), ties (|d| <= tie), rises exactly (d > 0)
    d = defaultdict(lambda: [0, 0, 0, 0, 0])
    for lad in ladders:
        if lad["type"] not in INCREASE:
            continue
        ids = lad["item_ids"]
        if any(i not in scores for i in ids):
            continue
        for i, j in itertools.combinations(range(3), 2):
            lo, hi = scores[ids[i]], scores[ids[j]]
            g = A.dominates(hi["graph"], lo["graph"], hi["xv"], tol=tol)
            rev = A.dominates(lo["graph"], hi["graph"], hi["xv"], tol=tol)
            kind = ("forward" if g == "strict" else "equal" if g == "equal" else
                    "backward" if rev == "strict" else "incomparable")
            dc, t = hi[key] - lo[key], _thr(hi[key], lo[key], REL_TIE, True)
            d[kind][0] += dc > t
            d[kind][1] += 1
            d[kind][2] += dc < -t
            d[kind][3] += abs(dc) <= t
            d[kind][4] += dc > 0
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


#: Review N10: explicit discourse markers in the ladders' fillers. A lexical baseline that counts
#: inference markers minus incompatibility markers in the response text.
MARKERS_INF = ("because", "therefore", "thus", "hence", "as a result", "in consequence",
               "consequently", "which shows", "shows that", "follows that", "so that", "since")
MARKERS_INC = ("rules out", "cannot both", "cannot be squared", "contradicts", "inconsistent with",
               "incompatible", "cannot have", "is false", "refutes", "but")


def marker_score(text):
    t = " " + text.lower() + " "
    cnt = lambda ms: sum(len(re.findall(r"\b" + re.escape(m) + r"\b", t)) for m in ms)  # noqa: E731
    return float(cnt(MARKERS_INF) - cnt(MARKERS_INC))


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
            ok += abs(b["score"] - a["score"]) <= sd + _thr(a["score"], b["score"], REL_TIE, True)
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
        # LCS readouts are decided on LCS itself (relative tie); the mean marginal on its own
        # value (absolute tie).
        res[arm] = {
            ro: assess(ladders, lambda i, ro=ro, sc=sc: sc[i][unit[ro]] if i in sc else None,
                       DECIDE[ro][0], rel=DECIDE[ro][1],
                       tol_value=lambda i, ro=ro, sc=sc: sc[i][unit[ro]] if i in sc else None)
            for ro in ("lambda", "lambda_assert", "mean_marginal")
        }
        for ro in SIGNED:
            res[arm][ro] = assess(ladders, lambda i, ro=ro, sc=sc: sc[i][ro] if i in sc else None,
                                  DECIDE[ro][0], rel=DECIDE[ro][1])
        res[arm]["dominance"] = dominance(ladders, sc)
        res[arm]["dominance_wsum"] = dominance(ladders, sc, key="wsum")
        res[arm]["dominance_tol3"] = dominance(ladders, sc, tol=1e-3)
        res[arm]["dominance_tol3_wsum"] = dominance(ladders, sc, key="wsum", tol=1e-3)
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
    base["control_markers"] = {iid: {"score": marker_score(it["response"])} for iid, it in items.items()}
    bres = {}
    for name, rows in base.items():
        bres[name] = assess(ladders, lambda i, rows=rows: rows[i]["score"] if i in rows else None,
                            REL_TIE, rel=True)
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
    unit = {"lambda": "lcs", "lambda_assert": "lcs_assert", "mean_marginal": "mean_marginal",
            **{ro: ro for ro in SIGNED}}
    pairs = {arm: {ro: pair_rows(ladders, lambda i, ro=ro, sc=sc: sc[i][unit[ro]] if i in sc
                                 else None, DECIDE[ro][0],
                                 lambda i, ro=ro, sc=sc: sc[i][unit[ro]] if i in sc else None,
                                 rel=DECIDE[ro][1])
                   for ro in unit} for arm, sc in scores.items()}
    pairs_base = {name: pair_rows(ladders, lambda i, rows=rows: rows[i]["score"] if i in rows
                                  else None, REL_TIE, rel=True)
                  for name, rows in base.items()}
    dom_rows = {arm: dominance_rows(ladders, sc) for arm, sc in scores.items() if arm != "gold"}
    # The SIGN ladder rung by rung on gold graphs, under each readout (Figure: polarity).
    sign = {ro: [[scores["gold"][l["item_ids"][k]][ro] for k in range(3)]
                 for l in ladders if l["type"] == "SIGN"]
            for ro in ("lcs", "lcs_assert", "mean_marginal")}
    # The LCS tie rule, checked against two alternatives on every rung pair of every arm and
    # both polarities: (a) deciding on Lambda at TIE, the log-ratio scale the theorems use,
    # where no compression occurs -- the rule should agree everywhere; (b) nearby relative
    # tolerances; and (c) an absolute tolerance TIE on LCS, which fails near LCS = 0.
    def cls(d, t):
        return 1 if d > t else (-1 if d < -t else 0)
    alts = (1e-11, 1e-12, 1e-13)
    rule = {"pairs": 0, "dis_lambda": 0, "dis_alt": {str(a): 0 for a in alts},
            "dis_abs": {}}
    for arm, sc in scores.items():
        n_abs = 0
        for ro, lo in (("lambda", "lcs"), ("lambda_assert", "lcs_assert")):
            for lad in ladders:
                ids = lad["item_ids"]
                if any(i not in sc for i in ids):
                    continue
                for i, j in itertools.combinations(range(3), 2):
                    a, b = sc[ids[i]][lo], sc[ids[j]][lo]
                    ours = cls(b - a, _thr(a, b, REL_TIE, True))
                    rule["pairs"] += 1
                    rule["dis_lambda"] += ours != cls(sc[ids[j]][ro] - sc[ids[i]][ro], TIE)
                    for al in alts:
                        rule["dis_alt"][str(al)] += ours != cls(b - a, _thr(a, b, al, True))
                    if ro == "lambda":
                        n_abs += ours != cls(b - a, TIE)
        rule["dis_abs"][arm] = n_abs
    rule["pairs_per_arm"] = 3 * len(ladders)
    out = {"arms": res, "baselines": bres, "gold_shape": gold_shape, 
           "pairs": pairs, "pairs_baseline": pairs_base, "dominance_rows": dom_rows,
           "lcs_rule": rule,
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
        for ro, (st, _disp) in SIGNED.items():
            c = r[ro]
            for key in ("inc", "inv", "lad_inc", "lad_inv") + tuple(f"pair_{t}" for t in TYPES):
                kk = {"inc": "Inc", "inv": "Inv", "lad_inc": "LadInc",
                      "lad_inv": "LadInv"}.get(key, key.replace("pair_", "").title())
                m(f"sThree{tg}{st}{kk}", _frac(c.get(key)))
        for dk, dt in (("dominance_wsum", "DomW"), ("dominance_tol3", "DomT"),
                       ("dominance_tol3_wsum", "DomTW")):
            for kind in ("forward", "incomparable", "backward", "equal"):
                d = tuple(r[dk].get(kind, (0, 0, 0, 0, 0)))
                m(f"sThree{tg}{dt}{kind.title()}", d[1])
                m(f"sThree{tg}{dt}{kind.title()}Ok", d[0])
                m(f"sThree{tg}{dt}{kind.title()}Falls", d[2])
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
    rl = out["lcs_rule"]
    m("sThreeRelTie", f"10^{{{round(math.log10(REL_TIE))}}}")
    m("sThreeLcsRulePairs", f"{rl['pairs']:,}".replace(",", "{,}"))
    m("sThreeLcsRuleLamDis", rl["dis_lambda"])
    m("sThreeLcsRuleAltMaxDis", max(rl["dis_alt"].values()))
    for arm, k in rl["dis_abs"].items():
        m(f"sThree{tags[arm]}LcsAbsDis", f"{k}/{rl['pairs_per_arm']}")
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
    # Gold SIGN ladders at the asserted world, by rung pair (AB = rho0->rho1, BC = rho1->rho2,
    # AC = rho0->rho2): how the stance-based readout rises, ties or falls on each step.
    sp = [r for r in out["pairs"]["gold"]["lambda_assert"] if r["type"] == "SIGN"]
    for (i, j), tag in (((0, 1), "AB"), ((1, 2), "BC"), ((0, 2), "AC")):
        pr = [r for r in sp if tuple(r["pair"]) == (i, j)]
        m(f"sThreeGoldLamASign{tag}Rise", f"{sum(r['ok'] for r in pr)}/{len(pr)}")
        m(f"sThreeGoldLamASign{tag}Tie", f"{sum(r['tie'] for r in pr)}/{len(pr)}")
        m(f"sThreeGoldLamASign{tag}Fall",
          f"{sum(not r['ok'] and not r['tie'] for r in pr)}/{len(pr)}")
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
        for ro in SIGNED:
            c = r[ro]
            print(f"  {ro:14s} inc {_frac(c.get('inc')):7s} inv {_frac(c.get('inv')):7s} "
                  f"lad_inc {_frac(c.get('lad_inc')):6s} | "
                  + " ".join(f"{t}:{_frac(c.get('pair_'+t))}" for t in TYPES))
        print(f"  dominance_wsum {r['dominance_wsum']}")
        print(f"  dominance_tol3 {r['dominance_tol3']}  (wsum {r['dominance_tol3_wsum']})")
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
