"""Analyze the LoCoBench human validation study (pre-registered in
``docs/arr2027/locobench/study/preregistration.md``).

Inputs: the gold key (``data/locobench-study/key/key.json``, checked against its committed
SHA-256) and each annotator's Label Studio JSON exports, one file per task:
``<exports>/annotator_<k>/task1.json`` and ``task2.json``.

Outputs: a Markdown report, LaTeX macros (prefix ``\\hs``) and a JSON file with every number.

* **H1** -- admitted fillers read as intended: the human verdict on a filler mirrors the
  committee's SV5 rule exactly (a rater's reading is OK when its link matches the intended one by
  ``slots.label_matches`` and every owned claim has the intended stance; a filler passes when a
  majority of raters are OK and no majority reports a link to another statement).
* **H2** -- declared increases: per distinct rung pair, the sum over raters of +1 (prefers the
  higher rung), 0 (about equal), -1 (prefers the lower); agree / tie / disagree by its sign; a
  one-sided sign test of agree vs disagree.
* **H3** -- CONTROL invariances: share of pairs whose majority says "about equal", and a
  two-sided sign test for a side preference among non-equal judgements.
* **H4** -- committee vs people on all 120 fillers: agreement, Cohen's kappa, sensitivity and
  specificity reweighted to the population of SV5 readings.
* Reliability: Krippendorff's alpha (nominal link, ordinal preference). Timing from
  ``lead_time``. Secondary: each reference metric's direction on the increase pairs against the
  human majority (from the companion paper's ``eval.json``).

``--simulate`` writes synthetic exports from the key (for testing this script end to end).

    .venv/bin/python scripts/locobench_study_analyze.py --exports data/locobench-study/exports
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import locobench_study_build as B  # noqa: E402  (the option lists: one source of truth)
from fact_reasoner.locobench import slots as S  # noqa: E402

LINK = dict(B.LINK_OPTIONS)
STANCE = dict(B.STANCE_OPTIONS)
PREF = {B.PREFERENCE_OPTIONS[0]: "A", B.PREFERENCE_OPTIONS[1]: "=", B.PREFERENCE_OPTIONS[2]: "B"}
H1_NULL = 0.80      # H1: the rate of admitted fillers read as intended exceeds this (one-sided)
EVAL = "results/arr2027_slot3_test/eval.json"


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def binom_tail(k, n, p=0.5):
    """P(X >= k) for X ~ Bin(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test(pos, neg, two_sided=False):
    n = pos + neg
    if n == 0:
        return float("nan")
    if two_sided:
        return min(1.0, 2 * binom_tail(max(pos, neg), n))
    return binom_tail(pos, n)


def kappa(pairs):
    """Cohen's kappa for two binary raters given [(a, b), ...]."""
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def krippendorff(units, metric="nominal"):
    """Krippendorff's alpha. ``units``: {unit: [values from different coders]}."""
    units = {u: v for u, v in units.items() if len(v) >= 2}
    vals = sorted({x for v in units.values() for x in v})
    if len(vals) < 2:
        return float("nan")
    idx = {v: i for i, v in enumerate(vals)}
    o = [[0.0] * len(vals) for _ in vals]
    for v in units.values():
        m = len(v)
        for a, b in itertools.permutations(range(m), 2):
            o[idx[v[a]]][idx[v[b]]] += 1 / (m - 1)
    nc = [sum(r) for r in o]
    n = sum(nc)

    def delta(c, k):
        if metric == "nominal":
            return 0.0 if c == k else 1.0
        # ordinal (Krippendorff 2004), over ranks
        lo, hi = min(c, k), max(c, k)
        s = sum(nc[g] for g in range(lo, hi + 1)) - (nc[lo] + nc[hi]) / 2
        return s * s

    do = sum(o[c][k] * delta(c, k) for c in range(len(vals)) for k in range(len(vals)))
    de = sum(nc[c] * nc[k] * delta(c, k) for c in range(len(vals)) for k in range(len(vals)))
    return 1 - (n - 1) * do / de if de else float("nan")


# ---------------------------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------------------------
def parse_results(result):
    out = {}
    for r in result:
        v = r.get("value", {})
        if "choices" in v and v["choices"]:
            out[r["from_name"]] = v["choices"][0]
        elif "text" in v and v["text"]:
            out[r["from_name"]] = v["text"][0] if isinstance(v["text"], list) else v["text"]
    return out


def load_exports(exports):
    """{task: [(annotator, task_id, answers, lead_time)]}"""
    out = {"task1": [], "task2": []}
    for d in sorted(os.listdir(exports)):
        if not d.startswith("annotator_"):
            continue
        a = int(d.split("_")[1])
        for task in ("task1", "task2"):
            p = os.path.join(exports, d, f"{task}.json")
            if not os.path.exists(p):
                continue
            for t in json.load(open(p)):
                anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
                if not anns:
                    continue
                x = anns[-1]
                out[task].append((a, t["data"]["task_id"], parse_results(x.get("result", [])),
                                  float(x.get("lead_time") or 0.0)))
    return out


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def rater_ok(k, ans):
    """One rater's reading of a filler, scored as SV5 scores a committee rater."""
    got = LINK.get(ans.get("link"))
    link_ok = got is not None and S.label_matches(k["expected_link"], got)
    st = {k["x_pos"]: STANCE.get(ans.get("stance_x")), k["y_pos"]: STANCE.get(ans.get("stance_y"))}
    want = {int(p): v for p, v in k["expected_stance"].items()}
    status_ok = all(st.get(p) == want[p] for p in k["owned"])
    return link_ok and status_ok, ans.get("other") == "Yes", got


def analyze(key, ex, excluded):
    R = {}
    # ---- attention checks and exclusions -----------------------------------------------
    fails = collections.Counter()
    for a, t, ans, _ in ex["task1"]:
        k = key["task1"].get(t)
        if k and k["check"]:
            ok, other, _ = rater_ok(k, ans)
            fails[a] += not (ok and not other)
    for a, t, ans, _ in ex["task2"]:
        k = key["task2"].get(t)
        if k and k["check"]:
            fails[a] += ans.get("preference") != k["expected"]
    excluded = set(excluded) | {a for a, f in fails.items() if f >= 2}
    R["check_fails"] = dict(fails)
    R["excluded"] = sorted(excluded)
    # ---- Task 1 ------------------------------------------------------------------------
    per = collections.defaultdict(list)
    links = collections.defaultdict(list)
    for a, t, ans, _ in ex["task1"]:
        k = key["task1"].get(t)
        if not k or k["check"] or a in excluded:
            continue
        ok, other, got = rater_ok(k, ans)
        per[t].append((ok, other))
        if got:
            links[t].append(got)
    verdict = {}
    for t, v in per.items():
        n = len(v)
        verdict[t] = sum(o for o, _ in v) * 2 > n and not sum(x for _, x in v) * 2 > n
    adm = [t for t in verdict if key["task1"][t]["committee_admitted"]]
    rej = [t for t in verdict if not key["task1"][t]["committee_admitted"]]
    k1 = sum(verdict[t] for t in adm)
    R["H1"] = {"n": len(adm), "pass": k1, "rate_ci": wilson(k1, len(adm)),
               "null": H1_NULL, "p": binom_tail(k1, len(adm), H1_NULL) if adm else float("nan"),
               "by_combo": {f"{r}/{kd}": (sum(verdict[t] for t in adm if (key["task1"][t]["role"],
                                                                         key["task1"][t]["kind"])
                                                == (r, kd)),
                                          sum((key["task1"][t]["role"], key["task1"][t]["kind"])
                                              == (r, kd) for t in adm)) for r, kd in B.COMBOS}}
    R["H1"]["passes"] = R["H1"]["p"] < 0.05
    # H4: committee vs human verdict, reweighted to the population of SV5 readings
    pairs = [(int(key["task1"][t]["committee_admitted"]), int(verdict[t])) for t in verdict]
    pop = key.get("population", {})
    tp = sum(1 for c, h in pairs if c and h)
    tn = sum(1 for c, h in pairs if not c and not h)
    R["H4"] = {"n": len(pairs), "agree_ci": wilson(sum(c == h for c, h in pairs), len(pairs)),
               "kappa": kappa(pairs),
               "human_pass_of_admitted": wilson(tp, len(adm)),
               "human_fail_of_rejected": wilson(tn, len(rej))}
    if pop:  # rates reweighted from the stratified sample to the population of SV5 readings
        for side, want in (("admitted", True), ("rejected", False)):
            num = den = 0.0
            for r, kd in B.COMBOS:
                ts = [t for t in verdict if key["task1"][t]["committee_admitted"] == want
                      and (key["task1"][t]["role"], key["task1"][t]["kind"]) == (r, kd)]
                if not ts:
                    continue
                w = pop[f"{r}/{kd}"][side]
                rate = sum(verdict[t] for t in ts) / len(ts)
                num += w * (rate if want else 1 - rate)
                den += w
            R["H4"][f"weighted_{side}"] = num / den if den else float("nan")
    # reliability on the link label (raw and at the coupling level)
    R["alpha_link"] = krippendorff({t: v for t, v in links.items()})
    R["alpha_link_coupling"] = krippendorff(
        {t: [S.LABEL_COUPLING.get(x, "none") for x in v] for t, v in links.items()})
    # ---- Task 2 ------------------------------------------------------------------------
    by_pair = collections.defaultdict(list)
    prefs = collections.defaultdict(list)
    for a, t, ans, _ in ex["task2"]:
        k = key["task2"].get(t)
        if not k or k["check"] or a in excluded:
            continue
        p = PREF.get(ans.get("preference"))
        if p is None:
            continue
        hi = max(k["rungs"])  # the declared-higher rung of an increase pair
        chosen = None if p == "=" else (k["a_is_rung"] if p == "A" else k["b_is_rung"])
        score = 0 if chosen is None else (1 if chosen == hi else -1)
        u = (k["ladder"], tuple(k["rungs"]))
        by_pair[u].append(score)
        prefs[u].append(score)
        by_pair[u + ("type",)] = k["type"]
    inc = {u: v for u, v in by_pair.items() if len(u) == 2
           and by_pair[u + ("type",)] in ("CONFLICT", "CHAIN", "SIGN")}
    ctl = {u: v for u, v in by_pair.items() if len(u) == 2 and by_pair[u + ("type",)] == "CONTROL"}
    H2 = {}
    for name, sel in [("all", lambda t: True)] + [(t, (lambda tt: lambda t: t == tt)(t))
                                                   for t in ("CONFLICT", "CHAIN", "SIGN")]:
        us = [u for u in inc if sel(by_pair[u + ("type",)])]
        agree = sum(sum(inc[u]) > 0 for u in us)
        dis = sum(sum(inc[u]) < 0 for u in us)
        H2[name] = {"n": len(us), "agree": agree, "disagree": dis, "tie": len(us) - agree - dis,
                    "agree_ci": wilson(agree, len(us)), "p_sign": sign_test(agree, dis)}
    # Holm correction over the three per-type tests (the pooled test is primary)
    types = sorted(("CONFLICT", "CHAIN", "SIGN"), key=lambda t: H2[t]["p_sign"])
    running = 0.0
    for i, t in enumerate(types):
        running = max(running, min(1.0, (3 - i) * H2[t]["p_sign"]))
        H2[t]["p_holm"] = running
    H2["all"]["p_holm"] = H2["all"]["p_sign"]
    R["H2"] = H2
    eq_major = sum(sum(s == 0 for s in v) * 2 > len(v) for v in ctl.values())
    hi_n = sum(s > 0 for v in ctl.values() for s in v)
    lo_n = sum(s < 0 for v in ctl.values() for s in v)
    R["H3"] = {"n": len(ctl), "equal_majority_ci": wilson(eq_major, len(ctl)),
               "later_rung": hi_n, "earlier_rung": lo_n, "p_side": sign_test(hi_n, lo_n, True)}
    R["alpha_pref"] = krippendorff({u: v for u, v in prefs.items()}, metric="ordinal")
    # ---- timing --------------------------------------------------------------------------
    tm = collections.defaultdict(float)
    for task in ("task1", "task2"):
        for a, _t, _ans, lt in ex[task]:
            tm[(a, task)] += lt / 60
    R["minutes"] = {f"{a}/{task}": round(v, 1) for (a, task), v in sorted(tm.items())}
    R["_inc"] = {f"{u[0]}:{u[1][0]}-{u[1][1]}": sum(v) for u, v in inc.items()}
    return R


def rescore_metrics(R, eval_path):
    """Secondary: each metric's direction on the study's increase pairs vs the human majority."""
    if not os.path.exists(eval_path):
        return {}
    ev = json.load(open(eval_path))
    human = {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in R["_inc"].items()}
    out = {}

    def rows(name, rs):
        m = {f"{r['ladder']}:{r['pair'][0]}-{r['pair'][1]}":
             (1 if r["ok"] else 0 if r["tie"] else -1) for r in rs}
        both = [k for k in human if k in m and human[k] != 0]
        out[name] = {"n": len(both), "agree": sum(m[k] == human[k] for k in both)}

    for arm, ro in ev.get("pairs", {}).items():
        rows(f"{arm}:lambda", ro.get("lambda", []))
    for name, rs in ev.get("pairs_baseline", {}).items():
        rows(name, rs)
    return out


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------
def pct(ci):
    p, lo, hi = ci
    return f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def report(R, M):
    L = ["# LoCoBench human validation study: results", ""]
    L.append(f"Excluded annotators: {R['excluded'] or 'none'}; attention-check failures "
             f"{R['check_fails']}.")
    h1 = R["H1"]
    L += ["", f"**H1** admitted fillers read as intended: {h1['pass']}/{h1['n']} = "
          f"{pct(h1['rate_ci'])}; one-sided exact test against {100 * h1['null']:.0f}%: "
          f"p = {h1['p']:.2g} ({'PASS' if h1['passes'] else 'FAIL'})", ""]
    L += ["| role/kind | read as intended |", "|---|---|"]
    L += [f"| {c} | {k}/{n} |" for c, (k, n) in h1["by_combo"].items()]
    L += ["", "**H2** declared increases (agree / tie / disagree, one-sided sign test):", ""]
    L += ["| ladder | n | agree | tie | disagree | agree rate | p | p (Holm) |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {t} | {v['n']} | {v['agree']} | {v['tie']} | {v['disagree']} | "
          f"{pct(v['agree_ci'])} | {v['p_sign']:.2g} | {v['p_holm']:.2g} |"
          for t, v in R["H2"].items()]
    h3 = R["H3"]
    L += ["", f"**H3** CONTROL pairs judged equal by a majority: {pct(h3['equal_majority_ci'])} "
          f"of {h3['n']}; side preference among non-equal judgements: later rung "
          f"{h3['later_rung']}, earlier {h3['earlier_rung']} (two-sided p = {h3['p_side']:.2g})."]
    h4 = R["H4"]
    L += ["", f"**H4** committee vs human verdict on {h4['n']} fillers: agreement "
          f"{pct(h4['agree_ci'])}, kappa {h4['kappa']:.2f}; people pass "
          f"{pct(h4['human_pass_of_admitted'])} of admitted fillers and fail "
          f"{pct(h4['human_fail_of_rejected'])} of rejected ones (reweighted to the population "
          f"of committee readings: {100 * h4.get('weighted_admitted', float('nan')):.0f}% and "
          f"{100 * h4.get('weighted_rejected', float('nan')):.0f}%)."]
    L += ["", f"Reliability: Krippendorff alpha, link {R['alpha_link']:.2f} (coupling level "
          f"{R['alpha_link_coupling']:.2f}); preference (ordinal) {R['alpha_pref']:.2f}.",
          "", f"Minutes per annotator and task: {R['minutes']}"]
    if M:
        L += ["", "**Secondary**: metric direction vs human majority on non-tied increase pairs:",
              "", "| metric | agree |", "|---|---|"]
        L += [f"| {k} | {v['agree']}/{v['n']} |" for k, v in sorted(M.items())]
    return "\n".join(L) + "\n"


def macros(R):
    L = ["% Generated by scripts/locobench_study_analyze.py -- do not edit by hand."]

    def m(n, v):
        L.append(f"\\newcommand{{\\hs{n}}}{{{v}}}")

    p, lo, hi = R["H1"]["rate_ci"]
    m("HOneRate", f"{100 * p:.0f}")
    m("HOneCI", f"[{100 * lo:.0f}, {100 * hi:.0f}]")
    m("HOneN", R["H1"]["n"])
    m("HOneP", f"{R['H1']['p']:.2g}")
    for t, v in R["H2"].items():
        tag = t.title()
        m(f"HTwo{tag}Agree", f"{v['agree']}/{v['n']}")
        m(f"HTwo{tag}Tie", v["tie"])
        m(f"HTwo{tag}P", f"{v['p_sign']:.2g}")
    p, lo, hi = R["H3"]["equal_majority_ci"]
    m("HThreeEqual", f"{100 * p:.0f}")
    m("HFourKappa", f"{R['H4']['kappa']:.2f}")
    p, lo, hi = R["H4"]["agree_ci"]
    m("HFourAgree", f"{100 * p:.0f}")
    m("AlphaLink", f"{R['alpha_link']:.2f}")
    m("AlphaPref", f"{R['alpha_pref']:.2f}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------
# Simulation (testing only)
# ---------------------------------------------------------------------------------------------
def simulate(key, bundles, out, seed=1):
    """Synthetic Label Studio exports: noisy raters who mostly read fillers as intended."""
    rng = random.Random(seed)
    inv_link = {v: k for k, v in B.LINK_OPTIONS}
    inv_st = {v: k for k, v in B.STANCE_OPTIONS}
    for a in B.ANNOTATORS:
        d = os.path.join(out, f"annotator_{a}")
        os.makedirs(d, exist_ok=True)
        for task, fname in (("task1", "task1_fillers.json"), ("task2", "task2_pairs.json")):
            tasks = json.load(open(os.path.join(bundles, f"annotator_{a}", fname)))
            res = []
            for i, t in enumerate(tasks):
                k = key[task][t["data"]["task_id"]]
                if task == "task1":
                    good = rng.random() < (0.9 if k["committee_admitted"] else 0.45)
                    link = k["expected_link"] if good else rng.choice(list(inv_link))
                    st = {int(p): v for p, v in k["expected_stance"].items()}
                    r = [{"from_name": "link", "value": {"choices": [inv_link[link]]}},
                         {"from_name": "stance_x", "value": {"choices": [inv_st[st[k["x_pos"]]]]}},
                         {"from_name": "stance_y", "value": {"choices": [inv_st[st[k["y_pos"]]]]}},
                         {"from_name": "other", "value": {"choices": [
                             "Yes" if rng.random() < 0.07 else "No"]}}]
                else:
                    if k["check"]:
                        ch = "About equal"
                    elif k["type"] == "CONTROL":
                        ch = rng.choices(B.PREFERENCE_OPTIONS, [0.15, 0.7, 0.15])[0]
                    else:
                        hi_is_a = k["a_is_rung"] == max(k["rungs"])
                        u = rng.random()
                        win = "A" if hi_is_a else "B"
                        pick = win if u < 0.72 else ("=" if u < 0.9 else ("B" if win == "A"
                                                                          else "A"))
                        ch = {v: kk for kk, v in PREF.items()}[pick]
                    r = [{"from_name": "preference", "value": {"choices": [ch]}}]
                res.append({"id": i + 1, "data": t["data"], "annotations": [
                    {"completed_by": a, "lead_time": rng.uniform(35, 80), "result": r}]})
            json.dump(res, open(os.path.join(d, f"{task}.json"), "w"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="data/locobench-study/key/key.json")
    ap.add_argument("--exports", default="data/locobench-study/exports")
    ap.add_argument("--bundles", default="data/locobench-study/bundles")
    ap.add_argument("--out", default="data/locobench-study/results")
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    raw = open(args.key, "rb").read()
    want = open(os.path.join(os.path.dirname(args.key), "key.sha256")).read().split()[0]
    assert hashlib.sha256(raw).hexdigest() == want, "the key does not match its committed hash"
    key = json.loads(raw)
    if args.simulate:
        simulate(key, args.bundles, args.exports)
    R = analyze(key, load_exports(args.exports), args.exclude)
    M = rescore_metrics(R, EVAL)
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "report.md"), "w").write(report(R, M))
    open(os.path.join(args.out, "numbers_study.tex"), "w").write(macros(R))
    json.dump({"results": R, "metrics": M}, open(os.path.join(args.out, "results.json"), "w"),
              indent=1, default=str)
    print(report(R, M))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
