"""The decomposition of Lambda into per-relation evidence and an interaction term (coherence
paper, Sec. 2.4 and 3.1, App. H).

    Lambda(R) = sum_r lambda_r - I_R,   lambda_r = Lambda({r}),   I_R = log Z_R - sum_r log Z_{r}

lambda_r has a closed form (``arr_coherence_eval.relation_evidence``), and I_R = 0 when no two
relations share a claim. The signed weight sum is, up to a factor 2, the small-weight expansion of
sum_r lambda_r with every relation taken to forbid half the prior mass of its pair. Three scores
therefore form an ablation: the signed sum, the independent evidence sum_r lambda_r (exact weights,
no interaction), and Lambda / LCS (exact weights and interaction). This script reports:

* ``weights``      lambda_r for a satisfied and a violated relation at the gold weight, flat priors;
* ``interaction``  the distribution of I_R over the LoCoBench test items (gold and GPT-6 graphs);
* ``attribution``  on the GPT-6 graphs, the rung pairs where LCS and the signed sum disagree, and
                   which of the two parts (weights or interaction) each disagreement comes from;
* ``examples``     one disagreement each way, chosen by a fixed rule (below);
* ``counts``       Spearman correlation of each score with the number of relations and with
                   length, on the ExpertQA pool of the human study (expert verdicts, GPT-6 graphs).

The ladder rates with verdict noise and their paired intervals are in scripts/arr_phaseb.py.

    .venv/bin/python scripts/arr_decomp.py --macros docs/arr2027/coherence/numbers_decomp.tex
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import statistics as st
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import arr_coherence_eval as A  # noqa: E402
import arr_slot3_eval as E  # noqa: E402
import asap_lcs as S  # noqa: E402

P = 0.925  # the strong band's midpoint, the weight of most gold relations


# Palette and style of fig_lcs_lambda.pdf (scripts/arr_lcs_lambda.py): validated ordinal ramp and
# categorical pair.
RAMP = ("#86b6ef", "#2a78d6", "#104281")
GOLD, MINED = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#52514E", "#898781", "#E4E2DD"


def figure(scat, path):
    """(a) lambda_r against the weight, with the signed sum's linear terms; (b) Lambda against the
    independent evidence (the interaction is the distance from the diagonal); (c) the independent
    evidence against the signed sum (what the weights change). (b) and (c): every LoCoBench test
    response at the verdict world, gold and GPT-6 graphs."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    # (a)
    ax = axes[0]
    p = np.linspace(0.0, 0.985, 400)
    for v, col, name in ((0.25, RAMP[1], r"ENT, CON, CNC ($v_r=1/4$)"),
                         (0.5, RAMP[2], r"EQV, EXC ($v_r=1/2$)")):
        ax.plot(p, -np.log(1 - p * v), color=col, lw=1.6, zorder=3, label=name)
        ax.plot(p, np.log(1 - p) - np.log(1 - p * v), color=col, lw=1.6, ls=(0, (3, 2)), zorder=3)
    ax.plot(p, p / 2, color=MUTED, lw=1.0, ls=":", zorder=2, label=r"$\pm p_r/2$ (the terms of $\Sigma_{\pm}$)")
    ax.plot(p, -p / 2, color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-4.2, 1.3)
    ax.set_xlabel(r"weight $p_r$")
    ax.set_ylabel(r"$\lambda_r$ (nats)")
    ax.set_title(r"(a) one relation's evidence $\lambda_r$", fontsize=7.5, color=INK, loc="left")
    ax.text(0.03, 0.35, "satisfied (solid)", fontsize=6, color=INK, va="bottom")
    ax.text(0.03, -0.55, "violated (dashed)", fontsize=6, color=INK, va="top")
    ax.legend(fontsize=5.8, frameon=False, loc="lower left", handlelength=1.8)
    # (b): the interaction term itself, against the independent evidence
    lo = -9.5  # below every response's Lambda_perp (min -9.1), so nothing is clipped
    ax = axes[1]
    for tag, col, lab in (("Gold", GOLD, "gold graphs"), ("Gpt", MINED, "graph-level miner, GPT-6")):
        pts = np.array(scat[tag])
        ax.scatter(np.maximum(pts[:, 0], lo), pts[:, 0] - pts[:, 1], s=5, color=col, alpha=0.5,
                   lw=0, zorder=3, label=lab)
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_xlim(lo - 0.3, 4.5)
    top = 1.15 * max(abs(r[0] - r[1]) for t in scat for r in scat[t])
    ax.set_ylim(-top, top)
    ax.set_xlabel(r"$\Lambda^{\perp}=\sum_r\lambda_r$ (nats)")
    ax.set_ylabel(r"$I_{\mathcal{R}}=\Lambda^{\perp}-\Lambda$ (nats)")
    ax.set_title(r"(b) the interaction $I_{\mathcal{R}}$", fontsize=7.5, color=INK, loc="left")
    ax.text(lo, 0.97 * top, "$I_{\\mathcal{R}}>0$: relations overlap (discount)", fontsize=6,
            color=MUTED, va="top")
    ax.text(lo, -0.97 * top, "$I_{\\mathcal{R}}<0$: relations reinforce", fontsize=6,
            color=MUTED, va="bottom")
    # (c): what the weights change
    ax = axes[2]
    for tag, col, lab in (("Gold", GOLD, "gold graphs"), ("Gpt", MINED, "graph-level miner, GPT-6")):
        pts = np.array(scat[tag])
        ax.scatter(pts[:, 2], np.maximum(pts[:, 0], lo), s=5, color=col, alpha=0.5, lw=0, zorder=3,
                   label=lab)
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_ylim(lo - 0.3, 4.5)
    ax.set_xlabel(r"$\Sigma_{\pm}$ (sum of $\pm p_r$)")
    ax.set_ylabel(r"$\Lambda^{\perp}$ (nats)")
    ax.set_title(r"(c) the weights: $\Lambda^{\perp}$ against $\Sigma_{\pm}$", fontsize=7.5,
                 color=INK, loc="left")
    ax.legend(fontsize=5.8, frameon=False, loc="lower right", markerscale=2.2, handletextpad=0.3)
    fig.savefig(path)
    print(f"wrote {path}")


def tup(rels):
    return [(E._idx(q["source"]), E._idx(q["target"]), q["type"], float(q["probability"]))
            for q in rels]


def world(it):
    return np.array([1 if a["factual"] else 0 for a in sorted(it["atoms"], key=lambda a: a["pos"])])


def three(n, g, x):
    pu = np.full(n, 0.5)
    lam = A.confirmation(pu, g, x)
    ind = A.independent_evidence(pu, g, x)
    ws = float(sum(p * (1 if A.is_support((s, t, ty, p), x) else -1) for s, t, ty, p in g))
    return {"lcs": A.lcs_flat(lam, n), "lam": lam, "indep": ind, "I": ind - lam, "wsum": ws}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--macros", default=None)
    ap.add_argument("--figure", default=None, help="Write the decomposition figure (PDF).")
    args = ap.parse_args()
    M = {}
    SCAT = {}
    # ---- weights: one-cell couplings (ent, con, cnc) forbid 1/4 of the flat prior, two-cell
    # couplings (eqv, exc) 1/2
    x2 = {"entailment": (1, 1), "equivalence": (1, 1)}
    for ty, tag in (("entailment", "One"), ("equivalence", "Two")):
        pu = np.full(2, 0.5)
        gain = A.relation_evidence(pu, (0, 1, ty, P), np.array(x2[ty]))
        cost = -A.relation_evidence(pu, (0, 1, ty, P), np.array((1, 0)))
        M[f"Gain{tag}"] = f"{gain:.2f}"
        M[f"Cost{tag}"] = f"{cost:.2f}"
        M[f"Ratio{tag}"] = f"{cost / gain:.1f}"
    M["P"] = f"{P}"
    # ---- LoCoBench test split
    E.DATA = os.path.join(REPO, "data", "locobench-slot3-test")
    E.ROOT = os.path.join(REPO, "results", "arr2027_slot3_test")
    items, _f, ladders = E.load()
    mined = E.mined_relations("graph_gpt6")
    for src, graphs, tag in (("gold", {i: tup(E.gold_relations(it)) for i, it in items.items()}, "Gold"),
                             ("gpt6", {i: tup(r) for i, r in mined.items()}, "Gpt")):
        T = [three(len(items[i]["atoms"]), g, world(items[i])) for i, g in graphs.items()
             if i in items]
        SCAT[tag] = [(t["indep"], t["lam"], t["wsum"]) for t in T]
        # The bands of the figure's panel (c): at the same signed sum, each further violated
        # relation lowers Lambda_perp (residual of a linear fit of Lambda_perp on the sum).
        keys = [i for i in graphs if i in items]
        nviol = np.array([sum(not A.is_support(r, world(items[i])) for r in graphs[i]) for i in keys])
        xs_, ys_ = np.array([t["wsum"] for t in T]), np.array([t["indep"] for t in T])
        res_ = ys_ - np.polyval(np.polyfit(xs_, ys_, 1), xs_)
        for k, word in enumerate(("Zero", "One", "Two", "Three")):
            if (nviol == k).any():
                M[f"Band{tag}{word}"] = f"{res_[nviol == k].mean():+.2f}"
        M[f"Band{tag}Corr"] = f"{np.corrcoef(res_, nviol)[0, 1]:.2f}"
        M[f"Rho{tag}IndepLam"] = f"{S.spearman([t['indep'] for t in T], [t['lam'] for t in T]):.2f}"
        M[f"Rho{tag}WsumIndep"] = f"{S.spearman([t['wsum'] for t in T], [t['indep'] for t in T]):.2f}"
        I = [t["I"] for t in T]
        M[f"I{tag}AbsMedian"] = f"{st.median(np.abs(I)):.2f}"
        M[f"I{tag}AbsPninety"] = f"{np.percentile(np.abs(I), 90):.2f}"
        M[f"Lam{tag}AbsMedian"] = f"{st.median([abs(t['lam']) for t in T]):.2f}"
        M[f"I{tag}N"] = f"{len(I):,}".replace(",", "{,}")
        M[f"I{tag}Median"] = f"{st.median(I):.2f}"
        M[f"I{tag}Pninety"] = f"{np.percentile(I, 90):.2f}"
        M[f"I{tag}AbsSmall"] = f"{100 * np.mean(np.abs(I) < 0.05):.0f}"
        M[f"I{tag}Neg"] = f"{100 * np.mean(np.array(I) < -1e-9):.0f}"
        M[f"I{tag}Pos"] = f"{100 * np.mean(np.array(I) > 1e-9):.0f}"
        print(src, "I_R", M[f"I{tag}Median"], M[f"I{tag}Pninety"], M[f"I{tag}AbsSmall"])
    # ---- attribution on the GPT-6 graphs (all increase rung pairs, the ladders' tie rule)
    rows = []
    for lad in ladders:
        if lad["type"] not in E.INCREASE or any(i not in mined for i in lad["item_ids"]):
            continue
        V = [three(len(items[i]["atoms"]), tup(mined[i]), world(items[i])) for i in lad["item_ids"]]
        for i, j in itertools.combinations(range(3), 2):
            ok = {k: E.pair_ok(lad["type"], V[i][k], V[j][k], E.REL_TIE, True)
                  for k in ("lcs", "indep", "wsum")}
            rows.append({"ladder": lad["ladder_id"], "pair": (i, j), "ok": ok, "V": (V[i], V[j]),
                         "ids": (lad["item_ids"][i], lad["item_ids"][j])})
    M["AttrPairs"] = len(rows)
    for k in ("lcs", "indep", "wsum"):
        M[f"Attr{k.title()}"] = f"{100 * np.mean([r['ok'][k] for r in rows]):.1f}"
    dis = [r for r in rows if r["ok"]["lcs"] != r["ok"]["wsum"]]
    sum_w = [r for r in dis if r["ok"]["wsum"]]
    lcs_w = [r for r in dis if r["ok"]["lcs"]]
    M["AttrDisagree"] = len(dis)
    M["AttrSumWins"] = len(sum_w)
    M["AttrLcsWins"] = len(lcs_w)
    # a disagreement comes from the interaction term if sum_r lambda_r sides with the signed sum
    # (the weights agree with it), from the weights if it sides with LCS
    M["AttrSumWinsInteraction"] = sum(r["ok"]["indep"] == r["ok"]["wsum"] for r in sum_w)
    M["AttrSumWinsWeights"] = sum(r["ok"]["indep"] == r["ok"]["lcs"] for r in sum_w)
    M["AttrLcsWinsInteraction"] = sum(r["ok"]["indep"] == r["ok"]["wsum"] for r in lcs_w)
    M["AttrLcsWinsWeights"] = sum(r["ok"]["indep"] == r["ok"]["lcs"] for r in lcs_w)
    print("attribution", {k: v for k, v in M.items() if k.startswith("Attr")})

    # ---- examples. Rule: among the disagreements won by each score, the one whose two graphs
    # differ in the fewest relations (ties broken by ladder id); for the signed sum, among those
    # decided by the weights (the case the text explains).
    def ndiff(r):
        a, b = (set(tup(mined[i])) for i in r["ids"])
        return len(a ^ b)

    def pick(cands):
        return min(cands, key=lambda r: (ndiff(r), r["ladder"], r["pair"])) if cands else None

    ex = {"Lcs": pick(lcs_w),
          "Sum": pick([r for r in sum_w if r["ok"]["indep"] == r["ok"]["lcs"]])}
    for tag, r in ex.items():
        if r is None:
            continue
        a, b = r["V"]
        M[f"Ex{tag}Ladder"] = r["ladder"].replace("-", "--")
        M[f"Ex{tag}Type"] = r["ladder"].split("-")[1].lower()
        M[f"Ex{tag}RungA"], M[f"Ex{tag}RungB"] = r["pair"]
        for k, f in (("lcs", "{:.3f}"), ("wsum", "{:.2f}"), ("indep", "{:.2f}"), ("I", "{:+.2f}")):
            M[f"Ex{tag}{k.title()}A"] = f.format(a[k])
            M[f"Ex{tag}{k.title()}B"] = f.format(b[k])
        ga, gb = (set(tup(mined[i])) for i in r["ids"])
        x = world(items[r["ids"][1]])
        n = len(x)
        added = sorted(gb - ga)
        removed = sorted(ga - gb)
        print(tag, r["ladder"], r["pair"], {k: (a[k], b[k]) for k in ("lcs", "wsum", "indep", "I")})
        print("   added", [(s, t, ty, p, A.is_support((s, t, ty, p), x)) for s, t, ty, p in added])
        print("   removed", removed)
        M[f"Ex{tag}Added"] = len(added)
        M[f"Ex{tag}NA"], M[f"Ex{tag}NB"] = len(ga), len(gb)
        M[f"Ex{tag}By"] = "weights" if r["ok"]["indep"] == r["ok"]["lcs"] else "interaction"
        M[f"Ex{tag}Removed"] = len(removed)
        if tag == "Sum":
            sup = [q for q in added if A.is_support(q, x)]
            viol = [q for q in added if not A.is_support(q, x)]
            if len(sup) == 1 and len(viol) == 1:
                pu = np.full(n, 0.5)
                M["ExSumSupP"] = f"{sup[0][3]:.3f}"
                M["ExSumViolP"] = f"{viol[0][3]:.3f}"
                M["ExSumViolType"] = viol[0][2]
                M["ExSumSupGain"] = f"{A.relation_evidence(pu, sup[0], x):.2f}"
                M["ExSumViolCost"] = f"{-A.relation_evidence(pu, viol[0], x):.2f}"
    # ---- counts: ExpertQA pool, expert verdicts, GPT-6 graphs
    import ext_eval as X
    its = X.items("expertqa")
    recs, _ = X.records("expertqa", "graph_gpt6")
    R = []
    for iid, it in sorted(its.items()):
        if iid not in recs:
            continue
        n = it["num_atoms"]
        g = X.rels_of(recs[iid])
        x = np.array([1 if a["factual"] else 0 for a in sorted(it["atoms"], key=lambda a: a["pos"])])
        v = three(n, g, x)
        v.update(nrel=len(g), words=it["expected"]["words"])
        R.append(v)
    M["CountN"] = len(R)
    for k in ("lcs", "indep", "wsum"):
        M[f"Count{k.title()}Nrel"] = f"{S.spearman([r[k] for r in R], [r['nrel'] for r in R]):.2f}"
        M[f"Count{k.title()}Words"] = f"{S.spearman([r[k] for r in R], [r['words'] for r in R]):.2f}"
    print("counts", {k: v for k, v in M.items() if k.startswith("Count")})
    if args.figure:
        figure(SCAT, args.figure)
    if args.macros:
        L = ["% Generated by scripts/arr_decomp.py -- do not edit by hand."]
        # signed values in math mode, so a minus is a minus in running text too
        fmt = lambda v: f"\\ensuremath{{{v}}}" if isinstance(v, str) and re.match(r"^[+-]\d", v) else v  # noqa: E731
        L += [f"\\newcommand{{\\dc{k}}}{{{fmt(v)}}}" for k, v in M.items()]
        open(args.macros, "w").write("\n".join(L) + "\n")
        print(f"wrote {args.macros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
