"""Figure for the coherence paper's External Checks (GPT-6 graph miner).

    uv run --with matplotlib python scripts/ext_figures.py \\
        --eval results/ext/eval.json --out docs/arr2027/coherence/figures/fig_external.pdf

(a) Conflicts: how LCS (every claim true) moves when a contradicting sentence is added, on
    ContraDoc insertion pairs and ChatProtect contradictory pairs, split by whether the miner
    found the contradiction (falls / ties / rises, as shares of the pairs).
(b) Order, matched designs: share of permutations that keep a score within tau, next to the share
    an unchanged text keeps when mined (LCS) or judged (G-Eval) again in the same run.
(c) ASAP 2.0 essays: LCS (every claim true) with length removed, by human score level; the
    GDTB breakdown is a table in the appendix.
Palette: the validated categorical slots 1-4 (blue, orange, aqua, yellow) plus a neutral gray for
ties; every bar carries a value label (two slots are below 3:1 against the surface).
"""

from __future__ import annotations

import argparse
import json

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#52514E", "#898781", "#E4E2DD"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="graph_gpt6")
    ap.add_argument("--asap", default="results/ext/asap_120/eval.json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = json.load(open(args.eval))
    c1, c2, c3 = ev["check1"], ev["check2"][f"2:{args.arm}"], ev["check3"][f"3:{args.arm}"]
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.2, 1.05, 0.95]})

    def label(ax, bars, horizontal=False):
        for b in bars:
            v = b.get_width() if horizontal else b.get_height()
            if horizontal:
                ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center",
                        ha="left", fontsize=6.3, color=INK)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=6.3, color=INK)

    # (a) conflicts: falls / ties / rises of LCS when the contradicting sentence is added
    ax = axes[0]
    cd_all, cd_edge = c2["asserted"], c2["asserted_edge"]
    cd_found = (cd_edge["decrease"], cd_edge["tie"], cd_edge["increase"])
    cd_miss = tuple(cd_all[k] - cd_edge[k] for k in ("decrease", "tie", "increase"))
    bf, cp_all = c3["asserted_by_found"], c3["asserted"]
    cp_found = (bf["found_fall"], 0, bf["found_rise"])
    cp_miss = (bf["notfound_fall"], cp_all["n"] - sum(cp_found) - bf["notfound_fall"]
               - bf["notfound_rise"], bf["notfound_rise"])
    rows_ = [("ContraDoc, found", cd_found), ("ContraDoc, missed", cd_miss),
             ("ChatProtect, found", cp_found), ("ChatProtect, missed", cp_miss)]
    ys = [3.6, 2.6, 1.0, 0.0]
    for y, (name, cnt) in zip(ys, rows_):
        n = sum(cnt)
        left = 0.0
        for v, color, lab in zip(cnt, (BLUE, GRID, ORANGE), ("falls", "ties", "rises")):
            wv = 100 * v / n
            ax.barh(y, wv, left=left, height=0.72, color=color, edgecolor="white", lw=1.0,
                    label=lab if y == ys[0] else None, zorder=2)
            if wv >= 9:
                ax.text(left + wv / 2, y, f"{v}", ha="center", va="center", fontsize=6.0,
                        color="white" if color == BLUE else INK, zorder=3)
            left += wv
    ax.set_yticks(ys, [f"{nm} ({sum(c)})" for nm, c in rows_], fontsize=6.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 4.75)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of pairs (%)")
    ax.tick_params(axis="y", length=0)
    ax.set_title("(a) conflicts", fontsize=7.5, color=INK, loc="left")
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", ncol=3, handlelength=1.0,
              borderaxespad=0.0, columnspacing=0.9, bbox_to_anchor=(0.5, 1.02))

    # (b) order
    ax = axes[1]
    # Matched designs (each text mined, and judged, twice in one run): 1g LoCoBench, 1f ContraDoc
    # documents permuted by whole sections (third review).
    series = [("LCS, permuted", BLUE, lambda r: r["perm"]["lcs"]["invariant"]),
              ("LCS, re-mined", AQUA, lambda r: r["noise_floor"]["lcs"]["invariant"]),
              ("G-Eval, permuted", ORANGE,
               lambda r: r.get("judge_perm", {}).get("lcs", {}).get("invariant")),
              ("G-Eval, re-judged", YELLOW,
               lambda r: r.get("judge_noise", {}).get("lcs", {}).get("invariant"))]
    groups = [("LoCoBench", c1[f"1g:{args.arm}"]), ("ContraDoc", c1[f"1f:{args.arm}"])]
    w = 0.19
    for k, (name, color, f) in enumerate(series):
        xs, ys = [], []
        for g, (_, r) in enumerate(groups):
            v = f(r)
            if v is not None:
                xs.append(g + (k - 1.5) * w)
                ys.append(100 * v)
        bars = ax.bar(xs, ys, width=w * 0.92, color=color, label=name, zorder=2)
        label(ax, bars)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(r"kept within $\tau$ (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_title("(b) order", fontsize=7.5, color=INK, loc="left")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", ncol=2, handlelength=1.0,
              borderaxespad=0.1, labelspacing=0.25, columnspacing=0.8)

    # (c) ASAP 2.0 essays: LCS with every claim true, length removed (rank residual on word
    # count, as a share of the sample), one dot per essay and a median bar per score level.
    ax = axes[2]
    import os, random, statistics as st, sys
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from asap_lcs import SEED, _rank
    rows = json.load(open(args.asap))["arms"]["graph_gpt6"]["rows"]
    wr = np.array(_rank([r["words"] for r in rows]), float)
    yr = np.array(_rank([r["asserted"]["lcs"] for r in rows]), float)
    A = np.vstack([wr, np.ones_like(wr)]).T
    resid = (yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]) / len(rows)
    rng = random.Random(SEED)
    for lvl in range(1, 7):
        v = [x for x, r in zip(resid, rows) if r["score"] == lvl]
        ax.scatter([lvl + rng.uniform(-0.2, 0.2) for _ in v], v, s=6, color=BLUE, alpha=0.65,
                   lw=0, zorder=2)
        ax.plot([lvl - 0.32, lvl + 0.32], [st.median(v)] * 2, color=INK, lw=1.4, zorder=3,
                solid_capstyle="round")
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("human essay score")
    ax.set_ylabel("LCS rank, length removed")
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(c) ASAP essays", fontsize=7.5, color=INK, loc="left")
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
