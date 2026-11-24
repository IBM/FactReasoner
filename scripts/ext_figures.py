"""Figure for the coherence paper's External Checks (GPT-6 graph miner).

    uv run --with matplotlib python scripts/ext_figures.py \\
        --eval results/ext/eval_interim.json --out docs/arr2027/coherence/figures/fig_external.pdf

(a) Order: share of cases a score keeps within tau when paragraphs are permuted, next to the
    share an unchanged text keeps when mined again (the miner's noise floor).
(b) ContraDoc: recall of the inserted or replaced contradiction, by scope and by edit.
(c) GDTB: share of relations that receive a coupling, by category; for the first group that is
    the right answer, for the second (senses that compile to none) it is an error.
Palette: the validated categorical slots 1-4 (blue, orange, aqua, yellow); every bar carries a
value label (two slots are below 3:1 against the surface).
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
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = json.load(open(args.eval))
    c1, c2, c4 = ev["check1"], ev["check2"][f"2:{args.arm}"], ev["check4"][f"4b:{args.arm}"]
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})

    def label(ax, bars, horizontal=False):
        for b in bars:
            v = b.get_width() if horizontal else b.get_height()
            if horizontal:
                ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center",
                        ha="left", fontsize=6.3, color=INK)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=6.3, color=INK)

    # (a) order
    ax = axes[0]
    series = [("LCS, permuted", BLUE, lambda r: r["perm"]["lcs"]["invariant"]),
              ("LCS, re-mined", AQUA, lambda r: r["noise_floor"]["lcs"]["invariant"]),
              ("entity graph", YELLOW,
               lambda r: r["perm"].get("discourse_entity_graph", {}).get("invariant")),
              ("G-Eval (GPT-6)", ORANGE,
               lambda r: r["perm"].get("judge_geval@gpt-6-sol", {}).get("invariant"))]
    groups = [("LoCoBench", c1[f"1a:{args.arm}"]), ("ContraDoc", c1[f"1c:{args.arm}"])]
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
    ax.set_title("(a) order", fontsize=7.5, color=INK, loc="left")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", ncol=2, handlelength=1.0,
              borderaxespad=0.1, labelspacing=0.25, columnspacing=0.8)

    # (b) ContraDoc recall
    ax = axes[1]
    R = c2["recall"]
    cats = [("within\nsent.", "scope=intra"), ("local", "scope=local"), ("global", "scope=global"),
            ("insert", "plug=Insert"), ("replace", "plug=Replace")]
    xs = [0, 1.35, 2.7, 4.25, 5.75]
    bars = ax.bar(xs, [100 * R[k]["recall"] for _, k in cats], width=0.7, color=BLUE, zorder=2)
    label(ax, bars)
    ax.set_xticks(xs, [c for c, _ in cats], fontsize=6.2)
    ax.set_ylim(0, 75)
    ax.set_ylabel("contradiction found (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.text(1.35, 70, "scope", ha="center", fontsize=6.5, color=MUTED)
    ax.text(5.0, 70, "edit", ha="center", fontsize=6.5, color=MUTED)
    ax.set_title("(b) ContraDoc recall", fontsize=7.5, color=INK, loc="left")

    # (c) GDTB
    ax = axes[2]
    by = c4["by_category"]
    coup = ["Cause-Effect", "Condition", "Instantiation", "Restatement", "Disjunction"]
    none = ["Contrast", "Concession", "Temporal", "EntRel/NoRel"]
    names = coup + none
    ys = list(range(len(names)))[::-1]
    vals = [100 * by[c]["any"] for c in names]
    colors = [BLUE] * len(coup) + [ORANGE] * len(none)
    bars = ax.barh(ys, vals, height=0.7, color=colors, zorder=2)
    label(ax, bars, horizontal=True)
    ax.set_yticks(ys, [n.replace("EntRel/NoRel", "EntRel, NoRel") for n in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("receives a coupling (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="sense has a coupling"),
                       Patch(color=ORANGE, label="sense compiles to none")],
              fontsize=5.6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36),
              handlelength=1.0)
    ax.set_title("(c) GDTB", fontsize=7.5, color=INK, loc="left")
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
