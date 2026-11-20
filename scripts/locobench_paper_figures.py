"""Data figures for the LoCoBench dataset paper, from ``scripts/locobench_paper_stats.py --json``.

* ``fig_composition.pdf`` -- what is in the release: families per domain and ladder type,
  gold relations by discourse sense (coloured by the coupling they compile to), and response
  length.
* ``fig_gates.pdf``       -- how selective validation is: pass rate of every gate over every
  evaluation, and the committee's first reading of each filler kind.
* ``fig_gold_lcs.pdf``    -- the gold graphs realize the declared orders: gold LCS per rung for
  every ladder, by type.

Same palette and type as the companion paper's figures (the dataviz reference palette).

    uv run --with matplotlib python scripts/locobench_paper_figures.py \\
        --json docs/arr2027/locobench/stats.json --out docs/arr2027/locobench/figures
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
PURPLE, GOLD = "#7b5ea7", "#c59a1f"
GRAY, LIGHT = "#898781", "#d9d8d2"
INK, INK2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

TYPES = ["CONFLICT", "CHAIN", "SIGN", "ORDER"]
COUPLING_COL = {"entailment": BLUE, "contradiction": ORANGE, "equivalence": AQUA,
                "co_necessity": PURPLE, "exclusive": GOLD, "none": LIGHT}
COUPLING_NAME = {"entailment": "entailment", "contradiction": "contradiction",
                 "equivalence": "equivalence", "co_necessity": "co-necessity",
                 "exclusive": "exclusion", "none": "no factor"}
SENSE_COUPLING = {"Evidence": "entailment", "Cause-Effect": "entailment",
                  "Refutation": "contradiction", "Restatement": "equivalence",
                  "Disjunction": "co_necessity", "Alternative": "exclusive",
                  "Contrast": "none", "Concession": "none", "Precedence": "none",
                  "Succession": "none"}


def _clean(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)


def fig_composition(S, out):
    R = S["release"]
    fig = plt.figure(figsize=(7.0, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=0.55)
    # (a) domain x type grid.
    ax = fig.add_subplot(gs[0])
    doms = sorted(R["domain_x_type"])
    vmax = max(v for d in doms for v in R["domain_x_type"][d].values())
    for i, d in enumerate(doms):
        for j, t in enumerate(TYPES):
            v = R["domain_x_type"][d][t]
            shade = 0.15 + 0.75 * v / vmax
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.9, color=BLUE, alpha=shade, lw=0))
            ax.text(j + 0.47, i + 0.45, str(v), ha="center", va="center", fontsize=7,
                    color="white" if shade > 0.55 else INK)
        tot = sum(R["domain_x_type"][d].values())
        ax.text(len(TYPES) + 0.15, i + 0.45, str(tot), ha="left", va="center", fontsize=7,
                color=INK2)
    ax.set_xlim(0, len(TYPES) + 0.7)
    ax.set_ylim(len(doms), -0.15)
    ax.set_xticks([j + 0.47 for j in range(len(TYPES))])
    ax.set_xticklabels(["conf.", "chain", "sign", "order"], fontsize=6.3)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.45 for i in range(len(doms))])
    ax.set_yticklabels([d.replace(" & ", " &\n") for d in doms], fontsize=6.5)
    ax.tick_params(length=0)
    _clean(ax, keep=())
    ax.set_title("(a) families by domain and primary ladder", fontsize=7.2, color=INK,
                 loc="left", pad=16)
    # (b) relations by sense, coloured by coupling.
    ax = fig.add_subplot(gs[1])
    sense = sorted(R["relations"]["sense"].items(), key=lambda kv: kv[1])
    ys = range(len(sense))
    ax.barh(list(ys), [v for _k, v in sense],
            color=[COUPLING_COL[SENSE_COUPLING[k]] for k, _v in sense], height=0.7, lw=0)
    for y, (k, v) in zip(ys, sense):
        ax.text(v + 30, y, f"{v:,}", va="center", fontsize=6.2, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([k for k, _v in sense], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(v for _k, v in sense) * 1.22)
    ax.set_xlabel("gold relations (all responses)", fontsize=6.5)
    _clean(ax)
    seen = []
    for k, _v in sense[::-1]:
        c = SENSE_COUPLING[k]
        if c not in seen:
            seen.append(c)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COUPLING_COL[c]) for c in seen]
    ax.legend(handles, [COUPLING_NAME[c] for c in seen], loc="lower right", frameon=False,
              fontsize=5.8, handlelength=0.9, handleheight=0.7, title="compiles to",
              title_fontsize=5.8, borderaxespad=0.0)
    ax.set_title("(b) gold relations by discourse sense", fontsize=7.2, color=INK, loc="left")
    # (c) response length.
    ax = fig.add_subplot(gs[2])
    w = R["dist"]["words"]
    ax.hist(w, bins=range(360, 610, 10), color=BLUE, edgecolor="white", lw=0.5)
    ax.axvline(st.median(w), color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(st.median(w) + 4, ax.get_ylim()[1] * 0.92, f"median {st.median(w):.0f}",
            fontsize=6.2, color=INK2)
    ax.set_xlabel("words per response", fontsize=6.5)
    ax.set_ylabel("responses", fontsize=6.5)
    _clean(ax, keep=("bottom", "left"))
    ax.set_title("(c) response length", fontsize=7.2, color=INK, loc="left")
    fig.savefig(os.path.join(out, "fig_composition.pdf"))
    plt.close(fig)


GATES = [  # (gate key, label, stage)
    ("P2", "claim pool parses", "generate"),
    ("S3.gate", "plan gate (deterministic)", "generate"),
    ("S4.structure", "template structure", "generate"),
    ("V3.scores", "V3 fluency, formality, organization", "committee"),
    ("V3.spans", "V3 no hedging or annotation leakage", "committee"),
    ("V3.scores@cover", "V3 scores, coverage splices", "committee"),
    ("V4", "V4 every claim asserted", "committee"),
    ("V4.false_atoms", "V4 false claims asserted as given", "committee"),
    ("SV1", "SV1 fixed relations confirmed (blind)", "committee"),
    ("SV5", "SV5 filler read as intended (blind)", "committee"),
    ("ORDER.permutation", "ORDER permutation keeps relations", "committee"),
]
READ = [("conflict_tt/conflict", "conflict: sets two truths against each other"),
        ("conflict_tt/none", "conflict: no link"),
        ("chain_link/support", "chain: draws the link"),
        ("chain_link/none", "chain: no link"),
        ("sign_tf/conflict", "sign: derives the falsehood"),
        ("sign_tf/none", "sign: no link"),
        ("sign_tf/support", "sign: refutes the falsehood"),
        ("control/para1", "control: wording 1"),
        ("control/para2", "control: wording 2"),
        ("control/para3", "control: wording 3")]


def fig_gates(S, out):
    G = S["gates"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"wspace": 1.15})
    for ax, rows, src, title in (
            (axes[0], GATES, G["gate"], "(a) every evaluation of every gate (retries included)"),
            (axes[1], READ, G["sv5_first"], "(b) committee's first reading of each filler kind")):
        ys = range(len(rows))
        vals = [src[k][0] / src[k][1] for k, *_ in rows]
        cols = [GRAY if (len(r) > 2 and r[2] == "generate") else BLUE for r in rows]
        if src is G["sv5_first"]:
            cols = [ORANGE if v < 0.7 else BLUE for v in vals]
        ax.barh(list(ys), [100 * v for v in vals], color=cols, height=0.68, lw=0)
        for y, (k, *_r) in zip(ys, rows):
            ok, n = src[k]
            ax.text(100 * ok / n + 1.5, y, f"{ok}/{n}", va="center", fontsize=5.9, color=INK2)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([r[1] for r in rows], fontsize=6.2)
        ax.set_ylim(len(rows) - 0.5, -0.6)
        ax.set_xlim(0, 122)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("passed (%)", fontsize=6.5)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _clean(ax)
        ax.set_title(title, fontsize=7.0, color=INK, loc="left")
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, color=GRAY),
                    plt.Rectangle((0, 0), 1, 1, color=BLUE)],
                   ["deterministic / generator", "committee (majority of three)"],
                   loc="lower left", bbox_to_anchor=(-0.02, -0.42), ncol=2, frameon=False,
                   fontsize=5.9, handlelength=0.9, handleheight=0.7)
    fig.savefig(os.path.join(out, "fig_gates.pdf"))
    plt.close(fig)


def fig_gold_lcs(S, out):
    R = S["release"]
    types = ["CONFLICT", "CHAIN", "SIGN", "ORDER", "CONTROL"]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.55), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, t in zip(axes, types):
        lads = R["gold_lcs"][t]
        for v in lads:
            ax.plot([0, 1, 2], v, color=GRAY, lw=0.45, alpha=0.35, zorder=1)
        mean = [st.mean(v[k] for v in lads) for k in range(3)]
        col = BLUE if t in ("CONFLICT", "CHAIN", "SIGN") else INK2
        ax.plot([0, 1, 2], mean, color=col, lw=2.0, zorder=3, solid_capstyle="round")
        ax.scatter([0, 1, 2], mean, s=16, color=col, edgecolor="white", lw=0.8, zorder=4)
        declared = "declared: increasing" if t in ("CONFLICT", "CHAIN", "SIGN") \
            else "declared: invariant"
        ax.set_title(f"{t.lower()} ({len(lads)})\n{declared}", fontsize=6.6, color=INK,
                     loc="left", pad=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["$\\rho_0$", "$\\rho_1$", "$\\rho_2$"], fontsize=6.5)
        ax.set_xlim(-0.25, 2.25)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _clean(ax, keep=("bottom", "left") if t == "CONFLICT" else ("bottom",))
    axes[0].set_ylabel("gold LCS", fontsize=6.5)
    fig.savefig(os.path.join(out, "fig_gold_lcs.pdf"))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/arr2027/locobench/stats.json")
    ap.add_argument("--out", default="docs/arr2027/locobench/figures")
    args = ap.parse_args()
    S = json.load(open(args.json))
    os.makedirs(args.out, exist_ok=True)
    fig_composition(S, args.out)
    fig_gates(S, args.out)
    fig_gold_lcs(S, args.out)
    print(f"wrote fig_composition.pdf, fig_gates.pdf, fig_gold_lcs.pdf to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
