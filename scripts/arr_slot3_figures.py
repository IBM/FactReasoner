"""Figures for the ARR coherence paper from `scripts/arr_slot3_eval.py --json` output.

Three static vector figures, one job each, emphasis form (our score in the accent hue, every
other score in the de-emphasis gray), 95% intervals from a family-level bootstrap:

* ``fig_ladders.pdf``  -- per score, the share of declared increases on which it rises and on
  which it falls, and of invariances it keeps (within tau): the headline comparison.
LCS is the score; its rises and falls are decided on Lambda, its log-ratio form (the same
order, without the logistic's compression near 0 and 1), so ``rows["dl"]`` is a Lambda step.

* ``fig_dominance.pdf`` -- per mined arm, its Increase pairs split by how the two extracted
  graphs compare in the coherence order: no error falls on a correctly ordered pair.
* ``fig_sign.pdf``     -- the SIGN ladder rung by rung on gold graphs, under the two
  polarities: only the verdict-polarity score rises from "derives the falsehood" to "refutes it".

Palette: the dataviz skill's reference palette (validated: blue/aqua/orange pass every gate;
aqua sits below 3:1 contrast, so its segments carry direct labels).

Run (matplotlib is not a project dependency)::

    uv run --with matplotlib python scripts/arr_slot3_figures.py \\
        --json results/arr2027_slot3_test/eval.json --out docs/arr2027/coherence/figures
"""

from __future__ import annotations

import argparse
import json
import os
import random

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"   # categorical slots 1, 3, 2
GRAY = "#898781"                                       # de-emphasis / muted ink
INK, INK2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

#: (key, label, kind): kind 'ours' rows are drawn in the accent hue.
ROWS = [
    ("header", "Our score: LCS at verdict polarity", None),
    ("arm:gold:lambda", "gold graphs", "ours"),
    ("arm:graph_gpt6:lambda", "graph-level miner, GPT-6", "ours"),
    ("arm:graph_gptoss:lambda", "graph-level miner, gpt-oss-120b", "ours"),
    ("arm:pair_gptoss:lambda", "pairwise miner, gpt-oss-120b", "ours"),
    ("arm:pair_llama:lambda", "pairwise miner, Llama-3.3-70B", "ours"),
    ("header", "The same model at another world (gold graphs)", None),
    ("arm:gold:lambda_assert", "LCS at the asserted world", "other"),
    ("header", "Baselines (read the text)", None),
    ("base:judge_geval@gpt-6-sol", "LLM judge, G-Eval, GPT-6", "other"),
    ("base:judge_direct@gpt-6-sol", "LLM judge, 1–5 rating, GPT-6", "other"),
    ("base:judge_geval@gpt-oss-120b", "LLM judge, G-Eval, gpt-oss-120b", "other"),
    ("base:judge_direct@gpt-oss-120b", "LLM judge, 1–5 rating, gpt-oss-120b", "other"),
    ("base:discourse_rc", "DiscoScore RC", "other"),
    ("base:control_length", "response length", "other"),
    ("base:nli_contradiction", "NLI contradiction (claims)", "other"),
    ("base:roscoe_sc", "ROSCOE self-consistency (claims)", "other"),
]


#: Baselines that read only the claims, which are identical across a family's rungs.
TRIVIAL = {"base:nli_contradiction", "base:roscoe_sc"}


def _flag(r, key):
    if key == "falls":  # an Increase pair the score moves the wrong way on (not a tie)
        return not r["ok"] and not r["tie"]
    return bool(r[key])


def boot_share(rows, key, n=2000, seed=0):
    """Share of rows with flag ``key``, and a 95% family-level bootstrap interval."""
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault(r["family"], []).append(_flag(r, key))
    fams = list(by)
    point = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        ok = tot = 0
        for _f in range(len(fams)):
            v = by[fams[rng.randrange(len(fams))]]
            ok += sum(v)
            tot += len(v)
        stats.append(ok / tot)
    stats.sort()
    return point, stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def boot_mean(values, n=2000, seed=0):
    rng = random.Random(seed)
    m = sum(values) / len(values)
    st = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                for _ in range(n))
    return m, st[int(0.025 * n)], st[int(0.975 * n) - 1]


def rows_for(d, key, group):
    src, *rest = key.split(":")
    if src == "arm":
        arm, ro = rest
        rows = d["pairs"].get(arm, {}).get(ro, [])
    else:
        rows = d["pairs_baseline"].get(rest[0], [])
    inc = ("CONFLICT", "CHAIN", "SIGN")
    return [r for r in rows if (r["type"] in inc) == (group == "inc")]


def partial(d, arm, label):
    """Mark an arm that has not scored every response (e.g. a run still in progress)."""
    n = d["arms"].get(arm, {}).get("n_items", d["n_items"])
    return label if n >= d["n_items"] else f"{label} (partial: {n} of {d['n_items']})"


def fig_ladders(d, out):
    rows = [(k, partial(d, k.split(":")[1], lab) if k.startswith("arm:") else lab, kind)
            for k, lab, kind in ROWS]
    n = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.165 * n + 0.8), sharey=True,
                             gridspec_kw={"wspace": 0.10, "width_ratios": [1.0, 0.62, 1.0]})
    panels = [("inc", "ok", "Declared increases: score rises (%)"),
              ("inc", "falls", "… score falls (%)"),
              ("inv", "ok_tol", "Invariances kept, within τ (%)")]
    for ax, (group, okkey, title) in zip(axes, panels):
        for y, (key, label, kind) in enumerate(rows):
            if key == "header":
                continue
            st = boot_share(rows_for(d, key, group), okkey)
            if st is None:
                continue
            p, lo, hi = (100 * v for v in st)
            col = BLUE if kind == "ours" else GRAY
            # Hollow: invariance "kept" for a reason that is not the score's: the pairwise
            # arms' LCS is pinned near 0 on both rungs (near-certain weights saturate it), and
            # the claims-only baselines never move at all.
            hollow = group == "inv" and (key.startswith("arm:pair_") or key in TRIVIAL)
            hi = min(hi, 100)
            ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
            ax.scatter([p], [y], s=26, color="white" if hollow else col,
                       edgecolor=col if hollow else "white", linewidth=1.2, zorder=3)
            if kind == "ours" or group == "inc" or hollow:
                span = 62 if okkey == "falls" else 100
                ax.text(hi + 0.02 * span, y, f"{p:.0f}", va="center", ha="left",
                        fontsize=6.5, color=INK2)
        if okkey == "falls":
            ax.set_xlim(-1.2, 62)
            ax.set_xticks([0, 25, 50])
        else:
            ax.set_xlim(-2, 112)
            ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=4)
        if group == "inc" and okkey == "ok":
            ax.axvline(50, color=AXIS, lw=0.8, zorder=1)
            ax.text(51, -0.55, "coin flip", ha="left", va="center", fontsize=6, color=GRAY)
    ax0 = axes[0]
    ax0.set_ylim(n - 0.5, -0.7)
    ax0.set_yticks(range(n))
    labels = []
    for key, label, kind in rows:
        labels.append(label)
    ax0.set_yticklabels(labels)
    for tick, (key, _l, _k) in zip(ax0.get_yticklabels(), rows):
        if key == "header":
            tick.set_fontweight("bold")
            tick.set_color(INK)
        else:
            tick.set_color(INK2)
    fig.savefig(os.path.join(out, "fig_ladders.pdf"))
    plt.close(fig)


ARMS = [("graph_gpt6", "graph-level, GPT-6"), ("graph_gptoss", "graph-level, gpt-oss-120b"),
        ("pair_gptoss", "pairwise, gpt-oss-120b"), ("pair_llama", "pairwise, Llama-3.3-70B")]


def fig_dominance(d, out):
    segs = [("ordered", BLUE, "ordered as declared: LCS rises"),
            ("incomp_ok", AQUA, "incomparable: LCS rises"),
            ("incomp_bad", ORANGE, "incomparable: LCS does not rise"),
            ("backward", GRAY, "equal or reversed: LCS ties or falls")]
    fig, ax = plt.subplots(figsize=(3.3, 1.35))
    for y, (arm, label) in enumerate(ARMS):
        rows = d["dominance_rows"].get(arm, [])
        c = {"ordered": 0, "incomp_ok": 0, "incomp_bad": 0, "backward": 0}
        for r in rows:
            if r["kind"] == "forward":
                c["ordered"] += 1
                # The theorem: a strictly better graph strictly raises Λ (exact sign; a rise
                # below the 1e-9 counting resolution still counts as a rise here) ...
                assert r.get("dl", 1.0) > 0, f"{arm}: Λ did not rise on an ordered pair"
            elif r["kind"] == "incomparable":
                c["incomp_ok" if r["rises"] else "incomp_bad"] += 1
            else:
                c["backward"] += 1
                # ... an identical graph leaves it unchanged and a worse one lowers it.
                ok = (abs(r.get("dl", 0.0)) <= 1e-9 if r["kind"] == "equal"
                      else r.get("dl", -1.0) < 0)
                assert ok, f"{arm}: Λ moved against the order on a {r['kind']} pair"
        total = sum(c.values()) or 1
        left = 0.0
        for key, col, _lab in segs:
            w = 100 * c[key] / total
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0)
            if w >= 7:
                ax.text(left + w / 2, y, f"{c[key]}", ha="center", va="center", fontsize=6.3,
                        color="white" if col != AQUA else INK)
            left += w
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([partial(d, a, lab) for a, lab in ARMS], color=INK2)
    ax.set_ylim(len(ARMS) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Increase pairs (%)", fontsize=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _k, col, _l in segs]
    ax.legend(handles, [lab for _k, _c, lab in segs], loc="lower left",
              title="the two rungs' extracted graphs are …", title_fontsize=6.3,
              alignment="left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=6.3,
              handlelength=1.0, handleheight=0.8, columnspacing=1.0, borderaxespad=0.2)
    fig.savefig(os.path.join(out, "fig_dominance.pdf"))
    plt.close(fig)


def fig_sign(d, out):
    """Each SIGN ladder's change from its unlinked rung, averaged, with 95% intervals."""
    panels = [("lcs", "Our score: LCS at verdict polarity", BLUE),
              ("lcs_assert", "LCS at the asserted world", GRAY)]
    xs = [0, 1, 2]
    names = ["derives\nthe falsehood", "no link", "refutes\nthe falsehood"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.6), gridspec_kw={"wspace": 0.35})
    for ax, (ro, title, col) in zip(axes, panels):
        lad = d["sign_rungs"][ro]
        stats = [boot_mean([l[k] - l[1] for l in lad], seed=k) for k in xs]
        m = [s[0] for s in stats]
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        for x, (mean, lo, hi) in zip(xs, stats):
            if x != 1:
                ax.plot([x, x], [lo, hi], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(xs, m, color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(xs, m, s=30, color=col, edgecolor="white", linewidth=1.0, zorder=3)
        if m[0] < 0 < m[2]:
            verdict = "derivation lowers it, refutation raises it"
        elif m[0] > 0 > m[2]:
            verdict = "the reverse of the declared order"
        else:
            verdict = "derivation does not lower it" if m[0] >= 0 else "not as declared"
        ax.set_title(f"{title}\n{verdict}", fontsize=7.0, color=INK, loc="left", pad=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=6.3, color=INK2)
        ax.set_xlim(-0.3, 2.3)
        lim = max(abs(v) for st in stats for v in st) * 1.15 or 1
        ax.set_ylim(-lim, lim)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=6.3, colors=INK2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("change from the\nunlinked rung", fontsize=6.5)
    fig.savefig(os.path.join(out, "fig_sign.pdf"))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.json))
    os.makedirs(args.out, exist_ok=True)
    fig_ladders(d, args.out)
    fig_dominance(d, args.out)
    fig_sign(d, args.out)
    print(f"wrote fig_ladders.pdf, fig_dominance.pdf, fig_sign.pdf to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
