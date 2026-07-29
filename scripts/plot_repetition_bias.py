"""
Plot repetition bias results mirroring Schuster et al. Figure 8 style.
Generates two figures:
  1. Part 1: controlled experiment (Figure 8 analog)
  2. Part 2: full dataset SP distribution by label type
"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_JSON = "/u/samit/repetition_bias_results.json"

with open(OUT_JSON) as f:
    data = json.load(f)

# ── Figure 1: Controlled experiment (mirrors Schuster et al. Fig 8) ───────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#1C2B4A')
ax.set_facecolor('#1C2B4A')

conditions = [r["condition"].split("\n")[0] for r in data["part1_controlled"]]
sp_trust   = [r["sp_trust"]   for r in data["part1_controlled"]]
sp_vanilla = [r["sp_vanilla"] for r in data["part1_controlled"]]

# Skip baseline (SP=0 by definition)
conds  = conditions[1:]
sp_t   = sp_trust[1:]
sp_v   = sp_vanilla[1:]

x = np.arange(len(conds))
w = 0.35

bars_t = ax.bar(x - w/2, sp_t, w, label="Trust Fusion (DynaTD+UTD)",
                color="#16A34A", alpha=0.85, zorder=3)
bars_v = ax.bar(x + w/2, sp_v, w, label="Vanilla FactReasoner (all fp=0.9)",
                color="#D97706", alpha=0.85, zorder=3)

# Annotate bars
for bar in bars_t:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f"{h:+.3f}", ha='center', va='bottom',
            color="#16A34A", fontsize=9, fontweight='bold')
for bar in bars_v:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f"{h:+.3f}", ha='center', va='bottom',
            color="#D97706", fontsize=9, fontweight='bold')

# Zero line
ax.axhline(0, color='white', linewidth=0.8, alpha=0.4, zorder=2)

# Threshold line: where verdict flips from NS to S
base_trust_p = data["part1_controlled"][0]["trust_p"]
base_van_p   = data["part1_controlled"][0]["vanilla_p"]
flip_t = 0.5 - base_trust_p   # SP needed to flip Trust to S
flip_v = 0.5 - base_van_p
ax.axhline(flip_t, color="#16A34A", linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)
ax.axhline(flip_v, color="#D97706", linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)
ax.text(len(conds)-0.1, flip_t + 0.005, "Trust flips to S →",
        color="#16A34A", fontsize=8, alpha=0.8, ha='right')
ax.text(len(conds)-0.1, flip_v - 0.018, "Vanilla flips to S →",
        color="#D97706", fontsize=8, alpha=0.8, ha='right')

ax.set_xticks(x)
ax.set_xticklabels(conds, color='white', fontsize=10)
ax.set_ylabel("SP  (ΔP(S) from baseline)", color='white', fontsize=11)
ax.set_title("Repetition Bias on Chinese State Media Dataset\n"
             "Claim: '336 bio-labs' (GT = Not Supported)   ·   Baseline: 1 fact-checker vs 1 state-media",
             color='white', fontsize=12, pad=12)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#475569')
ax.yaxis.label.set_color('white')
ax.grid(axis='y', color='#475569', alpha=0.3, zorder=1)

legend = ax.legend(loc='upper left', framealpha=0.2, labelcolor='white',
                   facecolor='#253761', edgecolor='#475569', fontsize=10)

# Annotations
ax.annotate("Paper finding:\nrepetition flips\npreference",
            xy=(1, max(sp_t[1], sp_v[1])), xytext=(1.5, max(sp_t[1], sp_v[1]) + 0.06),
            color='white', fontsize=8, alpha=0.7,
            arrowprops=dict(arrowstyle='->', color='white', alpha=0.5))

fig.tight_layout()
fig.savefig("/u/samit/fig1_repetition_bias_controlled.png", dpi=150,
            bbox_inches='tight', facecolor='#1C2B4A')
print("Saved fig1")
plt.close()

# ── Figure 2: Full dataset SP distribution ────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor('#1C2B4A')

p2 = data["part2_full_dataset"]
labels_order = ["factual", "false", "biased", "biased/false"]
label_colors = {"factual": "#16A34A", "false": "#DC2626",
                "biased": "#D97706", "biased/false": "#C084FC"}

for ax_idx, (key, system_label, col) in enumerate([
    ("sp_trust_repeat",   "Trust Fusion",          "#16A34A"),
    ("sp_van_repeat",     "Vanilla FactReasoner",   "#D97706"),
]):
    ax = axes[ax_idx]
    ax.set_facecolor('#1C2B4A')

    by_label = {}
    for r in p2:
        lbl = r["raw_label"]
        by_label.setdefault(lbl, []).append(r[key])

    xs, ys, cs, ls = [], [], [], []
    for i, lbl in enumerate(labels_order):
        if lbl not in by_label: continue
        vals = by_label[lbl]
        for j, v in enumerate(vals):
            xs.append(i + np.random.uniform(-0.15, 0.15))
            ys.append(v)
            cs.append(label_colors[lbl])
            ls.append(lbl)

    ax.scatter(xs, ys, c=cs, s=60, alpha=0.8, zorder=3, edgecolors='white', linewidths=0.3)

    # Mean per label
    for i, lbl in enumerate(labels_order):
        if lbl not in by_label: continue
        mean_val = np.mean(by_label[lbl])
        ax.plot([i-0.3, i+0.3], [mean_val, mean_val],
                color=label_colors[lbl], linewidth=2.5, zorder=4)
        ax.text(i+0.32, mean_val, f"{mean_val:+.3f}",
                color=label_colors[lbl], fontsize=9, va='center', fontweight='bold')

    ax.axhline(0, color='white', linewidth=0.8, alpha=0.4)
    ax.set_xticks(range(len(labels_order)))
    ax.set_xticklabels(labels_order, color='white', fontsize=11)
    ax.set_ylabel("SP  (ΔP(S) when top entailment repeated)",
                  color='white', fontsize=10)
    ax.set_title(f"{system_label}\nRepetition Bias by Label Type",
                 color='white', fontsize=11, pad=8)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#475569')
    ax.grid(axis='y', color='#475569', alpha=0.3)

    # Annotation
    ax.text(0.02, 0.97,
            "SP > 0 on false = fooled\nSP > 0 on factual = correct confidence",
            transform=ax.transAxes, color='#94A3B8', fontsize=8,
            va='top', alpha=0.8)

fig2.suptitle("Repetition Bias Susceptibility — Full Dataset (29 atoms)\n"
              "SP = shift in P(S) when top entailing context is duplicated",
              color='white', fontsize=12)
fig2.tight_layout()
fig2.savefig("/u/samit/fig2_repetition_bias_full_dataset.png", dpi=150,
             bbox_inches='tight', facecolor='#1C2B4A')
print("Saved fig2")
plt.close()
