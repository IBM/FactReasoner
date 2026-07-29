"""
plot_conditions.py
==================
Generates downloadable PNG figures:
  Fig 1: Schuster et al. Figure 8 style strip plot — SP by condition × system
  Fig 2: Per-atom strip plot — SP under 4 conditions for each atom
  Fig 3: Aggregate SP bar chart by label type

Run on server:
    python3 /u/samit/FactReasoner/scripts/plot_conditions.py
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

IN_JSON  = "/u/samit/real_dataset_sp_results.json"
OUT_DIR  = "/u/samit/"

with open(IN_JSON) as f:
    data = json.load(f)

sp_data   = data["sp_conditions"]
judge_acc = data["summary"]["judge_acc"]
trust_acc = data["summary"]["trust_acc"]
van_acc   = data["summary"]["vanilla_acc"]

CONDS     = ["baseline", "1tm", "2tm", "repeated"]
COND_LABS = ["Baseline\n(1 govt vs\n1 state-media)", "1-Table\nMajority",
             "2-Table\nMajority", "Repeated\n(same source ×2)"]

# ── Colour / marker scheme ────────────────────────────────────────────────────
TRUST_COL  = "#2a78d6"   # blue circle
VAN_COL    = "#eda100"   # amber triangle-up
GUARD_COL  = "#1baf7a"   # teal diamond
JUDGE_COL  = "#e24b4a"   # red square

LABEL_COLS = {
    "factual":      "#3B6D11",
    "false":        "#A32D2D",
    "biased":       "#BA7517",
    "biased/false": "#534AB7",
}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Schuster et al. Figure 8 style
#  Rows = conditions (Baseline, 1TM, 2TM, Repeated)
#  X    = SP (percentage points, ±50)
#  Dots = 3 systems; one dot per system per row
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5.5))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Aggregate SP per condition per system
def avg_sp(cond, key):
    vals = [r["conditions"][cond][key] for r in sp_data
            if cond in r["conditions"]]
    return np.mean(vals) if vals else 0.0

trust_sps = [avg_sp(c, "sp_trust")   for c in CONDS]
van_sps   = [avg_sp(c, "sp_vanilla") for c in CONDS]
# LLM-as-a-judge: doesn't have per-condition SP, show as flat reference line
judge_sp_overall = (judge_acc - 0.5) * 100  # crude mapping

y_positions = [0, 1, 2, 3]

for yi, cond in enumerate(CONDS):
    # Horizontal row separator
    ax1.axhline(yi - 0.5, color='#e1e0d9', linewidth=0.7, zorder=1)

    # Individual atom dots (jittered vertically within row)
    atom_trust = [r["conditions"][cond]["sp_trust"]   for r in sp_data if cond in r["conditions"]]
    atom_van   = [r["conditions"][cond]["sp_vanilla"] for r in sp_data if cond in r["conditions"]]

    for j, (at, av) in enumerate(zip(atom_trust, atom_van)):
        jitter = np.random.uniform(-0.12, 0.12)
        lbl = sp_data[j]["raw_label"] if j < len(sp_data) else "factual"
        col = LABEL_COLS.get(lbl, "#888")
        ax1.scatter(at, yi + jitter,      color=TRUST_COL, s=55, zorder=4,
                    alpha=0.6, linewidths=0.5, edgecolors='white', marker='o')
        ax1.scatter(av, yi + jitter*0.5,  color=VAN_COL,   s=40, zorder=4,
                    alpha=0.5, linewidths=0.5, edgecolors='white', marker='^')

    # Mean markers (larger, opaque)
    ax1.scatter(trust_sps[yi], yi, color=TRUST_COL, s=140, zorder=5,
                edgecolors='white', linewidths=1.5, marker='o')
    ax1.scatter(van_sps[yi],   yi, color=VAN_COL,   s=110, zorder=5,
                edgecolors='white', linewidths=1.5, marker='^')

    # Mean value labels
    ax1.annotate(f"{trust_sps[yi]:+.1f}",
                 (trust_sps[yi], yi), xytext=(0, 9),
                 textcoords='offset points', ha='center', fontsize=8.5,
                 color=TRUST_COL, fontweight='bold')
    ax1.annotate(f"{van_sps[yi]:+.1f}",
                 (van_sps[yi], yi), xytext=(0, -14),
                 textcoords='offset points', ha='center', fontsize=8.5,
                 color=VAN_COL, fontweight='bold')

# Zero line
ax1.axvline(0, color='#888', linewidth=1, linestyle='--', alpha=0.6, zorder=2)

# Shading: negative = biased toward state media
ax1.axvspan(-55, 0, alpha=0.04, color='red', zorder=0)
ax1.text(-52, 3.6, '← biased toward\nstate media', fontsize=7.5,
         color='#A32D2D', alpha=0.7, va='top')

# Judge reference line
ax1.axvline(judge_sp_overall, color=JUDGE_COL, linewidth=1.5, linestyle=':',
            alpha=0.7, label=f'LLM-as-a-judge (overall, no conditions)')
ax1.text(judge_sp_overall + 1, -0.48, f'Judge\n{judge_sp_overall:+.1f}',
         color=JUDGE_COL, fontsize=7.5, va='bottom')

ax1.set_yticks(y_positions)
ax1.set_yticklabels(COND_LABS, fontsize=10)
ax1.set_xlabel("SP̂  (ΔP towards S — positive = prefers institutional source)",
               fontsize=10)
ax1.set_xlim(-55, 55)
ax1.set_ylim(-0.65, 3.65)
ax1.set_title("Source Preference under Repetition Bias Conditions\n"
              "Chinese State Media Dataset  ·  9 mixed-evidence atoms",
              fontsize=11, pad=10)

# Right-side labels
ax2r = ax1.twinx()
ax2r.set_ylim(-0.65, 3.65)
ax2r.set_yticks(y_positions)
ax2r.set_yticklabels(
    ["1 govt vs 1 state-media",
     "1 govt vs 2 social (merged)",
     "1 govt vs 2 social (separate)",
     "1 govt vs same source ×2"],
    fontsize=8, color='#898781'
)

# Legend
handles = [
    mlines.Line2D([],[], color=TRUST_COL, marker='o', ms=8, ls='none',
                  label=f'Trust Fusion  ({trust_acc*100:.1f}% acc)'),
    mlines.Line2D([],[], color=VAN_COL, marker='^', ms=8, ls='none',
                  label=f'Vanilla FR  ({van_acc*100:.1f}% acc)'),
    mlines.Line2D([],[], color=JUDGE_COL, lw=1.5, ls=':',
                  label=f'LLM-as-a-judge  ({judge_acc*100:.1f}% acc)'),
]
ax1.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(axis='x', color='#e1e0d9', linewidth=0.5, zorder=1)
ax1.spines[['top','right']].set_visible(False)

fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}fig_conditions_strip.png", dpi=180, bbox_inches='tight')
print(f"Saved fig1 → {OUT_DIR}fig_conditions_strip.png")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Per-atom trajectory: SP across 4 conditions
#  Each atom = one colored line; color = label type
#  Shows how SP evolves from Baseline → 1TM → 2TM → Repeated
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig2.patch.set_facecolor('white')

for ax_i, (key, sys_label) in enumerate([
    ("sp_trust",   f"Trust Fusion  ({trust_acc*100:.1f}% acc on 9 atoms)"),
    ("sp_vanilla", f"Vanilla FactReasoner  ({van_acc*100:.1f}% acc on 9 atoms)"),
]):
    ax = axes2[ax_i]
    ax.set_facecolor('white')

    for r in sp_data:
        lbl = r["raw_label"]
        col = LABEL_COLS.get(lbl, "#888")
        ys  = [r["conditions"][c][key] for c in CONDS]
        xs  = [0, 1, 2, 3]
        ax.plot(xs, ys, color=col, linewidth=1.5, alpha=0.7, marker='o',
                markersize=6, markerfacecolor=col, markeredgecolor='white',
                markeredgewidth=1)
        # Label the atom at the end
        ax.annotate(
            f"{r['account'][:10]}\n({r['ground_truth']})",
            (3, ys[-1]), xytext=(5, 0), textcoords='offset points',
            fontsize=7, color=col, va='center'
        )

    ax.axhline(0, color='#888', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhspan(-60, 0, alpha=0.04, color='red')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(["Baseline", "1TM", "2TM", "Repeated"], fontsize=9)
    ax.set_ylabel("SP̂  (pp)" if ax_i==0 else "", fontsize=10)
    ax.set_title(sys_label, fontsize=10, pad=8)
    ax.set_ylim(-60, 60)
    ax.set_xlim(-0.2, 3.8)
    ax.grid(color='#e1e0d9', linewidth=0.5)
    ax.spines[['top','right']].set_visible(False)

# Legend
legend_handles = [mpatches.Patch(color=c, label=l)
                  for l,c in LABEL_COLS.items()]
axes2[0].legend(handles=legend_handles, fontsize=9, title="Label",
                loc='lower left', framealpha=0.9)

fig2.suptitle("Per-Atom SP Trajectory Across 4 Conditions\n"
              "Each line = one atom  ·  Red zone = biased toward state media",
              fontsize=11)
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}fig_conditions_trajectories.png", dpi=180, bbox_inches='tight')
print(f"Saved fig2 → {OUT_DIR}fig_conditions_trajectories.png")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Aggregate SP by label type × condition (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
labels_present = list(dict.fromkeys(r["raw_label"] for r in sp_data))
fig3, ax3 = plt.subplots(figsize=(9, 4.5))
fig3.patch.set_facecolor('white')
ax3.set_facecolor('white')

x    = np.arange(len(CONDS))
w    = 0.22
offsets = np.linspace(-(len(labels_present)-1)*w/2,
                       (len(labels_present)-1)*w/2,
                       len(labels_present))

for i, lbl in enumerate(labels_present):
    rs  = [r for r in sp_data if r["raw_label"] == lbl]
    sps = [np.mean([r["conditions"][c]["sp_trust"] for r in rs]) for c in CONDS]
    col = LABEL_COLS.get(lbl, "#888")
    bars = ax3.bar(x + offsets[i], sps, w*0.9, color=col, alpha=0.8,
                   label=f"{lbl} (n={len(rs)})", zorder=3)
    for bar, val in zip(bars, sps):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 val + (1.5 if val >= 0 else -3.5),
                 f"{val:+.1f}", ha='center', va='bottom' if val>=0 else 'top',
                 fontsize=7.5, color=col, fontweight='bold')

ax3.axhline(0, color='#888', linewidth=1, linestyle='--', alpha=0.6)
ax3.axhspan(-60, 0, alpha=0.04, color='red')
ax3.text(-0.45, -52, '← biased toward state media', fontsize=7.5,
         color='#A32D2D', alpha=0.7)

# LLM-as-a-judge reference
ax3.axhline(judge_sp_overall, color=JUDGE_COL, linewidth=1.5, linestyle=':',
            alpha=0.8, label=f'LLM-as-a-judge ({judge_acc*100:.1f}% acc overall)')

ax3.set_xticks(x)
ax3.set_xticklabels(["Baseline", "1-Table Majority", "2-Table Majority",
                     "Repeated (same source ×2)"], fontsize=10)
ax3.set_ylabel("Average SP̂ (Trust Fusion, pp)", fontsize=10)
ax3.set_title("Average Source Preference by Label Type × Condition\n"
              "Trust Fusion  ·  9 mixed-evidence atoms from Chinese state media dataset",
              fontsize=11, pad=10)
ax3.set_ylim(-60, 60)
ax3.legend(fontsize=9, framealpha=0.9, loc='upper left')
ax3.grid(axis='y', color='#e1e0d9', linewidth=0.5, zorder=1)
ax3.spines[['top','right']].set_visible(False)

fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}fig_conditions_by_label.png", dpi=180, bbox_inches='tight')
print(f"Saved fig3 → {OUT_DIR}fig_conditions_by_label.png")
plt.close(fig3)

print("\nAll figures saved. Transfer with:")
print(f"  scp samit@ccc-login5.pok.ibm.com:/u/samit/fig_conditions_*.png ~/Desktop/")
