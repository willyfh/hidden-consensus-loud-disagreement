"""Generate Part 1 figures. Palette + mark specs per the dataviz skill:
categorical slot 1 (blue #2a78d6) / slot 2 (orange #eb6834), validated for CVD safety;
thin marks, rounded bar ends, muted axis/gridlines, direct labels over legend where it fits.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"      # slot 1 -- concrete
ORANGE = "#eb6834"    # slot 2 -- abstract

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

df = pd.read_csv(EXPERIMENT_DIR / "results_with_canonical.csv")
SPECIFICITY = {"H1": "abstract", "H2": "concrete", "H3": "abstract",
               "H4": "abstract", "H5": "concrete", "H6": "abstract"}
HYP_ORDER = ["H2", "H5", "H1", "H3", "H4", "H6"]  # concrete first, then abstract

# ---------------------------------------------------------------------------
# Figure 1: measure-choice diversity per hypothesis (collapsed across arms)
# ---------------------------------------------------------------------------
rows = []
for hyp in HYP_ORDER:
    sub = df[df["hypothesis_id"] == hyp]
    rows.append({
        "hyp": hyp,
        "specificity": SPECIFICITY[hyp],
        "unique_canonical_per_20": sub["metric_canonical"].nunique(),
    })
fig1_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 4.2))
colors = [BLUE if s == "concrete" else ORANGE for s in fig1_df["specificity"]]
x = np.arange(len(fig1_df))
bar_width = 0.6
max_val = 20

bars = ax.bar(x, fig1_df["unique_canonical_per_20"], width=bar_width, color=colors,
              zorder=3)
# 4px rounded data-end, square at baseline -> approximate with rounded top via boxstyle
for bar in bars:
    bar.set_zorder(3)

for i, v in enumerate(fig1_df["unique_canonical_per_20"]):
    ax.text(i, v + 0.4, str(v), ha="center", va="bottom", fontsize=9.5,
             color=TEXT_PRIMARY, fontweight="medium")

ax.set_xticks(x)
ax.set_xticklabels(fig1_df["hyp"], fontsize=10)
ax.set_ylabel("Unique measure framings (out of 20 replicates)", fontsize=9.5, color=TEXT_SECONDARY)
ax.set_ylim(0, max_val + 2)
ax.set_title("Measure-choice diversity is far higher for abstractly-worded questions",
             fontsize=11, color=TEXT_PRIMARY, pad=14, loc="left")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.yaxis.grid(True, color=BASELINE, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0)

legend_handles = [
    mpatches.Patch(color=BLUE, label="Concrete (measure specified)"),
    mpatches.Patch(color=ORANGE, label="Abstract (measure unspecified)"),
]
ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9.5,
          bbox_to_anchor=(0.0, 1.0))

fig.tight_layout()
fig1_out = EXPERIMENT_DIR / "paper_notes"
fig1_out.mkdir(parents=True, exist_ok=True)
fig.savefig(fig1_out / "fig1_diversity.png", dpi=200)
fig.savefig(fig1_out / "fig1_diversity.pdf")
print("Saved fig1_diversity.{png,pdf}")

# ---------------------------------------------------------------------------
# Figure 2: H2 / H5 replicate-level value distributions by arm
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=False)

for ax, hyp, title in zip(
    axes, ["H2", "H5"],
    ["H2 (concrete): RF − LogReg ROC-AUC", "H5 (concrete): SMOTE − none, minority F1"],
):
    sub = df[df["hypothesis_id"] == hyp]
    for arm_i, (arm, label, color) in enumerate([
        (False, "No-verify", BLUE), (True, "Verify", ORANGE)
    ]):
        vals = sub[sub["verify_arm"] == arm]["primary_metric_value"].dropna().values
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), arm_i) + jitter, vals, s=34, color=color,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3, alpha=0.9)
        mean_v = vals.mean()
        ax.plot([arm_i - 0.22, arm_i + 0.22], [mean_v, mean_v], color=TEXT_PRIMARY,
                linewidth=2, zorder=4, solid_capstyle="round")

    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No-verify", "Verify"], fontsize=10)
    ax.set_title(title, fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(axis="both", length=0)
    ax.set_xlim(-0.5, 1.5)

axes[0].set_ylabel("Reported effect (points)", fontsize=9.5, color=TEXT_SECONDARY)
fig.suptitle("Verification tightens a robust effect (H2) and shrinks a fragile one toward zero (H5)",
             fontsize=11, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.02)
fig.tight_layout()
fig.savefig(fig1_out / "fig2_verify_arm.png", dpi=200, bbox_inches="tight")
fig.savefig(fig1_out / "fig2_verify_arm.pdf", bbox_inches="tight")
print("Saved fig2_verify_arm.{png,pdf}")
