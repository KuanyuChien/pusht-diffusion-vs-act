"""DP vs ACT: 500-episode success vs. training step, from results/<run>/success_vs_step.csv (checkpoint_curve.py).

Usage: python compare_policies.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path.home() / "projects/pusht/results"
RUNS = [("dp_run1", "Diffusion Policy", "#2a78d6"), ("act_run1", "ACT", "#eb6834")]  # categorical slots 1, 2
BASELINE = 59.4

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})

fig, ax = plt.subplots(figsize=(9, 5))
for run, label, color in RUNS:
    d = pd.read_csv(RES / run / "success_vs_step.csv").sort_values("step")
    ax.fill_between(d.step / 1000, d.ci_low, d.ci_high, color=color, alpha=0.15, linewidth=0)
    ax.plot(d.step / 1000, d.pc_success, "-o", color=color, linewidth=2, markersize=7, label=label)
    last = d.iloc[-1]
    ax.annotate(f"{label}: {last.pc_success:.1f}%", (last.step / 1000, last.pc_success), xytext=(-8, 12),
                textcoords="offset points", ha="right", color=INK)
ax.axhline(BASELINE, color=INK_2, linewidth=1, linestyle=":")
ax.text(26, BASELINE + 2, f"official DP, our pipeline: {BASELINE}%", color=INK_2, fontsize=9)
ax.set(ylim=(0, 100), xlabel="training step (thousands)", ylabel="eval success (%), 500 episodes",
       title="PushT: Diffusion Policy vs ACT (same data, steps, seed, eval; shaded = 95% CI)")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(RES / "dp_vs_act.png", dpi=150)
print("saved", RES / "dp_vs_act.png")
