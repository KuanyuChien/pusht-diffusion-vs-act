"""Success vs. training step: 500-episode evals (eval_ckpt.slurm) vs. the 50-episode evals logged during training.

Usage: python checkpoint_curve.py <run_name> [baseline_success_pct]
  e.g. python checkpoint_curve.py dp_run1 59.4
Needs results/<run>/eval_500ep_step*.json and results/<run>/eval_during_training.csv (from training_log.py).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN = sys.argv[1]
baseline = float(sys.argv[2]) if len(sys.argv) > 2 else None
RES = Path.home() / "projects/pusht/results" / RUN

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})

rows = []
for f in RES.glob("eval_500ep_step*.json"):
    o = json.loads(f.read_text())["overall"]
    rows.append({"step": int(f.stem.split("step")[1]), "pc_success": o["pc_success"],
                 "ci_low": o["pc_success_ci95"][0], "ci_high": o["pc_success_ci95"][1],
                 "avg_max_reward": o["avg_max_reward"]})
e500 = pd.DataFrame(rows).sort_values("step")
e50 = pd.read_csv(RES / "eval_during_training.csv")
table = e500.merge(e50[["step", "pc_success"]].rename(columns={"pc_success": "pc_success_50ep"}), on="step", how="left")
table.to_csv(RES / "success_vs_step.csv", index=False)
print(table.round(1).to_string(index=False))

k = 1000
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(e500.step / k, e500.ci_low, e500.ci_high, color=ORANGE, alpha=0.15, linewidth=0,
                label="95% CI, 500 episodes")
ax.plot(e500.step / k, e500.pc_success, "-o", color=ORANGE, linewidth=2, markersize=8, label="500 episodes (eval_ckpt)")
ax.plot(e50.step / k, e50.pc_success, "--o", color=BLUE, linewidth=1.5, markersize=7, label="50 episodes (during training)")
for s, p, p50 in zip(e500.step / k, e500.pc_success, table.pc_success_50ep):
    above = pd.notna(p50) and p > p50  # put the label on the side away from the 50-episode point
    ax.annotate(f"{p:.1f}", (s, p), xytext=(0, 10 if above else -16), textcoords="offset points",
                ha="center", color=INK_2, fontsize=9)
if baseline is not None:
    ax.axhline(baseline, color=INK_2, linewidth=1, linestyle=":")
    ax.text(e500.step.max() / k, baseline - 3, f"official model, our pipeline: {baseline}%", color=INK_2,
            fontsize=9, ha="right", va="top")
ax.set(ylim=(0, 100), xlabel="training step (thousands)", ylabel="eval success (%)",
       title=f"{RUN}: success vs. training step (50-episode evals over- and under-shoot)")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(RES / "success_vs_step.png", dpi=150)
print("saved", RES / "success_vs_step.png")
