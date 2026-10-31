"""Parse a lerobot-train log into tables + a training-curve figure.

Usage: python training_log.py <log file> <out dir> [baseline_success_pct]
  e.g. python training_log.py ~/projects/pusht/logs/train_dp-2260235.out ~/projects/pusht/results/dp_run1 59.4
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

log, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
baseline = float(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})

num = r"([-+0-9.eE]+)"
rows, evals, saves = [], [], []
for line in log.read_text().splitlines():
    if " step:" in line and " loss:" in line:
        # "step:" is rounded (e.g. 6K), so the step is the line count: one line every log_freq=100 steps
        r = {"step": (len(rows) + 1) * 100, "time": datetime.strptime(line[5:24], "%Y-%m-%d %H:%M:%S")}
        for k in ["loss", "grdn", "lr", "data_s", "updt_s", "step_s", "mem_gb"]:
            r[k] = float(re.search(rf"\b{k}:{num}", line).group(1))
        rows.append(r)
    elif "Checkpoint policy after step" in line:
        saves.append(int(line.rsplit(" ", 1)[1]))
    elif "Suite overall aggregated" in line:
        g = lambda k: float(re.search(rf"'{k}': {num}", line).group(1))
        ci = re.search(rf"'pc_success_ci95': \[{num}, {num}\]", line)
        evals.append({"step": saves[-1], "pc_success": g("pc_success"), "ci_low": float(ci.group(1)),
                      "ci_high": float(ci.group(2)), "avg_max_reward": g("avg_max_reward"),
                      "n_episodes": int(g("n_episodes")), "eval_s": g("eval_s")})

train, ev = pd.DataFrame(rows), pd.DataFrame(evals)
train.to_csv(out / "train_log.csv", index=False)
ev.to_csv(out / "eval_during_training.csv", index=False)

wall = (train.time.iloc[-1] - train.time.iloc[0]).total_seconds()
summary = {
    "steps": int(train.step.iloc[-1]),
    "wall_clock_h": round(wall / 3600, 2),
    "median_step_s": float(train.step_s.median()),
    "eval_total_min": round(ev.eval_s.sum() / 60, 1),
    "loss_first_last": [float(train.loss.iloc[0]), float(train.loss.iloc[-1])],
    "loss_at": {f"{s // 1000}k": float(train.loc[train.step == s, "loss"].iloc[0])
                for s in [1000, 10000, 25000, 50000, 100000, 150000, 200000] if s in set(train.step)},
    "max_mem_gb": float(train.mem_gb.max()),
    "eval": evals,
}
(out / "training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "eval"}, indent=2))
print(ev.to_string(index=False))

# ── Figure: loss / eval success / learning rate, shared x (three charts, never a dual axis) ──
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 3, 1.4]})
k = 1000
a1.plot(train.step / k, train.loss, color=BLUE, linewidth=0.8, alpha=0.35)
a1.plot(train.step / k, train.loss.rolling(20, center=True).mean(), color=BLUE, linewidth=2)
a1.set(yscale="log", ylabel="training loss (log)", title="Training loss: keeps falling the whole run")
a1.text(train.step.iloc[-1] / k, train.loss.iloc[-1] * 1.6, f"{train.loss.iloc[-1]:.3f}", ha="right", color=INK_2)

a2.fill_between(ev.step / k, ev.ci_low, ev.ci_high, color=ORANGE, alpha=0.15, linewidth=0,
                label=f"95% CI ({ev.n_episodes.iloc[0]} episodes each)")
a2.plot(ev.step / k, ev.pc_success, "-o", color=ORANGE, linewidth=2, markersize=7, label="success rate")
for s, p in zip(ev.step / k, ev.pc_success):
    a2.annotate(f"{p:.0f}%", (s, p), xytext=(0, 9), textcoords="offset points", ha="center", color=INK_2)
if baseline is not None:
    a2.axhline(baseline, color=INK_2, linewidth=1, linestyle="--")
    a2.text(train.step.iloc[-1] / k, baseline - 5, f"official model, our pipeline (500 ep): {baseline}%",
            color=INK_2, fontsize=9, ha="right", va="top")
a2.set(ylim=(0, 100), ylabel="eval success (%)", title="Eval success: plateaus after ~100k steps")
a2.legend(loc="lower right", frameon=False)

a3.plot(train.step / k, train.lr, color=INK_2, linewidth=1.5)
a3.set(ylabel="learning rate", xlabel="training step (thousands)", title="Learning rate: warmup, then cosine decay")
a3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.tight_layout()
fig.savefig(out / "training_curves.png", dpi=150)
print("saved", out / "training_curves.png")
