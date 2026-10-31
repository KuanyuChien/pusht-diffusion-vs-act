"""Teaching figure: one episode up close, showing how action (blue) and agent position (orange) relate step by step."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/scratch/general/vast/u1470293/pusht/data/lerobot/pusht")
OUT = Path.home() / "projects/pusht/results/phase4"
EPISODE = 205  # shortest demo: 49 steps

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

df = pd.read_parquet(ROOT / "data/chunk-000/file-000.parquet")
e = df[df.episode_index == EPISODE]
s = np.stack(e["observation.state"].to_numpy())
a = np.stack(e["action"].to_numpy())

fig, ax = plt.subplots(figsize=(8, 7))
for t in range(len(e)):  # gray link: action and agent position at the same step
    ax.plot([s[t, 0], a[t, 0]], [s[t, 1], a[t, 1]], color="#b9b8b3", linewidth=0.8, zorder=1)
ax.plot(a[:, 0], a[:, 1], "-o", color=BLUE, linewidth=1.5, markersize=4, label="action (where the mouse pointed)", zorder=2)
ax.plot(s[:, 0], s[:, 1], "-o", color=ORANGE, linewidth=1.5, markersize=4, label="agent position (where the agent was)", zorder=3)
for t in range(0, len(e), 8):
    ax.annotate(f"t={t}", s[t], xytext=(6, 6), textcoords="offset points", fontsize=9, color=INK)
ax.annotate("start", s[0], xytext=(10, -14), textcoords="offset points", fontweight="bold")
ax.annotate("end", s[-1], xytext=(10, -14), textcoords="offset points", fontweight="bold")
ax.set_xlim(a[:, 0].min() - 25, a[:, 0].max() + 40)
ax.set_ylim(a[:, 1].max() + 25, a[:, 1].min() - 25)  # y down, like the screen
ax.set_aspect("equal")
ax.set(xlabel="x (px)", ylabel="y (px)")
ax.set_title(f"Episode {EPISODE} up close: one dot = one step (0.1 s); gray line = gap at the same step",
             fontweight="bold", fontsize=11)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "5_trajectory_zoom.png", dpi=150)
print("saved", OUT / "5_trajectory_zoom.png")
