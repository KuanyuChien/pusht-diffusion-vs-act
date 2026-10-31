"""Phase 4: explore the lerobot/pusht dataset. Writes summary.json and 4 figures to results/phase4/.

Runs on the login node: the dataset is 7 MB and only 12 video frames get decoded.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/scratch/general/vast/u1470293/pusht/data/lerobot/pusht")
OUT = Path.home() / "projects/pusht/results/phase4"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (light mode): categorical slots 1-2, text inks, surface
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})

# ── Load all frames (one parquet file) ───────────────────────────────────────
df = pd.read_parquet(ROOT / "data/chunk-000/file-000.parquet")
state = np.stack(df["observation.state"].to_numpy())  # agent (x, y)
action = np.stack(df["action"].to_numpy())            # target (x, y) the agent moves toward
df["sx"], df["sy"], df["ax"], df["ay"] = state[:, 0], state[:, 1], action[:, 0], action[:, 1]
df["gap"] = np.hypot(df.ax - df.sx, df.ay - df.sy)    # how far ahead of the agent the target is

ep = df.groupby("episode_index").agg(
    length=("frame_index", "size"),
    max_reward=("next.reward", "max"),
    final_reward=("next.reward", "last"),
    success=("next.success", "any"),
)
stats = json.loads((ROOT / "meta/stats.json").read_text())

summary = {
    "episodes": int(len(ep)),
    "frames": int(len(df)),
    "fps": 10,
    "episode_length": {k: float(getattr(ep.length, k)()) for k in ["min", "median", "mean", "max"]},
    "episodes_reaching_success": int(ep.success.sum()),
    "episode_max_reward_mean": float(ep.max_reward.mean()),
    "action_minus_state_distance_px": {"median": float(df.gap.median()), "p90": float(df.gap.quantile(0.9))},
    "action_range": {"min": stats["action"]["min"], "max": stats["action"]["max"]},
    "state_range": {"min": stats["observation.state"]["min"], "max": stats["observation.state"]["max"]},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

# ── ① Episode length distribution ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(ep.length, bins=np.arange(0, ep.length.max() + 10, 10), color=BLUE, edgecolor=SURFACE, linewidth=2)
med = ep.length.median()
ax.axvline(med, color=INK_2, linewidth=1, linestyle="--")
ax.text(med + 3, ax.get_ylim()[1] * 0.92, f"median {med:.0f} steps ({med / 10:.1f} s)", color=INK_2)
ax.set(title="Episode length: 206 human demonstrations", xlabel="steps per episode (10 steps = 1 s)",
       ylabel="episodes")
fig.tight_layout()
fig.savefig(OUT / "1_episode_lengths.png", dpi=150)

# ── ② Trajectories: action vs agent position, 6 episodes ─────────────────────
picks = ep.length.sort_values().index[np.linspace(0, len(ep) - 1, 6).astype(int)]  # short → long
fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
for a, i in zip(axes.flat, picks):
    e = df[df.episode_index == i]
    a.plot(e.ax, e.ay, color=BLUE, linewidth=1.5, label="action (target point)")
    a.plot(e.sx, e.sy, color=ORANGE, linewidth=1.5, label="agent position")
    a.scatter(e.sx.iloc[0], e.sy.iloc[0], s=40, color=INK, zorder=3, label="start")
    a.set(title=f"episode {i}: {len(e)} steps", xlim=(0, 512), ylim=(512, 0), aspect="equal")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.suptitle("Action leads, agent follows (512×512 workspace, y down)", y=0.985, fontweight="bold")
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "2_trajectories.png", dpi=150)

# ── ③ Where actions go: density over the workspace ──────────────────────────
fig, ax = plt.subplots(figsize=(5.4, 5))
h = ax.hist2d(df.ax, df.ay, bins=48, range=[[0, 512], [0, 512]], cmap="Blues", cmin=1)
ax.set(title="Action density (all 25,650 frames)", xlim=(0, 512), ylim=(512, 0), aspect="equal",
       xlabel="x (px)", ylabel="y (px)")
ax.grid(False)
fig.colorbar(h[3], ax=ax, label="frames per cell", shrink=0.8)
fig.tight_layout()
fig.savefig(OUT / "3_action_density.png", dpi=150)

# ── ④ What the policy sees: 96×96 frames at start / middle / end ─────────────
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/pusht", root=ROOT)
ep_from = ds.meta.episodes["dataset_from_index"]
ep_to = ds.meta.episodes["dataset_to_index"]
show = list(picks[[0, 2, 3, 5]])
fig, axes = plt.subplots(len(show), 3, figsize=(6.6, 2.3 * len(show)))
for row, i in zip(axes, show):
    lo, hi = ep_from[i], ep_to[i] - 1
    for a, (name, idx) in zip(row, [("start", lo), ("middle", (lo + hi) // 2), ("end", hi)]):
        a.imshow(ds[idx]["observation.image"].permute(1, 2, 0).numpy())
        a.set_title(f"ep {i} · {name}", fontsize=9, fontweight="normal")
        a.axis("off")
fig.suptitle("Policy input: 96×96 RGB (gray T = block, green T = target, blue = agent)",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "4_frames.png", dpi=150)

print("saved figures to", OUT)
