"""2x3 demo video: top row = DP, bottom row = ACT, each column = the same starting state (same eval seed).

Usage: python make_video_grid.py
Short episodes freeze on their last frame until the longest one ends. Played at 2x speed.
"""

import json
import subprocess
from pathlib import Path

EVAL = Path("/scratch/general/vast/u1470293/pusht/outputs/eval")
RES = Path.home() / "projects/pusht/results"
OUT = RES / "demo/dp_vs_act_grid.mp4"
FONT = Path.home() / "envs/lerobot/fonts/DejaVuSans.ttf"
ROWS = [  # (label, results run dir, eval output dir with videos)
    ("DP", "dp_run1", "dp_run1_200000_500ep_2260931"),
    ("ACT", "act_run1", "act_run1_200000_500ep_2261507"),
]
EPISODES = [4, 2, 0]  # both succeed / DP succeeds, ACT fails / DP near miss, ACT succeeds
SPEED = 2.0

OUT.parent.mkdir(parents=True, exist_ok=True)
inputs, filters, labels = [], [], []
for label, run, eval_dir in ROWS:
    m = json.loads((RES / run / "eval_500ep_step200000.json").read_text())["per_task"][0]["metrics"]
    for ep in EPISODES:
        video = EVAL / eval_dir / f"videos/pusht_0/eval_episode_{ep}.mp4"
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(video)], capture_output=True, text=True).stdout)
        result = "success" if m["successes"][ep] else f"fail (max {m['max_rewards'][ep]:.2f})"
        text = f"{label}  episode {ep}  {result}  {dur:.1f} s"
        labels.append(text)
        i = len(filters)  # ffmpeg input index (inputs holds two items per video: "-i", path)
        inputs += ["-i", str(video)]
        filters.append(
            f"[{i}:v]tpad=stop_mode=clone:stop_duration=30,trim=duration=30,setpts=PTS/{SPEED},"
            f"drawtext=fontfile={FONT}:text='{text}':x=8:y=8:fontsize=15:fontcolor=white:"
            f"box=1:boxcolor=black@0.6:boxborderw=5[v{i}]"
        )

layout = "0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0"  # 3 across, 2 down
graph = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(6)) + f"xstack=inputs=6:layout={layout}[out]"
cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(OUT)]
subprocess.run(cmd, check=True)
print("\n".join(labels))
print("saved", OUT)
