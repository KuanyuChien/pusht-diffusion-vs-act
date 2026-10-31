"""Teaching example: pull numbers out of a training log with regex → CSV → plot.

Run: python regex_example.py
A simpler version of training_log.py, written to be read top to bottom.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOG = Path.home() / "projects/pusht/logs/train_dp-2260235.out"
OUT = Path.home() / "projects/pusht/results/regex_example"
OUT.mkdir(parents=True, exist_ok=True)

# ── Step 1: try the regex on ONE line first ──────────────────────────────────
line = "INFO 2026-10-27 18:16:20 ot_train.py:972 step:500 smpl:32K ep:257 epch:1.25 loss:0.071 grdn:2.107 lr:9.0e-05"

m = re.search(r"loss:([0-9.]+)", line)
#              loss:              → the literal text "loss:"
#                   (      )      → capture group: the part we want back
#                    [0-9.]       → one character that is a digit or a dot
#                          +      → one or more of them
print("Step 1  whole match:", m.group(0), "| captured:", m.group(1), "| as float:", float(m.group(1)))

lr = re.search(r"lr:([0-9.eE+-]+)", line).group(1)  # lr uses scientific notation: 9.0e-05
print("Step 1  lr:", lr, "→", float(lr))

# ── Step 2: apply it to every line of the log ────────────────────────────────
losses = []
for text in LOG.read_text().splitlines():
    m = re.search(r"loss:([0-9.]+)", text)
    if m:  # None when the line has no "loss:" (config lines, eval lines, ...)
        losses.append(float(m.group(1)))
print(f"Step 2  found {len(losses)} loss values; first 3: {losses[:3]}, last: {losses[-1]}")

# Eval lines look like: ... Suite overall aggregated: {'avg_sum_reward': ..., 'pc_success': 66.0, ...}
success = [float(x) for x in re.findall(r"overall aggregated.*?'pc_success': ([0-9.]+)", LOG.read_text())]
#                                          .*?  → any characters, as few as possible, up to the next part
#   re.findall returns every captured value in the whole text at once
print(f"Step 2  found {len(success)} eval results: {success}")

# ── Step 3: put them in tables and save as CSV ───────────────────────────────
train = pd.DataFrame({"step": [(i + 1) * 100 for i in range(len(losses))], "loss": losses})  # one line per 100 steps
evals = pd.DataFrame({"step": [(i + 1) * 25000 for i in range(len(success))], "pc_success": success})
train.to_csv(OUT / "loss.csv", index=False)
evals.to_csv(OUT / "success.csv", index=False)
print("Step 3  saved CSVs; preview:\n", evals.to_string(index=False))

# ── Step 4: plot ─────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
a1.plot(train.step, train.loss, linewidth=1)
a1.set(yscale="log", ylabel="loss", title="Training loss")
a2.plot(evals.step, evals.pc_success, "-o")
a2.set(ylim=(0, 100), xlabel="step", ylabel="success (%)", title="Eval success")
fig.tight_layout()
fig.savefig(OUT / "curves.png", dpi=120)
print("Step 4  saved", OUT / "curves.png")
