#!/usr/bin/env python3
"""Regenerate dark-mode crossover plot with proper dark background."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Parse data
rows = []
with open("artifacts/crossover_sweep/crossover.csv") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("embed_dim"):
            continue
        parts = line.split(",")
        if len(parts) >= 9:
            mf = float(parts[2])
            t = int(parts[3])
            median_ns = float(parts[5])
            rows.append((mf, t, median_ns))

data: dict[tuple[float, int], float] = {}
for mf, t, ns in rows:
    data[(mf, t)] = ns

sizes = sorted(set(mf for mf, _, _ in rows))
baselines = {mf: ns for mf, t, ns in rows if t == 1}

# Dark mode colors — site theme: --bg #090b0e, --surface #0d1014, --text #f3efe5
BG = "#0d1014"
SURFACE = "#12151a"
TEXT = "#f3efe5"
MUTED = "#a6a6ad"
ACCENT = "#9b8fd1"

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(SURFACE)

colors = {2: "#64B5F6", 4: "#81C784", 8: "#FFB74D"}
markers = {2: "s", 4: "D", 8: "^"}

for t in [2, 4, 8]:
    xs, ys = [], []
    for mf in sizes:
        if (mf, t) in data and mf in baselines:
            xs.append(mf)
            ys.append(baselines[mf] / data[(mf, t)])
    ax.plot(xs, ys, marker=markers[t], color=colors[t], linewidth=2, markersize=8,
            label=f"{t} threads")

ax.axhline(y=1.0, color="#666666", linestyle="--", alpha=0.6, label="no speedup")
ax.axhline(y=2.0, color="#444444", linestyle=":", alpha=0.4)
ax.axhline(y=4.0, color="#444444", linestyle=":", alpha=0.4)

ax.set_xlabel("MFLOPs per matmul (Q8_0 decode)", fontsize=12, color=TEXT)
ax.set_ylabel("Speedup vs 1 thread", fontsize=12, color=TEXT)
ax.set_title("Thread-parallelism crossover — matmul_q8_0_decode\n(i5-1135G7, Q8_0, aspect ratio 1:3)", fontsize=13, color=TEXT)
ax.legend(loc="upper left", fontsize=11, facecolor=SURFACE, edgecolor="#333", labelcolor=TEXT)
ax.grid(True, alpha=0.15, color="#555")

ax.tick_params(colors=MUTED)
for spine in ax.spines.values():
    spine.set_color("#333")

ax.axvspan(20, 30, alpha=0.15, color="#EF5350")
ax.annotate("crossover\nthreshold", xy=(25, 1.5), fontsize=9, color="#EF9A9A",
            ha="center", fontstyle="italic")

fig.tight_layout()
fig.savefig("docs/plots/crossover_plot.png", dpi=120, facecolor=fig.get_facecolor())
print("Regenerated docs/plots/crossover_plot.png with dark background")
