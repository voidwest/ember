#!/usr/bin/env python3
"""Generate light-mode crossover plot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

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

fig, ax = plt.subplots(figsize=(10, 6))

# Light-mode colors
colors = {2: "#1565C0", 4: "#2E7D32", 8: "#E65100"}
markers = {2: "s", 4: "D", 8: "^"}

for t in [2, 4, 8]:
    xs, ys = [], []
    for mf in sizes:
        if (mf, t) in data and mf in baselines:
            xs.append(mf)
            ys.append(baselines[mf] / data[(mf, t)])
    ax.plot(xs, ys, marker=markers[t], color=colors[t], linewidth=2, markersize=8,
            label=f"{t} threads")

ax.axhline(y=1.0, color="#666666", linestyle="--", alpha=0.5, label="no speedup")
ax.axhline(y=2.0, color="#999999", linestyle=":", alpha=0.3)
ax.axhline(y=4.0, color="#999999", linestyle=":", alpha=0.3)

ax.set_xlabel("MFLOPs per matmul (Q8_0 decode)", fontsize=12, color="#333")
ax.set_ylabel("Speedup vs 1 thread", fontsize=12, color="#333")
ax.set_title("Thread-parallelism crossover — matmul_q8_0_decode\n(i5-1135G7, Q8_0, aspect ratio 1:3)", fontsize=13, color="#222")
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, alpha=0.3, color="#ccc")

# Light background
ax.set_facecolor("#fafaf7")
fig.patch.set_facecolor("#fafaf7")
ax.tick_params(colors="#333")
ax.spines["bottom"].set_color("#999")
ax.spines["top"].set_color("#999")
ax.spines["left"].set_color("#999")
ax.spines["right"].set_color("#999")

# Annotate threshold
ax.axvspan(20, 30, alpha=0.1, color="#c62828")
ax.annotate("crossover\nthreshold", xy=(25, 1.5), fontsize=9, color="#b71c1c",
            ha="center", fontstyle="italic")

fig.tight_layout()
fig.savefig("docs/plots/crossover_plot_light.png", dpi=120, facecolor=fig.get_facecolor())
print("Saved docs/plots/crossover_plot_light.png")
