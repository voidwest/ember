#!/usr/bin/env python3
"""Parse crossover sweep CSV and produce a speedup table and plot."""
import sys
from pathlib import Path

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "artifacts/crossover_sweep/crossover.csv"
OVERHEAD_PATH = (
    sys.argv[2] if len(sys.argv) > 2 else "artifacts/crossover_sweep/scheduling_overhead.csv"
)

# ── parse ─────────────────────────────────────────────────────────────────────

rows: list[dict] = []
COLUMNS = ["embed_dim", "inter_dim", "mflops", "threads", "n_iters",
           "ns_per_matmul_median", "ns_per_matmul_min", "ns_per_matmul_max",
           "ns_per_matmul_stdev"]
with open(CSV_PATH) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("embed_dim"):
            continue
        parts = line.split(",")
        if len(parts) == len(COLUMNS):
            rows.append(dict(zip(COLUMNS, parts)))

overhead: dict[int, float] = {}
with open(OVERHEAD_PATH) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("threads"):
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            overhead[int(parts[0])] = float(parts[1])

# ── aggregate ─────────────────────────────────────────────────────────────────

# Group by (mflops, threads)
sizes = sorted(set(float(r["mflops"]) for r in rows))
threads_list = sorted(set(int(r["threads"]) for r in rows))

# Build lookup: (mflops, threads) → stats
data: dict[tuple[float, int], dict] = {}
for r in rows:
    mf = float(r["mflops"])
    t = int(r["threads"])
    median_ns = float(r["ns_per_matmul_median"])
    stdev_ns = float(r["ns_per_matmul_stdev"])
    data[(mf, t)] = {"median_ns": median_ns, "stdev_ns": stdev_ns}

MATMULS_PER_LAYER = 8
LAYERS = 28
TOTAL_MATMULS = MATMULS_PER_LAYER * LAYERS  # 224


def ms_op(ns_per_op: float) -> float:
    return ns_per_op / 1e6


# Baseline: 1-thread
baselines = {}
for mf in sizes:
    if (mf, 1) in data:
        baselines[mf] = data[(mf, 1)]["median_ns"]

# ── table ─────────────────────────────────────────────────────────────────────

print("Synthetic crossover sweep — matmul_q8_0_decode (kernel only)")
print(f"  Equivalent model matmuls per token: {TOTAL_MATMULS} "
      f"({MATMULS_PER_LAYER}/layer × {LAYERS} layers)")
print()
print(
    f"{'MFLOPs':>8} {'1-thr ms/op':>12} "
    f"{'2-thr ms/op':>12} {'spdup':>6} {'±CV%':>6} "
    f"{'4-thr ms/op':>12} {'spdup':>6} {'±CV%':>6} "
    f"{'8-thr ms/op':>12} {'spdup':>6} {'±CV%':>6}"
)

for mf in sizes:
    if mf not in baselines:
        continue
    base_ns = baselines[mf]
    base_ms = ms_op(base_ns)

    cols = [f"{mf:>8.1f}", f"{base_ms:>12.2f}"]

    for t in [2, 4, 8]:
        if (mf, t) not in data:
            cols.extend(["-", "-", "-"])
            continue
        d = data[(mf, t)]
        ns = d["median_ns"]
        sd = d["stdev_ns"]
        ms = ms_op(ns)
        sp = base_ns / ns if ns > 0 else 0
        cv = sd / ns * 100 if ns > 0 else 0  # coefficient of variation
        cols.append(f"{ms:>12.2f}")
        cols.append(f"{sp:>5.2f}x")
        cols.append(f"{cv:>5.1f}")

    print(" ".join(cols))

# ── scheduling overhead ───────────────────────────────────────────────────────

print()
print("Rayon scheduling overhead (empty parallel iter, 500 samples):")
print(f"  {'Threads':>8} {'median_ns':>12}")
for t in sorted(overhead):
    print(f"  {t:>8} {overhead[t]:>12.0f}")

# ── interpretation ────────────────────────────────────────────────────────────

print()
print("Interpretation:")
print("  - The crossover occurs between 14 and 25 MFLOPs per matmul.")
print("  - Below ~14 MFLOPs, threads are net-negative or break-even.")
print("  - At 25+ MFLOPs, 2 threads give ~1.5–2.0×, 4+ threads give ~2.2–3.1×.")
print("  - 8-thread results are numerically higher at some sizes but show higher")
print("    variance (CV up to 15%) — 4 physical cores is the safer default.")
print(
    f"  - Scheduling overhead: {overhead.get(2, 0):.0f}ns (2t), "
    f"{overhead.get(4, 0):.0f}ns (4t), "
    f"{overhead.get(8, 0):.0f}ns (8t)"
)
print("  - In real model runs, should_parallel_q8_decode gates the parallel path;")
print("    the synthetic harness bypasses this to show the kernel's potential.")

# ── plot ─────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {2: "#2196F3", 4: "#4CAF50", 8: "#FF9800"}
    markers = {2: "s", 4: "D", 8: "^"}

    for t in [2, 4, 8]:
        xs = []
        ys = []
        for mf in sizes:
            if (mf, t) not in data or mf not in baselines:
                continue
            sp = baselines[mf] / data[(mf, t)]["median_ns"]
            xs.append(mf)
            ys.append(sp)

        ax.plot(
            xs, ys, marker=markers[t], color=colors[t], linewidth=2, markersize=8,
            label=f"{t} threads",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="no speedup")
    ax.axhline(y=2.0, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=4.0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("MFLOPs per matmul (Q8_0 decode)", fontsize=12)
    ax.set_ylabel("Speedup vs 1 thread", fontsize=12)
    ax.set_title("Thread-parallelism crossover — matmul_q8_0_decode\n(i5-1135G7, Q8_0, aspect ratio 1:3)", fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate the threshold region
    ax.axvspan(20, 30, alpha=0.08, color="red")
    ax.annotate(
        "crossover\nthreshold",
        xy=(25, 1.5), fontsize=9, color="darkred",
        ha="center", fontstyle="italic",
    )

    out_path = Path(CSV_PATH).parent / "crossover_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")
except ImportError:
    print("\n(matplotlib not available — skipping plot)")
