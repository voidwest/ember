"""Plot the September 12 measurements recorded in the Phase 3 report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.23, top=0.72, wspace=0.85)
    fig.suptitle("Startup work, measured separately", x=0.05, y=0.96,
                 ha="left", fontsize=20, weight="bold")
    fig.text(0.05, 0.85, "Llama-3.2-1B Q8_0 · Intel i5-1135G7 · four workers · lower is better",
             color="#475569", fontsize=11)

    panels = [
        (axes[0], "Model build", ["Cache off", "First cache write", "Warm cache hit"],
         [670, 749, 83], ["670 ms", "749 ms", "83 ms"], 900,
         ["#64748b", "#b7791f", "#0f766e"]),
        (axes[1], "GGUF metadata parse", ["Before", "Validate + skip"],
         [37.7, 10.8], ["37.7 ms", "10.8 ms"], 48,
         ["#64748b", "#0f766e"]),
    ]
    for ax, title, labels, values, annotations, limit, colors in panels:
        ax.barh(labels, values, color=colors, height=0.5, zorder=3)
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.set_title(title, loc="left", pad=16, weight="bold", fontsize=13)
        ax.set_xlabel("Milliseconds")
        ax.grid(axis="x", color="#e2e8f0", zorder=0)
        ax.tick_params(axis="both", length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, (value, label) in enumerate(zip(values, annotations)):
            ax.text(value + limit * 0.025, i, label, va="center", fontsize=11)

    fig.text(0.05, 0.07, "Independent comparisons; different axis scales. Build bars show reported/derived medians.\n"
             "Source: docs/phase3-optimization-report.md, September 12 follow-ups. Not full time to first token.",
             color="#475569", fontsize=10, linespacing=1.6)
    directory = Path(__file__).resolve().parent
    fig.savefig(directory / "startup-phase-timings.png", dpi=180, facecolor="white")
    fig.savefig(directory / "startup-phase-timings.svg", facecolor="white")
    svg = directory / "startup-phase-timings.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
