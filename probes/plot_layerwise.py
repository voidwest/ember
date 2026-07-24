"""Generate layerwise probe accuracy curves for POS, gender, number across models.

Reads baseline_probe_summary.json from each model directory and produces a
multi-panel figure showing per-layer accuracy for the three low-cardinality
tasks that survive heldout evaluation.

Usage:
    python probes/plot_layerwise.py \
        --qwen3 data/arabic_morph_real/probe_baseline_qwen3_5k/baseline_probe_summary.json \
        --llama data/arabic_morph_real/probe_baseline_llama32_5k/baseline_probe_summary.json \
        --output paper/figures/layerwise_probe_curves.png
"""

import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import LIGHT, LIGHT_CYCLE, matplotlib_style  # noqa: E402

# -- style -------------------------------------------------------------
STYLE = matplotlib_style(dark=False, dpi=200)

MODEL_COLORS = {
    "Qwen3-0.6B": LIGHT_CYCLE[0],
    "Llama-3.2-1B": LIGHT_CYCLE[1],
    "Qwen2.5-1.5B": LIGHT_CYCLE[2],
}

TASK_NAMES = {
    "pos": "POS (3 classes)",
    "features.gender": "Gender (2 classes)",
    "features.number": "Number (3 classes)",
}


def load_layerwise(path, task):
    with open(path) as f:
        data = json.load(f)
    t = data["tasks"].get(task)
    if t is None:
        return None
    lw = t.get("layerwise_accuracy", [])
    info = {
        "num_examples": t.get("num_examples", 0),
        "num_classes": t.get("num_classes", 0),
        "majority": t.get("majority_baseline_accuracy", 0),
    }
    return np.array(lw), info


def main():
    parser = argparse.ArgumentParser(description="layerwise probe curve plot")
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--qwen25")
    parser.add_argument("--output", default="paper/figures/layerwise_probe_curves.png")
    args = parser.parse_args()

    plt.rcParams.update(STYLE)

    models = {
        "Qwen3-0.6B": args.qwen3,
        "Llama-3.2-1B": args.llama,
    }
    if args.qwen25:
        models["Qwen2.5-1.5B"] = args.qwen25

    tasks = ["pos", "features.gender", "features.number"]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    axes = axes.flatten()
    fig.subplots_adjust(wspace=0.28, hspace=0.38)
    legend_handles = {}

    for ax, task in zip(axes, tasks):
        ax.set_facecolor(LIGHT.surface)
        result = None
        for model_name, path in models.items():
            result = load_layerwise(path, task)
            if result is None:
                continue
            lw, info = result
            layers = np.arange(len(lw))
            color = MODEL_COLORS[model_name]
            line, = ax.plot(layers, lw, color=color, linewidth=1.6,
                            marker="o", markersize=2.4, label=model_name)
            legend_handles[model_name] = line
            # Mark best layer
            best = int(np.argmax(lw))
            ax.scatter([best], [lw[best]], color=color, s=30,
                       edgecolors=LIGHT.heading, linewidths=0.45, zorder=5)

        # Majority baseline
        if result:
            majority = ax.axhline(info["majority"], color=LIGHT.subtle, linestyle=":",
                                  linewidth=1.0, alpha=0.8, label="majority baseline")
            legend_handles.setdefault("Majority baseline", majority)

        ax.set_title(TASK_NAMES[task], fontsize=9, pad=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.02)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="center",
        frameon=True,
        fontsize=8,
        borderpad=0.6,
        handlelength=2.2,
        labelspacing=0.5,
    )
    legend_ax.text(
        0.0, 0.24,
        "Dots mark each model's best layer.\nLayer 0 is the first transformer block output.",
        transform=legend_ax.transAxes,
        fontsize=7.4,
        color=LIGHT.muted,
        va="top",
    )

    fig.suptitle("Layerwise probe accuracy (random CV)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
