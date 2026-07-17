"""Generate layerwise probe accuracy curves for POS, gender, number across models.

Reads baseline_probe_summary.json from each model directory and produces a
multi-panel figure showing per-layer accuracy for the three low-cardinality
tasks that survive heldout evaluation.

Usage:
    python probes/plot_layerwise.py \
        --qwen3 data/arabic_morph_real/probe_baseline_qwen3_1416/baseline_probe_summary.json \
        --llama data/arabic_morph_real/probe_baseline_llama32_1b/baseline_probe_summary.json \
        --qwen25 data/arabic_morph_real/probe_baseline_qwen25_15b/baseline_probe_summary.json \
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
    parser.add_argument("--qwen25", required=True)
    parser.add_argument("--output", default="paper/figures/layerwise_probe_curves.png")
    args = parser.parse_args()

    plt.rcParams.update(STYLE)

    models = {
        "Qwen3-0.6B": args.qwen3,
        "Llama-3.2-1B": args.llama,
        "Qwen2.5-1.5B": args.qwen25,
    }

    tasks = ["pos", "features.gender", "features.number"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.subplots_adjust(wspace=0.28)

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
            ax.plot(layers, lw, color=color, linewidth=1.6,
                    marker="o", markersize=2.5, label=model_name)
            # Mark best layer
            best = int(np.argmax(lw))
            ax.scatter([best], [lw[best]], color=color, s=30,
                       edgecolors="white", linewidths=0.5, zorder=5)

        # Majority baseline
        if result:
            ax.axhline(info["majority"], color="#484f58", linestyle=":",
                       linewidth=0.8, alpha=0.7, label=f"majority ({info['majority']:.2f})")

        ax.set_title(TASK_NAMES[task], fontsize=10, pad=6)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.5, loc="lower right",
                  framealpha=0.85, borderpad=0.3,
                  handletextpad=0.5, labelspacing=0.2)
        ax.tick_params(labelsize=7)

    fig.suptitle("Layerwise probe accuracy (random CV, best layer only)",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
