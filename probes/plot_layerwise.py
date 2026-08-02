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

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .train_linear_probe import atomic_save_figure
except ImportError:  # direct script execution
    from train_linear_probe import atomic_save_figure

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


def load_report(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {source}")

    data = json.loads(source.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), dict):
        raise ValueError(f"probe summary must contain a tasks object: {source}")
    return data


def prompt_status(data, source: str) -> str:
    audit = data.get("prompt_leakage_audit")
    if not isinstance(audit, dict) or not isinstance(audit.get("status"), str):
        return "unverifiable"
    status = audit["status"]
    if status not in {
        "passed",
        "not_applicable",
        "label_revealed",
        "not_checked_missing_probe_template_metadata",
    }:
        raise ValueError(f"unsupported prompt leakage status {status!r}: {source}")
    return status


def load_layerwise(data, task):
    t = data["tasks"].get(task)
    if t is None:
        return None
    if not isinstance(t, dict):
        raise ValueError(f"task {task!r} must be an object")
    lw = t.get("layerwise_accuracy", [])
    lw = np.asarray(lw, dtype=np.float64)
    if lw.ndim != 1 or lw.size == 0 or not np.isfinite(lw).all():
        raise ValueError(f"task {task!r} layerwise_accuracy must be a finite vector")
    if np.any((lw < 0.0) | (lw > 1.0)):
        raise ValueError(f"task {task!r} accuracies are outside [0, 1]")
    majority = t.get("majority_baseline_accuracy")
    if isinstance(majority, bool) or not isinstance(majority, (int, float)):
        raise ValueError(f"task {task!r} has no numeric majority baseline")
    majority = float(majority)
    if not math.isfinite(majority) or not 0.0 <= majority <= 1.0:
        raise ValueError(f"task {task!r} majority baseline is invalid")
    info = {
        "num_examples": t.get("num_examples", 0),
        "num_classes": t.get("num_classes", 0),
        "majority": majority,
    }
    return lw, info


def main():
    parser = argparse.ArgumentParser(description="layerwise probe curve plot")
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--qwen25")
    parser.add_argument("--output", default="paper/figures/layerwise_probe_curves.png")
    parser.add_argument("--allow-unverified-inputs", action="store_true")
    parser.add_argument("--allow-label-revealed-inputs", action="store_true")
    args = parser.parse_args()

    plt.rcParams.update(STYLE)

    model_paths = {
        "Qwen3-0.6B": args.qwen3,
        "Llama-3.2-1B": args.llama,
    }
    if args.qwen25:
        model_paths["Qwen2.5-1.5B"] = args.qwen25
    models = {name: load_report(path) for name, path in model_paths.items()}
    hashes = {data.get("stimuli_sha256") for data in models.values()}
    if None in hashes:
        if not args.allow_unverified_inputs:
            raise ValueError(
                "all summaries require stimuli_sha256 provenance; use "
                "--allow-unverified-inputs only for externally checked legacy files"
            )
        hashes.discard(None)
    if len(hashes) > 1:
        raise ValueError("probe summaries were produced from different stimuli files")
    statuses = {name: prompt_status(data, model_paths[name]) for name, data in models.items()}
    if "label_revealed" in statuses.values() and not args.allow_label_revealed_inputs:
        raise ValueError(
            "a probe summary used label-revealed prompts; allow it only as a positive control"
        )
    if any(
        status in {"unverifiable", "not_checked_missing_probe_template_metadata"}
        for status in statuses.values()
    ) and not args.allow_unverified_inputs:
        raise ValueError("all probe summaries require a verifiable prompt leakage audit")
    configurations = [data.get("config") for data in models.values()]
    if any(not isinstance(config, dict) for config in configurations):
        if not args.allow_unverified_inputs:
            raise ValueError("all probe summaries require classifier/CV configuration metadata")
    elif len({json.dumps(config, sort_keys=True) for config in configurations}) > 1:
        raise ValueError("probe summaries use incompatible classifier/CV configurations")

    tasks = ["pos", "features.gender", "features.number"]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    axes = axes.flatten()
    fig.subplots_adjust(wspace=0.28, hspace=0.38)
    legend_handles = {}

    for ax, task in zip(axes, tasks):
        ax.set_facecolor(LIGHT.surface)
        plotted = False
        for model_name, data in models.items():
            result = load_layerwise(data, task)
            if result is None:
                continue
            plotted = True
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
            majority = ax.axhline(
                info["majority"],
                color=color,
                linestyle=":",
                linewidth=0.8,
                alpha=0.55,
                label=f"{model_name} majority",
            )
            legend_handles.setdefault(f"{model_name} majority", majority)
        if not plotted:
            raise ValueError(f"task {task!r} is absent from every model summary")

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

    title = "Layerwise probe accuracy (random CV; descriptive layer selection)"
    if "label_revealed" in statuses.values():
        title = f"POSITIVE CONTROL — LABEL-REVEALED PROMPTS\n{title}"
    elif any(status != "passed" for status in statuses.values()):
        title = f"UNVERIFIED PROMPT CONTRACT\n{title}"
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_figure(fig, out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
