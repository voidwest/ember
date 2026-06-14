"""Plot layerwise morphology probe accuracy charts from saved probe NPZ files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BG = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
DIM = "#8b949e"
ACCENT = "#f78166"
ACCENT2 = "#d2a8ff"
BLUE = "#79c0ff"
GREEN = "#7ee787"
RED = "#ff7b72"
YELLOW = "#d29922"
DARK_CYCLE = [BLUE, ACCENT, GREEN, ACCENT2, YELLOW, RED, "#a5d6ff"]

DEFAULT_MODELS = [
    "qwen3_06b",
    "qwen25_15b",
    "qwen3_8b",
    "llama_1b",
    "llama_3b",
    "llama_8b",
    "gemma_e2b",
]


def setup_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT,
            "xtick.color": DIM,
            "ytick.color": DIM,
            "text.color": TEXT,
            "axes.titlecolor": ACCENT,
            "grid.color": BORDER,
            "legend.labelcolor": TEXT,
            "axes.prop_cycle": plt.cycler(color=DARK_CYCLE),
        }
    )


def load_peak_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as f:
        return {(row["model"], row["task"]): row for row in csv.DictReader(f)}


def require_key(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"{path} is missing required key {key!r}; keys={data.files}")
    return np.asarray(data[key], dtype=float)


def style_axis(ax: plt.Axes, dark: bool = False) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    if dark:
        ax.tick_params(colors=DIM)
        for spine in ax.spines.values():
            spine.set_color(BORDER)


def plot_layerwise(
    model: str,
    probes_path: Path,
    peak_rows: dict[tuple[str, str], dict[str, str]],
    output_path: Path,
    dark: bool = False,
) -> None:
    data = np.load(probes_path, allow_pickle=True)
    root = require_key(data, "root_accuracy", probes_path)
    pattern = require_key(data, "pattern_accuracy", probes_path)
    if root.shape != pattern.shape:
        raise ValueError(f"{probes_path}: root and pattern curves have different shapes")

    layers = np.arange(root.shape[0])
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    ax.plot(layers, root, marker="o", linewidth=2, markersize=3.5, label="root_accuracy")
    ax.plot(
        layers,
        pattern,
        marker="s",
        linewidth=2,
        markersize=3.5,
        label="pattern_accuracy",
    )

    root_peak = int(peak_rows[(model, "root")]["peak_layer"])
    pattern_peak = int(peak_rows[(model, "pattern")]["peak_layer"])
    final_layer = int(root.shape[0] - 1)

    markers = [
        (root_peak, "root peak", BLUE if dark else "#1f77b4", 0.90),
        (pattern_peak, "pattern peak", ACCENT if dark else "#ff7f0e", 0.82),
        (final_layer, "final", DIM if dark else "#4d4d4d", 0.74),
    ]
    for layer, label, color, ypos in markers:
        ax.axvline(layer, color=color, linestyle="--", linewidth=1.1, alpha=0.75)
        ax.annotate(
            f"{label}: {layer}",
            xy=(layer, ypos),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=8,
            rotation=90,
            va="top",
        )

    style_axis(ax, dark=dark)
    ax.set_title(f"Layerwise probe accuracy: {model}")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG if dark else "white")
    plt.close(fig)


def plot_combined(
    curves: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    ylabel: str = "Accuracy",
    normalized: bool = False,
    dark: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=160)
    for model, values in curves.items():
        if normalized:
            x = np.linspace(0.0, 1.0, len(values))
        else:
            x = np.arange(len(values))
        ax.plot(x, values, marker="o", linewidth=1.8, markersize=3, label=model)

    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("Relative layer depth" if normalized else "Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    if dark:
        ax.tick_params(colors=DIM)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG if dark else "white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--peak-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--dark", action="store_true", help="use voidwest dark chart styling")
    args = parser.parse_args()
    if args.dark:
        setup_dark_theme()

    peak_rows = load_peak_table(args.peak_table)
    root_curves: dict[str, np.ndarray] = {}
    pattern_curves: dict[str, np.ndarray] = {}

    for model in args.models:
        probes_path = args.metrics_dir / f"{model}_probes.npz"
        if not probes_path.exists():
            raise FileNotFoundError(f"Missing probe metrics: {probes_path}")
        data = np.load(probes_path, allow_pickle=True)
        root = require_key(data, "root_accuracy", probes_path)
        pattern = require_key(data, "pattern_accuracy", probes_path)
        root_curves[model] = root
        pattern_curves[model] = pattern
        plot_layerwise(
            model,
            probes_path,
            peak_rows,
            args.output_dir / "layerwise" / f"{model}_layerwise_probe_curves.png",
            dark=args.dark,
        )

    plot_combined(
        root_curves,
        args.output_dir / "root_layerwise_all_models.png",
        "Root probe accuracy across layers",
        dark=args.dark,
    )
    layer_counts = {len(v) for v in root_curves.values()}
    if len(layer_counts) > 1:
        plot_combined(
            root_curves,
            args.output_dir / "root_layerwise_all_models_normalized.png",
            "Root probe accuracy across normalized layer depth",
            normalized=True,
            dark=args.dark,
        )

    plot_combined(
        pattern_curves,
        args.output_dir / "pattern_layerwise_all_models.png",
        "Pattern probe accuracy across layers",
        dark=args.dark,
    )


if __name__ == "__main__":
    main()
