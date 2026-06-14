"""Plot peak-layer and peak-vs-final summaries from morphology probe CSV output."""

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
        }
    )


def finish_axes(ax: plt.Axes, dark: bool) -> None:
    if not dark:
        return
    ax.tick_params(colors=DIM)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["model"], row["task"]): row for row in rows}


def grouped_peak_layers(rows_by_key: dict[tuple[str, str], dict[str, str]], models: list[str], output: Path, dark: bool = False) -> None:
    x = np.arange(len(models))
    width = 0.36
    root = [float(rows_by_key[(m, "root")]["peak_layer"]) for m in models]
    pattern = [float(rows_by_key[(m, "pattern")]["peak_layer"]) for m in models]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=160)
    ax.bar(x - width / 2, root, width, label="root", color=BLUE if dark else None)
    ax.bar(x + width / 2, pattern, width, label="pattern", color=ACCENT if dark else None)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Peak layer")
    ax.set_title("Peak probe layer by task")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish_axes(ax, dark)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG if dark else "white")
    plt.close(fig)


def connected_peak_final(
    rows_by_key: dict[tuple[str, str], dict[str, str]],
    models: list[str],
    output: Path,
    tasks: list[str],
    title: str,
    dark: bool = False,
) -> None:
    labels = [f"{m}\n{task}" for m in models for task in tasks]
    peak = [float(rows_by_key[(m, task)]["peak_score"]) for m in models for task in tasks]
    final = [float(rows_by_key[(m, task)]["final_layer_score"]) for m in models for task in tasks]
    x = np.arange(len(labels))

    fig_width = max(8.5, len(labels) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), dpi=160)
    for i, (p, f) in enumerate(zip(peak, final, strict=True)):
        ax.plot([i, i], [p, f], color=DIM if dark else "#808080", linewidth=1.5, alpha=0.75)
    ax.scatter(x, peak, label="peak accuracy", s=38, zorder=3, color=BLUE if dark else None)
    ax.scatter(x, final, label="final-layer accuracy", s=38, marker="s", zorder=3, color=ACCENT if dark else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    finish_axes(ax, dark)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG if dark else "white")
    plt.close(fig)


def drop_bars(
    rows_by_key: dict[tuple[str, str], dict[str, str]],
    models: list[str],
    output: Path,
    tasks: list[str],
    title: str,
    dark: bool = False,
) -> None:
    labels = [f"{m}\n{task}" for m in models for task in tasks]
    drops = [
        float(rows_by_key[(m, task)]["final_layer_score"])
        - float(rows_by_key[(m, task)]["peak_score"])
        for m in models
        for task in tasks
    ]
    colors = [RED if v < 0 else GREEN for v in drops] if dark else ["#d62728" if v < 0 else "#2ca02c" for v in drops]
    x = np.arange(len(labels))

    fig_width = max(8.5, len(labels) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), dpi=160)
    ax.bar(x, drops, color=colors)
    ax.axhline(0, color=DIM if dark else "#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final accuracy minus peak accuracy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    finish_axes(ax, dark)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG if dark else "white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--dark", action="store_true", help="use voidwest dark chart styling")
    args = parser.parse_args()
    if args.dark:
        setup_dark_theme()

    rows = read_rows(args.peak_table)
    rows_by_key = row_map(rows)
    missing = [(m, t) for m in args.models for t in ("root", "pattern") if (m, t) not in rows_by_key]
    if missing:
        raise KeyError(f"Missing model/task rows in {args.peak_table}: {missing}")

    grouped_peak_layers(rows_by_key, args.models, args.output_dir / "peak_layer_summary.png", dark=args.dark)
    connected_peak_final(
        rows_by_key,
        args.models,
        args.output_dir / "peak_vs_final_accuracy.png",
        ["root", "pattern"],
        "Peak vs final-layer probe accuracy",
        dark=args.dark,
    )
    connected_peak_final(
        rows_by_key,
        args.models,
        args.output_dir / "root_peak_vs_final_accuracy.png",
        ["root"],
        "Root peak vs final-layer probe accuracy",
        dark=args.dark,
    )
    connected_peak_final(
        rows_by_key,
        args.models,
        args.output_dir / "pattern_peak_vs_final_accuracy.png",
        ["pattern"],
        "Pattern peak vs final-layer probe accuracy",
        dark=args.dark,
    )
    drop_bars(
        rows_by_key,
        args.models,
        args.output_dir / "final_minus_peak_drop.png",
        ["root", "pattern"],
        "Final minus peak probe accuracy",
        dark=args.dark,
    )
    drop_bars(
        rows_by_key,
        args.models,
        args.output_dir / "root_final_minus_peak_drop.png",
        ["root"],
        "Root final minus peak probe accuracy",
        dark=args.dark,
    )


if __name__ == "__main__":
    main()
