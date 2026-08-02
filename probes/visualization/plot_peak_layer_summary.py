"""Plot peak-layer and peak-vs-final summaries from morphology probe CSV output."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import BLUE, DARK, GREEN, PURPLE, RED, apply_matplotlib_theme  # noqa: E402
sys.path.insert(0, str(ROOT / "probes"))
try:
    from ..analysis_common import (
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )
    from ..train_linear_probe import atomic_save_figure
except ImportError:  # direct script execution
    from analysis_common import (  # noqa: E402
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )
    from train_linear_probe import atomic_save_figure  # noqa: E402

BG = DARK.bg
SURFACE = DARK.surface
BORDER = DARK.border
TEXT = DARK.text
DIM = DARK.muted
ACCENT = DARK.accent
ACCENT2 = PURPLE

DEFAULT_MODELS = [
    "qwen3_06b",
    "qwen25_15b",
    "qwen3_8b",
    "llama_1b",
    "llama_3b",
    "llama_8b",
    "gemma_e2b",
]
SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def setup_dark_theme() -> None:
    apply_matplotlib_theme(dark=True)


def finish_axes(ax: plt.Axes, dark: bool) -> None:
    if not dark:
        return
    ax.tick_params(colors=DIM)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "model",
            "task",
            "peak_layer",
            "peak_score",
            "final_layer_score",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"peak table is missing columns: {sorted(required)}")
        rows = list(reader)
    seen = set()
    for line_number, row in enumerate(rows, start=2):
        key = (row["model"], row["task"])
        if key in seen:
            raise ValueError(f"duplicate model/task row {key!r} at line {line_number}")
        seen.add(key)
        try:
            layer = int(row["peak_layer"])
            peak = float(row["peak_score"])
            final = float(row["final_layer_score"])
        except ValueError as error:
            raise ValueError(f"invalid numeric value at line {line_number}") from error
        if layer < 0 or not all(math.isfinite(value) for value in (peak, final)):
            raise ValueError(f"invalid peak metrics at line {line_number}")
        if not 0.0 <= peak <= 1.0 or not 0.0 <= final <= 1.0:
            raise ValueError(f"accuracy outside [0, 1] at line {line_number}")
        if final > peak + 1e-9:
            raise ValueError(f"peak_score is smaller than final_layer_score at line {line_number}")
    return rows


def row_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["model"], row["task"]): row for row in rows}


def save_figure(fig, output: Path, dark: bool) -> None:
    atomic_save_figure(fig, output, facecolor=BG if dark else "white")
    plt.close(fig)


def validate_probe_sources(
    rows_by_key: dict[tuple[str, str], dict[str, str]],
    models: list[str],
    metrics_dir: Path,
) -> tuple[list[Path], set[str]]:
    paths = []
    stimuli_hashes: set[str] = set()
    for model in models:
        path = metrics_dir / f"{model}_probes.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing source probe artifact: {path}")
        paths.append(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                if "stimuli_sha256" not in data:
                    raise ValueError(f"{path} is missing stimuli_sha256")
                sha = np.asarray(data["stimuli_sha256"])
                if sha.size != 1:
                    raise ValueError(f"{path}:stimuli_sha256 must be scalar")
                stimuli_hashes.add(str(sha.reshape(-1)[0]))
                for task in ("root", "pattern"):
                    key = f"{task}_accuracy"
                    if key not in data:
                        raise ValueError(f"{path} is missing {key}")
                    values = np.asarray(data[key], dtype=np.float64)
                    if (
                        values.ndim != 1
                        or values.size == 0
                        or not np.isfinite(values).all()
                        or np.any((values < 0.0) | (values > 1.0))
                    ):
                        raise ValueError(f"{path}:{key} is not a valid accuracy vector")
                    row = rows_by_key[(model, task)]
                    peak_layer = int(np.argmax(values))
                    if int(row["peak_layer"]) != peak_layer:
                        raise ValueError(f"peak table has a stale {task} layer for {model}")
                    if not np.isclose(
                        float(row["peak_score"]), values[peak_layer], rtol=0.0, atol=5e-7
                    ) or not np.isclose(
                        float(row["final_layer_score"]), values[-1], rtol=0.0, atol=5e-7
                    ):
                        raise ValueError(f"peak table has stale {task} scores for {model}")
        except ValueError as error:
            raise ValueError(f"unsafe or invalid source probe artifact: {path}") from error
    if len(stimuli_hashes) != 1:
        raise ValueError("source probe artifacts do not share one stimuli SHA-256")
    return paths, stimuli_hashes


def grouped_peak_layers(rows_by_key: dict[tuple[str, str], dict[str, str]], models: list[str], output: Path, dark: bool = False, warning: str | None = None) -> None:
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
    title = "Peak probe layer by task (descriptive selection)"
    ax.set_title(f"{warning}\n{title}" if warning else title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish_axes(ax, dark)
    fig.tight_layout()
    save_figure(fig, output, dark)


def connected_peak_final(
    rows_by_key: dict[tuple[str, str], dict[str, str]],
    models: list[str],
    output: Path,
    tasks: list[str],
    title: str,
    dark: bool = False,
    warning: str | None = None,
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
    ax.set_title(f"{warning}\n{title}" if warning else title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    finish_axes(ax, dark)
    fig.tight_layout()
    save_figure(fig, output, dark)


def drop_bars(
    rows_by_key: dict[tuple[str, str], dict[str, str]],
    models: list[str],
    output: Path,
    tasks: list[str],
    title: str,
    dark: bool = False,
    warning: str | None = None,
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
    ax.set_title(f"{warning}\n{title}" if warning else title)
    ax.grid(True, axis="y", alpha=0.25)
    finish_axes(ax, dark)
    fig.tight_layout()
    save_figure(fig, output, dark)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        help="directory containing MODEL_probes.npz files used to verify the CSV",
    )
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--dark", action="store_true", help="use voidwest dark chart styling")
    parser.add_argument("--allow-unverified-inputs", action="store_true")
    parser.add_argument("--allow-label-revealed-inputs", action="store_true")
    args = parser.parse_args()
    if not args.models or len(args.models) != len(set(args.models)) or any(
        not SAFE_MODEL.fullmatch(model) for model in args.models
    ):
        parser.error("--models must be non-empty unique safe identifiers")
    if args.dark:
        setup_dark_theme()

    rows = read_rows(args.peak_table)
    rows_by_key = row_map(rows)
    missing = [(m, t) for m in args.models for t in ("root", "pattern") if (m, t) not in rows_by_key]
    if missing:
        raise KeyError(f"Missing model/task rows in {args.peak_table}: {missing}")

    warning = None
    if args.metrics_dir is None:
        if not args.allow_unverified_inputs:
            parser.error("--metrics-dir is required unless --allow-unverified-inputs is set")
        warning = "UNVERIFIED SOURCE TABLE"
    else:
        probe_paths, _ = validate_probe_sources(rows_by_key, args.models, args.metrics_dir)
        statuses = enforce_probe_prompt_contracts(
            probe_paths,
            allow_label_revealed=args.allow_label_revealed_inputs,
            allow_unverifiable=args.allow_unverified_inputs,
        )
        if "label_revealed" in statuses:
            warning = "POSITIVE CONTROL — LABEL-REVEALED PROMPTS"
        elif any(status in UNVERIFIABLE_PROMPT_AUDIT_STATUSES for status in statuses):
            warning = "UNVERIFIED PROMPT CONTRACT"

    grouped_peak_layers(rows_by_key, args.models, args.output_dir / "peak_layer_summary.png", dark=args.dark, warning=warning)
    connected_peak_final(
        rows_by_key,
        args.models,
        args.output_dir / "peak_vs_final_accuracy.png",
        ["root", "pattern"],
        "Peak vs final-layer probe accuracy",
        dark=args.dark,
        warning=warning,
    )
    connected_peak_final(
        rows_by_key,
        args.models,
        args.output_dir / "root_peak_vs_final_accuracy.png",
        ["root"],
        "Root peak vs final-layer probe accuracy",
        dark=args.dark,
        warning=warning,
    )
    connected_peak_final(
        rows_by_key,
        args.models,
        args.output_dir / "pattern_peak_vs_final_accuracy.png",
        ["pattern"],
        "Pattern peak vs final-layer probe accuracy",
        dark=args.dark,
        warning=warning,
    )
    drop_bars(
        rows_by_key,
        args.models,
        args.output_dir / "final_minus_peak_drop.png",
        ["root", "pattern"],
        "Final minus peak probe accuracy",
        dark=args.dark,
        warning=warning,
    )
    drop_bars(
        rows_by_key,
        args.models,
        args.output_dir / "root_final_minus_peak_drop.png",
        ["root"],
        "Root final minus peak probe accuracy",
        dark=args.dark,
        warning=warning,
    )


if __name__ == "__main__":
    main()
