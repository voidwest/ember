"""Plot layerwise morphology probe accuracy charts from saved probe NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import (  # noqa: E402
    BLUE, DARK, DARK_CYCLE, GREEN, PURPLE, RED, YELLOW, apply_matplotlib_theme
)
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


def load_peak_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"model", "task", "peak_layer", "peak_score", "final_layer_score"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"peak table is missing columns: {sorted(required)}")
        result = {}
        for line_number, row in enumerate(reader, start=2):
            key = (row["model"], row["task"])
            if key in result:
                raise ValueError(f"duplicate peak-table key {key!r} at line {line_number}")
            result[key] = row
        return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except ValueError as error:
        raise ValueError(f"unsafe or invalid NPZ artifact: {path}") from error


def require_key(data: dict[str, np.ndarray], key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{path} is missing required key {key!r}; keys={sorted(data)}")
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{path}:{key} must be a non-empty finite vector")
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError(f"{path}:{key} contains values outside [0, 1]")
    return value


def _scalar_text(data, key):
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return str(value.reshape(-1)[0])


def _atomic_figure(fig, output_path: Path, dark: bool) -> None:
    atomic_save_figure(fig, output_path, facecolor=BG if dark else "white")
    plt.close(fig)


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
    warning: str | None = None,
) -> None:
    data = load_npz(probes_path)
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
    if not 0 <= root_peak < len(root) or not 0 <= pattern_peak < len(pattern):
        raise ValueError(f"peak table selects an out-of-range layer for {model}")
    expected = {
        "root": (root_peak, float(root[root_peak]), float(root[-1])),
        "pattern": (pattern_peak, float(pattern[pattern_peak]), float(pattern[-1])),
    }
    for task, (peak_layer, peak_score, final_score) in expected.items():
        if peak_layer != int(np.argmax(root if task == "root" else pattern)):
            raise ValueError(f"peak table has stale {task} peak layer for {model}")
        row = peak_rows[(model, task)]
        if not np.isclose(float(row["peak_score"]), peak_score, rtol=0.0, atol=5e-7):
            raise ValueError(f"peak table has stale {task} peak score for {model}")
        if not np.isclose(float(row["final_layer_score"]), final_score, rtol=0.0, atol=5e-7):
            raise ValueError(f"peak table has stale {task} final score for {model}")

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
    title = f"Layerwise probe accuracy: {model}"
    if warning:
        title = f"{warning}\n{title}"
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _atomic_figure(fig, output_path, dark)


def plot_combined(
    curves: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    ylabel: str = "Accuracy",
    normalized: bool = False,
    dark: bool = False,
    warning: str | None = None,
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
    ax.set_title(f"{warning}\n{title}" if warning else title)
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    if dark:
        ax.tick_params(colors=DIM)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
    fig.tight_layout()
    _atomic_figure(fig, output_path, dark)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--peak-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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

    peak_rows = load_peak_table(args.peak_table)
    root_curves: dict[str, np.ndarray] = {}
    pattern_curves: dict[str, np.ndarray] = {}
    stimuli_hashes = set()
    split_policies = {}

    probe_paths = [args.metrics_dir / f"{model}_probes.npz" for model in args.models]
    statuses = enforce_probe_prompt_contracts(
        probe_paths,
        allow_label_revealed=args.allow_label_revealed_inputs,
        allow_unverifiable=args.allow_unverified_inputs,
    )
    warning = None
    if "label_revealed" in statuses:
        warning = "POSITIVE CONTROL — LABEL-REVEALED PROMPTS"
    elif any(status in UNVERIFIABLE_PROMPT_AUDIT_STATUSES for status in statuses):
        warning = "UNVERIFIED PROMPT CONTRACT"

    for model, probes_path in zip(args.models, probe_paths, strict=True):
        if not probes_path.exists():
            raise FileNotFoundError(f"Missing probe metrics: {probes_path}")
        data = load_npz(probes_path)
        root = require_key(data, "root_accuracy", probes_path)
        pattern = require_key(data, "pattern_accuracy", probes_path)
        if root.shape != pattern.shape:
            raise ValueError(f"{probes_path}: root and pattern layer counts differ")
        stimuli_hashes.add(_scalar_text(data, "stimuli_sha256"))
        policy_text = _scalar_text(data, "task_split_policy_json")
        if policy_text is None:
            policy_text = _scalar_text(data, "split_policy_json")
        if policy_text is not None:
            def reject_constant(value):
                raise ValueError(
                    f"non-standard JSON constant {value!r} in {probes_path}"
                )

            policies = json.loads(policy_text, parse_constant=reject_constant)
            if not isinstance(policies, list):
                raise ValueError(f"{probes_path}: split policy metadata must be a list")
            split_policies[model] = {
                row["task"]: row.get("effective_policy")
                for row in policies
                if isinstance(row, dict) and row.get("task") in {"root", "pattern"}
            }
        else:
            split_policies[model] = None
        root_curves[model] = root
        pattern_curves[model] = pattern
        plot_layerwise(
            model,
            probes_path,
            peak_rows,
            args.output_dir / "layerwise" / f"{model}_layerwise_probe_curves.png",
            dark=args.dark,
            warning=warning,
        )

    if None in stimuli_hashes:
        if not args.allow_unverified_inputs:
            raise ValueError("all probe artifacts require stimuli_sha256 provenance")
        stimuli_hashes.discard(None)
    if len(stimuli_hashes) > 1:
        raise ValueError("probe artifacts use different stimuli files")
    if any(value is None for value in split_policies.values()) and not args.allow_unverified_inputs:
        raise ValueError("all probe artifacts require split-policy metadata")
    verified_policies = [value for value in split_policies.values() if value is not None]
    if verified_policies and len({json.dumps(value, sort_keys=True) for value in verified_policies}) > 1:
        raise ValueError("probe artifacts use different root/pattern split policies")

    plot_combined(
        root_curves,
        args.output_dir / "root_layerwise_all_models.png",
        "Root probe accuracy across layers",
        dark=args.dark,
        warning=warning,
    )
    layer_counts = {len(v) for v in root_curves.values()}
    if len(layer_counts) > 1:
        plot_combined(
            root_curves,
            args.output_dir / "root_layerwise_all_models_normalized.png",
            "Root probe accuracy across normalized layer depth",
            normalized=True,
            dark=args.dark,
            warning=warning,
        )

    plot_combined(
        pattern_curves,
        args.output_dir / "pattern_layerwise_all_models.png",
        "Pattern probe accuracy across layers",
        dark=args.dark,
        warning=warning,
    )


if __name__ == "__main__":
    main()
