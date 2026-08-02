"""Plot root probe accuracy across explicitly supplied model runs.

All runs must have matching stimuli, label sets, and split policies. Artifacts
from label-revealed prompts are rejected unless explicitly requested as a
positive-control visualization.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analysis_common import (
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )
    from .train_linear_probe import atomic_save_figure
except ImportError:  # direct script execution
    from analysis_common import (
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )
    from train_linear_probe import atomic_save_figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import BLUE, DARK, GREEN, RED, apply_matplotlib_theme  # noqa: E402

DARK_BG = DARK.bg
DARK_SURFACE = DARK.surface
DARK_BORDER = DARK.border
DARK_TEXT = DARK.text
DARK_DIM = DARK.muted
DARK_BLUE = BLUE
DARK_ACCENT = DARK.accent
DARK_GREEN = GREEN
DARK_RED = RED


COLORS = [DARK_BLUE, DARK_ACCENT, DARK_GREEN, DARK_RED]


def setup_dark_theme():
    apply_matplotlib_theme(dark=True)


def parse_run(value: str):
    if ":" not in value:
        raise argparse.ArgumentTypeError("runs must be LABEL:PATH")
    label, path = value.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("runs must be LABEL:PATH")
    return label, path


def load_root_accuracy(path: str):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with np.load(source, allow_pickle=False) as data:
            if "root_accuracy" not in data.files:
                raise KeyError(f"{path} missing root_accuracy; keys: {data.files}")
            accuracy = np.asarray(data["root_accuracy"], dtype=np.float64)
            chance = float(np.asarray(data["root_chance"]).item()) if "root_chance" in data else None
            stimuli_sha = (
                str(np.asarray(data["stimuli_sha256"]).item())
                if "stimuli_sha256" in data
                else None
            )
            classes = (
                tuple(str(value) for value in np.asarray(data["root_classes"]).tolist())
                if "root_classes" in data
                else None
            )
            policy_text = None
            for key in ("task_split_policy_json", "split_policy_json"):
                if key in data:
                    raw = np.asarray(data[key])
                    if raw.size != 1:
                        raise ValueError(f"{path}:{key} must be scalar")
                    policy_text = str(raw.reshape(-1)[0])
                    break
            probe_kind = (
                str(np.asarray(data["probe_kind"]).item())
                if "probe_kind" in data
                else None
            )
    except ValueError as error:
        raise ValueError(f"unsafe or invalid probe artifact: {source}") from error
    if accuracy.ndim != 1 or accuracy.size < 2 or not np.isfinite(accuracy).all():
        raise ValueError(f"{path} root_accuracy must be a finite vector with at least 2 layers")
    if np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError(f"{path} root_accuracy is outside [0, 1]")
    if chance is not None and (not np.isfinite(chance) or not 0.0 <= chance <= 1.0):
        raise ValueError(f"{path} root_chance is invalid")
    policy = None
    if policy_text is not None:
        def reject_constant(value):
            raise ValueError(f"non-standard JSON constant {value!r} in {path}")

        policy_rows = json.loads(policy_text, parse_constant=reject_constant)
        if not isinstance(policy_rows, list):
            raise ValueError(f"{path} split policy metadata must be a list")
        root_rows = [row for row in policy_rows if isinstance(row, dict) and row.get("task") == "root"]
        if len(root_rows) != 1:
            raise ValueError(f"{path} must contain exactly one root split-policy record")
        policy = root_rows[0]
    return accuracy, chance, stimuli_sha, classes, policy, probe_kind


def main():
    parser = argparse.ArgumentParser(
        description="plot root probe accuracy across LLaMA model scales"
    )
    parser.add_argument(
        "--output",
        default="docs/plots/root_probe_scale_comparison.png",
        help="output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="LABEL:PATH probe artifact (repeat for every model)",
    )
    parser.add_argument("--title", default="Root probe accuracy across normalized layer depth")
    parser.add_argument("--annotate-minima", action="store_true")
    parser.add_argument("--allow-unverified-inputs", action="store_true")
    parser.add_argument("--allow-label-revealed-inputs", action="store_true")
    args = parser.parse_args()

    if args.dpi < 1:
        parser.error("--dpi must be positive")
    setup_dark_theme()

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    configured_runs = args.run
    if len(configured_runs) < 2:
        parser.error("at least two --run artifacts are required for a scale comparison")
    if len(configured_runs) != len({label for label, _ in configured_runs}):
        parser.error("run labels must be unique")
    loaded = []
    for index, (label, path) in enumerate(configured_runs):
        accuracy, chance, stimuli_sha, classes, policy, probe_kind = load_root_accuracy(path)
        color = COLORS[index % len(COLORS)]
        loaded.append(
            (label, accuracy, chance, stimuli_sha, classes, policy, probe_kind, color)
        )
    hashes = {row[3] for row in loaded}
    if None in hashes:
        if not args.allow_unverified_inputs:
            raise ValueError("all probe artifacts require stimuli_sha256 provenance")
        hashes.discard(None)
    if len(hashes) > 1:
        raise ValueError("probe artifacts use different stimuli files")
    contracts = {(row[4], json.dumps(row[5], sort_keys=True), row[6]) for row in loaded}
    if any(value is None for row in loaded for value in (row[4], row[5], row[6])):
        if not args.allow_unverified_inputs:
            raise ValueError(
                "all probe artifacts require root classes, split policy, and probe-kind metadata"
            )
    elif len(contracts) != 1:
        raise ValueError("probe artifacts use incompatible labels, split policies, or classifiers")
    statuses = enforce_probe_prompt_contracts(
        [path for _, path in configured_runs],
        allow_label_revealed=args.allow_label_revealed_inputs,
        allow_unverifiable=args.allow_unverified_inputs,
    )

    for label, root_acc, chance, _, _, _, _, color in loaded:
        x = np.linspace(0.0, 1.0, len(root_acc))
        ax.plot(
            x,
            root_acc,
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            color=color,
            label=label,
        )
        trough_idx = int(np.argmin(root_acc))
        if chance is not None:
            ax.axhline(
                chance,
                color=color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.45,
                label=f"{label} chance ({chance:.1%})",
            )
        if args.annotate_minima:
            y = float(root_acc[trough_idx])
            ax.annotate(
                f"{label} min: {y:.1%}",
                xy=(x[trough_idx], y),
                xytext=(6, 8),
                textcoords="offset points",
                color=DARK_TEXT,
                fontsize=8,
            )

    title = args.title
    if "label_revealed" in statuses:
        title = f"POSITIVE CONTROL — LABEL-REVEALED PROMPTS\n{title}"
    elif any(status in UNVERIFIABLE_PROMPT_AUDIT_STATUSES for status in statuses):
        title = f"UNVERIFIED PROMPT CONTRACT\n{title}"
    ax.set_title(
        title,
        pad=14,
        color=DARK_TEXT,
        fontweight="bold",
    )
    ax.set_xlabel("normalized layer depth")
    ax.set_ylabel("root probe accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.92)

    for spine in ax.spines.values():
        spine.set_color(DARK_BORDER)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    atomic_save_figure(
        fig,
        out_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor=DARK_BG,
        edgecolor="none",
    )
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
