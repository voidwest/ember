"""Inspect saved CCA/RSA NPZ files and plot supported layerwise heatmaps."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import (  # noqa: E402
    DARK, apply_matplotlib_theme, diverging_cmap, sequential_cmap, similarity_norm
)
sys.path.insert(0, str(ROOT / "probes"))
try:
    from ..analysis_common import (
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )
    from ..train_linear_probe import atomic_save_figure, atomic_write_text
except ImportError:  # direct script execution
    from analysis_common import (  # noqa: E402
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )
    from train_linear_probe import atomic_save_figure, atomic_write_text  # noqa: E402

BG = DARK.bg
SURFACE = DARK.surface
BORDER = DARK.border
TEXT = DARK.text
DIM = DARK.muted
ACCENT = DARK.accent

DEFAULT_MODELS = [
    "qwen3_06b",
    "qwen25_15b",
    "qwen3_8b",
    "llama_1b",
    "llama_3b",
    "llama_8b",
    "gemma_e2b",
]

PAIRWISE_PLAN = [
    ("llama_1b", "llama_8b"),
    ("qwen3_06b", "qwen3_8b"),
    ("llama_8b", "qwen3_8b"),
    ("gemma_e2b", "llama_8b"),
]
SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def setup_dark_theme() -> None:
    apply_matplotlib_theme(dark=True)


@dataclass
class ArraySummary:
    key: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | None
    maximum: float | None
    mean: float | None


def summarize_npz(path: Path) -> list[ArraySummary]:
    summaries: list[ArraySummary] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.number):
                    if arr.size == 0 or not np.isfinite(arr).all():
                        raise ValueError(f"{path}:{key} is empty or non-finite")
                    summaries.append(
                        ArraySummary(
                            key=key,
                            shape=arr.shape,
                            dtype=str(arr.dtype),
                            minimum=float(np.min(arr)),
                            maximum=float(np.max(arr)),
                            mean=float(np.mean(arr)),
                        )
                    )
                else:
                    summaries.append(
                        ArraySummary(
                            key=key,
                            shape=arr.shape,
                            dtype=str(arr.dtype),
                            minimum=None,
                            maximum=None,
                            mean=None,
                        )
                    )
    except ValueError as error:
        raise ValueError(f"unsafe or invalid NPZ artifact: {path}") from error
    return summaries


def matrix_key(path: Path, preferred_key: str) -> tuple[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if preferred_key not in data.files:
                return None
            arr = np.asarray(data[preferred_key], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"unsafe or invalid NPZ artifact: {path}") from error
    if (
        arr.ndim != 2
        or arr.shape[0] != arr.shape[1]
        or arr.shape[0] == 0
        or not np.isfinite(arr).all()
    ):
        raise ValueError(f"{path}:{preferred_key} must be a finite non-empty square matrix")
    bounds = (0.0, 1.0) if preferred_key.startswith("cca") else (-1.0, 1.0)
    if np.any(arr < bounds[0] - 1e-8) or np.any(arr > bounds[1] + 1e-8):
        raise ValueError(f"{path}:{preferred_key} is outside {bounds}")
    return preferred_key, arr


def plot_heatmap(
    matrix: np.ndarray,
    output: Path,
    title: str,
    colorbar_label: str,
    *,
    kind: str,
    dark: bool = False,
    warning: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=160)
    values_are_similarity = kind == "cca"
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=sequential_cmap(dark=dark) if values_are_similarity else diverging_cmap(dark=dark),
        norm=similarity_norm() if values_are_similarity else None,
        vmin=None if values_are_similarity else -1.0,
        vmax=None if values_are_similarity else 1.0,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"{warning}\n{title}" if warning else title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if values_are_similarity:
        cbar.set_ticks([0.0, 0.6, 0.8, 0.9, 1.0])
    cbar.set_label(colorbar_label)
    if dark:
        ax.tick_params(colors=DIM)
        cbar.ax.tick_params(colors=DIM)
        cbar.ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
    fig.tight_layout()
    atomic_save_figure(fig, output, facecolor=BG if dark else "white")
    plt.close(fig)


def scalar_text(path: Path, key: str) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                return None
            value = np.asarray(data[key])
            if value.size != 1:
                raise ValueError(f"{path}:{key} must be scalar")
            return str(value.reshape(-1)[0])
    except ValueError as error:
        raise ValueError(f"unsafe or invalid NPZ artifact: {path}") from error


def validate_metric_source(metric_path: Path, probe_path: Path, *, allow_unverified: bool) -> None:
    metric_hash = scalar_text(metric_path, "activations_a_sha256")
    probe_hash = scalar_text(probe_path, "activations_sha256")
    if metric_hash is None or probe_hash is None:
        if not allow_unverified:
            raise ValueError(
                f"{metric_path} and {probe_path} require activation SHA-256 provenance"
            )
    elif metric_hash != probe_hash:
        raise ValueError(f"{metric_path} and {probe_path} refer to different activations")


def array_bytes(path: Path) -> tuple[tuple[int, ...], str, int]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        arr.ndim != 3
        or any(size <= 0 for size in arr.shape)
        or arr.dtype.kind != "f"
    ):
        raise ValueError(f"activation array must be [samples, layers, hidden]: {path}")
    return arr.shape, str(arr.dtype), int(np.prod(arr.shape) * arr.dtype.itemsize)


def mib(value: int) -> float:
    return value / (1024 * 1024)


def write_schema(
    metrics_dir: Path,
    output_dir: Path,
    models: list[str],
    generated_rsa: list[Path],
    generated_cca: list[Path],
) -> None:
    lines = [
        "# Geometry NPZ Schema",
        "",
        "Numeric min/max/mean values are computed from the saved arrays. Chart safety is based on explicit layer-by-layer matrix keys.",
        "",
    ]
    for model in models:
        lines.append(f"## {model}")
        for kind, preferred in (("CCA", "cca_layer_matrix"), ("RSA", "rsa_layer_matrix")):
            path = metrics_dir / f"{model}_{kind.lower()}.npz"
            lines.append(f"### `{path.name}`")
            if not path.exists():
                lines.append("- Missing file; no chart generated.")
                lines.append("")
                continue
            for summary in summarize_npz(path):
                if summary.minimum is None:
                    stat = "non-numeric"
                else:
                    stat = (
                        f"min={summary.minimum:.6f}, "
                        f"max={summary.maximum:.6f}, mean={summary.mean:.6f}"
                    )
                lines.append(
                    f"- `{summary.key}`: shape={summary.shape}, dtype={summary.dtype}, {stat}"
                )
            safe = matrix_key(path, preferred)
            if safe is None:
                lines.append(f"- Safe chart: skipped; no unambiguous square `{preferred}` matrix.")
            else:
                key, matrix = safe
                label = "similarity" if float(np.mean(np.diag(matrix))) >= 0.9 else "score"
                rel = (
                    Path("rsa") / f"{model}_rsa_heatmap.png"
                    if kind == "RSA"
                    else Path("cca") / f"{model}_cca_heatmap.png"
                )
                lines.append(
                    f"- Safe chart: `{rel}` from `{key}` as within-model layerwise {label}."
                )
            if kind == "CCA":
                lines.append(
                    "- Cross-model note: this file is not a pairwise cross-model CCA file; "
                    "it contains within-model layer CCA and per-layer root-pattern probe-weight CCA when present."
                )
            if kind == "RSA":
                lines.append(
                    "- Cross-model note: this file is not a pairwise cross-model RSA file; it contains within-model layer RSA."
                )
            lines.append("")

    if generated_rsa or generated_cca:
        lines.append("## Generated Geometry Charts")
        for path in generated_rsa + generated_cca:
            lines.append(f"- `{path.relative_to(output_dir)}`")
        lines.append("")

    atomic_write_text(output_dir / "geometry_npz_schema.md", "\n".join(lines) + "\n")


def write_pairwise_plan(run_dir: Path, output_dir: Path) -> None:
    hidden_dir = run_dir / "hidden_states"
    lines = [
        "# Pairwise Cross-Model Geometry Plan",
        "",
        "This document records proposed cross-model comparisons and validates only the saved activation shapes listed below.",
        "",
        "Existing scripts that appear relevant: `probes/cca_analysis.py`, `probes/rsa_analysis.py`, `probes/cross_model_geometry.py`.",
        "",
    ]
    for a, b in PAIRWISE_PLAN:
        path_a = hidden_dir / f"{a}_layers.npy"
        path_b = hidden_dir / f"{b}_layers.npy"
        lines.append(f"## `{a}` vs `{b}`")
        lines.append(f"- Expected input activation files: `{path_a}`, `{path_b}`.")
        lines.append(f"- Proposed output path: `{output_dir / 'pairwise' / f'{a}_vs_{b}_geometry.npz'}`.")
        if path_a.exists() and path_b.exists():
            shape_a, dtype_a, bytes_a = array_bytes(path_a)
            shape_b, dtype_b, bytes_b = array_bytes(path_b)
            lines.append(f"- Activation A: shape={shape_a}, dtype={dtype_a}, size={mib(bytes_a):.1f} MiB.")
            lines.append(f"- Activation B: shape={shape_b}, dtype={dtype_b}, size={mib(bytes_b):.1f} MiB.")
            same_rows = shape_a[0] == shape_b[0]
            same_dim = shape_a[2] == shape_b[2]
            lines.append(
                f"- Row-count compatibility: {'same count' if same_rows else 'different counts'} "
                f"({shape_a[0]} vs {shape_b[0]}). Matching row identities must still be "
                "verified from activation metadata before analysis."
            )
            lines.append(
                "- Layer dimensions: "
                + (
                    f"same hidden dimension ({shape_a[2]})."
                    if same_dim
                    else f"different hidden dimensions ({shape_a[2]} vs {shape_b[2]}); "
                    "CCA supports different feature dimensions and RSA compares pairwise "
                    "geometry once rows are aligned."
                )
            )
            pair_count = shape_a[1] * shape_b[1]
            lines.append(
                f"- Estimated compute/memory risk: moderate for CCA ({pair_count} layer pairs, "
                f"{mib(bytes_a + bytes_b):.1f} MiB raw activations); lower for RSA after RSM vectorization."
            )
            lines.append(
                "- Suggested venue: local is reasonable for a single pair with these saved arrays; use AWS for batch runs, repeated bootstraps, or larger activation sets."
            )
        else:
            lines.append("- Estimated compute/memory risk: unknown because one or both activation files are missing.")
            lines.append("- Layer compatibility: unknown.")
            lines.append("- Suggested venue: inspect locally first, then decide.")
        lines.append("")

    atomic_write_text(output_dir / "pairwise_geometry_plan.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metrics-dir", required=True, type=Path)
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

    active_models = [
        model
        for model in args.models
        if (args.metrics_dir / f"{model}_cca.npz").exists()
        or (args.metrics_dir / f"{model}_rsa.npz").exists()
    ]
    probe_paths = [args.metrics_dir / f"{model}_probes.npz" for model in active_models]
    missing_probes = [path for path in probe_paths if not path.is_file()]
    if missing_probes and not args.allow_unverified_inputs:
        raise FileNotFoundError(
            "geometry plots require linked probe artifacts for prompt-contract auditing: "
            + ", ".join(str(path) for path in missing_probes)
        )
    verified_probe_paths = [path for path in probe_paths if path.is_file()]
    statuses = enforce_probe_prompt_contracts(
        verified_probe_paths,
        allow_label_revealed=args.allow_label_revealed_inputs,
        allow_unverifiable=args.allow_unverified_inputs,
    )
    warning = None
    if "label_revealed" in statuses:
        warning = "POSITIVE CONTROL — LABEL-REVEALED PROMPTS"
    elif missing_probes or any(
        status in UNVERIFIABLE_PROMPT_AUDIT_STATUSES for status in statuses
    ):
        warning = "UNVERIFIED PROMPT CONTRACT"

    generated_rsa: list[Path] = []
    generated_cca: list[Path] = []
    for model in args.models:
        rsa_path = args.metrics_dir / f"{model}_rsa.npz"
        if rsa_path.exists():
            probe_path = args.metrics_dir / f"{model}_probes.npz"
            if probe_path.is_file():
                validate_metric_source(
                    rsa_path, probe_path, allow_unverified=args.allow_unverified_inputs
                )
            selected = matrix_key(rsa_path, "rsa_layer_matrix")
            if selected is not None:
                key, matrix = selected
                output = args.output_dir / "rsa" / f"{model}_rsa_heatmap.png"
                plot_heatmap(
                    matrix,
                    output,
                    f"RSA heatmap: {model}",
                    f"{key} correlation",
                    kind="rsa",
                    dark=args.dark,
                    warning=warning,
                )
                generated_rsa.append(output)

        cca_path = args.metrics_dir / f"{model}_cca.npz"
        if cca_path.exists():
            probe_path = args.metrics_dir / f"{model}_probes.npz"
            if probe_path.is_file():
                validate_metric_source(
                    cca_path, probe_path, allow_unverified=args.allow_unverified_inputs
                )
            selected = matrix_key(cca_path, "cca_layer_matrix")
            if selected is not None:
                key, matrix = selected
                output = args.output_dir / "cca" / f"{model}_cca_heatmap.png"
                plot_heatmap(
                    matrix,
                    output,
                    f"Cross-validated CCA heatmap: {model}",
                    f"{key} similarity",
                    kind="cca",
                    dark=args.dark,
                    warning=warning,
                )
                generated_cca.append(output)

    write_schema(args.metrics_dir, args.output_dir, args.models, generated_rsa, generated_cca)
    write_pairwise_plan(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
