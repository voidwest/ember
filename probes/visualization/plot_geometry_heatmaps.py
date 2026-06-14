"""Inspect saved CCA/RSA NPZ files and plot supported layerwise heatmaps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
        }
    )


@dataclass
class ArraySummary:
    key: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | None
    maximum: float | None
    mean: float | None


def summarize_npz(path: Path) -> list[ArraySummary]:
    data = np.load(path, allow_pickle=True)
    summaries: list[ArraySummary] = []
    for key in data.files:
        arr = data[key]
        if np.issubdtype(arr.dtype, np.number):
            summaries.append(
                ArraySummary(
                    key=key,
                    shape=arr.shape,
                    dtype=str(arr.dtype),
                    minimum=float(np.nanmin(arr)),
                    maximum=float(np.nanmax(arr)),
                    mean=float(np.nanmean(arr)),
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
    return summaries


def matrix_key(path: Path, preferred_key: str) -> tuple[str, np.ndarray] | None:
    data = np.load(path, allow_pickle=True)
    if preferred_key in data.files:
        arr = np.asarray(data[preferred_key], dtype=float)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return preferred_key, arr
    square = []
    for key in data.files:
        arr = np.asarray(data[key])
        if np.issubdtype(arr.dtype, np.number) and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            square.append((key, np.asarray(arr, dtype=float)))
    if len(square) == 1:
        return square[0]
    return None


def plot_heatmap(matrix: np.ndarray, output: Path, title: str, colorbar_label: str, dark: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=160)
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    if dark:
        ax.tick_params(colors=DIM)
        cbar.ax.tick_params(colors=DIM)
        cbar.ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG if dark else "white")
    plt.close(fig)


def array_bytes(path: Path) -> tuple[tuple[int, ...], str, int]:
    arr = np.load(path, mmap_mode="r")
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
                label = "similarity" if float(np.nanmean(np.diag(matrix))) >= 0.9 else "score"
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

    (output_dir / "geometry_npz_schema.md").write_text("\n".join(lines) + "\n")


def write_pairwise_plan(run_dir: Path, output_dir: Path) -> None:
    hidden_dir = run_dir / "hidden_states"
    lines = [
        "# Pairwise Cross-Model Geometry Plan",
        "",
        "No pairwise cross-model CCA/RSA output files were found in the completed run, so no full pairwise geometry job was launched.",
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
                f"- Row compatibility: {'compatible' if same_rows else 'requires row alignment'} "
                f"({shape_a[0]} vs {shape_b[0]} stimuli)."
            )
            lines.append(
                "- Layer dimensions: "
                + (
                    f"same hidden dimension ({shape_a[2]})."
                    if same_dim
                    else f"different hidden dimensions ({shape_a[2]} vs {shape_b[2]}); CCA requires projection/alignment, RSA can compare RSM vectors after row alignment."
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

    (output_dir / "pairwise_geometry_plan.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--dark", action="store_true", help="use voidwest dark chart styling")
    args = parser.parse_args()
    if args.dark:
        setup_dark_theme()

    generated_rsa: list[Path] = []
    generated_cca: list[Path] = []
    for model in args.models:
        rsa_path = args.metrics_dir / f"{model}_rsa.npz"
        if rsa_path.exists():
            selected = matrix_key(rsa_path, "rsa_layer_matrix")
            if selected is not None:
                key, matrix = selected
                output = args.output_dir / "rsa" / f"{model}_rsa_heatmap.png"
                plot_heatmap(matrix, output, f"RSA heatmap: {model}", f"{key} similarity", dark=args.dark)
                generated_rsa.append(output)

        cca_path = args.metrics_dir / f"{model}_cca.npz"
        if cca_path.exists():
            selected = matrix_key(cca_path, "cca_layer_matrix")
            if selected is not None:
                key, matrix = selected
                output = args.output_dir / "cca" / f"{model}_cca_heatmap.png"
                plot_heatmap(matrix, output, f"CCA heatmap: {model}", f"{key} similarity", dark=args.dark)
                generated_cca.append(output)

    write_schema(args.metrics_dir, args.output_dir, args.models, generated_rsa, generated_cca)
    write_pairwise_plan(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
