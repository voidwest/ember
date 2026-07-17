"""Generate PCA projection charts from saved hidden-state arrays when labels map safely."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import DARK, apply_matplotlib_theme, categorical_cmap  # noqa: E402

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


def setup_dark_theme() -> None:
    apply_matplotlib_theme(dark=True)


def load_peak_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as f:
        return {(row["model"], row["task"]): row for row in csv.DictReader(f)}


def resolve_repo_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_mapping(
    model: str,
    run_dir: Path,
    repo_root: Path,
    activations: np.ndarray,
) -> tuple[bool, list[str], list[dict[str, str]], dict]:
    reasons: list[str] = []
    metadata_path = run_dir / "hidden_states" / f"{model}_layers_metadata.json"
    if not metadata_path.exists():
        return False, [f"missing metadata file `{metadata_path}`"], [], {}
    metadata = load_json(metadata_path)

    if tuple(metadata.get("activation_shape", ())) != tuple(activations.shape):
        reasons.append(
            f"metadata activation_shape={metadata.get('activation_shape')} does not match actual shape={activations.shape}"
        )

    stimuli_text = metadata.get("stimuli_path")
    if not stimuli_text:
        reasons.append("metadata has no stimuli_path")
        return False, reasons, [], metadata
    stimuli_path = resolve_repo_path(stimuli_text, repo_root)
    if not stimuli_path.exists():
        reasons.append(f"stimuli file not found: `{stimuli_path}`")
        return False, reasons, [], metadata
    stimuli = load_json(stimuli_path)
    if not isinstance(stimuli, list):
        reasons.append(f"stimuli file is not a list: `{stimuli_path}`")
        return False, reasons, [], metadata
    if len(stimuli) != activations.shape[0]:
        reasons.append(f"stimuli rows={len(stimuli)} but activations rows={activations.shape[0]}")

    for i, item in enumerate(stimuli):
        if not isinstance(item, dict) or "root" not in item or "pattern" not in item:
            reasons.append(f"stimulus row {i} lacks root/pattern labels")
            break

    token_selections = metadata.get("token_selections")
    if not isinstance(token_selections, list) or len(token_selections) != activations.shape[0]:
        reasons.append("metadata token_selections length does not match activation rows")
    else:
        bad = [
            i
            for i, item in enumerate(token_selections)
            if not isinstance(item, dict) or item.get("index") != i
        ]
        if bad:
            reasons.append(f"token_selections index mismatch at first bad row {bad[0]}")

    correctness_text = metadata.get("correctness_path")
    if correctness_text:
        correctness_path = resolve_repo_path(correctness_text, repo_root)
        if correctness_path.exists():
            correctness = load_json(correctness_path)
            if not isinstance(correctness, list) or len(correctness) != activations.shape[0]:
                reasons.append("correctness file length does not match activation rows")
            else:
                for i, (stim, corr) in enumerate(zip(stimuli, correctness, strict=True)):
                    if corr.get("index") != i:
                        reasons.append(f"correctness index mismatch at row {i}")
                        break
                    if corr.get("root") != stim.get("root") or corr.get("pattern") != stim.get("pattern"):
                        reasons.append(f"correctness labels do not match stimuli at row {i}")
                        break
        else:
            reasons.append(f"correctness file not found: `{correctness_path}`")

    return not reasons, reasons, stimuli, metadata


def selected_layers(model: str, rows: dict[tuple[str, str], dict[str, str]], n_layers: int) -> list[tuple[str, int]]:
    root_peak = int(rows[(model, "root")]["peak_layer"])
    choices = [("early", 0), ("root_peak", root_peak), ("final", n_layers - 1)]
    seen: set[int] = set()
    deduped: list[tuple[str, int]] = []
    for role, layer in choices:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"{model}: selected {role} layer {layer} outside n_layers={n_layers}")
        if layer not in seen:
            deduped.append((role, layer))
            seen.add(layer)
    return deduped


def label_colors(labels: list[str], *, dark: bool = False) -> dict[str, tuple[float, float, float, float]]:
    unique = sorted(set(labels))
    cmap = categorical_cmap(dark=dark)
    return {label: cmap(i % cmap.N) for i, label in enumerate(unique)}


def plot_projection(
    xy: np.ndarray,
    labels: list[str],
    label_name: str,
    model: str,
    layer: int,
    role: str,
    variance: np.ndarray,
    output: Path,
    dark: bool = False,
) -> None:
    colors = label_colors(labels, dark=dark)
    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=160)
    for label in sorted(colors):
        idx = np.array([value == label for value in labels])
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=22,
            alpha=0.82,
            color=colors[label],
            edgecolors="none",
            label=label,
        )
    ax.axhline(0, color=DIM if dark else "#cccccc", linewidth=0.8)
    ax.axvline(0, color=DIM if dark else "#cccccc", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% var.)")
    ax.set_title(f"PCA projection: {model} layer {layer} ({role}) by {label_name}")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        markerscale=0.9,
    )
    if dark:
        ax.tick_params(colors=DIM)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG if dark else "white")
    plt.close(fig)


def silhouette_or_nan(xy: np.ndarray, labels: list[str]) -> float:
    unique = set(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(xy, labels))


def write_projection_plan(output_dir: Path, entries: list[dict[str, object]]) -> None:
    lines = [
        "# Projection Plan",
        "",
        "PCA plots are generated only when hidden-state rows can be matched to the stimuli and labels with metadata checks.",
        "",
    ]
    for entry in entries:
        lines.append(f"## {entry['model']}")
        lines.append(f"- Hidden-state array: `{entry['hidden_path']}`.")
        lines.append(f"- Expected/observed shape: `{entry['shape']}`.")
        lines.append(f"- Row mapping: {entry['row_mapping']}")
        lines.append(f"- Root/pattern labels: {entry['labels']}")
        lines.append(f"- Selected layers: {entry['selected_layers']}")
        lines.append(f"- PCA generation: {entry['pca_status']}")
        if entry.get("caveats"):
            lines.append(f"- Caveats: {entry['caveats']}")
        lines.append("")
    (output_dir / "projection_plan.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--peak-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--dark", action="store_true", help="use voidwest dark chart styling")
    args = parser.parse_args()
    if args.dark:
        setup_dark_theme()

    repo_root = args.repo_root.resolve()
    peak_rows = load_peak_rows(args.peak_table)
    plan_entries: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for model in args.models:
        hidden_path = args.run_dir / "hidden_states" / f"{model}_layers.npy"
        if not hidden_path.exists():
            plan_entries.append(
                {
                    "model": model,
                    "hidden_path": hidden_path,
                    "shape": "missing",
                    "row_mapping": "not checked",
                    "labels": "not loaded",
                    "selected_layers": "not selected",
                    "pca_status": "skipped",
                    "caveats": "hidden-state file is missing",
                }
            )
            continue

        activations = np.load(hidden_path, mmap_mode="r")
        ok, reasons, stimuli, metadata = validate_mapping(model, args.run_dir, repo_root, activations)
        if ok:
            layers = selected_layers(model, peak_rows, activations.shape[1])
            layer_text = ", ".join(f"{role}={layer}" for role, layer in layers)
            plan_entries.append(
                {
                    "model": model,
                    "hidden_path": hidden_path,
                    "shape": tuple(activations.shape),
                    "row_mapping": (
                        f"clear: row i matches `stimuli/nonce_root_pattern.json` item i; "
                        "token_selections and correctness indices are sequential"
                    ),
                    "labels": "loaded from stimulus `root` and `pattern` fields",
                    "selected_layers": layer_text,
                    "pca_status": "generated",
                    "caveats": "PCA is illustrative and uses only the first two principal components.",
                }
            )
        else:
            plan_entries.append(
                {
                    "model": model,
                    "hidden_path": hidden_path,
                    "shape": tuple(activations.shape),
                    "row_mapping": "unclear",
                    "labels": "not safely loaded",
                    "selected_layers": "not selected",
                    "pca_status": "skipped",
                    "caveats": "; ".join(reasons),
                }
            )
            continue

        roots = [str(item["root"]) for item in stimuli]
        patterns = [str(item["pattern"]) for item in stimuli]
        for role, layer in layers:
            x = np.asarray(activations[:, layer, :], dtype=np.float32)
            pca = PCA(n_components=2)
            xy = pca.fit_transform(x)
            variance = pca.explained_variance_ratio_
            for label_name, labels in (("root", roots), ("pattern", patterns)):
                output = args.output_dir / "pca" / f"{model}_layer_{layer}_by_{label_name}.png"
                plot_projection(xy, labels, label_name, model, layer, role, variance, output, dark=args.dark)
                metric_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "layer_role": role,
                        "label": label_name,
                        "pc1_explained_variance": variance[0],
                        "pc2_explained_variance": variance[1],
                        "silhouette_score": silhouette_or_nan(xy, labels),
                        "n_samples": len(labels),
                        "n_classes": len(set(labels)),
                    }
                )

    write_projection_plan(args.output_dir, plan_entries)
    metrics_path = args.output_dir / "pca" / "pca_cluster_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "layer",
        "layer_role",
        "label",
        "pc1_explained_variance",
        "pc2_explained_variance",
        "silhouette_score",
        "n_samples",
        "n_classes",
    ]
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)


if __name__ == "__main__":
    main()
