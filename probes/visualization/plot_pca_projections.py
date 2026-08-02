"""Generate PCA projection charts from saved hidden-state arrays when labels map safely."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
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
sys.path.insert(0, str(ROOT / "probes"))
try:
    from ..train_linear_probe import (
        atomic_save_figure,
        atomic_write_text,
        enforce_prompt_contract,
        sha256_file,
    )
except ImportError:  # direct script execution
    from train_linear_probe import (  # noqa: E402
        atomic_save_figure,
        atomic_write_text,
        enforce_prompt_contract,
        sha256_file,
    )

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
SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def setup_dark_theme() -> None:
    apply_matplotlib_theme(dark=True)


def load_peak_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"model", "task", "peak_layer"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"peak table is missing columns: {sorted(required)}")
        result = {}
        for line_number, row in enumerate(reader, start=2):
            key = (row["model"], row["task"])
            if key in result:
                raise ValueError(f"duplicate peak table row {key!r} at line {line_number}")
            result[key] = row
        return result


def resolve_repo_path(path_text: str, repo_root: Path, *, allow_external: bool = False) -> Path:
    path = Path(path_text)
    resolved = (path if path.is_absolute() else repo_root / path).resolve()
    if not allow_external and not resolved.is_relative_to(repo_root):
        raise ValueError(f"metadata path escapes repository root: {path_text}")
    return resolved


def load_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )


def validate_mapping(
    model: str,
    run_dir: Path,
    repo_root: Path,
    activations: np.ndarray,
    *,
    allow_external_paths: bool = False,
    allow_unverified_metadata: bool = False,
    allow_label_revealed: bool = False,
) -> tuple[bool, list[str], list[dict[str, str]], dict]:
    reasons: list[str] = []
    metadata_path = run_dir / "hidden_states" / f"{model}_layers_metadata.json"
    if not metadata_path.exists():
        return False, [f"missing metadata file `{metadata_path}`"], [], {}
    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        return False, [f"metadata is not a JSON object: `{metadata_path}`"], [], {}

    if tuple(metadata.get("activation_shape", ())) != tuple(activations.shape):
        reasons.append(
            f"metadata activation_shape={metadata.get('activation_shape')} does not match actual shape={activations.shape}"
        )
    hidden_path = run_dir / "hidden_states" / f"{model}_layers.npy"
    recorded_activation_sha = metadata.get("activations_sha256")
    if recorded_activation_sha is None:
        if not allow_unverified_metadata:
            reasons.append("metadata has no activations_sha256")
    elif recorded_activation_sha != sha256_file(hidden_path):
        reasons.append("metadata activations_sha256 does not match the hidden-state file")

    stimuli_text = metadata.get("stimuli_path")
    if not stimuli_text:
        reasons.append("metadata has no stimuli_path")
        return False, reasons, [], metadata
    stimuli_path = resolve_repo_path(
        stimuli_text, repo_root, allow_external=allow_external_paths
    )
    if not stimuli_path.exists():
        reasons.append(f"stimuli file not found: `{stimuli_path}`")
        return False, reasons, [], metadata
    source_stimuli = load_json(stimuli_path)
    if not isinstance(source_stimuli, list):
        reasons.append(f"stimuli file is not a list: `{stimuli_path}`")
        return False, reasons, [], metadata
    row_indices = metadata.get("row_indices")
    if row_indices is None and len(source_stimuli) == activations.shape[0]:
        row_indices = list(range(len(source_stimuli)))
    if (
        not isinstance(row_indices, list)
        or len(row_indices) != activations.shape[0]
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(source_stimuli)
            for index in row_indices
        )
        or len(set(row_indices)) != len(row_indices)
    ):
        reasons.append("metadata row_indices do not map activation rows to source stimuli")
        return False, reasons, [], metadata
    stimuli = [source_stimuli[index] for index in row_indices]
    recorded_sha = metadata.get("stimuli_sha256")
    if recorded_sha is None:
        if not allow_unverified_metadata:
            reasons.append("metadata has no stimuli_sha256")
    elif recorded_sha != sha256_file(stimuli_path):
        reasons.append("metadata stimuli_sha256 does not match the stimuli file")

    for i, item in enumerate(stimuli):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("root"), str)
            or not item.get("root")
            or not isinstance(item.get("pattern"), str)
            or not item.get("pattern")
        ):
            reasons.append(f"stimulus row {i} lacks root/pattern labels")
            break

    token_selections = metadata.get("token_selections")
    if not isinstance(token_selections, list) or len(token_selections) != activations.shape[0]:
        reasons.append("metadata token_selections length does not match activation rows")
    else:
        bad = [
            i
            for i, (item, source_index) in enumerate(
                zip(token_selections, row_indices, strict=True)
            )
            if not isinstance(item, dict) or item.get("index") != source_index
        ]
        if bad:
            reasons.append(f"token_selections index mismatch at first bad row {bad[0]}")

    correctness_text = metadata.get("correctness_path")
    if correctness_text:
        correctness_path = resolve_repo_path(
            correctness_text, repo_root, allow_external=allow_external_paths
        )
        if correctness_path.exists():
            correctness = load_json(correctness_path)
            if not isinstance(correctness, list) or len(correctness) != activations.shape[0]:
                reasons.append("correctness file length does not match activation rows")
            else:
                for i, (stim, corr, source_index) in enumerate(
                    zip(stimuli, correctness, row_indices, strict=True)
                ):
                    if not isinstance(corr, dict):
                        reasons.append(f"correctness row {i} is not an object")
                        break
                    if corr.get("index") != source_index:
                        reasons.append(f"correctness index mismatch at row {i}")
                        break
                    if corr.get("root") != stim.get("root") or corr.get("pattern") != stim.get("pattern"):
                        reasons.append(f"correctness labels do not match stimuli at row {i}")
                        break
            recorded_correctness_sha = metadata.get("correctness_sha256")
            if recorded_correctness_sha is None:
                if not allow_unverified_metadata:
                    reasons.append("metadata has no correctness_sha256")
            elif recorded_correctness_sha != sha256_file(correctness_path):
                reasons.append("metadata correctness_sha256 does not match correctness file")
        else:
            reasons.append(f"correctness file not found: `{correctness_path}`")

    try:
        prompt_audit = enforce_prompt_contract(
            stimuli,
            ["root", "pattern"],
            metadata,
            allow_label_revealed=allow_label_revealed,
            allow_unverifiable=allow_unverified_metadata,
            context="PCA visualization",
        )
        metadata = {**metadata, "pca_prompt_leakage_audit": prompt_audit}
    except ValueError as error:
        reasons.append(str(error))

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
    warning: str | None = None,
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
    title = f"PCA projection: {model} layer {layer} ({role}) by {label_name}"
    ax.set_title(f"{warning}\n{title}" if warning else title)
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
    atomic_save_figure(fig, output, facecolor=BG if dark else "white")
    plt.close(fig)


def silhouette_or_none(xy: np.ndarray, labels: list[str]) -> float | None:
    counts = {label: labels.count(label) for label in set(labels)}
    if len(counts) < 2 or len(counts) >= len(labels) or min(counts.values()) < 2:
        return None
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
    atomic_write_text(output_dir / "projection_plan.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--peak-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--dark", action="store_true", help="use voidwest dark chart styling")
    parser.add_argument("--allow-external-metadata-paths", action="store_true")
    parser.add_argument("--allow-unverified-metadata", action="store_true")
    parser.add_argument("--allow-label-revealed-inputs", action="store_true")
    args = parser.parse_args()
    if not args.models or len(args.models) != len(set(args.models)) or any(
        not SAFE_MODEL.fullmatch(model) for model in args.models
    ):
        parser.error("--models must be non-empty unique safe identifiers")
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

        activations = np.load(hidden_path, mmap_mode="r", allow_pickle=False)
        if activations.ndim != 3 or any(size <= 0 for size in activations.shape):
            raise ValueError(f"invalid activation tensor: {hidden_path} {activations.shape}")
        for start in range(0, activations.shape[0], 256):
            if not np.isfinite(activations[start : start + 256]).all():
                raise ValueError(f"activation tensor contains non-finite values: {hidden_path}")
        ok, reasons, stimuli, metadata = validate_mapping(
            model,
            args.run_dir,
            repo_root,
            activations,
            allow_external_paths=args.allow_external_metadata_paths,
            allow_unverified_metadata=args.allow_unverified_metadata,
            allow_label_revealed=args.allow_label_revealed_inputs,
        )
        if ok:
            layers = selected_layers(model, peak_rows, activations.shape[1])
            layer_text = ", ".join(f"{role}={layer}" for role, layer in layers)
            audit = metadata["pca_prompt_leakage_audit"]
            warning = (
                "POSITIVE CONTROL — LABEL-REVEALED PROMPTS"
                if audit["status"] == "label_revealed"
                else "UNVERIFIED PROMPT CONTRACT"
                if audit["status"] == "not_checked_missing_probe_template_metadata"
                else None
            )
            plan_entries.append(
                {
                    "model": model,
                    "hidden_path": hidden_path,
                    "shape": tuple(activations.shape),
                    "row_mapping": (
                        f"clear: row i matches `{metadata.get('stimuli_path')}` item i; "
                        "token_selections and correctness indices are sequential"
                    ),
                    "labels": "loaded from stimulus `root` and `pattern` fields",
                    "selected_layers": layer_text,
                    "pca_status": "generated",
                    "caveats": (
                        "PCA and silhouette values are descriptive; PCA uses the same rows it displays. "
                        f"Prompt audit status: {audit['status']}."
                    ),
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
            if x.shape[0] < 3 or x.shape[1] < 2 or np.allclose(np.var(x, axis=0), 0.0):
                raise ValueError(f"{model} layer {layer} cannot support a 2-D PCA projection")
            pca = PCA(n_components=2, svd_solver="full")
            xy = pca.fit_transform(x)
            variance = pca.explained_variance_ratio_
            if not np.isfinite(xy).all() or not np.isfinite(variance).all():
                raise ValueError(f"PCA produced non-finite output for {model} layer {layer}")
            for label_name, labels in (("root", roots), ("pattern", patterns)):
                output = args.output_dir / "pca" / f"{model}_layer_{layer}_by_{label_name}.png"
                plot_projection(
                    xy,
                    labels,
                    label_name,
                    model,
                    layer,
                    role,
                    variance,
                    output,
                    dark=args.dark,
                    warning=warning,
                )
                metric_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "layer_role": role,
                        "label": label_name,
                        "pc1_explained_variance": variance[0],
                        "pc2_explained_variance": variance[1],
                        "silhouette_score": silhouette_or_none(xy, labels),
                        "n_samples": len(labels),
                        "n_classes": len(set(labels)),
                        "prompt_contract_status": audit["status"],
                        "evaluation": "descriptive_in_sample_pca_and_silhouette",
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
        "prompt_contract_status",
        "evaluation",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metric_rows)
    atomic_write_text(metrics_path, buffer.getvalue())


if __name__ == "__main__":
    main()
