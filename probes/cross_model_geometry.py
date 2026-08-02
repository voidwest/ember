#!/usr/bin/env python3
"""Run pairwise CCA/RSA geometry comparisons across saved activation tensors."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analysis_common import assert_row_alignment
    from .cca_analysis import cca_cross_model
    from .rsa_analysis import rsa_cross_model
    from .train_linear_probe import (
        atomic_savez,
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        load_rows,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from analysis_common import assert_row_alignment
    from cca_analysis import cca_cross_model
    from rsa_analysis import rsa_cross_model
    from train_linear_probe import (
        atomic_savez,
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        load_rows,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )


def load_activations(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        activations = np.load(source, mmap_mode="r", allow_pickle=False)
    elif source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if "activations" not in archive:
                raise ValueError(f"activation archive has no 'activations' array: {path}")
            activations = np.array(archive["activations"], copy=True)
    else:
        raise ValueError(f"unsupported activation format: {source.suffix}")
    activations = validate_activation_tensor(activations, path)
    return activations


def activation_prompt_audit(
    path: str,
    shape: tuple[int, int, int],
    tasks: list[str],
    *,
    allow_label_revealed: bool,
    allow_unverifiable: bool,
) -> dict:
    metadata = load_activation_metadata(path)
    if not metadata:
        if allow_unverifiable:
            return {"status": "unverifiable_missing_activation_metadata"}
        raise ValueError(f"activation prompt contract requires metadata for {path}")
    stimuli_path = metadata.get("stimuli_path", metadata.get("benchmark"))
    if not isinstance(stimuli_path, str) or not stimuli_path:
        if allow_unverifiable:
            return {"status": "unverifiable_missing_stimuli_path"}
        raise ValueError(f"activation metadata has no stimuli path: {path}")
    rows = load_rows(stimuli_path)
    validate_activation_provenance(
        path,
        shape,
        stimuli_path,
        metadata,
        require=not allow_unverifiable,
    )
    row_indices = metadata.get("row_indices")
    if row_indices is None and len(rows) == shape[0]:
        row_indices = list(range(len(rows)))
    if (
        not isinstance(row_indices, list)
        or len(row_indices) != shape[0]
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(rows)
            for index in row_indices
        )
        or len(set(row_indices)) != len(row_indices)
    ):
        if allow_unverifiable:
            return {"status": "unverifiable_row_mapping"}
        raise ValueError(f"activation metadata has invalid row_indices: {path}")
    selected_rows = [rows[index] for index in row_indices]
    return enforce_prompt_contract(
        selected_rows,
        tasks,
        metadata,
        allow_label_revealed=allow_label_revealed,
        allow_unverifiable=allow_unverifiable,
        context=f"cross-model geometry ({path})",
    )


def normalized_layer_alignment(matrix: np.ndarray) -> list[dict]:
    rows = []
    if matrix.size == 0:
        return rows
    denom_a = max(matrix.shape[0] - 1, 1)
    denom_b = max(matrix.shape[1] - 1, 1)
    for layer_a in range(matrix.shape[0]):
        layer_b = int(np.argmax(matrix[layer_a]))
        maximum = float(matrix[layer_a, layer_b])
        rows.append(
            {
                "layer_a": layer_a,
                "layer_b": layer_b,
                "layer_a_norm": layer_a / denom_a,
                "layer_b_norm": layer_b / denom_b,
                "score": maximum,
                "n_tied_best_layers": int(
                    np.count_nonzero(np.isclose(matrix[layer_a], maximum, rtol=0.0, atol=1e-12))
                ),
                "selection": "descriptive_rowwise_argmax_same_matrix",
            }
        )
    return rows


def parse_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL:ACTIVATIONS")
    label, path = value.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must be LABEL:ACTIVATIONS")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in label):
        raise argparse.ArgumentTypeError("model labels may contain only letters, numbers, '_' and '-'")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description="pairwise cross-model CCA/RSA")
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        metavar="LABEL:ACTIVATIONS",
        help="model label and activation .npy/.npz path; may be repeated",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="correlation")
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--assume-row-aligned", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["root", "pattern"],
        help="target fields used only to audit whether source prompts revealed labels",
    )
    parser.add_argument("--allow-label-revealed-inputs", action="store_true")
    parser.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    args = parser.parse_args()

    if len(args.model) < 2:
        parser.error("at least two --model arguments are required")
    labels = [label for label, _ in args.model]
    if len(labels) != len(set(labels)):
        parser.error("model labels must be unique")
    if args.metric not in {"correlation", "cosine", "euclidean"}:
        parser.error("--metric must be correlation, cosine, or euclidean")
    if args.n_components < 1:
        parser.error("--n-components must be at least 1")
    if not np.isfinite(args.reg) or args.reg < 0.0:
        parser.error("--reg must be finite and non-negative")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if not args.tasks or len(args.tasks) != len(set(args.tasks)) or any(
        not isinstance(task, str) or not task for task in args.tasks
    ):
        parser.error("--tasks must be non-empty and unique")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activations = {
        label: load_activations(path)
        for label, path in args.model
    }
    prompt_audits = {
        label: activation_prompt_audit(
            path,
            tuple(activations[label].shape),
            args.tasks,
            allow_label_revealed=args.allow_label_revealed_inputs,
            allow_unverifiable=args.allow_unverifiable_prompt_contract,
        )
        for label, path in args.model
    }
    manifest = {
        "schema_version": 3,
        "models": [
            {
                "label": label,
                "path": path,
                "sha256": sha256_file(path),
                "shape": list(activations[label].shape),
                "prompt_leakage_audit": prompt_audits[label],
            }
            for label, path in args.model
        ],
        "pairs": [],
        "status": "running",
    }

    manifest_path = out_dir / "cross_model_geometry_manifest.json"

    def checkpoint() -> None:
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )

    checkpoint()

    paths = dict(args.model)
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            acts_a = activations[label_a]
            acts_b = activations[label_b]
            alignment_evidence = assert_row_alignment(
                paths[label_a],
                paths[label_b],
                acts_a.shape[0],
                allow_assumed=args.assume_row_aligned,
            )
            pair_name = f"{label_a}__{label_b}"
            npz_path = out_dir / f"{pair_name}_geometry.npz"
            pair_record = {
                "label_a": label_a,
                "label_b": label_b,
                "path": str(npz_path),
                "status": "running",
            }
            manifest["pairs"].append(pair_record)
            checkpoint()
            try:
                cca = cca_cross_model(
                    acts_a,
                    acts_b,
                    n_components=args.n_components,
                    reg=args.reg,
                    cv_folds=args.cv_folds,
                )
                rsa = rsa_cross_model(acts_a, acts_b, args.metric)
            except BaseException:
                pair_record["status"] = "failed"
                checkpoint()
                raise
            atomic_savez(
                npz_path,
                schema_version=np.array(3, dtype=np.int64),
                cca_cross_model=cca,
                rsa_cross_model=rsa,
                alignment_evidence=np.array(alignment_evidence),
                cca_evaluation=np.array(
                    "deterministic_cross_validated_regularized_cca"
                ),
                cca_cv_folds=np.array(args.cv_folds, dtype=np.int64),
                rsa_evaluation=np.array(
                    "descriptive_pairwise_rdm_pearson_correlation"
                ),
                activations_a_sha256=np.array(sha256_file(paths[label_a])),
                activations_b_sha256=np.array(sha256_file(paths[label_b])),
                activations_a_shape=np.asarray(acts_a.shape, dtype=np.int64),
                activations_b_shape=np.asarray(acts_b.shape, dtype=np.int64),
                prompt_audit_a_json=np.array(
                    json.dumps(prompt_audits[label_a], sort_keys=True, allow_nan=False)
                ),
                prompt_audit_b_json=np.array(
                    json.dumps(prompt_audits[label_b], sort_keys=True, allow_nan=False)
                ),
            )
            pair_record.update(
                {
                    "status": "completed",
                    "sha256": sha256_file(npz_path),
                    "cca_shape": list(cca.shape),
                    "rsa_shape": list(rsa.shape),
                    "alignment_evidence": alignment_evidence,
                    "cca_alignment": normalized_layer_alignment(cca),
                    "rsa_alignment": normalized_layer_alignment(rsa),
                }
            )
            checkpoint()

    manifest["status"] = "completed"
    manifest["cca"] = {
        "n_components": args.n_components,
        "regularization": args.reg,
        "cv_folds": args.cv_folds,
        "evaluation": "deterministic_cross_validated_regularized_cca",
        "negative_heldout_correlation_policy": "clip_to_zero",
    }
    manifest["rsa"] = {
        "metric": args.metric,
        "evaluation": "descriptive_pairwise_rdm_pearson_correlation",
    }
    manifest["tasks_for_prompt_audit"] = args.tasks
    checkpoint()
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
