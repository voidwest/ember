"""correct-vs-incorrect divergence analysis.

Descriptively measures the layerwise separation between hidden states for
correct and incorrect generations. Separation does not identify when a model
"figures out" a task and is not an inferential or causal statistic.
"""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    from .train_linear_probe import (
        atomic_savez,
        load_activation_metadata,
        load_activations as load_validated_activations,
        sha256_file,
    )
except ImportError:  # direct script execution
    from train_linear_probe import (
        atomic_savez,
        load_activation_metadata,
        load_activations as load_validated_activations,
        sha256_file,
    )


def load_activations(path: str) -> np.ndarray:
    return load_validated_activations(path)


def divergence_curves(
    activations: np.ndarray,
    correctness: list[dict],
) -> dict:
    """compute per-layer distance between correct and incorrect mean states.

    returns dict with keys: layer, cos_dist, eucl_dist
    each is (n_layers,) array.
    """
    n_stimuli, n_layers, hidden_dim = activations.shape
    if not isinstance(correctness, list) or len(correctness) != n_stimuli:
        raise ValueError(
            f"correctness rows must match activations: {len(correctness) if isinstance(correctness, list) else 'not-a-list'} vs {n_stimuli}"
        )

    decisions = []
    for index, record in enumerate(correctness):
        if not isinstance(record, dict):
            raise ValueError(f"correctness row {index} must be an object")
        if "correct" in record:
            if not isinstance(record["correct"], bool):
                raise ValueError(f"correctness row {index} field 'correct' must be boolean")
            decisions.append(record["correct"])
        else:
            predicted = record.get("predicted")
            expected = record.get("expected")
            if not isinstance(predicted, str) or not isinstance(expected, str):
                raise ValueError(f"correctness row {index} lacks string predicted/expected fields")
            decisions.append(predicted.strip() == expected.strip())
    correct_mask = np.array(decisions, dtype=bool)
    n_correct = correct_mask.sum()
    n_incorrect = n_stimuli - n_correct

    if n_correct == 0 or n_incorrect == 0:
        raise ValueError(
            f"divergence requires both classes; found {n_correct} correct and {n_incorrect} incorrect"
        )

    cos_dist = np.zeros(n_layers)
    eucl_dist = np.zeros(n_layers)

    for layer in range(n_layers):
        correct_states = activations[correct_mask, layer, :]
        incorrect_states = activations[~correct_mask, layer, :]

        mean_correct = correct_states.mean(axis=0)
        mean_incorrect = incorrect_states.mean(axis=0)

        # cosine distance: 1 - cosine similarity
        dot = np.dot(mean_correct, mean_incorrect)
        norm_c = np.linalg.norm(mean_correct)
        norm_i = np.linalg.norm(mean_incorrect)
        if norm_c > 0 and norm_i > 0:
            cos_dist[layer] = 1.0 - dot / (norm_c * norm_i)
        else:
            raise ValueError(f"layer {layer} has a zero-norm group mean")

        eucl_dist[layer] = np.linalg.norm(mean_correct - mean_incorrect)

    return {
        "layer": np.arange(n_layers),
        "cos_dist": cos_dist,
        "eucl_dist": eucl_dist,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
    }


def load_strict_json(path: str):
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def validate_correctness_alignment(
    correctness: list[dict],
    metadata: dict,
    activation_path: str,
    correctness_path: str,
    row_count: int,
    *,
    allow_unlinked: bool,
) -> dict:
    """Bind behavioral rows to the exact activation tensor and row order."""
    if not isinstance(correctness, list) or len(correctness) != row_count:
        raise ValueError(
            f"correctness rows must match activations: "
            f"{len(correctness) if isinstance(correctness, list) else 'not-a-list'} vs {row_count}"
        )

    if metadata:
        declared_shape = metadata.get("activation_shape")
        if not isinstance(declared_shape, list) or len(declared_shape) != 3:
            raise ValueError("activation metadata has an invalid activation_shape")
        if declared_shape[0] != row_count:
            raise ValueError("activation metadata row count does not match activations")
        declared_activation_sha = metadata.get("activations_sha256")
        if declared_activation_sha is not None and (
            declared_activation_sha != sha256_file(activation_path)
        ):
            raise ValueError("activation SHA-256 does not match activation metadata")

    actual_correctness_sha = sha256_file(correctness_path)
    declared_correctness_sha = metadata.get("correctness_sha256") if metadata else None
    if declared_correctness_sha is None:
        if not allow_unlinked:
            raise ValueError(
                "activation metadata does not cryptographically link the correctness artifact; "
                "re-extract with current Ember or pass --allow-unlinked-correctness only after "
                "external row verification"
            )
        link_evidence = "user_allowed_legacy_unlinked_correctness"
    elif declared_correctness_sha != actual_correctness_sha:
        raise ValueError("correctness SHA-256 does not match activation metadata")
    else:
        link_evidence = "metadata_correctness_sha256"

    expected_indices = metadata.get("row_indices") if metadata else None
    if not isinstance(expected_indices, list) or len(expected_indices) != row_count:
        expected_indices = list(range(row_count))
    for row, (expected_index, record) in enumerate(
        zip(expected_indices, correctness, strict=True)
    ):
        if not isinstance(expected_index, int) or isinstance(expected_index, bool):
            raise ValueError(f"invalid activation row index at row {row}")
        if not isinstance(record, dict):
            raise ValueError(f"correctness row {row} must be an object")
        if record.get("index") != expected_index:
            raise ValueError(
                f"correctness identity mismatch at row {row}: expected source index "
                f"{expected_index}, got {record.get('index')!r}"
            )
        for field in ("probe_template", "probe_position"):
            declared = metadata.get(field) if metadata else None
            if declared is not None and record.get(field) != declared:
                raise ValueError(
                    f"correctness {field} mismatch at row {row}: "
                    f"metadata={declared!r}, row={record.get(field)!r}"
                )
    return {
        "evidence": link_evidence,
        "correctness_sha256": actual_correctness_sha,
        "row_indices_verified": True,
        "probe_template": metadata.get("probe_template") if metadata else None,
        "probe_position": metadata.get("probe_position") if metadata else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="divergence analysis of correct vs incorrect predictions"
    )
    parser.add_argument(
        "--activations", required=True,
        help="path to activations .npy or .npz"
    )
    parser.add_argument(
        "--correctness", required=True,
        help="path to correctness.json (from ember --probe)"
    )
    parser.add_argument(
        "--output", default="data/divergence.npz",
        help="path to save results"
    )
    parser.add_argument(
        "--allow-unlinked-correctness",
        action="store_true",
        help="permit legacy correctness without a matching SHA-256 in activation metadata",
    )
    args = parser.parse_args()

    activations = load_activations(args.activations)
    correctness = load_strict_json(args.correctness)
    activation_metadata = load_activation_metadata(args.activations)

    n_stimuli, n_layers, _ = activations.shape
    print(f"activations: {activations.shape}")
    print(f"correctness records: {len(correctness)}")

    alignment = validate_correctness_alignment(
        correctness,
        activation_metadata,
        args.activations,
        args.correctness,
        n_stimuli,
        allow_unlinked=args.allow_unlinked_correctness,
    )
    results = divergence_curves(activations, correctness)
    results["schema_version"] = np.array(3, dtype=np.int64)
    results["evaluation"] = np.array("descriptive_group_mean_separation")
    results["alignment_evidence_json"] = np.array(
        json.dumps(alignment, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    results["activations_sha256"] = np.array(sha256_file(args.activations))
    results["correctness_sha256"] = np.array(sha256_file(args.correctness))
    results["activation_shape"] = np.asarray(activations.shape, dtype=np.int64)

    print(f"\ncorrect: {results['n_correct']}, incorrect: {results['n_incorrect']}")

    if not np.isnan(results["cos_dist"]).all():
        # find layer of maximum divergence
        max_div_layer = np.argmax(results["cos_dist"])
        print(f"\nmax cosine divergence at layer {max_div_layer}: "
              f"{results['cos_dist'][max_div_layer]:.4f}")
        print(f"max euclidean divergence at layer "
              f"{np.argmax(results['eucl_dist'])}: "
              f"{results['eucl_dist'][np.argmax(results['eucl_dist'])]:.2f}")

        print("\nper-layer divergence:")
        for i in range(n_layers):
            print(f"  layer {i:2d}: cos_dist={results['cos_dist'][i]:.4f}  "
                  f"eucl_dist={results['eucl_dist'][i]:.4f}")

    atomic_savez(args.output, **results)
    print(f"\nsaved results to {args.output}")


if __name__ == "__main__":
    main()
