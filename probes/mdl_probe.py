"""data-efficiency / MDL-style probing curves.

This is a lightweight practical proxy for full online coding MDL: for each
layer and task, train the same probe on increasing fractions of the training
data and report held-out accuracy plus area under the data-efficiency curve.
Features that are easily extractable should reach high accuracy with less data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from .train_linear_probe import (
        SPLIT_CHOICES,
        atomic_savez,
        enforce_prompt_contract,
        groups_for_task,
        load_activation_metadata,
        load_activations,
        load_available_labels,
        load_rows,
        make_probe,
        make_splits,
        normalize_split_policy,
        safe_key,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from train_linear_probe import (
        SPLIT_CHOICES,
        atomic_savez,
        enforce_prompt_contract,
        groups_for_task,
        load_activation_metadata,
        load_activations,
        load_available_labels,
        load_rows,
        make_probe,
        make_splits,
        normalize_split_policy,
        safe_key,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )


DEFAULT_FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.8]


def train_size_for_fraction(y_train: np.ndarray, fraction: float) -> int:
    classes = len(set(y_train))
    return max(classes, int(round(len(y_train) * fraction)))


def stratified_subset(y_train: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Return a deterministic prefix of a class-covering randomized order."""
    if size >= len(y_train):
        return np.arange(len(y_train))
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    if size < len(classes):
        raise ValueError("stratified subset size is smaller than the number of classes")
    selected = [int(rng.choice(np.flatnonzero(y_train == label))) for label in classes]
    selected_set = set(selected)
    remaining = np.asarray(
        [index for index in range(len(y_train)) if index not in selected_set],
        dtype=np.int64,
    )
    order = np.concatenate([np.asarray(selected, dtype=np.int64), rng.permutation(remaining)])
    return np.sort(order[: min(size, len(y_train))])


def run_task(
    activations: np.ndarray,
    labels: list[str],
    fractions: list[float],
    probe_kind: str,
    seed: int,
    splits=None,
    n_folds: int = 5,
) -> dict:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if np.bincount(y).min() < 2:
        raise ValueError("MDL-style held-out evaluation requires at least 2 examples per class")
    if splits is None:
        splits = make_splits(
            y,
            n_folds=n_folds,
            split_name="MDL",
            random_state=seed,
        )

    n_layers = activations.shape[1]
    curves = np.zeros((n_layers, len(fractions)), dtype=np.float32)
    # Use identical training subsets for every layer. Varying the sampled rows
    # by layer confounds representation quality with subset luck.
    subsets = {}
    for fold_index, (train_idx, _) in enumerate(splits):
        y_train = y[train_idx]
        for fi, fraction in enumerate(fractions):
            size = train_size_for_fraction(y_train, fraction)
            subsets[(fold_index, fi)] = stratified_subset(
                y_train,
                size,
                seed + fold_index,
            )

    for li in range(n_layers):
        for fi, fraction in enumerate(fractions):
            prediction = np.full(y.shape, -1, dtype=np.int64)
            for fold_index, (train_idx, test_idx) in enumerate(splits):
                y_train = y[train_idx]
                subset = subsets[(fold_index, fi)]
                probe = make_probe(probe_kind)
                probe.fit(activations[train_idx[subset], li, :], y_train[subset])
                prediction[test_idx] = probe.predict(activations[test_idx, li, :])
            if np.any(prediction < 0):
                raise RuntimeError("MDL cross-validation did not predict every sample")
            curves[li, fi] = np.mean(prediction == y)

    auc = np.trapezoid(curves, x=np.asarray(fractions), axis=1) / (
        fractions[-1] - fractions[0]
    )
    return {
        "classes": le.classes_.tolist(),
        "fractions": fractions,
        "accuracy_curve": curves,
        "data_efficiency_auc": auc.astype(np.float32),
        "effective_folds": len(splits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="run MDL-style data-efficiency probes")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--tasks", nargs="+", default=["root", "pattern"])
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--probe-kind", choices=["linear", "sgd", "mlp"], default="linear")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-field", default=None)
    parser.add_argument("--split-policy", choices=SPLIT_CHOICES, default="random")
    parser.add_argument("--root-split", choices=SPLIT_CHOICES, default="pattern")
    parser.add_argument("--pattern-split", choices=SPLIT_CHOICES, default="root")
    parser.add_argument("--require-activation-provenance", action="store_true")
    parser.add_argument("--allow-label-revealed-prompts", action="store_true")
    parser.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if len(args.fractions) < 2 or any(
        not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0
        for fraction in args.fractions
    ):
        parser.error("--fractions require at least two finite values in (0, 1]")
    if args.fractions != sorted(set(args.fractions)):
        parser.error("--fractions must be unique and strictly increasing")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("--tasks must not contain duplicates")

    activations = load_activations(args.activations)
    activation_metadata = load_activation_metadata(args.activations)
    rows = load_rows(args.stimuli)
    validate_activation_provenance(
        args.activations,
        tuple(activations.shape),
        args.stimuli,
        activation_metadata,
        require=args.require_activation_provenance,
    )
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
        activations = activations[: args.max_rows]
    activations = validate_activation_tensor(
        activations,
        args.activations,
        expected_rows=len(rows),
    )
    leakage_report = enforce_prompt_contract(
        rows,
        args.tasks,
        activation_metadata,
        allow_label_revealed=args.allow_label_revealed_prompts,
        allow_unverifiable=args.allow_unverifiable_prompt_contract,
        context="MDL",
    )
    save = {
        "schema_version": np.array(2, dtype=np.int64),
        "evaluation": np.array("cross_validated_data_efficiency_curve"),
        "tasks": np.array(args.tasks, dtype=str),
        "fractions": np.array(args.fractions, dtype=np.float32),
        "probe_kind": args.probe_kind,
        "activations_sha256": np.array(sha256_file(args.activations)),
        "stimuli_sha256": np.array(sha256_file(args.stimuli)),
        "activation_shape": np.asarray(activations.shape, dtype=np.int64),
        "seed": np.array(args.seed, dtype=np.int64),
        "folds_requested": np.array(args.folds, dtype=np.int64),
        "prompt_leakage_audit_json": np.array(
            json.dumps(
                leakage_report,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        ),
        "label_revealed_prompt_allowed": np.array(args.allow_label_revealed_prompts),
        "unverifiable_prompt_contract_allowed": np.array(
            args.allow_unverifiable_prompt_contract
        ),
    }

    for task in args.tasks:
        task_indices, labels = load_available_labels(rows, task)
        task_activations = activations[task_indices]
        task_rows = [rows[index] for index in task_indices]
        if len(task_indices) != len(rows):
            print(f"{task}: usable rows {len(task_indices)} / {len(rows)}")
        split = (
            args.root_split
            if task == "root"
            else args.pattern_split
            if task == "pattern"
            else args.split_policy
        )
        groups, _, split_metadata = groups_for_task(
            task,
            normalize_split_policy(split),
            task_rows,
            args.group_field,
            activation_metadata=activation_metadata,
        )
        encoded_labels = LabelEncoder().fit_transform(labels)
        splits = make_splits(
            encoded_labels,
            n_folds=args.folds,
            groups=groups,
            split_name=f"MDL-{task}",
            random_state=args.seed,
        )
        result = run_task(
            task_activations,
            labels,
            args.fractions,
            args.probe_kind,
            args.seed,
            splits=splits,
            n_folds=args.folds,
        )
        key = safe_key(task)
        save[f"{key}_accuracy_curve"] = result["accuracy_curve"]
        save[f"{key}_data_efficiency_auc"] = result["data_efficiency_auc"]
        save[f"{key}_classes"] = np.array(result["classes"], dtype=str)
        save[f"{key}_effective_folds"] = np.array(
            result["effective_folds"], dtype=np.int64
        )
        save[f"{key}_split_policy"] = np.array(split_metadata["effective_policy"])
        save[f"{key}_split_metadata_json"] = np.array(
            json.dumps(
                {
                    **split_metadata,
                    "fold_sizes": [
                        {"train": int(len(train)), "test": int(len(test))}
                        for train, test in splits
                    ],
                    "layer_comparison_uses_identical_subsets": True,
                    "fraction_subsets_are_nested": True,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        best_layer = int(np.argmax(result["data_efficiency_auc"]))
        best_auc = float(result["data_efficiency_auc"][best_layer])
        print(f"{task}: best AUC={best_auc:.3f} at layer {best_layer}")

    atomic_savez(args.output, **save)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
