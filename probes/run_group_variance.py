"""Compute heldout probe accuracy variance from group-shuffled splits.

Fast version — only probes the best layer (from existing heldout results)
rather than scanning all layers. Uses simple fold-level random assignment
instead of GroupKFold to vary split composition.

Usage:
    python probes/run_group_variance.py \
        --activations <acts.npy> --stimuli <stimuli.json> \
        --heldout-results <heldout_probe_results.json> \
        --output-dir <dir> --n-configs 20 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .run_heldout_probes import (
        extract_labels,
        get_group_values,
        load_activations,
        load_stimuli,
    )
    from .train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from run_heldout_probes import (
        extract_labels,
        get_group_values,
        load_activations,
        load_stimuli,
    )
    from train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import t as student_t


DEFAULT_TASKS = ["pos", "features.gender", "features.number"]


def load_json_object(path: str | Path) -> dict:
    source = Path(path)
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} in {source}")

    value = json.loads(
        source.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return value


def shuffled_group_folds(groups, n_folds=5, seed=42):
    """Assign groups to non-empty, approximately size-balanced random folds."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0 or n_folds < 2:
        return None
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < n_folds:
        return None
    group_sizes = {group: int(np.sum(groups == group)) for group in unique_groups}
    tie_breaks = {group: float(rng.random_sample()) for group in unique_groups}
    ordered = sorted(unique_groups, key=lambda group: (-group_sizes[group], tie_breaks[group]))
    fold_groups = [set() for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]
    for group in ordered:
        fold = min(range(n_folds), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold].add(group)
        fold_sizes[fold] += group_sizes[group]
    all_indices = np.arange(len(groups))
    splits = []
    for fold in fold_groups:
        test_mask = np.isin(groups, list(fold))
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]
        if not len(test_idx) or not len(train_idx):
            return None
        splits.append((train_idx, test_idx))
    return splits


def probe_one_layer(X, y, splits, layer):
    """Train the same standardized logistic probe as heldout selection."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_layer = X[:, layer, :]
    predictions = np.full_like(y_enc, -1)
    for train_idx, test_idx in splits:
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ])
        probe.fit(X_layer[train_idx], y_enc[train_idx])
        predictions[test_idx] = probe.predict(X_layer[test_idx])
    if np.any(predictions < 0):
        raise RuntimeError("group folds did not predict every sample exactly once")
    return float(np.mean(predictions == y_enc))


def load_best_layers(heldout_path, tasks):
    """Extract best layer per task/strategy from heldout results."""
    data = load_json_object(heldout_path)
    best = {}
    for task in tasks:
        if task not in data:
            continue
        if not isinstance(data[task], dict):
            raise ValueError(f"heldout task {task!r} must be an object")
        best[task] = {}
        strategies = data[task].get("strategies", {})
        if not isinstance(strategies, dict):
            raise ValueError(f"heldout task {task!r} strategies must be an object")
        for strat_name, strat_data in strategies.items():
            if not isinstance(strat_data, dict):
                raise ValueError(f"heldout strategy {task}/{strat_name} must be an object")
            if strat_name in ("lemma-heldout", "root-heldout"):
                bl = strat_data.get("probe_best_layer")
                if isinstance(bl, int) and not isinstance(bl, bool) and bl >= 0:
                    best[task][strat_name] = bl
                elif bl is not None:
                    raise ValueError(
                        f"heldout best layer for {task}/{strat_name} must be non-negative integer"
                    )
    return best


def validate_heldout_provenance(heldout_path, activations_path, stimuli_path, shape):
    path = Path(heldout_path)
    provenance_path = path.with_name(f"{path.stem}_provenance.json")
    if not provenance_path.is_file():
        raise ValueError(f"heldout results lack required provenance sidecar: {provenance_path}")
    provenance = load_json_object(provenance_path)
    if provenance.get("results") != path.name:
        raise ValueError("heldout provenance results filename mismatch")
    if provenance.get("results_sha256") != sha256_file(path):
        raise ValueError("heldout results SHA-256 does not match its provenance sidecar")
    if provenance.get("activation_shape") != list(shape):
        raise ValueError("heldout provenance activation shape mismatch")
    if provenance.get("activations_sha256") != sha256_file(activations_path):
        raise ValueError("heldout results were selected from different activations")
    if provenance.get("stimuli_sha256") != sha256_file(stimuli_path):
        raise ValueError("heldout results were selected from different stimuli")
    return provenance


def main():
    parser = argparse.ArgumentParser(description="group-shuffled heldout CI (fast)")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--heldout-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--min-examples-per-label", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-activation-provenance", action="store_true")
    parser.add_argument("--allow-label-revealed-prompts", action="store_true")
    parser.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    args = parser.parse_args()

    if args.min_examples_per_label < 2:
        parser.error("--min-examples-per-label must be at least 2")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.n_configs < 2:
        parser.error("--n-configs must be at least 2 for a variance estimate")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading...")
    activations = load_activations(args.activations)
    rows = load_stimuli(args.stimuli)
    activation_metadata = load_activation_metadata(args.activations)
    validate_activation_provenance(
        args.activations,
        tuple(activations.shape),
        args.stimuli,
        activation_metadata,
        require=args.require_activation_provenance,
    )
    activations = validate_activation_tensor(
        activations,
        args.activations,
        expected_rows=len(rows),
    )
    print(f"  activations: {activations.shape}")
    print(f"  stimuli: {len(rows)} rows")

    heldout_provenance = validate_heldout_provenance(
        args.heldout_results,
        args.activations,
        args.stimuli,
        activations.shape,
    )
    leakage_report = enforce_prompt_contract(
        rows,
        args.tasks,
        activation_metadata,
        allow_label_revealed=args.allow_label_revealed_prompts,
        allow_unverifiable=args.allow_unverifiable_prompt_contract,
        context="group variance probe",
    )
    recorded_leakage = heldout_provenance.get("prompt_leakage_audit")
    if recorded_leakage is not None and recorded_leakage != leakage_report:
        raise ValueError("heldout provenance prompt audit differs from the current audit")

    best_layers = load_best_layers(args.heldout_results, args.tasks)
    print(f"  best layers loaded from {args.heldout_results}")
    for task, strats in best_layers.items():
        print(f"    {task}: {strats}")

    strategies = {"lemma-heldout": "lemma", "root-heldout": "root"}
    results = {}

    for task_key in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_key}")
        task_results = {}
        try:
            indices, labels, info = extract_labels(
                rows, task_key, args.min_examples_per_label
            )
        except ValueError as error:
            print(f"  SKIP: {error}")
            continue
        labels_arr = np.array(labels)
        print(f"  examples: {info['num_examples']}, classes: {info['num_classes']}")
        # Get best layer for this task from heldout results
        task_layers = best_layers.get(task_key, {})
        # Determine best layer: use lemma-heldout if available
        best_layer = task_layers.get("lemma-heldout", task_layers.get("root-heldout"))
        if best_layer is None:
            print("  SKIP: heldout results do not declare a best layer")
            continue
        if best_layer < 0 or best_layer >= activations.shape[1]:
            raise ValueError(f"best layer {best_layer} is outside activation shape {activations.shape}")
        print(f"  probing layer: {best_layer}")

        for strategy_name, group_field in strategies.items():
            # If we have a strategy-specific best layer, use it
            layer = task_layers.get(strategy_name, best_layer)
            if layer < 0 or layer >= activations.shape[1]:
                raise ValueError(f"probe layer {layer} is outside activation shape {activations.shape}")
            print(f"\n  Strategy: {strategy_name} (layer {layer})")
            task_rows = [rows[index] for index in indices]
            group_arr = np.array(get_group_values(task_rows, group_field))

            config_accs = []
            for cfg_idx in range(args.n_configs):
                cfg_seed = args.seed + cfg_idx * 1000
                splits = shuffled_group_folds(group_arr, n_folds=args.folds, seed=cfg_seed)
                if splits is None:
                    print(f"    config {cfg_idx}: split failed")
                    continue

                # Check for unseen labels
                has_unseen = any(
                    bool(set(labels_arr[test_idx]) - set(labels_arr[train_idx]))
                    for train_idx, test_idx in splits
                )
                if has_unseen:
                    # Just skip — rare for low-cardinality tasks
                    continue

                acc = probe_one_layer(activations[indices], labels_arr, splits, layer)
                config_accs.append(acc)
                if (cfg_idx + 1) % 5 == 0 or cfg_idx == 0:
                    print(f"    config {cfg_idx}: acc={acc:.4f}")

            if len(config_accs) >= 2:
                arr = np.array(config_accs)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1))
                critical = float(student_t.ppf(0.975, df=len(arr) - 1))
                ci95 = critical * std / np.sqrt(len(arr))
                ci_low = max(0.0, mean - ci95)
                ci_high = min(1.0, mean + ci95)
                task_results[strategy_name] = {
                    "mean": mean,
                    "std": std,
                    "n_configs": len(config_accs),
                    "n_attempted_configs": args.n_configs,
                    "probe_layer": layer,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ci_method": "student_t_over_valid_split_configurations",
                    "ci_scope": (
                        "conditional_valid_closed_set_split_configuration_sensitivity; "
                        "not population uncertainty"
                    ),
                    "layer_selection": "preselected_on_heldout_results",
                    "per_config": [float(value) for value in config_accs],
                }
                print(f"    => mean={mean:.4f}, std={std:.4f}, "
                      f"95% CI=[{ci_low:.4f}, {ci_high:.4f}]")
            else:
                task_results[strategy_name] = {
                    "error": "fewer than 2 valid closed-set configurations",
                    "n_configs": len(config_accs),
                    "n_attempted_configs": args.n_configs,
                }

        results[task_key] = task_results

    output_path = out_dir / "heldout_group_variance.json"
    if not results:
        raise ValueError("no group-variance task could be evaluated")
    atomic_write_text(
        output_path,
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    provenance_path = out_dir / "heldout_group_variance_provenance.json"
    atomic_write_text(
        provenance_path,
        json.dumps(
            {
                "schema_version": 2,
                "activations_sha256": sha256_file(args.activations),
                "stimuli_sha256": sha256_file(args.stimuli),
                "heldout_results_sha256": sha256_file(args.heldout_results),
                "implementation_sha256": sha256_file(__file__),
                "probe_classifier": "standardized_logistic_regression",
                "activation_shape": list(activations.shape),
                "prompt_leakage_audit": leakage_report,
                "config": {
                    "tasks": args.tasks,
                    "folds": args.folds,
                    "n_configs": args.n_configs,
                    "seed": args.seed,
                    "min_examples_per_label": args.min_examples_per_label,
                },
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
