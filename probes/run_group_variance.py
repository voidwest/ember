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

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heldout_probes import (
    load_activations, load_stimuli, extract_labels, get_group_values,
)
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


DEFAULT_TASKS = ["pos", "features.gender", "features.number"]


def shuffled_group_folds(groups, n_folds=5, seed=42):
    """Randomly assign each unique group to a fold."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < n_folds:
        return None
    fold_ids = rng.randint(0, n_folds, size=n_groups)
    group_to_fold = dict(zip(unique_groups, fold_ids))
    fold_indices = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        fold_indices[group_to_fold[g]].append(i)
    splits = []
    for fold in range(n_folds):
        test_idx = np.array(fold_indices[fold])
        train_idx = np.concatenate([np.array(fold_indices[f]) for f in range(n_folds) if f != fold])
        splits.append((train_idx, test_idx))
    return splits


def probe_one_layer(X, y, splits, layer):
    """Train Ridge on one layer, return mean CV accuracy."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_layer = X[:, layer, :]
    fold_accs = []
    for train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifier(alpha=1.0)),
        ])
        probe.fit(X_layer[train_idx], y_enc[train_idx])
        fold_accs.append(float(probe.score(X_layer[test_idx], y_enc[test_idx])))
    return float(np.mean(fold_accs)) if fold_accs else 0.0


def load_best_layers(heldout_path, tasks):
    """Extract best layer per task/strategy from heldout results."""
    with open(heldout_path) as f:
        data = json.load(f)
    best = {}
    for task in tasks:
        if task not in data:
            continue
        best[task] = {}
        strategies = data[task].get("strategies", {})
        for strat_name, strat_data in strategies.items():
            if strat_name in ("lemma-heldout", "root-heldout"):
                bl = strat_data.get("probe_best_layer")
                if bl is not None:
                    best[task][strat_name] = bl
    return best


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
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading...")
    activations = load_activations(args.activations)
    rows = load_stimuli(args.stimuli)
    print(f"  activations: {activations.shape}")
    print(f"  stimuli: {len(rows)} rows")

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
        indices, labels, info = extract_labels(rows, task_key, args.min_examples_per_label)
        labels_arr = np.array(labels)
        print(f"  examples: {info['num_examples']}, classes: {info['num_classes']}")
        if info["num_classes"] > 10:
            print(f"  SKIP: high-cardinality, not evaluable")
            continue

        # Get best layer for this task from heldout results
        task_layers = best_layers.get(task_key, {})
        # Determine best layer: use lemma-heldout if available
        best_layer = task_layers.get("lemma-heldout", task_layers.get("root-heldout"))
        if best_layer is None:
            print(f"  No best layer found, using layer 0")
            best_layer = 0
        print(f"  probing layer: {best_layer}")

        for strategy_name, group_field in strategies.items():
            # If we have a strategy-specific best layer, use it
            layer = task_layers.get(strategy_name, best_layer)
            print(f"\n  Strategy: {strategy_name} (layer {layer})")
            group_vals = get_group_values(rows, group_field)
            group_arr = np.array(group_vals)[indices]

            config_accs = []
            for cfg_idx in range(args.n_configs):
                cfg_seed = args.seed + cfg_idx * 1000
                splits = shuffled_group_folds(group_arr, n_folds=args.folds, seed=cfg_seed)
                if splits is None:
                    print(f"    config {cfg_idx}: split failed")
                    continue

                # Check for unseen labels
                has_unseen = any(
                    len(set(labels_arr[train_idx]) - set(labels_arr[test_idx])) > 0
                    for train_idx, test_idx in splits
                )
                if has_unseen:
                    # Just skip — rare for low-cardinality tasks
                    continue

                acc = probe_one_layer(activations[indices], labels_arr, splits, layer)
                config_accs.append(round(acc, 6))
                if (cfg_idx + 1) % 5 == 0 or cfg_idx == 0:
                    print(f"    config {cfg_idx}: acc={acc:.4f}")

            if config_accs:
                arr = np.array(config_accs)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1))
                ci95 = 1.96 * std / np.sqrt(len(arr))
                task_results[strategy_name] = {
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "n_configs": len(config_accs),
                    "probe_layer": layer,
                    "ci95_low": round(mean - ci95, 6),
                    "ci95_high": round(mean + ci95, 6),
                    "per_config": config_accs,
                }
                print(f"    => mean={mean:.4f}, std={std:.4f}, "
                      f"95% CI=[{mean-ci95:.4f}, {mean+ci95:.4f}]")
            else:
                task_results[strategy_name] = {"error": "no valid configs", "n_configs": 0}

        results[task_key] = task_results

    output_path = out_dir / "heldout_group_variance.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
