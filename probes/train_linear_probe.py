"""train linear classifiers on per-layer activations
to predict arabic root and pattern from hidden states.

supports:
  - standard cross-validated linear probing
  - random-label control tasks (selectivity à la Hewitt & Liang 2019)
  - root-disjoint train/test splits (hold out entire roots)
  - selectivity score reporting (real accuracy / max(control, chance))
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline


def load_activations(path: str) -> np.ndarray:
    """load activation tensor.

    supports .npz (key: "activations") and .npy (raw 3d array).
    shape: (n_stimuli, n_layers, hidden_dim).
    """
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(path)
        return data["activations"]
    elif p.suffix == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"unsupported activation format: {p.suffix}")


def load_labels(stimuli_path: str):
    """load root and pattern labels from stimuli json.

    expects format from generate_stimuli.py: each stimulus has
    'root' (str) and 'pattern' (str) fields.
    """
    with open(stimuli_path, encoding="utf-8") as f:
        stimuli = json.load(f)
    roots = [s["root"] for s in stimuli]
    patterns = [s["pattern"] for s in stimuli]
    return roots, patterns


def train_probes(activations, labels, n_folds=5, groups=None):
    """train linear probes on each layer's activations.

    returns per-layer accuracy and trained models.
    if groups is provided, uses GroupKFold (groups define
    disjoint sets like roots that must not span folds).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(set(y))

    # auto-reduce folds when there aren't enough samples per class
    min_per_class = min(np.bincount(y))
    if groups is not None:
        n_groups = len(set(groups))
        effective_folds = min(n_folds, n_groups, min_per_class)
    else:
        effective_folds = min(n_folds, min_per_class, len(y) // n_classes)

    if effective_folds < 2:
        print(f"  warning: only {min_per_class} samples per class, "
              f"skipping cross-validation (using train accuracy)")
        effective_folds = 0

    n_layers = activations.shape[1]
    accuracies = []
    probes = []

    for layer in range(n_layers):
        X = activations[:, layer, :]
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        if effective_folds >= 2:
            if groups is not None:
                gkf = GroupKFold(n_splits=effective_folds)
                scores = []
                for train_idx, test_idx in gkf.split(X, y, groups=groups):
                    probe_clone = make_pipeline(
                        StandardScaler(), LogisticRegression(max_iter=1000)
                    )
                    probe_clone.fit(X[train_idx], y[train_idx])
                    scores.append(probe_clone.score(X[test_idx], y[test_idx]))
                acc = np.mean(scores)
            else:
                scores = cross_val_score(probe, X, y, cv=effective_folds)
                acc = scores.mean()
        else:
            probe.fit(X, y)
            acc = probe.score(X, y)  # train accuracy (optimistic)
        accuracies.append(acc)
        probe.fit(X, y)  # refit on all data for export
        probes.append(probe)

    return np.array(accuracies), probes, le


def run_control(activations, labels, n_folds=5, groups=None, n_repeats=5):
    """run random-label control: shuffle labels, train probes, report accuracy.

    repeats n_repeats times and returns mean + std across repeats.
    a good probe should score far above the control accuracy.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_layers = activations.shape[1]
    all_acc = np.zeros((n_repeats, n_layers))

    for repeat in range(n_repeats):
        # shuffle labels to break any real signal
        y_shuffled = y.copy()
        rng = np.random.RandomState(repeat * 31 + 7)
        rng.shuffle(y_shuffled)

        if groups is not None:
            groups_shuffled = groups.copy()
            rng.shuffle(groups_shuffled)

        acc, _, _ = train_probes(
            activations, le.inverse_transform(y_shuffled),
            n_folds=n_folds,
            groups=groups_shuffled if groups is not None else None,
        )
        all_acc[repeat] = acc

    return all_acc.mean(axis=0), all_acc.std(axis=0)


def compute_selectivity(real_acc, control_acc_mean, chance):
    """compute selectivity score.

    selectivity = (real - control_mean) / (1 - max(control_mean, chance))
    clamps negative values to 0.

    this follows the spirit of Hewitt & Liang (2019): a good probe
    should do well on the real task and poorly on control tasks.
    """
    denominator = 1.0 - np.maximum(control_acc_mean, chance)
    # avoid division by zero
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    selectivity = (real_acc - control_acc_mean) / denominator
    return np.maximum(selectivity, 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="train linear probes on llm activations"
    )
    parser.add_argument(
        "--activations", required=True, help="path to .npy or .npz with activations"
    )
    parser.add_argument(
        "--stimuli", required=True, help="path to stimuli json"
    )
    parser.add_argument(
        "--output", default=None, help="path to save probe weights (.npz)"
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="cv folds"
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="run random-label control tasks and report selectivity",
    )
    parser.add_argument(
        "--control-repeats",
        type=int,
        default=5,
        help="number of random-label repeats for control (default: 5)",
    )
    parser.add_argument(
        "--split-root",
        action="store_true",
        help="use root-disjoint splits (GroupKFold by root) instead of random CV",
    )
    args = parser.parse_args()

    activations = load_activations(args.activations)
    roots, patterns = load_labels(args.stimuli)

    print(f"activations shape: {activations.shape}")
    print(f"stimuli: {len(roots)} roots, {len(set(roots))} unique roots, "
          f"{len(set(patterns))} unique patterns")
    if args.split_root:
        print("using root-disjoint splits (GroupKFold by root)")
    if args.control:
        print(f"running random-label control ({args.control_repeats} repeats)")

    # prepare groups for root-disjoint CV
    root_groups = None
    if args.split_root:
        root_le = LabelEncoder()
        root_groups = root_le.fit_transform(roots)

    # ── root probes ───────────────────────────────────────────────
    print("\n--- root probes ---")
    root_acc, root_probes, root_le = train_probes(
        activations, roots, args.folds, groups=root_groups
    )
    for i, acc in enumerate(root_acc):
        print(f"  layer {i:2d}: {acc:.3f}")

    root_control_mean = None
    root_control_std = None
    root_selectivity = None
    if args.control:
        print("\n--- root: random-label control ---")
        root_control_mean, root_control_std = run_control(
            activations, roots, args.folds, groups=root_groups,
            n_repeats=args.control_repeats,
        )
        chance_root = 1.0 / len(set(roots))
        root_selectivity = compute_selectivity(root_acc, root_control_mean, chance_root)
        for i, (real, ctrl, sel) in enumerate(
            zip(root_acc, root_control_mean, root_selectivity)
        ):
            print(f"  layer {i:2d}: real={real:.3f}  control={ctrl:.3f}  "
                  f"selectivity={sel:.3f}")
        print(f"  mean selectivity: {root_selectivity.mean():.3f} "
              f"(max: {root_selectivity.max():.3f} at layer {root_selectivity.argmax()})")

    # ── pattern probes ────────────────────────────────────────────
    print("\n--- pattern probes ---")
    pat_acc, pat_probes, pat_le = train_probes(
        activations, patterns, args.folds, groups=root_groups
    )
    for i, acc in enumerate(pat_acc):
        print(f"  layer {i:2d}: {acc:.3f}")

    pat_control_mean = None
    pat_control_std = None
    pat_selectivity = None
    if args.control:
        print("\n--- pattern: random-label control ---")
        pat_control_mean, pat_control_std = run_control(
            activations, patterns, args.folds, groups=root_groups,
            n_repeats=args.control_repeats,
        )
        chance_pat = 1.0 / len(set(patterns))
        pat_selectivity = compute_selectivity(pat_acc, pat_control_mean, chance_pat)
        for i, (real, ctrl, sel) in enumerate(
            zip(pat_acc, pat_control_mean, pat_selectivity)
        ):
            print(f"  layer {i:2d}: real={real:.3f}  control={ctrl:.3f}  "
                  f"selectivity={sel:.3f}")
        print(f"  mean selectivity: {pat_selectivity.mean():.3f} "
              f"(max: {pat_selectivity.max():.3f} at layer {pat_selectivity.argmax()})")

    # ── save ──────────────────────────────────────────────────────
    if args.output:
        save_dict = {
            "root_accuracy": root_acc,
            "pattern_accuracy": pat_acc,
            "root_probe_weights": [
                p.named_steps["logisticregression"].coef_
                for p in root_probes
            ],
            "pattern_probe_weights": [
                p.named_steps["logisticregression"].coef_
                for p in pat_probes
            ],
        }
        if args.control:
            save_dict["root_control_mean"] = root_control_mean
            save_dict["root_control_std"] = root_control_std
            save_dict["root_selectivity"] = root_selectivity
            save_dict["pat_control_mean"] = pat_control_mean
            save_dict["pat_control_std"] = pat_control_std
            save_dict["pat_selectivity"] = pat_selectivity
        np.savez(args.output, **save_dict)
        print(f"\nsaved probe weights to {args.output}")
        if args.control:
            print("  (includes control and selectivity arrays)")


if __name__ == "__main__":
    main()
