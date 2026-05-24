"""train linear classifiers on per-layer activations
to predict arabic root and pattern from hidden states."""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


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


def train_probes(activations, labels, n_folds=5):
    """train linear probes on each layer's activations.

    returns per-layer accuracy and trained models.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(set(y))

    # auto-reduce folds when there aren't enough samples per class
    min_per_class = min(np.bincount(y))
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
        probe = LogisticRegression(max_iter=1000)
        if effective_folds >= 2:
            scores = cross_val_score(probe, X, y, cv=effective_folds)
            acc = scores.mean()
        else:
            probe.fit(X, y)
            acc = probe.score(X, y)  # train accuracy (optimistic)
        accuracies.append(acc)
        probe.fit(X, y)  # refit on all data for export
        probes.append(probe)

    return np.array(accuracies), probes, le


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
    args = parser.parse_args()

    activations = load_activations(args.activations)
    roots, patterns = load_labels(args.stimuli)

    print(f"activations shape: {activations.shape}")
    print(f"stimuli: {len(roots)} roots, {len(set(patterns))} patterns")

    # root probes
    print("\n--- root probes ---")
    root_acc, root_probes, root_le = train_probes(
        activations, roots, args.folds
    )
    for i, acc in enumerate(root_acc):
        print(f"  layer {i:2d}: {acc:.3f}")

    # pattern probes
    print("\n--- pattern probes ---")
    pat_acc, pat_probes, pat_le = train_probes(
        activations, patterns, args.folds
    )
    for i, acc in enumerate(pat_acc):
        print(f"  layer {i:2d}: {acc:.3f}")

    if args.output:
        np.savez(
            args.output,
            root_accuracy=root_acc,
            pattern_accuracy=pat_acc,
            root_probe_weights=[p.coef_ for p in root_probes],
            pattern_probe_weights=[p.coef_ for p in pat_probes],
        )
        print(f"\nsaved probe weights to {args.output}")


if __name__ == "__main__":
    main()
