"""train linear classifiers on per-layer activations
to predict arabic root and pattern from hidden states."""

import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_activations(path: str) -> np.ndarray:
    """load activation tensor from .npz file. expects (n_stimuli, n_layers, hidden_dim)."""
    data = np.load(path)
    return data["activations"]


def load_labels(path: str):
    """load root and pattern labels from stimuli json."""
    with open(path) as f:
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

    n_layers = activations.shape[1]
    accuracies = []
    probes = []

    for layer in range(n_layers):
        X = activations[:, layer, :]
        probe = LogisticRegression(max_iter=1000, multi_class="multinomial")
        scores = cross_val_score(probe, X, y, cv=n_folds)
        accuracies.append(scores.mean())
        probe.fit(X, y)  # refit on all data for export
        probes.append(probe)

    return np.array(accuracies), probes, le


def main():
    parser = argparse.ArgumentParser(description="train linear probes on llm activations")
    parser.add_argument("--activations", required=True, help="path to .npz with activations")
    parser.add_argument("--labels", required=True, help="path to stimuli json")
    parser.add_argument("--output", default=None, help="path to save probe weights (.npz)")
    parser.add_argument("--folds", type=int, default=5, help="cv folds")
    args = parser.parse_args()

    activations = load_activations(args.activations)
    roots, patterns = load_labels(args.labels)

    print(f"activations shape: {activations.shape}")
    print(f"stimuli: {len(roots)} roots, {len(set(patterns))} patterns")

    # root probes
    print("\n--- root probes ---")
    root_acc, root_probes, root_le = train_probes(activations, roots, args.folds)
    for i, acc in enumerate(root_acc):
        print(f"  layer {i:2d}: {acc:.3f}")

    # pattern probes
    print("\n--- pattern probes ---")
    pat_acc, pat_probes, pat_le = train_probes(activations, patterns, args.folds)
    for i, acc in enumerate(pat_acc):
        print(f"  layer {i:2d}: {acc:.3f}")

    if args.output:
        np.savez(args.output,
                 root_accuracy=root_acc,
                 pattern_accuracy=pat_acc,
                 root_probe_weights=[p.coef_ for p in root_probes],
                 pattern_probe_weights=[p.coef_ for p in pat_probes])
        print(f"\nsaved probe weights to {args.output}")


if __name__ == "__main__":
    main()
