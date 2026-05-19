"""
train_linear_probe.py — Train linear classifiers on per-layer activations
to predict Arabic root and pattern from hidden states.

Usage:
    python probes/train_linear_probe.py --activations data/gpt2_activations.npz --labels stimuli/nonce_root_pattern.json

Input:
    - Activation tensor: (n_stimuli, n_layers, hidden_dim) per model
    - Labels: root (categorical, ~200 classes) and pattern (categorical, ~10 classes)

Output:
    - Per-layer classification accuracy for root and pattern
    - Saved probe weights for further analysis (CCA, RSA)
"""

import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_activations(path: str) -> np.ndarray:
    """Load activation tensor from .npz file.
    
    Expected shape: (n_stimuli, n_layers, hidden_dim)
    """
    data = np.load(path)
    # Assumes activations stored as {layer_N: array} or as single stacked array
    return data["activations"]


def load_labels(path: str):
    """Load root and pattern labels from stimuli JSON.
    
    Returns:
        roots: list of root strings (e.g., "q-l-z")
        patterns: list of pattern strings (e.g., "fa3ala")
    """
    with open(path) as f:
        stimuli = json.load(f)
    roots = [s["root"] for s in stimuli]
    patterns = [s["pattern"] for s in stimuli]
    return roots, patterns


def train_probes(activations, labels, n_folds=5):
    """Train linear probes on each layer's activations.
    
    Args:
        activations: (n_stimuli, n_layers, hidden_dim)
        labels: list of label strings
        n_folds: cross-validation folds
    
    Returns:
        per_layer_accuracy: list of accuracy scores, one per layer
        probes: list of trained LogisticRegression models
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
        # Refit on all data for export
        probe.fit(X, y)
        probes.append(probe)
    
    return np.array(accuracies), probes, le


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on LLM activations")
    parser.add_argument("--activations", required=True, help="Path to .npz file with activations")
    parser.add_argument("--labels", required=True, help="Path to stimuli JSON")
    parser.add_argument("--output", default=None, help="Path to save probe weights (.npz)")
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    args = parser.parse_args()
    
    activations = load_activations(args.activations)
    roots, patterns = load_labels(args.labels)
    
    print(f"Activations shape: {activations.shape}")
    print(f"Stimuli: {len(roots)} roots, {len(set(patterns))} patterns")
    
    # Root probes
    print("\n--- Root probes ---")
    root_acc, root_probes, root_le = train_probes(activations, roots, args.folds)
    for i, acc in enumerate(root_acc):
        print(f"  Layer {i:2d}: {acc:.3f}")
    
    # Pattern probes
    print("\n--- Pattern probes ---")
    pat_acc, pat_probes, pat_le = train_probes(activations, patterns, args.folds)
    for i, acc in enumerate(pat_acc):
        print(f"  Layer {i:2d}: {acc:.3f}")
    
    if args.output:
        np.savez(args.output,
                 root_accuracy=root_acc,
                 pattern_accuracy=pat_acc,
                 root_probe_weights=[p.coef_ for p in root_probes],
                 pattern_probe_weights=[p.coef_ for p in pat_probes])
        print(f"\nSaved probe weights to {args.output}")


if __name__ == "__main__":
    main()
