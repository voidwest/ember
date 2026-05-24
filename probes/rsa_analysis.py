"""representational similarity analysis
comparing pairwise similarity structures across layers and models.

rsa measures whether two representations organize the same stimuli
similarly, even if the raw vectors differ. this is a geometry-level
comparison: two layers can use different coordinate systems but
still encode the same relational structure.
"""

import argparse
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform


def rsa_matrix(activations: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """compute a representational similarity matrix for a set of activations.

    activations: (n_stimuli, hidden_dim)
    returns: (n_stimuli, n_stimuli) rsm (0=identical, 1=orthogonal, etc.)
    """
    rdm = squareform(pdist(activations, metric=metric))
    return 1 - rdm  # distance → similarity


def rsa_layer_matrix(activations: np.ndarray,
                     metric: str = "correlation") -> np.ndarray:
    """compute pairwise RSA between every pair of layers.

    activations: (n_stimuli, n_layers, hidden_dim)
    returns: (n_layers, n_layers) matrix of RSA correlations.
    """
    n_stimuli, n_layers, _ = activations.shape

    # compute RSM for each layer
    rsms = np.zeros((n_layers, n_stimuli, n_stimuli))
    for layer in range(n_layers):
        rsms[layer] = rsa_matrix(activations[:, layer, :], metric)

    # compare upper triangles
    triu_idx = np.triu_indices(n_stimuli, k=1)
    vecs = rsms[:, triu_idx[0], triu_idx[1]]  # (n_layers, n_pairs)

    sim = np.corrcoef(vecs)
    return sim


def rsa_cross_model(mat_a: np.ndarray, mat_b: np.ndarray,
                    metric: str = "correlation") -> np.ndarray:
    """compute RSA between layers of two different models.

    mat_a, mat_b: (n_stimuli, n_layers, hidden_dim)
    returns: (n_layers_a, n_layers_b) RSA similarity matrix.
    """
    n_stimuli = min(mat_a.shape[0], mat_b.shape[0])
    n_layers_a, n_layers_b = mat_a.shape[1], mat_b.shape[1]

    # compute RSMs
    triu_idx = np.triu_indices(n_stimuli, k=1)
    n_pairs = len(triu_idx[0])

    rsm_vecs_a = np.zeros((n_layers_a, n_pairs))
    rsm_vecs_b = np.zeros((n_layers_b, n_pairs))

    for i in range(n_layers_a):
        rsm = rsa_matrix(mat_a[:n_stimuli, i, :], metric)
        rsm_vecs_a[i] = rsm[triu_idx]

    for j in range(n_layers_b):
        rsm = rsa_matrix(mat_b[:n_stimuli, j, :], metric)
        rsm_vecs_b[j] = rsm[triu_idx]

    # correlation between each pair of RSM vectors
    sim = np.zeros((n_layers_a, n_layers_b))
    for i in range(n_layers_a):
        for j in range(n_layers_b):
            sim[i, j] = np.corrcoef(rsm_vecs_a[i], rsm_vecs_b[j])[0, 1]

    return sim


def main():
    parser = argparse.ArgumentParser(
        description="rsa analysis of layer representations"
    )
    parser.add_argument(
        "--activations", required=True,
        help="path to activations .npy or .npz"
    )
    parser.add_argument(
        "--activations-b", default=None,
        help="second model's activations for cross-model RSA"
    )
    parser.add_argument(
        "--output", default="data/rsa_results.npz",
        help="path to save results"
    )
    parser.add_argument(
        "--metric", default="correlation",
        help="distance metric for RSM (correlation, cosine, euclidean)"
    )
    args = parser.parse_args()

    # load
    if args.activations.endswith(".npz"):
        data_a = np.load(args.activations)
        acts_a = data_a[list(data_a.keys())[0]]
    else:
        acts_a = np.load(args.activations)

    n_stimuli, n_layers, hidden_dim = acts_a.shape
    print(f"activations: {acts_a.shape}")
    print(f"  {n_stimuli} stimuli, {n_layers} layers, {hidden_dim} dim")
    print(f"  metric: {args.metric}")

    results = {}

    # ── within-model RSA ───────────────────────────────────────
    print("\n--- within-model RSA ---")
    rsa = rsa_layer_matrix(acts_a, args.metric)
    results["rsa_layer_matrix"] = rsa

    # diagonal (self-similarity) should be 1.0
    # off-diagonal shows layer similarity structure
    for i in range(min(n_layers, 5)):
        print(f"  layer {i:2d} diagonal: {rsa[i, i]:.4f}")

    # ── cross-model RSA ────────────────────────────────────────
    if args.activations_b:
        print("\n--- cross-model RSA ---")
        if args.activations_b.endswith(".npz"):
            data_b = np.load(args.activations_b)
            acts_b = data_b[list(data_b.keys())[0]]
        else:
            acts_b = np.load(args.activations_b)
        print(f"  model B: {acts_b.shape}")

        rsa_cross = rsa_cross_model(acts_a, acts_b, args.metric)
        results["rsa_cross_model"] = rsa_cross

        best_per_a = np.argmax(rsa_cross, axis=1)
        for i in range(min(rsa_cross.shape[0], 8)):
            j = best_per_a[i]
            print(f"  A layer {i:2d} ↔ B layer {j:2d}: {rsa_cross[i, j]:.4f}")

    # ── save ───────────────────────────────────────────────────
    np.savez(args.output, **results)
    print(f"\nsaved results to {args.output}")


if __name__ == "__main__":
    main()
