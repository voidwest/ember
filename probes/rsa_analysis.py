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

try:
    from .analysis_common import assert_row_alignment
    from .train_linear_probe import atomic_savez, load_activations, sha256_file
except ImportError:  # direct script execution
    from analysis_common import assert_row_alignment
    from train_linear_probe import atomic_savez, load_activations, sha256_file


def rsa_matrix(activations: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """compute a representational similarity matrix for a set of activations.

    activations: (n_stimuli, hidden_dim)
    returns: (n_stimuli, n_stimuli) rsm (0=identical, 1=orthogonal, etc.)
    """
    activations = np.asarray(activations)
    if activations.ndim != 2 or activations.shape[0] < 3 or activations.shape[1] == 0:
        raise ValueError(f"RSA requires a [samples>=3, features>0] matrix, got {activations.shape}")
    if not np.isfinite(activations).all():
        raise ValueError("RSA inputs contain non-finite values")
    rdm = squareform(pdist(activations, metric=metric))
    if not np.isfinite(rdm).all():
        raise ValueError(
            f"RSA metric {metric!r} produced non-finite distances (for example from constant rows)"
        )
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

    sim = np.atleast_2d(np.corrcoef(vecs))
    if not np.isfinite(sim).all():
        raise ValueError("layer RSA produced non-finite correlations")
    return sim


def rsa_cross_model(mat_a: np.ndarray, mat_b: np.ndarray,
                    metric: str = "correlation") -> np.ndarray:
    """compute RSA between layers of two different models.

    mat_a, mat_b: (n_stimuli, n_layers, hidden_dim)
    returns: (n_layers_a, n_layers_b) RSA similarity matrix.
    """
    mat_a = np.asarray(mat_a)
    mat_b = np.asarray(mat_b)
    if mat_a.ndim != 3 or mat_b.ndim != 3:
        raise ValueError(
            f"cross-model RSA requires rank-3 tensors, got {mat_a.shape} and {mat_b.shape}"
        )
    if mat_a.shape[0] != mat_b.shape[0]:
        raise ValueError(
            f"cross-model RSA requires equal aligned sample counts, got {mat_a.shape[0]} and {mat_b.shape[0]}"
        )
    n_stimuli = mat_a.shape[0]
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
    if not np.isfinite(sim).all():
        raise ValueError("cross-model RSA produced non-finite correlations")

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
        "--metric", default="correlation", choices=["correlation", "cosine", "euclidean"],
        help="distance metric for RSM (correlation, cosine, euclidean)"
    )
    parser.add_argument(
        "--assume-row-aligned",
        action="store_true",
        help="permit cross-model analysis without verifiable row-identity metadata",
    )
    args = parser.parse_args()

    # load
    acts_a = load_activations(args.activations)

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
        acts_b = load_activations(args.activations_b)
        print(f"  model B: {acts_b.shape}")
        alignment_evidence = assert_row_alignment(
            args.activations,
            args.activations_b,
            acts_a.shape[0],
            allow_assumed=args.assume_row_aligned,
        )

        rsa_cross = rsa_cross_model(acts_a, acts_b, args.metric)
        results["rsa_cross_model"] = rsa_cross
        results["cross_model_alignment_evidence"] = np.array(alignment_evidence)

        best_per_a = np.argmax(rsa_cross, axis=1)
        for i in range(min(rsa_cross.shape[0], 8)):
            j = best_per_a[i]
            print(f"  A layer {i:2d} ↔ B layer {j:2d}: {rsa_cross[i, j]:.4f}")

    # ── save ───────────────────────────────────────────────────
    results["schema_version"] = np.array(3, dtype=np.int64)
    results["evaluation"] = np.array(
        "descriptive_pairwise_rdm_pearson_correlation"
    )
    results["metric"] = np.array(args.metric)
    results["activations_a_sha256"] = np.array(sha256_file(args.activations))
    results["activations_a_shape"] = np.asarray(acts_a.shape, dtype=np.int64)
    if args.activations_b:
        results["activations_b_sha256"] = np.array(sha256_file(args.activations_b))
        results["activations_b_shape"] = np.asarray(acts_b.shape, dtype=np.int64)
    atomic_savez(args.output, **results)
    print(f"\nsaved results to {args.output}")


if __name__ == "__main__":
    main()
