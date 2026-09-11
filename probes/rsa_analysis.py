"""representational similarity analysis
comparing pairwise similarity structures across layers and models.

rsa measures whether two representations organize the same stimuli
similarly, even if the raw vectors differ. this is a geometry-level
comparison: two layers can use different coordinate systems but
still encode the same relational structure.
"""

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    from .analysis_common import assert_row_alignment
    from .train_linear_probe import atomic_savez, load_activations, sha256_file
except ImportError:  # direct script execution
    from analysis_common import assert_row_alignment
    from train_linear_probe import atomic_savez, load_activations, sha256_file


def _rsa_distances(activations: np.ndarray, metric: str) -> np.ndarray:
    activations = np.asarray(activations)
    if activations.ndim != 2 or activations.shape[0] < 3 or activations.shape[1] == 0:
        raise ValueError(f"RSA requires a [samples>=3, features>0] matrix, got {activations.shape}")
    if not np.isfinite(activations).all():
        raise ValueError("RSA inputs contain non-finite values")
    distances = pdist(activations, metric=metric)
    if not np.isfinite(distances).all():
        raise ValueError(
            f"RSA metric {metric!r} produced non-finite distances (for example from constant rows)"
        )
    return distances


def rsa_matrix(activations: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Return the full [samples, samples] representational similarity matrix."""
    return 1 - squareform(_rsa_distances(activations, metric))


def _normalized_rsa_vectors(activations: np.ndarray, metric: str) -> np.ndarray:
    """Keep only condensed distances, centering/normalizing each layer once."""
    activations = np.asarray(activations)
    if activations.ndim != 3 or any(size == 0 for size in activations.shape):
        raise ValueError(f"RSA requires a non-empty rank-3 tensor, got {activations.shape}")
    n_stimuli, n_layers, _ = activations.shape
    vectors = np.empty((n_layers, n_stimuli * (n_stimuli - 1) // 2))
    for layer in range(n_layers):
        vectors[layer] = 1 - _rsa_distances(activations[:, layer, :], metric)
    vectors -= vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("RSA produced non-finite correlations (constant distance vectors)")
    vectors /= norms
    return vectors


def rsa_layer_matrix(activations: np.ndarray,
                     metric: str = "correlation") -> np.ndarray:
    """Return layer correlations without materializing square distance matrices."""
    vectors = _normalized_rsa_vectors(activations, metric)
    return np.clip(vectors @ vectors.T, -1.0, 1.0)


def rsa_cross_model(mat_a: np.ndarray, mat_b: np.ndarray,
                    metric: str = "correlation") -> np.ndarray:
    """Return RSA correlations between layers of two aligned activation tensors."""
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
    vectors_a = _normalized_rsa_vectors(mat_a, metric)
    vectors_b = _normalized_rsa_vectors(mat_b, metric)
    return np.clip(vectors_a @ vectors_b.T, -1.0, 1.0)


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
