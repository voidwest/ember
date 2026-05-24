"""canonical correlation analysis on hidden states and probe weights
to compare morphological subspaces across layers and models.

answers Q3 (are root and pattern disentangled?) and provides a
geometry-based complement to the linear probe accuracy curves.
"""

import argparse
import json
import numpy as np


def svd_cca(X, Y, n_components=10, reg=1e-4):
    """CCA via SVD with PCA pre-reduction for high-dimensional data.

    X: (n_samples, d_x)
    Y: (n_samples, d_y)

    reduces both to max_dim = min(n-1, d_x, d_y) via PCA,
    computes cross-covariance in the reduced space, and returns
    the leading singular values (canonical correlations).
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    dof = max(n - 1, 1)

    # PCA pre-reduction to full rank
    max_dim = min(n - 1, Xc.shape[1], Yc.shape[1])
    if Xc.shape[1] > max_dim:
        Ux, Sx, _ = np.linalg.svd(Xc, full_matrices=False)
        Xc = Ux[:, :max_dim] * Sx[:max_dim]
    if Yc.shape[1] > max_dim:
        Uy, Sy, _ = np.linalg.svd(Yc, full_matrices=False)
        Yc = Uy[:, :max_dim] * Sy[:max_dim]

    # after PCA, Xc.T @ Xc is diagonal: diag(Sx[:max_dim]^2)
    # so ridge-regularized inverse sqrt is element-wise
    Cxx = (Xc * Xc).sum(axis=0) / dof  # variance per component
    Cyy = (Yc * Yc).sum(axis=0) / dof
    inv_sqrt_x = 1.0 / np.sqrt(Cxx + reg)
    inv_sqrt_y = 1.0 / np.sqrt(Cyy + reg)

    # cross-covariance in whitened space
    Cxy = Xc.T @ Yc / dof
    K = (inv_sqrt_x[:, None] * Cxy) * inv_sqrt_y[None, :]

    _, s, _ = np.linalg.svd(K, full_matrices=False)
    effective_n = min(n_components, len(s))
    return np.clip(s[:effective_n], 0, 1)


def cca_layer_matrix(activations: np.ndarray) -> np.ndarray:
    """compute pairwise CCA similarity between every pair of layers.

    activations: (n_stimuli, n_layers, hidden_dim)
    returns: (n_layers, n_layers) matrix of mean canonical correlations.
    """
    n_layers = activations.shape[1]
    sim = np.zeros((n_layers, n_layers))

    for i in range(n_layers):
        for j in range(i, n_layers):
            c = svd_cca(activations[:, i, :], activations[:, j, :],
                        n_components=10)
            sim[i, j] = c.mean()
            sim[j, i] = sim[i, j]

    return sim


def cca_cross_model(mat_a: np.ndarray, mat_b: np.ndarray,
                    layers_a: list[int] | None = None,
                    layers_b: list[int] | None = None) -> np.ndarray:
    """compute CCA between layers of two different models.

    mat_a, mat_b: (n_stimuli, n_layers, hidden_dim)
    returns: (len(layers_a), len(layers_b)) CCA similarity matrix.
    """
    if layers_a is None:
        layers_a = list(range(mat_a.shape[1]))
    if layers_b is None:
        layers_b = list(range(mat_b.shape[1]))

    sim = np.zeros((len(layers_a), len(layers_b)))
    for ii, i in enumerate(layers_a):
        for jj, j in enumerate(layers_b):
            # align to fewer samples
            n = min(mat_a.shape[0], mat_b.shape[0])
            c = svd_cca(mat_a[:n, i, :], mat_b[:n, j, :], n_components=10)
            sim[ii, jj] = c.mean()

    return sim


def probe_weight_similarity(probes_path: str) -> tuple[np.ndarray, np.ndarray]:
    """compute subspace similarity between root and pattern probes per layer.

    loads .npz with keys: root_probe_weights, pattern_probe_weights
    each is a list of arrays shape (n_classes, hidden_dim).

    uses CCA on transposed weight matrices to measure the subspace
    angle between the root-discriminating and pattern-discriminating
    directions.

    returns: (per_layer_cca_mean, per_layer_frobenius_distances)
    """
    data = np.load(probes_path)
    root_w = [np.array(w) for w in data["root_probe_weights"]]
    pat_w = [np.array(w) for w in data["pattern_probe_weights"]]

    n_layers = len(root_w)
    cca_means = np.zeros(n_layers)

    for layer in range(n_layers):
        # compare subspaces: CCA between the weight matrices
        # (hidden_dim, n_classes_root) vs (hidden_dim, n_classes_pattern)
        c = svd_cca(root_w[layer].T, pat_w[layer].T, n_components=5)
        cca_means[layer] = c.mean()

    return cca_means


def main():
    parser = argparse.ArgumentParser(
        description="cca analysis of layer representations"
    )
    parser.add_argument(
        "--activations", required=True,
        help="path to activations .npy or .npz"
    )
    parser.add_argument(
        "--activations-b", default=None,
        help="second model's activations for cross-model comparison"
    )
    parser.add_argument(
        "--probes", default=None,
        help="path to probe weights .npz for root/pattern similarity"
    )
    parser.add_argument(
        "--output", default="data/cca_results.npz",
        help="path to save results"
    )
    parser.add_argument(
        "--n-components", type=int, default=10,
        help="number of CCA components"
    )
    args = parser.parse_args()

    # load activations
    if args.activations.endswith(".npz"):
        data_a = np.load(args.activations)
        acts_a = data_a[list(data_a.keys())[0]]
    else:
        acts_a = np.load(args.activations)

    n_stimuli, n_layers, hidden_dim = acts_a.shape
    print(f"activations: {acts_a.shape}")
    print(f"  {n_stimuli} stimuli, {n_layers} layers, {hidden_dim} dim")

    results = {}

    # ── within-model CCA ───────────────────────────────────────
    print("\n--- within-model CCA ---")
    cca_a = cca_layer_matrix(acts_a)
    results["cca_layer_matrix"] = cca_a

    for i in range(min(n_layers, 5)):
        print(f"  layer {i:2d} self:  {cca_a[i, i]:.4f}")

    # ── cross-model CCA ────────────────────────────────────────
    if args.activations_b:
        print("\n--- cross-model CCA ---")
        if args.activations_b.endswith(".npz"):
            data_b = np.load(args.activations_b)
            acts_b = data_b[list(data_b.keys())[0]]
        else:
            acts_b = np.load(args.activations_b)
        print(f"  model B: {acts_b.shape}")

        cca_cross = cca_cross_model(acts_a, acts_b)
        results["cca_cross_model"] = cca_cross

        # report best-matching layer pairs
        best_per_a = np.argmax(cca_cross, axis=1)
        for i in range(min(cca_cross.shape[0], 8)):
            j = best_per_a[i]
            print(f"  A layer {i:2d} ↔ B layer {j:2d}: {cca_cross[i, j]:.4f}")

    # ── probe weight similarity ────────────────────────────────
    if args.probes:
        print("\n--- probe weight similarity (Q3: disentanglement) ---")
        subspace_sim = probe_weight_similarity(args.probes)
        results["root_pattern_cca"] = subspace_sim

        for i in range(len(subspace_sim)):
            print(f"  layer {i:2d}: subspace CCA={subspace_sim[i]:.4f}")

        # disentanglement signal: low subspace CCA → root and pattern
        # probes use orthogonal subspaces. if CCA drops around mid
        # layers where probe accuracy is high, the model is
        # disentangling rather than encoding a fused vector.
        min_layer = np.argmin(subspace_sim)
        print(f"  min subspace CCA at layer {min_layer}: {subspace_sim[min_layer]:.4f}")

    # ── save ───────────────────────────────────────────────────
    np.savez(args.output, **results)
    print(f"\nsaved results to {args.output}")


if __name__ == "__main__":
    main()
