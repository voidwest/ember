"""representational similarity analysis
computing pairwise similarity matrices across layers and models."""

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform


def rsa_matrix(activations, metric="correlation"):
    """compute an rsa matrix for a set of activations.

    returns (n_stimuli, n_stimuli) representational similarity matrix
    (correlation distance converted to similarity).
    """
    rsm = squareform(pdist(activations, metric=metric))
    return 1 - rsm  # convert distance to similarity


def rsa_between_layers(activations_layer_a, activations_layer_b, metric="correlation"):
    """compare rsa matrices from two layers — returns correlation of upper triangles."""
    rsm_a = rsa_matrix(activations_layer_a, metric)
    rsm_b = rsa_matrix(activations_layer_b, metric)

    triu_idx = np.triu_indices_from(rsm_a, k=1)
    vec_a = rsm_a[triu_idx]
    vec_b = rsm_b[triu_idx]

    return np.corrcoef(vec_a, vec_b)[0, 1]


def main():
    parser = argparse.ArgumentParser(description="rsa analysis of layer representations")
    parser.add_argument("--activations", required=True, help="path to activations .npz")
    parser.add_argument("--output", default=None, help="path to save rsa matrices")
    args = parser.parse_args()

    data = np.load(args.activations)
    print(f"loaded activations: {data['activations'].shape}")

    # TODO: implement full rsa pipeline — per-model inter-layer rsa matrix,
    # cross-model rsa comparison
    print("rsa analysis scaffold — implement full pipeline.")


if __name__ == "__main__":
    main()
