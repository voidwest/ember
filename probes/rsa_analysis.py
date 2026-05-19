"""
rsa_analysis.py — Representational Similarity Analysis
computing pairwise similarity matrices across layers and models.

Usage:
    python probes/rsa_analysis.py --activations data/gpt2_activations.npz --output plots/rsa.png
"""

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform


def rsa_matrix(activations, metric="correlation"):
    """Compute RSA matrix for a set of activations.
    
    Args:
        activations: (n_stimuli, hidden_dim)
        metric: distance metric for pairwise comparison
    
    Returns:
        rsm: (n_stimuli, n_stimuli) representational similarity matrix
    """
    # Correlation distance: 1 - pearson r
    rsm = squareform(pdist(activations, metric=metric))
    return 1 - rsm  # convert distance to similarity


def rsa_between_layers(activations_layer_a, activations_layer_b, metric="correlation"):
    """Compare RSA matrices from two layers.
    
    Returns:
        correlation between the upper triangles of the two RSMs
    """
    rsm_a = rsa_matrix(activations_layer_a, metric)
    rsm_b = rsa_matrix(activations_layer_b, metric)
    
    # Flatten upper triangle
    triu_idx = np.triu_indices_from(rsm_a, k=1)
    vec_a = rsm_a[triu_idx]
    vec_b = rsm_b[triu_idx]
    
    return np.corrcoef(vec_a, vec_b)[0, 1]


def main():
    parser = argparse.ArgumentParser(description="RSA analysis of layer representations")
    parser.add_argument("--activations", required=True, help="Path to activations .npz")
    parser.add_argument("--output", default=None, help="Path to save RSA matrices")
    args = parser.parse_args()
    
    data = np.load(args.activations)
    print(f"Loaded activations: {data['activations'].shape}")
    
    # TODO: implement full RSA pipeline
    # - Per-model inter-layer RSA matrix
    # - Cross-model RSA comparison
    print("RSA analysis scaffold — implement full pipeline.")


if __name__ == "__main__":
    main()
