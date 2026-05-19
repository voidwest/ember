"""
cca_analysis.py — Canonical Correlation Analysis on probe weights
to compare morphological subspaces across layers and models.

Usage:
    python probes/cca_analysis.py --probes data/gpt2_probes.npz --output plots/cca.png
"""

import argparse
import numpy as np
from scipy.linalg import svd


def cca(X, Y, n_components=10):
    """Compute CCA between two matrices.
    
    Args:
        X, Y: (n_samples, n_features) — activation matrices from two layers/models
        n_components: number of CCA directions to compute
    
    Returns:
        correlations: CCA correlation coefficients (n_components,)
    """
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # SVD of cross-covariance
    U, s, Vt = svd(X.T @ Y, full_matrices=False)
    correlations = s[:n_components] / (np.linalg.norm(X, axis=0).sum() * np.linalg.norm(Y, axis=0).sum())
    # Normalize properly
    # ... TODO: proper CCA normalization
    
    return correlations


def main():
    parser = argparse.ArgumentParser(description="CCA analysis of layer representations")
    parser.add_argument("--probes", required=True, help="Path to probe weights .npz")
    parser.add_argument("--output", default=None, help="Path to save results")
    args = parser.parse_args()
    
    data = np.load(args.probes)
    print(f"Loaded probe data with keys: {list(data.keys())}")
    
    # TODO: implement full CCA comparison pipeline
    # - Within-model: compare probe subspaces across layers
    # - Across-model: compare corresponding layers
    print("CCA analysis scaffold — implement full pipeline.")


if __name__ == "__main__":
    main()
