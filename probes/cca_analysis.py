"""canonical correlation analysis on probe weights
to compare morphological subspaces across layers and models."""

import argparse
import numpy as np
from scipy.linalg import svd


def cca(X, Y, n_components=10):
    """compute cca between two matrices."""
    # center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # svd of cross-covariance
    U, s, Vt = svd(X.T @ Y, full_matrices=False)
    correlations = s[:n_components] / (np.linalg.norm(X, axis=0).sum() * np.linalg.norm(Y, axis=0).sum())
    # TODO: proper cca normalization

    return correlations


def main():
    parser = argparse.ArgumentParser(description="cca analysis of layer representations")
    parser.add_argument("--probes", required=True, help="path to probe weights .npz")
    parser.add_argument("--output", default=None, help="path to save results")
    args = parser.parse_args()

    data = np.load(args.probes)
    print(f"loaded probe data with keys: {list(data.keys())}")

    # TODO: implement full cca pipeline — within-model layer comparison,
    # across-model layer comparison
    print("cca analysis scaffold — implement full pipeline.")


if __name__ == "__main__":
    main()
