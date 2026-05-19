"""
plot_results.py — Visualization for probing results.

Usage:
    python probes/plot_results.py --probes data/gpt2_probes.npz --output plots/
"""

import argparse
import numpy as np

# Placeholder — matplotlib import at runtime when implemented


def main():
    parser = argparse.ArgumentParser(description="Plot probe results")
    parser.add_argument("--probes", required=True, help="Path to probe weights .npz")
    parser.add_argument("--output", default="plots/", help="Output directory for plots")
    args = parser.parse_args()
    
    data = np.load(args.probes)
    print(f"Keys: {list(data.keys())}")
    
    # TODO: implement plots
    # 1. Per-layer accuracy × model (bar or line chart)
    # 2. Scaling curve: peak accuracy × model size
    # 3. CCA heatmap: layer × layer within model
    # 4. RSA similarity matrix
    print("Plot scaffold — implement full visualization pipeline.")


if __name__ == "__main__":
    main()
