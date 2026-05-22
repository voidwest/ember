"""visualization for probing results."""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="plot probe results")
    parser.add_argument("--probes", required=True, help="path to probe weights .npz")
    parser.add_argument("--output", default="plots/", help="output directory for plots")
    args = parser.parse_args()

    data = np.load(args.probes)
    print(f"keys: {list(data.keys())}")

    # TODO: implement plots — per-layer accuracy, scaling curves, cca heatmap, rsa matrix
    print("plot scaffold — implement full visualization pipeline.")


if __name__ == "__main__":
    main()
