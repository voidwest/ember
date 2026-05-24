"""visualization for probing results.

plots all analysis outputs from the probing pipeline:
  (1) per-layer probe accuracy (root + pattern)
  (2) CCA layer similarity heatmap
  (3) RSA layer similarity heatmap
  (4) probe weight subspace similarity
  (5) correct-vs-incorrect divergence
  (6) cross-model comparison (if available)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_probe_accuracy(probes_path: str, ax_root, ax_pattern):
    """plot per-layer root and pattern probe accuracy."""
    data = np.load(probes_path)
    if "root_accuracy" in data:
        n_layers = len(data["root_accuracy"])
        layers = np.arange(n_layers)
        ax_root.plot(layers, data["root_accuracy"], "b-o", markersize=4, label="root")
        ax_root.axhline(1.0 / 20, color="gray", linestyle="--", alpha=0.5,
                        label="chance")
        ax_root.set_ylabel("accuracy")
        ax_root.set_title("root probe")
        ax_root.legend(fontsize=7)
        ax_root.grid(alpha=0.3)

        ax_pattern.plot(layers, data["pattern_accuracy"], "r-o", markersize=4, label="pattern")
        ax_pattern.axhline(1.0 / 10, color="gray", linestyle="--", alpha=0.5,
                           label="chance")
        ax_pattern.set_ylabel("accuracy")
        ax_pattern.set_title("pattern probe")
        ax_pattern.set_xlabel("layer")
        ax_pattern.legend(fontsize=7)
        ax_pattern.grid(alpha=0.3)


def plot_cca_heatmap(cca_path: str, ax):
    """plot CCA layer similarity matrix."""
    data = np.load(cca_path)
    if "cca_layer_matrix" in data:
        mat = data["cca_layer_matrix"]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        ax.set_title("CCA layer similarity")
        plt.colorbar(im, ax=ax, shrink=0.8)


def plot_rsa_heatmap(rsa_path: str, ax):
    """plot RSA layer similarity matrix."""
    data = np.load(rsa_path)
    if "rsa_layer_matrix" in data:
        mat = data["rsa_layer_matrix"]
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        ax.set_title("RSA layer similarity")
        plt.colorbar(im, ax=ax, shrink=0.8)


def plot_probe_subspace(cca_path: str, ax):
    """plot root-pattern probe subspace similarity."""
    data = np.load(cca_path)
    if "root_pattern_cca" in data:
        sim = data["root_pattern_cca"]
        layers = np.arange(len(sim))
        ax.plot(layers, sim, "g-o", markersize=4)
        ax.set_ylabel("subspace CCA")
        ax.set_xlabel("layer")
        ax.set_title("root-pattern subspace similarity (Q3)")
        ax.grid(alpha=0.3)


def plot_divergence(div_path: str, ax_cos, ax_euc):
    """plot correct-vs-incorrect divergence curves."""
    data = np.load(div_path)
    if "cos_dist" in data and not np.isnan(data["cos_dist"]).all():
        layers = data["layer"]
        ax_cos.plot(layers, data["cos_dist"], "m-o", markersize=4)
        ax_cos.set_ylabel("cosine distance")
        ax_cos.set_title("correct vs incorrect divergence (Q4)")
        ax_cos.grid(alpha=0.3)

        ax_euc.plot(layers, data["eucl_dist"], "c-o", markersize=4)
        ax_euc.set_ylabel("euclidean distance")
        ax_euc.set_xlabel("layer")
        ax_euc.grid(alpha=0.3)

        n_c = data["n_correct"]
        n_i = data["n_incorrect"]
        ax_cos.text(0.02, 0.98, f"correct={n_c}, incorrect={n_i}",
                    transform=ax_cos.transAxes, va="top", fontsize=7)


def main():
    parser = argparse.ArgumentParser(description="plot probe results")
    parser.add_argument("--probes", default=None,
                        help="path to probe weights .npz")
    parser.add_argument("--cca", default=None,
                        help="path to CCA results .npz")
    parser.add_argument("--rsa", default=None,
                        help="path to RSA results .npz")
    parser.add_argument("--divergence", default=None,
                        help="path to divergence results .npz")
    parser.add_argument("--output", default="data/plots/",
                        help="output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # determine layout based on available data
    n_plots = sum([
        args.probes is not None,
        args.cca is not None,
        args.rsa is not None,
        (args.cca is not None and args.probes is not None),  # subspace plot needs both
        args.divergence is not None,
    ])
    if n_plots == 0:
        print("no data provided; nothing to plot")
        return

    # layout: 2 columns, variable rows
    # probe accuracy (1 row, 2 cols), CCA heatmap + RSA heatmap (1 row),
    # subspace similarity + divergence (1 row)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("Arabic Morphology Probing Results", fontsize=14, fontweight="bold")

    # Row 1: probe accuracy
    if args.probes:
        plot_probe_accuracy(args.probes, axes[0, 0], axes[0, 1])
    else:
        axes[0, 0].set_visible(False)
        axes[0, 1].set_visible(False)

    # Row 2: CCA + RSA heatmaps
    if args.cca:
        plot_cca_heatmap(args.cca, axes[1, 0])
    else:
        axes[1, 0].set_visible(False)
    if args.rsa:
        plot_rsa_heatmap(args.rsa, axes[1, 1])
    else:
        axes[1, 1].set_visible(False)

    # Row 3: subspace similarity + divergence
    if args.cca and args.probes:
        plot_probe_subspace(args.cca, axes[2, 0])
    elif args.divergence:
        # if no subspace, put divergence in left column
        pass
    else:
        axes[2, 0].set_visible(False)

    if args.divergence:
        plot_divergence(args.divergence, axes[2, 1], axes[2, 1])
        # override: divergence takes both cols if subspace absent
    else:
        axes[2, 1].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(args.output, "probe_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved to {out_path}")

    plt.close()


if __name__ == "__main__":
    main()
