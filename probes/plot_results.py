"""visualization for probing results.

plots all analysis outputs from the probing pipeline:
  (1) per-layer probe accuracy (root + pattern)
  (2) CCA layer similarity heatmap
  (3) RSA layer similarity heatmap
  (4) probe weight subspace similarity
  (5) correct-vs-incorrect divergence
  (6) cross-model comparison (if available)

--dark flag produces dark-mode charts matching voidwest.dev styling.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── dark-mode palette (matches voidwest.dev CSS) ─────────────
DARK_BG       = "#0d1117"
DARK_SURFACE  = "#161b22"
DARK_BORDER   = "#30363d"
DARK_TEXT     = "#c9d1d9"
DARK_DIM      = "#8b949e"
DARK_ACCENT   = "#f78166"   # orange
DARK_ACCENT2  = "#d2a8ff"   # purple
DARK_GREEN    = "#7ee787"
DARK_BLUE     = "#79c0ff"
DARK_RED      = "#ff7b72"
DARK_YELLOW   = "#d29922"


def _setup_dark():
    """apply dark-mode rcParams."""
    matplotlib.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_SURFACE,
        "axes.edgecolor": DARK_BORDER,
        "axes.labelcolor": DARK_TEXT,
        "text.color": DARK_TEXT,
        "xtick.color": DARK_DIM,
        "ytick.color": DARK_DIM,
        "grid.color": DARK_BORDER,
        "grid.alpha": 0.6,
        "legend.facecolor": DARK_SURFACE,
        "legend.edgecolor": DARK_BORDER,
        "legend.labelcolor": DARK_TEXT,
        "figure.titlesize": 14,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    })


def plot_probe_accuracy(probes_path: str, ax_root, ax_pattern, dark: bool = False):
    """plot per-layer root and pattern probe accuracy."""
    data = np.load(probes_path)
    if "root_accuracy" not in data:
        return

    n_layers = len(data["root_accuracy"])
    layers = np.arange(n_layers)

    root_color = DARK_BLUE if dark else "b"
    pat_color  = DARK_ACCENT if dark else "r"

    ax_root.plot(layers, data["root_accuracy"], color=root_color, marker="o",
                 markersize=4, linewidth=1.4, label="root")
    ax_root.axhline(1.0 / 20, color=DARK_DIM if dark else "gray",
                    linestyle="--", alpha=0.5, label="chance (5%)")
    ax_root.set_ylabel("accuracy")
    ax_root.set_title("root probe")
    ax_root.legend(fontsize=7)
    ax_root.grid(alpha=0.3)
    ax_root.set_ylim(-0.02, 1.05)

    ax_pattern.plot(layers, data["pattern_accuracy"], color=pat_color, marker="o",
                    markersize=4, linewidth=1.4, label="pattern")
    ax_pattern.axhline(1.0 / 10, color=DARK_DIM if dark else "gray",
                       linestyle="--", alpha=0.5, label="chance (10%)")
    ax_pattern.set_ylabel("accuracy")
    ax_pattern.set_title("pattern probe")
    ax_pattern.set_xlabel("layer")
    ax_pattern.legend(fontsize=7)
    ax_pattern.grid(alpha=0.3)
    ax_pattern.set_ylim(-0.02, 1.05)


def plot_cca_heatmap(cca_path: str, ax, dark: bool = False):
    """plot CCA layer similarity matrix."""
    data = np.load(cca_path)
    if "cca_layer_matrix" not in data:
        return
    mat = data["cca_layer_matrix"]
    cmap = "YlOrRd"  # works well on both light and dark
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title("CCA layer similarity")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if dark:
        cbar.ax.yaxis.set_tick_params(color=DARK_DIM)
        cbar.outline.set_edgecolor(DARK_BORDER)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK_DIM)


def plot_rsa_heatmap(rsa_path: str, ax, dark: bool = False):
    """plot RSA layer similarity matrix."""
    data = np.load(rsa_path)
    if "rsa_layer_matrix" not in data:
        return
    mat = data["rsa_layer_matrix"]
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title("RSA layer similarity")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if dark:
        cbar.ax.yaxis.set_tick_params(color=DARK_DIM)
        cbar.outline.set_edgecolor(DARK_BORDER)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK_DIM)


def plot_probe_subspace(cca_path: str, ax, dark: bool = False):
    """plot root-pattern probe subspace similarity."""
    data = np.load(cca_path)
    if "root_pattern_cca" not in data:
        return
    sim = data["root_pattern_cca"]
    layers = np.arange(len(sim))
    color = DARK_GREEN if dark else "g"
    ax.plot(layers, sim, color=color, marker="o", markersize=4, linewidth=1.4)
    ax.set_ylabel("subspace CCA")
    ax.set_xlabel("layer")
    ax.set_title("root-pattern subspace (Q3)")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=-0.02)


def plot_divergence(div_path: str, ax_cos, ax_euc, dark: bool = False):
    """plot correct-vs-incorrect divergence curves."""
    data = np.load(div_path)
    if "cos_dist" not in data:
        return

    cos_ok = not np.isnan(data["cos_dist"]).all()
    n_c = int(data.get("n_correct", 0))
    n_i = int(data.get("n_incorrect", 0))

    dim_color = DARK_DIM if dark else "gray"

    if cos_ok:
        layers = data["layer"]
        cos_color = DARK_ACCENT2 if dark else "m"
        euc_color = DARK_BLUE if dark else "c"

        ax_cos.plot(layers, data["cos_dist"], color=cos_color, marker="o",
                    markersize=4, linewidth=1.4)
        ax_cos.set_ylabel("cosine distance")
        ax_cos.set_title("correct vs incorrect divergence (Q4)")
        ax_cos.grid(alpha=0.3)

        ax_euc.plot(layers, data["eucl_dist"], color=euc_color, marker="o",
                    markersize=4, linewidth=1.4)
        ax_euc.set_ylabel("euclidean distance")
        ax_euc.set_xlabel("layer")
        ax_euc.grid(alpha=0.3)

        ax_cos.text(0.02, 0.98, f"correct={n_c}, incorrect={n_i}",
                    transform=ax_cos.transAxes, va="top", fontsize=7, color=DARK_DIM if dark else "black")
    else:
        # no data — show placeholder
        for ax in (ax_cos, ax_euc):
            ax.text(0.5, 0.5, "N/A — 0 correct predictions",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=dim_color)
            ax.set_title("correct vs incorrect divergence (Q4)")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description="plot probe results")
    parser.add_argument("--probes", default=None, help="path to probe weights .npz")
    parser.add_argument("--cca", default=None, help="path to CCA results .npz")
    parser.add_argument("--rsa", default=None, help="path to RSA results .npz")
    parser.add_argument("--divergence", default=None, help="path to divergence results .npz")
    parser.add_argument("--output", default="data/plots/", help="output directory for plots")
    parser.add_argument("--dark", action="store_true", help="dark-mode styling")
    parser.add_argument("--title", default="Arabic Morphology Probing Results",
                        help="figure suptitle")
    parser.add_argument("--dpi", type=int, default=150, help="output DPI")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.dark:
        _setup_dark()

    n_plots = sum([
        args.probes is not None,
        args.cca is not None,
        args.rsa is not None,
        (args.cca is not None and args.probes is not None),
        args.divergence is not None,
    ])
    if n_plots == 0:
        print("no data provided; nothing to plot")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    title_color = DARK_ACCENT if args.dark else "black"
    fig.suptitle(args.title, fontsize=14, fontweight="bold", color=title_color)

    # Row 1: probe accuracy
    if args.probes:
        plot_probe_accuracy(args.probes, axes[0, 0], axes[0, 1], dark=args.dark)
    else:
        axes[0, 0].set_visible(False)
        axes[0, 1].set_visible(False)

    # Row 2: CCA + RSA heatmaps
    if args.cca:
        plot_cca_heatmap(args.cca, axes[1, 0], dark=args.dark)
    else:
        axes[1, 0].set_visible(False)
    if args.rsa:
        plot_rsa_heatmap(args.rsa, axes[1, 1], dark=args.dark)
    else:
        axes[1, 1].set_visible(False)

    # Row 3: subspace similarity + divergence
    if args.cca and args.probes:
        plot_probe_subspace(args.cca, axes[2, 0], dark=args.dark)
    else:
        axes[2, 0].set_visible(False)

    if args.divergence:
        plot_divergence(args.divergence, axes[2, 1], axes[2, 1], dark=args.dark)
        # override: divergence takes both cols if subspace absent
    else:
        axes[2, 1].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(args.output, "probe_results.png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                facecolor=DARK_BG if args.dark else "white",
                edgecolor="none")
    print(f"saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
