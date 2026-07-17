"""visualization for probing results.

plots all analysis outputs from the probing pipeline:
  (1) per-layer probe accuracy (root + pattern) with optional selectivity
  (2) CCA layer similarity heatmap
  (3) RSA layer similarity heatmap
  (4) probe weight subspace similarity
  (5) correct-vs-incorrect divergence
  (6) cross-model comparison overlay (--compare flag)
  (7) tokenizer fertility comparison (--fertility flag)

--compare label1:path1 label2:path2 ...  overlays probe accuracy from multiple models
--fertility path.json                      adds tokenizer fertility comparison chart

--dark flag produces dark-mode charts matching voidwest.dev styling.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import (  # noqa: E402
    BLUE, DARK, DARK_CYCLE, GREEN, LIGHT, LIGHT_CYCLE, PURPLE, RED, YELLOW,
    apply_matplotlib_theme,
    diverging_cmap, sequential_cmap, similarity_norm
)

# ── dark-mode palette (matches voidwest.dev CSS) ─────────────
DARK_BG       = DARK.bg
DARK_SURFACE  = DARK.surface
DARK_BORDER   = DARK.border
DARK_TEXT     = DARK.text
DARK_DIM      = DARK.muted
DARK_ACCENT   = DARK.accent
DARK_ACCENT2  = PURPLE
DARK_GREEN    = GREEN
DARK_BLUE     = BLUE
DARK_RED      = RED
DARK_YELLOW   = YELLOW

# cross-model palette
CM_COLORS = [
    (DARK_BLUE, DARK_RED),
    (DARK_ACCENT2, DARK_YELLOW),
    (DARK_ACCENT, "#ffa198"),
    (DARK_GREEN, DARK_CYCLE[5]),
    (DARK_TEXT, DARK_DIM),
]
CM_COLORS_LIGHT = [
    (LIGHT_CYCLE[1], LIGHT_CYCLE[4]),
    (LIGHT_CYCLE[0], LIGHT_CYCLE[3]),
    (LIGHT_CYCLE[2], LIGHT_CYCLE[5]),
]


def safe_key(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in value)


def task_label(task: str) -> str:
    return task.removeprefix("labels.")


def npz_has_key(path, key):
    if path is None or not os.path.exists(path):
        return False
    with np.load(path) as data:
        return key in data


def _setup_theme(*, dark: bool):
    """Apply the stylesheet-derived dark or light figure theme."""
    apply_matplotlib_theme(dark=dark)


def plot_probe_accuracy(probes_path, ax_root, ax_pattern, dark=False,
                        label=None, color_root=None, color_pat=None):
    """plot per-layer root and pattern probe accuracy.

    if label is provided, it's used in the legend (for cross-model comparison).
    if color_root/color_pat are provided, they override dark/light defaults.
    returns True if data was plotted.
    """
    data = np.load(probes_path, allow_pickle=True)
    # route to generic task rendering when a modern "tasks" manifest exists
    if "tasks" in data:
        return plot_generic_probe_metrics(data, ax_root, ax_pattern, dark=dark)
    if "root_accuracy" not in data or "pattern_accuracy" not in data:
        return plot_generic_probe_metrics(data, ax_root, ax_pattern, dark=dark)

    n_layers = len(data["root_accuracy"])
    layers = np.arange(n_layers)

    root_color = color_root or (DARK_BLUE if dark else LIGHT_CYCLE[1])
    pat_color = color_pat or (DARK_ACCENT if dark else LIGHT.accent)
    leg_label_root = f"root ({label})" if label else "root"
    leg_label_pat = f"pattern ({label})" if label else "pattern"

    ax_root.plot(layers, data["root_accuracy"], color=root_color, marker="o",
                 markersize=4, linewidth=1.4, label=leg_label_root)
    ax_root.axhline(1.0 / 20, color=DARK_DIM if dark else LIGHT.subtle,
                    linestyle="--", alpha=0.5, label="chance (5%)")
    ax_root.set_ylabel("accuracy")
    ax_root.set_title("root probe")
    ax_root.legend(fontsize=7)
    ax_root.grid(alpha=0.3)
    ax_root.set_ylim(-0.02, 1.05)

    ax_pattern.plot(layers, data["pattern_accuracy"], color=pat_color, marker="o",
                    markersize=4, linewidth=1.4, label=leg_label_pat)
    ax_pattern.axhline(1.0 / 10, color=DARK_DIM if dark else LIGHT.subtle,
                       linestyle="--", alpha=0.5, label="chance (10%)")
    ax_pattern.set_ylabel("accuracy")
    ax_pattern.set_title("pattern probe")
    ax_pattern.set_xlabel("layer")
    ax_pattern.legend(fontsize=7)
    ax_pattern.grid(alpha=0.3)
    ax_pattern.set_ylim(-0.02, 1.05)

    # plot selectivity on twin axis if available
    if "root_selectivity" in data:
        sel_color = DARK_GREEN if dark else LIGHT_CYCLE[2]
        ax_r2 = ax_root.twinx()
        ax_r2.plot(layers, data["root_selectivity"], color=sel_color,
                   marker="s", markersize=3, linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax_r2.set_ylabel("selectivity", color=sel_color, fontsize=7)
        ax_r2.tick_params(axis="y", colors=sel_color, labelsize=6)

    if "pat_selectivity" in data:
        sel_color = DARK_GREEN if dark else LIGHT_CYCLE[2]
        ax_p2 = ax_pattern.twinx()
        ax_p2.plot(layers, data["pat_selectivity"], color=sel_color,
                   marker="s", markersize=3, linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax_p2.set_ylabel("selectivity", color=sel_color, fontsize=7)
        ax_p2.tick_params(axis="y", colors=sel_color, labelsize=6)

    return True


def plot_generic_probe_metrics(data, ax_acc, ax_sel, dark=False):
    """plot all task accuracies/selectivities from a generic probe NPZ."""
    if "tasks" not in data:
        return False
    tasks = [str(t) for t in data["tasks"].tolist()]
    colors = [pair[0] for pair in CM_COLORS] + [pair[1] for pair in CM_COLORS]
    layers = None
    plotted_acc = False
    plotted_sel = False
    plotted_margin = False

    for i, task in enumerate(tasks):
        key = safe_key(task)
        acc_key = f"{key}_accuracy"
        if acc_key not in data:
            continue
        acc = data[acc_key]
        if layers is None:
            layers = np.arange(len(acc))
        color = colors[i % len(colors)]
        label_text = task_label(task)
        ax_acc.plot(
            layers,
            acc,
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=label_text,
        )
        plotted_acc = True

        margin_key = f"{key}_accuracy_minus_majority"
        if margin_key in data:
            ax_sel.plot(
                layers,
                data[margin_key],
                color=color,
                marker="s",
                markersize=3,
                linewidth=1.2,
                label=label_text,
            )
            plotted_margin = True

        sel_key = f"{key}_selectivity"
        if sel_key in data:
            ax_sel.plot(
                layers,
                data[sel_key],
                color=color,
                marker=None,
                linewidth=0.9,
                linestyle="--",
                alpha=0.45,
                label=f"{label_text} selectivity" if not plotted_margin else None,
            )
            plotted_sel = True

    if not plotted_acc:
        return False

    ax_acc.set_ylabel("accuracy")
    ax_acc.set_xlabel("layer")
    ax_acc.set_title("probe accuracy")
    ax_acc.set_ylim(-0.02, 1.05)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=7)

    if plotted_margin or plotted_sel:
        ax_sel.axhline(0.0, color=DARK_DIM if dark else LIGHT.subtle, linestyle="--", alpha=0.5)
        ax_sel.set_ylabel("score")
        ax_sel.set_xlabel("layer")
        ax_sel.set_title("accuracy - majority baseline")
        ax_sel.grid(alpha=0.3)
        ax_sel.legend(fontsize=7)
    else:
        ax_sel.set_visible(False)

    return True


def plot_cross_model_accuracy(compare_pairs, ax_root, ax_pattern, dark=False):
    """overlay probe accuracy from multiple models."""
    for i, (label, path) in enumerate(compare_pairs):
        palette = CM_COLORS if dark else CM_COLORS_LIGHT
        cr, cp = palette[i % len(palette)]
        plot_probe_accuracy(path, ax_root, ax_pattern, dark=dark,
                           label=label, color_root=cr, color_pat=cp)


def plot_cca_heatmap(cca_path, ax, dark=False):
    """plot CCA layer similarity matrix."""
    data = np.load(cca_path)
    if "cca_layer_matrix" not in data:
        return
    mat = data["cca_layer_matrix"]
    cmap = sequential_cmap(dark=dark)
    im = ax.imshow(mat, cmap=cmap, norm=similarity_norm(), aspect="auto")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title("CCA layer similarity")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0.0, 0.6, 0.8, 0.9, 1.0])
    if dark:
        cbar.ax.yaxis.set_tick_params(color=DARK_DIM)
        cbar.outline.set_edgecolor(DARK_BORDER)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK_DIM)


def plot_rsa_heatmap(rsa_path, ax, dark=False):
    """plot RSA layer similarity matrix."""
    data = np.load(rsa_path)
    if "rsa_layer_matrix" not in data:
        return
    mat = data["rsa_layer_matrix"]
    im = ax.imshow(mat, cmap=diverging_cmap(dark=dark), vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title("RSA layer similarity")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if dark:
        cbar.ax.yaxis.set_tick_params(color=DARK_DIM)
        cbar.outline.set_edgecolor(DARK_BORDER)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK_DIM)


def plot_probe_subspace(cca_path, ax, dark=False):
    """plot root-pattern probe subspace similarity."""
    data = np.load(cca_path)
    if "root_pattern_cca" not in data:
        return
    sim = data["root_pattern_cca"]
    layers = np.arange(len(sim))
    color = DARK_GREEN if dark else LIGHT_CYCLE[2]
    ax.plot(layers, sim, color=color, marker="o", markersize=4, linewidth=1.4)
    ax.set_ylabel("subspace CCA")
    ax.set_xlabel("layer")
    ax.set_title("root-pattern subspace (Q3)")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=-0.02)


def plot_divergence(div_path, ax_cos, ax_euc, dark=False):
    """plot correct-vs-incorrect divergence curves."""
    data = np.load(div_path)
    if "cos_dist" not in data:
        return

    cos_ok = not np.isnan(data["cos_dist"]).all()
    n_c = int(data.get("n_correct", 0))
    n_i = int(data.get("n_incorrect", 0))

    dim_color = DARK_DIM if dark else LIGHT.subtle

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
                    transform=ax_cos.transAxes, va="top", fontsize=7,
                    color=DARK_DIM if dark else LIGHT.text)
    else:
        for ax in (ax_cos, ax_euc):
            ax.text(0.5, 0.5, "N/A — 0 correct predictions",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=dim_color)
            ax.set_title("correct vs incorrect divergence (Q4)")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.2)


def plot_fertility_comparison(fertility_path, ax, dark=False):
    """plot tokenizer fertility comparison as a grouped bar chart."""
    with open(fertility_path, encoding="utf-8") as f:
        data = json.load(f)

    labels = [d["label"] for d in data]
    en_means = [d.get("en_mean_tokens", 0) for d in data]
    ar_means = [d.get("ar_mean_tokens", 0) for d in data]
    ratios = [d.get("en_ar_ratio", 0) for d in data]

    x = np.arange(len(labels))
    width = 0.35

    en_color = DARK_BLUE if dark else "steelblue"
    ar_color = DARK_ACCENT if dark else "darkorange"

    ax.bar(x - width / 2, en_means, width, label="en tokens",
           color=en_color, alpha=0.85)
    ax.bar(x + width / 2, ar_means, width, label="ar tokens",
           color=ar_color, alpha=0.85)

    for i, ratio in enumerate(ratios):
        ax.text(i, max(en_means[i], ar_means[i]) + 2,
                f"×{ratio:.1f}", ha="center", fontsize=8,
                color=DARK_DIM if dark else LIGHT.text)

    ax.set_ylabel("mean tokens/prompt")
    ax.set_title("tokenizer fertility (en vs ar prompts)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")


def main():
    parser = argparse.ArgumentParser(description="plot probe results")
    parser.add_argument("--probes", default=None, help="path to probe weights .npz")
    parser.add_argument("--cca", default=None, help="path to CCA results .npz")
    parser.add_argument("--rsa", default=None, help="path to RSA results .npz")
    parser.add_argument("--divergence", default=None,
                        help="path to divergence results .npz")
    parser.add_argument(
        "--compare", nargs="*", default=None,
        metavar="LABEL:PATH",
        help="cross-model comparison: label1:path1 label2:path2 ..."
    )
    parser.add_argument(
        "--fertility", default=None,
        help="path to fertility.json for tokenizer comparison chart"
    )
    parser.add_argument("--output", default="data/plots/",
                        help="output directory for plots")
    parser.add_argument("--output-file", default=None,
                        help="optional exact path for the main figure")
    parser.add_argument("--dark", action="store_true", help="dark-mode styling")
    parser.add_argument("--title", default="Arabic Morphology Probing Results",
                        help="figure suptitle")
    parser.add_argument("--dpi", type=int, default=150, help="output DPI")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.output_file:
        output_parent = os.path.dirname(args.output_file)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

    _setup_theme(dark=args.dark)

    # parse --compare pairs
    compare_pairs = None
    if args.compare:
        compare_pairs = []
        for item in args.compare:
            if ":" in item:
                label, path = item.split(":", 1)
                compare_pairs.append((label, path))
            else:
                print(f"warning: skipping malformed --compare item: {item}")

    # count how many plot rows we need
    has_single = args.probes is not None
    has_compare = compare_pairs is not None and len(compare_pairs) > 0
    has_cca = args.cca is not None
    has_rsa = args.rsa is not None
    has_subspace = has_cca and args.probes is not None and npz_has_key(args.cca, "root_pattern_cca")
    has_divergence = args.divergence is not None
    has_fertility = args.fertility is not None

    n_rows = 0
    row_assignments = []  # (row_idx, col0_fn, col1_fn, col0_args, col1_args)

    # row 0: single-model probe accuracy or cross-model comparison
    if has_single or has_compare:
        n_rows += 1
        if has_single and has_compare:
            # single model on left, cross-model on right
            row_assignments.append((
                "single_probe", args.probes, args.dark,
                "cross_model", compare_pairs, args.dark,
            ))
        elif has_single:
            row_assignments.append((
                "probe_accuracy", args.probes, None,
                None, None, None,
            ))
        else:
            row_assignments.append((
                "cross_model", compare_pairs, None,
                None, None, None,
            ))

    # row 1: CCA + RSA
    if has_cca or has_rsa:
        n_rows += 1
        row_assignments.append((
            "cca", args.cca,
            "rsa", args.rsa,
        ))

    # row 2: subspace + divergence or fertility
    if has_subspace or has_divergence or has_fertility:
        n_rows += 1
        row_assignments.append((
            "subspace" if has_subspace else "fertility",
            args.cca if has_subspace else args.fertility,
            "divergence" if has_divergence else "empty",
            args.divergence if has_divergence else None,
        ))

    if n_rows == 0:
        print("no data provided; nothing to plot")
        return

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2 + n_rows * 3.5))
    if n_rows == 1:
        axes = np.array([axes])

    title_color = DARK_ACCENT if args.dark else LIGHT.accent_strong
    fig.suptitle(args.title, fontsize=14, fontweight="bold", color=title_color)

    for row_idx, assignment in enumerate(row_assignments):
        ax_l, ax_r = axes[row_idx, 0], axes[row_idx, 1]
        kind_l, arg_l, kind_r, arg_r = \
            assignment[0], assignment[1], assignment[2], assignment[3]

        # left panel
        if kind_l == "single_probe":
            dark_arg = assignment[2]
            plot_probe_accuracy(arg_l, ax_l, ax_r, dark=dark_arg)
            # right panel gets cross-model
            cm_pairs, cm_dark = assignment[4], assignment[5]
            # cross-model takes new axes
            pass  # handled separately below
        elif kind_l == "probe_accuracy":
            plot_probe_accuracy(arg_l, ax_l, ax_r, dark=args.dark)
        elif kind_l == "cross_model":
            plot_cross_model_accuracy(arg_l, ax_l, ax_r, dark=args.dark)
        elif kind_l == "cca":
            plot_cca_heatmap(arg_l, ax_l, dark=args.dark)
        elif kind_l == "subspace":
            plot_probe_subspace(arg_l, ax_l, dark=args.dark)
        elif kind_l == "fertility":
            plot_fertility_comparison(arg_l, ax_l, dark=args.dark)
        else:
            ax_l.set_visible(False)

        # right panel
        if kind_r == "rsa":
            plot_rsa_heatmap(arg_r, ax_r, dark=args.dark)
        elif kind_r == "divergence":
            plot_divergence(arg_r, ax_r, ax_r, dark=args.dark)
        elif kind_r == "empty":
            ax_r.set_visible(False)
        elif kind_r is not None:
            ax_r.set_visible(False)

    # special case: single + cross-model on same row
    # re-do row 0 if both single and cross-model
    if has_single and has_compare:
        # left: single
        plot_probe_accuracy(args.probes, axes[0, 0], axes[0, 0], dark=args.dark)
        # right: cross-model
        # create a new fig for cross-model or use the right panel differently
        # actually, let's put cross-model on right panel with its own twin axes
        fig_cm, (ax_cm_root, ax_cm_pat) = plt.subplots(1, 2, figsize=(6, 3))
        plot_cross_model_accuracy(compare_pairs, ax_cm_root, ax_cm_pat, dark=args.dark)
        # save cross-model separately
        cm_path = os.path.join(args.output, "cross_model_comparison.png")
        fig_cm.savefig(cm_path, dpi=args.dpi, bbox_inches="tight",
                       facecolor=DARK_BG if args.dark else LIGHT.bg,
                       edgecolor="none")
        print(f"saved cross-model comparison to {cm_path}")
        plt.close(fig_cm)

    plt.tight_layout()
    out_path = args.output_file or os.path.join(args.output, "probe_results.png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                facecolor=DARK_BG if args.dark else LIGHT.bg,
                edgecolor="none")
    print(f"saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
