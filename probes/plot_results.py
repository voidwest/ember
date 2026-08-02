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
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .analysis_common import (
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )
except ImportError:  # direct script execution
    from analysis_common import (
        UNVERIFIABLE_PROMPT_AUDIT_STATUSES,
        enforce_probe_prompt_contracts,
    )

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
    if path is None or not Path(path).is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        return key in data


def load_npz(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except ValueError as error:
        raise ValueError(f"unsafe or invalid NPZ artifact: {source}") from error


def finite_vector(data, key, *, unit_interval=False):
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a non-empty finite vector")
    if unit_interval and np.any((value < 0.0) | (value > 1.0)):
        raise ValueError(f"{key} contains values outside [0, 1]")
    return value


def finite_matrix(data, key):
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a non-empty finite matrix")
    return value


def scalar_text(data, key):
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return str(value.reshape(-1)[0])


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
    data = load_npz(probes_path)
    # route to generic task rendering when a modern "tasks" manifest exists
    if "tasks" in data:
        colors = [value for value in (color_root, color_pat) if value is not None]
        return plot_generic_probe_metrics(
            data,
            ax_root,
            ax_pattern,
            dark=dark,
            model_label=label,
            colors_override=colors or None,
        )
    if "root_accuracy" not in data or "pattern_accuracy" not in data:
        return plot_generic_probe_metrics(data, ax_root, ax_pattern, dark=dark)

    root_accuracy = finite_vector(data, "root_accuracy", unit_interval=True)
    pattern_accuracy = finite_vector(data, "pattern_accuracy", unit_interval=True)
    if len(root_accuracy) != len(pattern_accuracy):
        raise ValueError("root and pattern accuracy vectors have different layer counts")
    n_layers = len(root_accuracy)
    layers = np.arange(n_layers)

    root_color = color_root or (DARK_BLUE if dark else LIGHT_CYCLE[1])
    pat_color = color_pat or (DARK_ACCENT if dark else LIGHT.accent)
    leg_label_root = f"root ({label})" if label else "root"
    leg_label_pat = f"pattern ({label})" if label else "pattern"

    ax_root.plot(layers, root_accuracy, color=root_color, marker="o",
                 markersize=4, linewidth=1.4, label=leg_label_root)
    root_chance = float(np.asarray(data["root_chance"]).item()) if "root_chance" in data else None
    if root_chance is not None:
        ax_root.axhline(root_chance, color=DARK_DIM if dark else LIGHT.subtle,
                        linestyle="--", alpha=0.5, label=f"chance ({root_chance:.1%})")
    ax_root.set_ylabel("accuracy")
    ax_root.set_title("root probe")
    ax_root.legend(fontsize=7)
    ax_root.grid(alpha=0.3)
    ax_root.set_ylim(-0.02, 1.05)

    ax_pattern.plot(layers, pattern_accuracy, color=pat_color, marker="o",
                    markersize=4, linewidth=1.4, label=leg_label_pat)
    pattern_chance = float(np.asarray(data["pattern_chance"]).item()) if "pattern_chance" in data else None
    if pattern_chance is not None:
        ax_pattern.axhline(pattern_chance, color=DARK_DIM if dark else LIGHT.subtle,
                           linestyle="--", alpha=0.5, label=f"chance ({pattern_chance:.1%})")
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
        selectivity = finite_vector(data, "root_selectivity")
        if len(selectivity) != n_layers:
            raise ValueError("root_selectivity layer count mismatch")
        ax_r2.plot(layers, selectivity, color=sel_color,
                   marker="s", markersize=3, linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax_r2.set_ylabel("selectivity", color=sel_color, fontsize=7)
        ax_r2.tick_params(axis="y", colors=sel_color, labelsize=6)

    pattern_selectivity_key = (
        "pattern_selectivity" if "pattern_selectivity" in data else "pat_selectivity"
        if "pat_selectivity" in data else None
    )
    if pattern_selectivity_key:
        sel_color = DARK_GREEN if dark else LIGHT_CYCLE[2]
        ax_p2 = ax_pattern.twinx()
        selectivity = finite_vector(data, pattern_selectivity_key)
        if len(selectivity) != n_layers:
            raise ValueError("pattern selectivity layer count mismatch")
        ax_p2.plot(layers, selectivity, color=sel_color,
                   marker="s", markersize=3, linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax_p2.set_ylabel("selectivity", color=sel_color, fontsize=7)
        ax_p2.tick_params(axis="y", colors=sel_color, labelsize=6)

    return True


def plot_generic_probe_metrics(
    data,
    ax_acc,
    ax_sel,
    dark=False,
    model_label=None,
    colors_override=None,
):
    """plot all task accuracies/selectivities from a generic probe NPZ."""
    if "tasks" not in data:
        return False
    tasks = [str(t) for t in data["tasks"].tolist()]
    colors = colors_override or ([pair[0] for pair in CM_COLORS] + [pair[1] for pair in CM_COLORS])
    plotted_acc = False
    plotted_sel = False
    plotted_margin = False

    for i, task in enumerate(tasks):
        key = safe_key(task)
        acc_key = f"{key}_accuracy"
        if acc_key not in data:
            continue
        acc = finite_vector(data, acc_key, unit_interval=True)
        layers = np.arange(len(acc))
        color = colors[i % len(colors)]
        label_text = task_label(task)
        if model_label:
            label_text = f"{label_text} ({model_label})"
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
            margin = finite_vector(data, margin_key)
            if len(margin) != len(acc):
                raise ValueError(f"{margin_key} layer count mismatch")
            ax_sel.plot(
                layers,
                margin,
                color=color,
                marker="s",
                markersize=3,
                linewidth=1.2,
                label=label_text,
            )
            plotted_margin = True

        sel_key = f"{key}_selectivity"
        if sel_key in data:
            selectivity = finite_vector(data, sel_key)
            if len(selectivity) != len(acc):
                raise ValueError(f"{sel_key} layer count mismatch")
            ax_sel.plot(
                layers,
                selectivity,
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
    data = load_npz(cca_path)
    if "cca_layer_matrix" not in data:
        return
    mat = finite_matrix(data, "cca_layer_matrix")
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("within-model CCA matrix must be square")
    if np.any((mat < 0.0) | (mat > 1.0)):
        raise ValueError("CCA similarities are outside [0, 1]")
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
    data = load_npz(rsa_path)
    if "rsa_layer_matrix" not in data:
        return
    mat = finite_matrix(data, "rsa_layer_matrix")
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("within-model RSA matrix must be square")
    if np.any((mat < -1.0) | (mat > 1.0)):
        raise ValueError("RSA correlations are outside [-1, 1]")
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
    data = load_npz(cca_path)
    if "root_pattern_cca" not in data:
        return
    sim = finite_vector(data, "root_pattern_cca", unit_interval=True)
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
    data = load_npz(div_path)
    if "cos_dist" not in data:
        return

    cos_dist = finite_vector(data, "cos_dist")
    eucl_dist = finite_vector(data, "eucl_dist")
    layers = finite_vector(data, "layer")
    if not (len(cos_dist) == len(eucl_dist) == len(layers)):
        raise ValueError("divergence vectors have different layer counts")
    if np.any(cos_dist < 0.0) or np.any(eucl_dist < 0.0):
        raise ValueError("divergence distances must be non-negative")
    if np.any(cos_dist > 2.0) or not np.array_equal(layers, np.arange(len(layers))):
        raise ValueError("divergence cosine distances or layer indices are invalid")
    cos_ok = True
    n_c = int(data.get("n_correct", 0))
    n_i = int(data.get("n_incorrect", 0))
    if n_c < 1 or n_i < 1:
        raise ValueError("divergence plot requires positive correct and incorrect counts")

    dim_color = DARK_DIM if dark else LIGHT.subtle

    if cos_ok:
        cos_color = DARK_ACCENT2 if dark else "m"
        euc_color = DARK_BLUE if dark else "c"

        ax_cos.plot(layers, cos_dist, color=cos_color, marker="o",
                    markersize=4, linewidth=1.4)
        ax_cos.set_ylabel("cosine distance")
        ax_cos.set_title("correct vs incorrect divergence (Q4)")
        ax_cos.grid(alpha=0.3)

        ax_euc.plot(layers, eucl_dist, color=euc_color, marker="o",
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
    source = Path(fertility_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {source}")

    data = json.loads(source.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(data, list) or not data or any(not isinstance(row, dict) for row in data):
        raise ValueError("fertility report must be a non-empty JSON array of objects")

    labels = [str(d["label"]) for d in data]
    if len(labels) != len(set(labels)) or any(not label for label in labels):
        raise ValueError("fertility labels must be non-empty and unique")

    def metric(field):
        values = np.asarray([row.get(field) for row in data], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"fertility field {field!r} must contain finite non-negative values")
        return values

    en_means = metric("en_mean_tokens")
    ar_means = metric("ar_mean_tokens")
    ratios = metric("en_ar_ratio")

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


def _require_artifact(path, label):
    if path is not None and not Path(path).is_file():
        raise FileNotFoundError(f"{label} artifact does not exist: {path}")


def _probe_source_identity(path):
    data = load_npz(path)
    return scalar_text(data, "stimuli_sha256"), scalar_text(data, "activations_sha256")


def _atomic_save_figure(fig, output, *, dpi, dark):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.tmp-",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        fig.savefig(
            temporary,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=DARK_BG if dark else LIGHT.bg,
            edgecolor="none",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


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
    parser.add_argument(
        "--allow-unverified-comparison",
        action="store_true",
        help="allow cross-model probe plots without matching stimuli SHA-256",
    )
    parser.add_argument(
        "--allow-label-revealed-inputs",
        action="store_true",
        help="plot label-revealed positive controls with an explicit warning title",
    )
    parser.add_argument(
        "--allow-unverifiable-prompt-contract",
        action="store_true",
        help="plot legacy probes only after their prompt contract was externally verified",
    )
    args = parser.parse_args()

    if args.dpi < 1:
        parser.error("--dpi must be positive")
    for path, label in (
        (args.probes, "probe"),
        (args.cca, "CCA"),
        (args.rsa, "RSA"),
        (args.divergence, "divergence"),
        (args.fertility, "fertility"),
    ):
        _require_artifact(path, label)
    artifact_activation_hashes = []
    for path, field in (
        (args.probes, "activations_sha256"),
        (args.cca, "activations_a_sha256"),
        (args.rsa, "activations_a_sha256"),
        (args.divergence, "activations_sha256"),
    ):
        if path is not None:
            artifact_activation_hashes.append(scalar_text(load_npz(path), field))
    if len(artifact_activation_hashes) > 1:
        if any(value is None for value in artifact_activation_hashes):
            if not args.allow_unverified_comparison:
                raise ValueError(
                    "combined plots require activation SHA-256 provenance for every artifact"
                )
        elif len(set(artifact_activation_hashes)) != 1:
            raise ValueError("combined analysis artifacts refer to different activation tensors")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    _setup_theme(dark=args.dark)

    # parse --compare pairs
    compare_pairs = []
    if args.compare:
        for item in args.compare:
            if ":" not in item:
                parser.error(f"malformed --compare item: {item!r}; expected LABEL:PATH")
            label, path = item.split(":", 1)
            if not label or not path:
                parser.error(f"malformed --compare item: {item!r}")
            _require_artifact(path, f"comparison {label}")
            compare_pairs.append((label, path))
        if len({label for label, _ in compare_pairs}) != len(compare_pairs):
            parser.error("--compare labels must be unique")
        identities = [_probe_source_identity(path)[0] for _, path in compare_pairs]
        if any(identity is None for identity in identities):
            if not args.allow_unverified_comparison:
                raise ValueError(
                    "cross-model probe artifacts require stimuli_sha256 provenance"
                )
        elif len(set(identities)) != 1:
            raise ValueError("cross-model probe artifacts use different stimuli files")

    prompt_audit_paths = []
    if args.probes:
        prompt_audit_paths.append(args.probes)
    prompt_audit_paths.extend(path for _, path in compare_pairs)
    statuses = enforce_probe_prompt_contracts(
        prompt_audit_paths,
        allow_label_revealed=args.allow_label_revealed_inputs,
        allow_unverifiable=args.allow_unverifiable_prompt_contract,
    ) if prompt_audit_paths else []

    # count how many plot rows we need
    has_single = args.probes is not None
    has_compare = bool(compare_pairs)
    has_cca = args.cca is not None
    has_rsa = args.rsa is not None
    has_subspace = has_cca and npz_has_key(args.cca, "root_pattern_cca")
    has_divergence = args.divergence is not None
    has_fertility = args.fertility is not None

    rows = []
    if has_single:
        rows.append(("probe", args.probes, None, None))
    if has_compare:
        rows.append(("compare", compare_pairs, None, None))
    if has_cca or has_rsa:
        rows.append(("cca" if has_cca else None, args.cca, "rsa" if has_rsa else None, args.rsa))
    if has_subspace or has_fertility:
        rows.append(
            (
                "subspace" if has_subspace else None,
                args.cca if has_subspace else None,
                "fertility" if has_fertility else None,
                args.fertility if has_fertility else None,
            )
        )
    if has_divergence:
        rows.append(("divergence", args.divergence, None, None))
    if not rows:
        parser.error("no data provided; nothing to plot")

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2 + len(rows) * 3.5))
    if len(rows) == 1:
        axes = np.array([axes])

    title_color = DARK_ACCENT if args.dark else LIGHT.accent_strong
    title = args.title
    if "label_revealed" in statuses:
        title = f"POSITIVE CONTROL — LABEL-REVEALED PROMPTS\n{title}"
    elif any(status in UNVERIFIABLE_PROMPT_AUDIT_STATUSES for status in statuses):
        title = f"UNVERIFIED PROMPT CONTRACT\n{title}"
    fig.suptitle(title, fontsize=14, fontweight="bold", color=title_color)

    for row_idx, (kind_l, arg_l, kind_r, arg_r) in enumerate(rows):
        ax_l, ax_r = axes[row_idx, 0], axes[row_idx, 1]
        if kind_l == "probe":
            plot_probe_accuracy(arg_l, ax_l, ax_r, dark=args.dark)
        elif kind_l == "compare":
            plot_cross_model_accuracy(arg_l, ax_l, ax_r, dark=args.dark)
        elif kind_l == "cca":
            plot_cca_heatmap(arg_l, ax_l, dark=args.dark)
        elif kind_l == "subspace":
            plot_probe_subspace(arg_l, ax_l, dark=args.dark)
        elif kind_l == "fertility":
            plot_fertility_comparison(arg_l, ax_l, dark=args.dark)
        elif kind_l == "divergence":
            plot_divergence(arg_l, ax_l, ax_r, dark=args.dark)
        else:
            ax_l.set_visible(False)

        # right panel
        if kind_r == "rsa":
            plot_rsa_heatmap(arg_r, ax_r, dark=args.dark)
        elif kind_r == "fertility":
            plot_fertility_comparison(arg_r, ax_r, dark=args.dark)
        elif kind_l not in {"probe", "compare", "divergence"}:
            ax_r.set_visible(False)

    fig.tight_layout()
    out_path = args.output_file or str(Path(args.output) / "probe_results.png")
    _atomic_save_figure(fig, out_path, dpi=args.dpi, dark=args.dark)
    print(f"saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
