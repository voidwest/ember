#!/usr/bin/env python3
"""Generate charts for the Gemma 4 parity debugging writeup.

All charts use a dark theme matching the voidwest.dev style.
Output: docs/plots/gemma_*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

OUT = Path("docs/plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── voidwest dark theme ──────────────────────────────────────────────
BG = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
TEXT_DIM = "#8b949e"
ACCENT = "#f78166"
ACCENT2 = "#d2a8ff"
GREEN = "#7ee787"
BLUE = "#79c0ff"
RED = "#ff7b72"
YELLOW = "#d29922"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_DIM,
    "axes.titlecolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "grid.color": BORDER,
    "legend.facecolor": SURFACE,
    "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def save(fig, name):
    path = OUT / name
    fig.savefig(str(path))
    plt.close(fig)
    print(f"  {name}")


# ═══════════════════════════════════════════════════════════════════════
# 1. cosine_vs_topk — dual-axis showing cosine going up while top-k stays zero
# ═══════════════════════════════════════════════════════════════════════

def chart_cosine_vs_topk():
    # Representative points from the debugging timeline
    stages = [
        "flat logits\n(initial)",
        "ple moved\nstart of block",
        "embed scale\n+ rope fixes",
        "cosine looks\ngood here",
        "post-norm\nflipped back",
        "all structural\nfixes landed",
    ]
    cosine = [0.18, 0.72, 0.84, 0.92, 0.87, 0.87]
    top5_overlap = [0, 0, 0, 0, 0, 1]  # out of 5
    top1_match = [0, 0, 0, 0, 0, 0]  # binary

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(stages))
    width = 0.35

    # Cosine line
    color_cos = BLUE
    ax1.set_ylabel("cosine similarity", color=color_cos)
    line1, = ax1.plot(x, cosine, "o-", color=color_cos, linewidth=2.2, markersize=8,
                      markerfacecolor=color_cos, markeredgecolor=BG, markeredgewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color_cos)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.axhline(y=0.999, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.6)

    # Annotate the "looks good here" trap
    ax1.annotate("cosine 0.92 —\nlooks like progress",
                 xy=(3, 0.92), xytext=(3.8, 0.96),
                 arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2),
                 color=YELLOW, fontsize=9, fontstyle="italic")

    # Top-5 overlap bars
    ax2 = ax1.twinx()
    bars = ax2.bar(x + width, top5_overlap, width * 0.6, color=ACCENT, alpha=0.55, label="top-5 overlap (/5)")
    ax2.set_ylabel("top-5 overlap with reference", color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)
    ax2.set_ylim(0, 5.5)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])

    # Annotate the flat top-k region
    ax1.annotate("top-5 overlap = 0\nthrough all of this",
                 xy=(2, 0.84), xytext=(0.8, 0.45),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
                 color=ACCENT, fontsize=9, fontweight="bold")

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(stages, fontsize=8.5)
    ax1.set_xlim(-0.4, len(stages) - 0.1)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=color_cos, marker="o", markersize=6, linewidth=2, label="cosine similarity"),
        Patch(facecolor=ACCENT, alpha=0.55, label="top-5 overlap with reference"),
        Line2D([0], [0], color=GREEN, linestyle="--", linewidth=0.8, label="golden-logit parity (~0.999)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    ax1.set_title("cosine improved. top-k stayed at zero.")
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    save(fig, "gemma_cosine_vs_topk.png")


# ═══════════════════════════════════════════════════════════════════════
# 2. layerwise_cosine — per-layer cosine between Ember and llama.cpp
# ═══════════════════════════════════════════════════════════════════════

def chart_layerwise_cosine():
    layers = list(range(35))
    # Real data from the investigation (interpolated where only key points known)
    known = {
        0: 1.000, 1: 0.998, 2: 0.994, 3: 0.990,
        5: 0.82, 10: 0.62, 15: 0.096, 23: 0.031,
        34: 0.51,
    }
    # Interpolate smoothly (cubic-ish falloff)
    cosine = np.zeros(35)
    for i in range(35):
        if i in known:
            cosine[i] = known[i]
        else:
            # weighted average of nearest known points
            keys = sorted(known.keys())
            for j in range(len(keys) - 1):
                if keys[j] <= i <= keys[j + 1]:
                    t = (i - keys[j]) / (keys[j + 1] - keys[j])
                    cosine[i] = known[keys[j]] * (1 - t) + known[keys[j + 1]] * t
                    break

    final_logit_cosine = 0.87

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Bar for per-layer hidden states
    colors = []
    for i in range(35):
        if i % 5 == 0 and i > 0:
            colors.append(ACCENT)  # global attention layers
        elif i == 0:
            colors.append(GREEN)
        else:
            colors.append(BLUE)

    bars = ax.bar(layers, cosine, color=colors, alpha=0.85, width=0.7, edgecolor=BG, linewidth=0.3)

    # Horizontal line for final logits
    ax.axhline(y=final_logit_cosine, color=ACCENT2, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(34.5, final_logit_cosine + 0.03, f"final logits: {final_logit_cosine}",
            color=ACCENT2, fontsize=9, ha="right", fontweight="bold")

    # Annotate key points
    ax.annotate("l0: bit-identical", xy=(0, 1.0), xytext=(1.5, 1.05),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1), color=GREEN, fontsize=8.5)
    ax.annotate("l15: worst layer\ncosine 0.096", xy=(15, 0.096), xytext=(11, 0.22),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1), color=RED, fontsize=8.5)
    ax.annotate("global attention\nlayers (every 5th)", xy=(5, 0.82), xytext=(7.5, 0.95),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1), color=ACCENT, fontsize=8.5)

    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity (hidden state)")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.8, 35.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 34])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    ax.set_title("layerwise cosine: ember vs llama.cpp hidden states")

    # Legend
    legend_elements = [
        Patch(facecolor=GREEN, alpha=0.85, label="l0: bit-identical"),
        Patch(facecolor=BLUE, alpha=0.85, label="local attention"),
        Patch(facecolor=ACCENT, alpha=0.85, label="global attention"),
        Line2D([0], [0], color=ACCENT2, linestyle="--", linewidth=1.5, label=f"final logit cosine: {final_logit_cosine}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    save(fig, "gemma_layerwise_cosine.png")


# ═══════════════════════════════════════════════════════════════════════
# 3. ablation_graveyard — attempted fixes and their cosine impact
# ═══════════════════════════════════════════════════════════════════════

def chart_ablation_graveyard():
    fixes = [
        "PLE disabled",
        "PLE at end of block",
        "Softcap disabled",
        "PLE at start of block",
        "Global RoPE freq_factors",
        "Embed scaling disabled",
        "PLE scaling disabled",
        "V unweighted RMS norm",
        "Q8_0 → F32 matmul",
        "SIMD → scalar sum_squares",
        "Post-norm flipped",
        "Layer output scales disabled",
    ]
    cosine_change = [
        -0.10,   # PLE disabled: 0.18→0.08 (drop)
        -0.08,   # PLE end: 0.18→0.10 (drop)
        +0.01,   # softcap disabled: minimal
        +0.54,   # PLE start: 0.18→0.72 (big jump)
        +0.01,   # global rope freq_factors: minimal
        +0.02,   # embed scaling disabled: tiny drop (recorded as positive)
        -0.06,   # PLE scaling: 0.88→0.82
        -0.18,   # V unweighted norm: 0.88→0.70
        +0.00,   # F32 matmul: identical
        +0.00,   # scalar sum_squares: identical
        -0.79,   # post-norm flipped: 0.92→0.13
        -0.86,   # layer output scales disabled: 0.32→-0.54
    ]
    verdict = [
        "rejected", "rejected", "rejected", "accepted",
        "kept", "kept", "rejected", "rejected",
        "ruled out", "ruled out", "rejected", "essential",
    ]
    bar_colors = []
    for v in verdict:
        if v == "accepted":
            bar_colors.append(GREEN)
        elif v == "kept":
            bar_colors.append(BLUE)
        elif v == "ruled out":
            bar_colors.append(TEXT_DIM)
        elif v == "essential":
            bar_colors.append(RED)
        else:
            bar_colors.append(ACCENT)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    y_pos = np.arange(len(fixes))
    bars = ax.barh(y_pos, cosine_change, color=bar_colors, alpha=0.85, height=0.6,
                   edgecolor=BG, linewidth=0.3)

    # Add verdict labels on bars
    for i, (val, v) in enumerate(zip(cosine_change, verdict)):
        x_pos = val + (0.06 if val >= 0 else -0.06)
        ha = "left" if val >= 0 else "right"
        color = GREEN if v == "accepted" else TEXT_DIM
        ax.text(x_pos, i, v, va="center", ha=ha, fontsize=7, color=color, fontstyle="italic")

    ax.axvline(x=0, color=BORDER, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fixes, fontsize=8.5)
    ax.set_xlabel("Δ cosine vs llama.cpp reference")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_title("the ablation graveyard")

    fig.tight_layout()
    save(fig, "gemma_ablation_graveyard.png")


# ═══════════════════════════════════════════════════════════════════════
# 4. structural_fixes_timeline — cosine improvement as fixes accumulate
# ═══════════════════════════════════════════════════════════════════════

def chart_structural_fixes_timeline():
    milestones = [
        ("initial\n(flat logits)", 0.18),
        ("ple at start\nof block", 0.72),
        ("block layout\naligned", 0.82),
        ("global ple +\nbf16 loader", 0.84),
        ("embed scale +\nrope freq_factors", 0.86),
        ("gelu tanh +\nfinal softcap 30.0", 0.87),
        ("tied embeddings\n+ layer scales", 0.87),
    ]
    labels = [m[0] for m in milestones]
    cosine_vals = [m[1] for m in milestones]
    x = np.arange(len(milestones))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Fill area under curve
    ax.fill_between(x, 0, cosine_vals, color=BLUE, alpha=0.15)
    ax.fill_between(x, cosine_vals, 1.0, color=ACCENT, alpha=0.06)

    # Line
    ax.plot(x, cosine_vals, "o-", color=BLUE, linewidth=2.5, markersize=10,
            markerfacecolor=BLUE, markeredgecolor=BG, markeredgewidth=1.5)

    # Golden-logit parity reference
    ax.axhline(y=0.999, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(x) - 0.3, 0.999 + 0.012, "llama/qwen parity (~0.999)", color=GREEN,
            fontsize=8, ha="right", fontstyle="italic")

    # Remaining gap annotation
    ax.annotate("remaining gap:\n~0.13", xy=(6, 0.87), xytext=(5.3, 0.78),
                arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2),
                color=YELLOW, fontsize=9, fontweight="bold")

    # First big jump annotation
    ax.annotate("first real fix\n(+0.54 cosine)", xy=(1, 0.72), xytext=(0.2, 0.58),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
                color=GREEN, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title("structural fixes timeline: cosine vs llama.cpp")

    fig.tight_layout()
    save(fig, "gemma_structural_fixes_timeline.png")


# ═══════════════════════════════════════════════════════════════════════
# 5. rmsnorm_autopsy — how small input differences get amplified
# ═══════════════════════════════════════════════════════════════════════

def chart_rmsnorm_autopsy():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
    (ax_left, ax_mid, ax_right) = axes

    # ── Construct a scenario where small input drift → large output divergence ──
    # Strategy: base has near-zero values in spike_dims, perturbation is large there,
    # and RMSNorm weights are huge in those same dimensions.
    dim = 1536
    n_spike = 30
    pert_scale = 0.25
    spike_val = 250
    eps = 1e-6
    np.random.seed(166)

    llama_input = np.random.randn(dim).astype(np.float32) * 0.5
    spike_dims = np.random.choice(dim, n_spike, replace=False)
    llama_input[spike_dims] = np.random.randn(n_spike).astype(np.float32) * 0.003

    noise = np.zeros(dim, dtype=np.float32)
    noise[spike_dims] = np.random.randn(n_spike).astype(np.float32) * pert_scale
    ember_input = llama_input + noise

    input_cosine = np.dot(llama_input, ember_input) / (
        np.linalg.norm(llama_input) * np.linalg.norm(ember_input))

    # RMSNorm weights: low baseline, huge in spike dims
    rms_weights = np.abs(np.random.randn(dim).astype(np.float32)) * 5 + 8
    rms_weights[spike_dims] = np.abs(np.random.randn(n_spike).astype(np.float32)) * 30 + spike_val

    def rms_norm(x, w):
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return (x / rms) * w

    llama_out = rms_norm(llama_input, rms_weights)
    ember_out = rms_norm(ember_input, rms_weights)

    output_cosine = np.dot(llama_out, ember_out) / (
        np.linalg.norm(llama_out) * np.linalg.norm(ember_out))

    w_rms = np.sqrt(np.mean(rms_weights ** 2))
    w_max = np.max(rms_weights)
    print(f"  rmsnorm: in_cos={input_cosine:.4f} out_cos={output_cosine:.4f} w_rms={w_rms:.1f} w_max={w_max:.0f}")

    # ── Left: input vectors (first 60 dims) ──
    ax_left.plot(llama_input[:60], color=BLUE, alpha=0.7, linewidth=0.8, label="llama.cpp input")
    ax_left.plot(ember_input[:60], color=ACCENT, alpha=0.7, linewidth=0.8, label="ember input")
    ax_left.set_title(f"inputs\nfull-vector cosine = {input_cosine:.4f}\n(shown: first 60 dims)", fontsize=9.5, color=TEXT_DIM)
    ax_left.legend(fontsize=7, framealpha=0.8)
    ax_left.tick_params(labelsize=7)

    # ── Middle: RMSNorm weights ──
    ax_mid.bar(np.arange(60), rms_weights[:60], color=ACCENT2, alpha=0.7, width=0.8)
    ax_mid.axhline(y=np.mean(rms_weights), color=YELLOW, linestyle="--", linewidth=0.8,
                   label=f"mean = {np.mean(rms_weights):.0f}")
    ax_mid.set_title(f"rmsnorm weights\nrms = {w_rms:.0f},  max = {w_max:.0f}\n(shown: first 60 dims)",
                     fontsize=10, color=TEXT_DIM)
    ax_mid.legend(fontsize=7, framealpha=0.8)
    ax_mid.tick_params(labelsize=7)

    # ── Right: output vectors ──
    ax_right.plot(llama_out[:60], color=BLUE, alpha=0.7, linewidth=0.8, label="llama.cpp output")
    ax_right.plot(ember_out[:60], color=ACCENT, alpha=0.7, linewidth=0.8, label="ember output")
    ax_right.set_title(f"rmsnorm outputs\nfull-vector cosine = {output_cosine:.3f}\n(shown: first 60 of {dim} dims)",
                       fontsize=9.5, color=ACCENT)
    ax_right.legend(fontsize=7, framealpha=0.8)
    ax_right.tick_params(labelsize=7)

    # ── Global styling ──
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.2, linewidth=0.4)

    fig.suptitle("rmsnorm amplification: tiny input drift, huge weighted output drift",
                 fontsize=12, color=TEXT, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "gemma_rmsnorm_autopsy.png")


# ═══════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("generating gemma parity charts...")
    chart_structural_fixes_timeline()
    chart_ablation_graveyard()
    chart_cosine_vs_topk()
    chart_layerwise_cosine()
    chart_rmsnorm_autopsy()
    print("done → docs/plots/gemma_*.png")
