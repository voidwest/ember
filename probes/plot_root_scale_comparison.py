"""plot root probe accuracy across LLaMA scales.

This produces a single LinkedIn-friendly chart from the per-model probe files:
root accuracy vs normalized layer depth for 1B, 3B, and 8B.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import BLUE, DARK, GREEN, RED, apply_matplotlib_theme  # noqa: E402

DARK_BG = DARK.bg
DARK_SURFACE = DARK.surface
DARK_BORDER = DARK.border
DARK_TEXT = DARK.text
DARK_DIM = DARK.muted
DARK_BLUE = BLUE
DARK_ACCENT = DARK.accent
DARK_GREEN = GREEN
DARK_RED = RED


RUNS = {
    "LLaMA 3.2 1B": ("data/llama1b_probes.npz", DARK_BLUE),
    "LLaMA 3.2 3B": ("data/llama3b_probes.npz", DARK_ACCENT),
    "LLaMA 3.1 8B": ("data/llama8b_probes.npz", DARK_GREEN),
}


def setup_dark_theme():
    apply_matplotlib_theme(dark=True)


def load_root_accuracy(path: str) -> np.ndarray:
    data = np.load(path)
    if "root_accuracy" not in data.files:
        raise KeyError(f"{path} missing root_accuracy; keys: {data.files}")
    return data["root_accuracy"]


def main():
    parser = argparse.ArgumentParser(
        description="plot root probe accuracy across LLaMA model scales"
    )
    parser.add_argument(
        "--output",
        default="docs/plots/root_probe_scale_comparison.png",
        help="output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    setup_dark_theme()

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    troughs = {}

    for label, (path, color) in RUNS.items():
        root_acc = load_root_accuracy(path)
        x = np.linspace(0.0, 1.0, len(root_acc))
        ax.plot(
            x,
            root_acc,
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            color=color,
            label=label,
        )
        trough_idx = int(np.argmin(root_acc))
        troughs[label] = (x[trough_idx], float(root_acc[trough_idx]))

    ax.axhline(
        1.0 / 20.0,
        color=DARK_DIM,
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        label="chance (5%)",
    )

    for label in ("LLaMA 3.2 3B", "LLaMA 3.1 8B"):
        x, y = troughs[label]
        short = "3B dip" if "3B" in label else "8B dip"
        y_offset = 0.07 if "3B" in label else -0.09
        pct = int(np.floor(y * 100.0 + 0.5))
        ax.annotate(
            f"{short}: {pct}%",
            xy=(x, y),
            xytext=(x + 0.06, y + y_offset),
            arrowprops={"arrowstyle": "->", "color": DARK_DIM, "lw": 1.0},
            color=DARK_TEXT,
            fontsize=9,
        )

    ax.set_title(
        "Root information becomes less linearly recoverable in mid-layers as scale increases",
        pad=14,
        color=DARK_TEXT,
        fontweight="bold",
    )
    ax.set_xlabel("normalized layer depth")
    ax.set_ylabel("root probe accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.92)

    for spine in ax.spines.values():
        spine.set_color(DARK_BORDER)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor=DARK_BG,
        edgecolor="none",
    )
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
