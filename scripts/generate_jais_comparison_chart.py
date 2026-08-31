#!/usr/bin/env python3
"""Generate the Jais 1/Jais 2 full-prompt comparison charts.

The values are the full_prompt_final macro-F1 means averaged over the
lemma-heldout and root-heldout splits, with each split value itself averaged
across five outer folds. Source: the immutable BF16 summary.csv files in
freezes/bf16-analysis-20260826-v2 and freezes/bf16-analysis-jais2-20260829-v1.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "plots"

TASKS = ("gender", "number", "POS")
JAIS_1 = (0.9259, 0.7177, 0.8612)
JAIS_2 = (0.9251, 0.9245, 0.8358)

THEMES = {
    "dark": {
        "bg": "#090b0e",
        "surface": "#0d1014",
        "text": "#f3efe5",
        "muted": "#a6a6ad",
        "border": "#4a4d53",
        "grid": "#3b3e44",
        "jais1": "#8eaed6",
        "jais2": "#d89151",
    },
    "light": {
        "bg": "#f3efe5",
        "surface": "#fbf8f1",
        "text": "#18191b",
        "muted": "#66676b",
        "border": "#bab4a9",
        "grid": "#d4cfc5",
        "jais1": "#526f96",
        "jais2": "#b86430",
    },
}


def render(theme_name: str, output: Path) -> None:
    theme = THEMES[theme_name]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": theme["text"],
            "axes.labelcolor": theme["muted"],
            "axes.edgecolor": theme["border"],
            "axes.facecolor": theme["surface"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["text"],
            "figure.facecolor": theme["bg"],
            "savefig.facecolor": theme["bg"],
            "savefig.edgecolor": theme["bg"],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    positions = list(range(len(TASKS)))
    height = 0.34
    bars1 = ax.barh(
        [position - height / 2 for position in positions],
        JAIS_1,
        height=height,
        color=theme["jais1"],
        label="Jais-13B",
    )
    bars2 = ax.barh(
        [position + height / 2 for position in positions],
        JAIS_2,
        height=height,
        color=theme["jais2"],
        label="JAIS-2-8B-Chat",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(TASKS)
    ax.invert_yaxis()
    ax.set_xlim(0.60, 1.0)
    ax.set_xlabel("Held-out macro-F1 (full_prompt_final)")
    ax.grid(axis="x", color=theme["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(theme["border"])
        spine.set_linewidth(0.7)
    for bars in (bars1, bars2):
        ax.bar_label(bars, fmt="%.4f", padding=4, color=theme["text"], fontsize=9)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=theme["surface"],
        edgecolor=theme["border"],
        labelcolor=theme["text"],
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.14)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("dark", "light", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    themes = ("dark", "light") if args.theme == "both" else (args.theme,)
    for theme in themes:
        suffix = "" if theme == "dark" else "_light"
        output = args.output_dir / f"jais_1_vs_jais_2_full_prompt{suffix}.png"
        render(theme, output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
