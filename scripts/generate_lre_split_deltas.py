#!/usr/bin/env python3
"""Generate the light and dark LRE lexical-split gap charts.

The values are the eleven model means from the manuscript's model-delta table.
Each value averages the three tasks and four representation interfaces.  The
site selects the generated image with ``.theme-chart-dark`` and
``.theme-chart-light`` in the page markup.
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

# Source: paper/paper1_lre_springer/main.tex, Table model-deltas.
MODEL_GAPS = (
    ("Qwen3-0.6B", 0.0977, 0.1118),
    ("Qwen2.5-1.5B", 0.1134, 0.1188),
    ("Qwen3-8B", 0.0783, 0.0750),
    ("Llama-3.2-1B", 0.1881, 0.2054),
    ("Llama-3.2-3B", 0.1276, 0.1279),
    ("Llama-3.1-8B", 0.0835, 0.0895),
    ("Gemma-4-E2B", 0.1382, 0.1518),
    ("Phi-3-mini", 0.1036, 0.1314),
    ("Mistral-7B", 0.0993, 0.1117),
    ("Jais-13B", 0.0711, 0.0918),
    ("ALLaM-7B", 0.0637, 0.0816),
)

THEMES = {
    "dark": {
        "bg": "#090b0e",
        "surface": "#0d1014",
        "text": "#f3efe5",
        "muted": "#a6a6ad",
        "border": "#4a4d53",
        "grid": "#3b3e44",
        "lemma": "#8eaed6",
        "root": "#d89151",
    },
    "light": {
        "bg": "#f3efe5",
        "surface": "#fbf8f1",
        "text": "#18191b",
        "muted": "#66676b",
        "border": "#bab4a9",
        "grid": "#d4cfc5",
        "lemma": "#526f96",
        "root": "#b86430",
    },
}


def render(theme_name: str, output: Path) -> None:
    theme = THEMES[theme_name]
    labels = [row[0] for row in MODEL_GAPS]
    lemma = [row[1] for row in MODEL_GAPS]
    root = [row[2] for row in MODEL_GAPS]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
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

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    y = list(range(len(labels)))
    height = 0.34
    ax.barh(
        [value - height / 2 for value in y],
        lemma,
        height=height,
        color=theme["lemma"],
        label="lemma-heldout",
    )
    ax.barh(
        [value + height / 2 for value in y],
        root,
        height=height,
        color=theme["root"],
        label="root-heldout",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.25)
    ax.set_xlabel("Macro-F1 gap: random − heldout")
    ax.grid(axis="x", color=theme["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(theme["border"])
        spine.set_linewidth(0.7)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=theme["surface"],
        edgecolor=theme["border"],
        labelcolor=theme["text"],
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.12)

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
    parser.add_argument(
        "--theme", choices=("dark", "light", "both"), default="both"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    themes = ("dark", "light") if args.theme == "both" else (args.theme,)
    for theme in themes:
        suffix = "" if theme == "dark" else "_light"
        output = args.output_dir / f"lre_split_deltas{suffix}.png"
        render(theme, output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
