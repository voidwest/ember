#!/usr/bin/env python3
"""Generate the docs crossover chart using the validated dark-theme renderer."""

from pathlib import Path

try:
    from plot_crossover import main
except ModuleNotFoundError:  # imported as scripts.gen_dark_chart
    from scripts.plot_crossover import main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                str(ROOT / "artifacts/crossover_sweep/crossover.csv"),
                str(ROOT / "artifacts/crossover_sweep/scheduling_overhead.csv"),
                "--theme",
                "dark",
                "--output",
                str(ROOT / "docs/plots/crossover_plot.png"),
            ]
        )
    )
