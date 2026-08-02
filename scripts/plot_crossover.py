#!/usr/bin/env python3
"""Validate a Q8 decode crossover sweep and plot measured speedups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from voidwest_theme import DARK_CYCLE, LIGHT_CYCLE, apply_matplotlib_theme
except ModuleNotFoundError:  # imported as scripts.plot_crossover
    from scripts.voidwest_theme import (
        DARK_CYCLE,
        LIGHT_CYCLE,
        apply_matplotlib_theme,
    )


COLUMNS = (
    "embed_dim",
    "inter_dim",
    "mflops",
    "threads",
    "n_iters",
    "median_ns",
    "min_ns",
    "max_ns",
    "stdev_ns",
)


@dataclass(frozen=True)
class SweepRow:
    embed_dim: int
    inter_dim: int
    mflops: float
    threads: int
    n_iters: int
    median_ns: float
    min_ns: float
    max_ns: float
    stdev_ns: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: str, *, context: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{context} must be an integer, got {value!r}") from error
    if result <= 0:
        raise ValueError(f"{context} must be greater than zero")
    return result


def _number(value: str, *, context: str, positive: bool = True) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{context} must be numeric, got {value!r}") from error
    if not math.isfinite(result) or (positive and result <= 0):
        comparator = "positive and finite" if positive else "finite"
        raise ValueError(f"{context} must be {comparator}")
    return result


def load_sweep(path: Path) -> list[SweepRow]:
    if not path.is_file():
        raise FileNotFoundError(f"crossover CSV does not exist: {path}")
    rows: list[SweepRow] = []
    seen: set[tuple[int, int, int]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle), start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if values[0].strip() == "embed_dim":
                continue
            if len(values) != len(COLUMNS):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(COLUMNS)} columns, got {len(values)}"
                )
            embed_dim = _integer(values[0], context=f"{path}:{line_number} embed_dim")
            inter_dim = _integer(values[1], context=f"{path}:{line_number} inter_dim")
            mflops = _number(values[2], context=f"{path}:{line_number} mflops")
            threads = _integer(values[3], context=f"{path}:{line_number} threads")
            n_iters = _integer(values[4], context=f"{path}:{line_number} n_iters")
            median_ns = _number(values[5], context=f"{path}:{line_number} median_ns")
            min_ns = _number(values[6], context=f"{path}:{line_number} min_ns")
            max_ns = _number(values[7], context=f"{path}:{line_number} max_ns")
            stdev_ns = _number(
                values[8], context=f"{path}:{line_number} stdev_ns", positive=False
            )
            if stdev_ns < 0:
                raise ValueError(f"{path}:{line_number}: stdev_ns must be non-negative")
            if not min_ns <= median_ns <= max_ns:
                raise ValueError(f"{path}:{line_number}: require min <= median <= max")
            computed_mflops = 2.0 * embed_dim * inter_dim / 1_000_000.0
            if not math.isclose(mflops, computed_mflops, rel_tol=0.01, abs_tol=0.11):
                raise ValueError(
                    f"{path}:{line_number}: mflops={mflops} disagrees with dimensions "
                    f"({computed_mflops:.3f})"
                )
            key = (embed_dim, inter_dim, threads)
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate measurement {key}")
            seen.add(key)
            rows.append(
                SweepRow(
                    embed_dim,
                    inter_dim,
                    mflops,
                    threads,
                    n_iters,
                    median_ns,
                    min_ns,
                    max_ns,
                    stdev_ns,
                )
            )
    if not rows:
        raise ValueError(f"crossover CSV contains no measurements: {path}")

    dimensions_by_mflops: dict[float, tuple[int, int]] = {}
    for row in rows:
        dimensions = (row.embed_dim, row.inter_dim)
        previous = dimensions_by_mflops.setdefault(row.mflops, dimensions)
        if previous != dimensions:
            raise ValueError(f"mflops value {row.mflops} maps to multiple matrix dimensions")
    sizes = sorted(dimensions_by_mflops)
    thread_counts = sorted({row.threads for row in rows})
    for size in sizes:
        missing = [
            thread for thread in thread_counts
            if not any(row.mflops == size and row.threads == thread for row in rows)
        ]
        if missing:
            raise ValueError(
                f"measurement size {size} MFLOPs is missing thread counts {missing}"
            )
        if 1 not in thread_counts:
            raise ValueError(f"measurement size {size} MFLOPs has no one-thread baseline")
    return rows


def load_real_sweep(path: Path, *, cache_state: str) -> list[SweepRow]:
    """Adapt q8_matmul JSONL rows to the common per-matmul work scale."""
    rows: list[SweepRow] = []
    seen: set[tuple[int, int]] = set()
    identity = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            def reject_constant(value):
                raise ValueError(
                    f"non-standard JSON constant {value!r} at {path}:{line_number}"
                )

            try:
                record = json.loads(line, parse_constant=reject_constant)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                raise ValueError(f"invalid q8 benchmark record at {path}:{line_number}")
            if record.get("benchmark") != "q8_gate_up" or record.get("exact_parity") is not True:
                raise ValueError(f"unverified q8 benchmark record at {path}:{line_number}")
            if record.get("cache_state") != cache_state:
                continue
            current_identity = (
                record.get("model"),
                record.get("layer"),
                record.get("first_tensor"),
                record.get("second_tensor"),
                record.get("input_features"),
                record.get("output_features"),
            )
            if identity is None:
                identity = current_identity
            elif identity != current_identity:
                raise ValueError("real-model sweep mixes model, layer, tensor, or shape identities")
            row_count = record.get("rows")
            threads = record.get("threads")
            input_features = record.get("input_features")
            output_features = record.get("output_features")
            n_samples = record.get("samples_per_mode")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (row_count, threads, input_features, output_features, n_samples)
            ):
                raise ValueError(f"invalid benchmark dimensions at {path}:{line_number}")
            paired = record.get("paired")
            separate = record.get("separate")
            if not isinstance(paired, dict) or not isinstance(separate, dict):
                raise ValueError(f"missing timing statistics at {path}:{line_number}")
            samples = paired.get("samples_ns")
            if (
                not isinstance(samples, list)
                or len(samples) != n_samples
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in samples)
            ):
                raise ValueError(f"invalid timing samples at {path}:{line_number}")
            ordered = sorted(samples)
            median_ns = float(ordered[len(ordered) // 2])
            if paired.get("median_ns") != int(median_ns):
                raise ValueError(f"paired median does not match samples at {path}:{line_number}")
            separate_median = separate.get("median_ns")
            speedup = record.get("paired_speedup")
            if (
                isinstance(separate_median, bool)
                or not isinstance(separate_median, int)
                or separate_median <= 0
                or isinstance(speedup, bool)
                or not isinstance(speedup, (int, float))
                or not math.isfinite(speedup)
                or not math.isclose(speedup, separate_median / median_ns, rel_tol=1e-12)
            ):
                raise ValueError(f"paired speedup is inconsistent at {path}:{line_number}")
            key = (row_count, threads)
            if key in seen:
                raise ValueError(f"duplicate real-model measurement {key}")
            seen.add(key)
            # Work per one projection. The benchmark times a pair, but this
            # scale preserves the historical chart's MFLOPs-per-matmul x-axis.
            mflops = 2.0 * row_count * input_features * output_features / 1_000_000.0
            rows.append(
                SweepRow(
                    embed_dim=row_count * input_features,
                    inter_dim=output_features,
                    mflops=mflops,
                    threads=threads,
                    n_iters=n_samples,
                    median_ns=median_ns,
                    min_ns=float(min(samples)),
                    max_ns=float(max(samples)),
                    stdev_ns=float(statistics.pstdev(samples)),
                )
            )
    if not rows:
        raise ValueError(f"no {cache_state!r} q8 benchmark rows found in {path}")
    thread_counts = sorted({row.threads for row in rows})
    for size in {row.mflops for row in rows}:
        missing = [
            thread for thread in thread_counts
            if not any(row.mflops == size and row.threads == thread for row in rows)
        ]
        if missing:
            raise ValueError(
                f"real-model workload {size} MFLOPs is missing thread counts {missing}"
            )
        if 1 not in thread_counts:
            raise ValueError(f"real-model workload {size} MFLOPs has no one-thread baseline")
    return rows


def load_measurements(path: Path, *, cache_state: str) -> tuple[list[SweepRow], str]:
    with path.open(encoding="utf-8") as handle:
        first = next((character for line in handle for character in line if not character.isspace()), "")
    if first == "{":
        return load_real_sweep(path, cache_state=cache_state), "real_model_jsonl"
    return load_sweep(path), "synthetic_csv"


def load_overhead(path: Path | None) -> dict[int, tuple[float, ...]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"scheduling-overhead CSV does not exist: {path}")
    result: dict[int, tuple[float, ...]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle), start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if values[0].strip() == "threads":
                continue
            if len(values) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least two columns")
            threads = _integer(values[0], context=f"{path}:{line_number} threads")
            samples = tuple(
                _number(value, context=f"{path}:{line_number} overhead column {index + 2}")
                for index, value in enumerate(values[1:])
            )
            if threads in result:
                raise ValueError(f"{path}:{line_number}: duplicate thread count {threads}")
            result[threads] = samples
    if not result:
        raise ValueError(f"scheduling-overhead CSV contains no measurements: {path}")
    return result


def summarize(rows: list[SweepRow]) -> dict[str, Any]:
    lookup = {(row.mflops, row.threads): row for row in rows}
    sizes = sorted({row.mflops for row in rows})
    threads = sorted({row.threads for row in rows})
    baselines = {size: lookup[(size, 1)].median_ns for size in sizes}
    speedups: dict[int, list[tuple[float, float]]] = {}
    first_observed: dict[int, dict[str, float | None]] = {}
    for thread_count in threads:
        if thread_count == 1:
            continue
        points = [
            (size, baselines[size] / lookup[(size, thread_count)].median_ns)
            for size in sizes
            if (size, thread_count) in lookup
        ]
        speedups[thread_count] = points
        first_index = next((index for index, (_, value) in enumerate(points) if value > 1), None)
        if first_index is None:
            first_observed[thread_count] = {
                "last_non_speedup_mflops": points[-1][0] if points else None,
                "first_speedup_mflops": None,
            }
        else:
            first_observed[thread_count] = {
                "last_non_speedup_mflops": points[first_index - 1][0]
                if first_index > 0
                else None,
                "first_speedup_mflops": points[first_index][0],
            }
    return {
        "sizes": sizes,
        "threads": threads,
        "lookup": lookup,
        "baselines": baselines,
        "speedups": speedups,
        "first_observed_speedup_bounds": first_observed,
    }


def print_report(
    summary: dict[str, Any],
    overhead: dict[int, tuple[float, ...]],
    *,
    source_format: str,
) -> None:
    lookup = summary["lookup"]
    threads = [thread for thread in summary["threads"] if thread != 1]
    label = "Real-model" if source_format == "real_model_jsonl" else "Synthetic"
    print(f"{label} crossover sweep — Q8 kernel only")
    headings = ["MFLOPs", "1t ms"] + [f"{thread}t ms / speedup / σ÷median" for thread in threads]
    print(" | ".join(headings))
    for size in summary["sizes"]:
        base = summary["baselines"][size]
        columns = [f"{size:.1f}", f"{base / 1e6:.3f}"]
        for thread in threads:
            row = lookup.get((size, thread))
            if row is None:
                columns.append("—")
                continue
            columns.append(
                f"{row.median_ns / 1e6:.3f} / {base / row.median_ns:.3f}× / "
                f"{100 * row.stdev_ns / row.median_ns:.1f}%"
            )
        print(" | ".join(columns))

    print("\nFirst observed median speedup (>1.0; not an inferred hardware threshold):")
    for thread, bounds in summary["first_observed_speedup_bounds"].items():
        lower = bounds["last_non_speedup_mflops"]
        upper = bounds["first_speedup_mflops"]
        if upper is None:
            print(f"  {thread} threads: not observed in the tested range")
        elif lower is None:
            print(f"  {thread} threads: at or below the first tested point ({upper:.1f} MFLOPs)")
        else:
            print(f"  {thread} threads: between tested points {lower:.1f} and {upper:.1f} MFLOPs")
    if overhead:
        print("\nScheduling-overhead samples (ns):")
        for thread, samples in sorted(overhead.items()):
            print(f"  {thread} threads: " + ", ".join(f"{sample:.0f}" for sample in samples))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def plot(summary: dict[str, Any], output: Path, *, theme: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dark = theme == "dark"
    visual = apply_matplotlib_theme(dark=dark, dpi=160)
    colors = DARK_CYCLE if dark else LIGHT_CYCLE
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ("s", "D", "^", "o", "v", "P")
    for index, (thread, points) in enumerate(sorted(summary["speedups"].items())):
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker=markers[index % len(markers)],
            color=colors[(index + 1) % len(colors)],
            linewidth=2,
            markersize=7,
            label=f"{thread} threads",
        )
    ax.axhline(y=1.0, color=visual.muted, linestyle="--", alpha=0.7, label="no speedup")
    ax.set_xlabel("MFLOPs per matmul (Q8_0 decode)")
    ax.set_ylabel("Median speedup vs one thread")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, format=output.suffix.lstrip(".") or "png", dpi=160)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="artifacts/crossover_sweep/crossover.csv")
    parser.add_argument("overhead", nargs="?", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument(
        "--title", default="Thread-parallelism crossover — measured Q8_0 decode kernels"
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cache-state", choices=("hot", "cold"), default="hot")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    overhead_path = Path(args.overhead) if args.overhead else None
    rows, source_format = load_measurements(csv_path, cache_state=args.cache_state)
    overhead = load_overhead(overhead_path)
    summary = summarize(rows)
    print_report(summary, overhead, source_format=source_format)

    if not args.no_plot:
        output = args.output or csv_path.parent / "crossover_plot.png"
        plot(summary, output, theme=args.theme, title=args.title)
        print(f"\nPlot saved to {output}")
    if args.summary_json:
        serializable = {
            "schema_version": 2,
            "source_format": source_format,
            "cache_state": args.cache_state if source_format == "real_model_jsonl" else None,
            "crossover_csv": str(csv_path.resolve()),
            "crossover_sha256": sha256_file(csv_path),
            "overhead_csv": str(overhead_path.resolve()) if overhead_path else None,
            "overhead_sha256": sha256_file(overhead_path) if overhead_path else None,
            "first_observed_speedup_bounds": summary["first_observed_speedup_bounds"],
            "speedups": {
                str(thread): [{"mflops": size, "speedup": value} for size, value in points]
                for thread, points in summary["speedups"].items()
            },
        }
        atomic_json(args.summary_json, serializable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
