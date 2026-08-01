#!/usr/bin/env python3
"""Run Ember's packed lifecycle and selective-residency ablations.

The binary performs all phase measurements. This driver provides fresh-process
isolation, optional CPU affinity, external wall time, deterministic parity
checks, median summaries, and generated-token break-even calculations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
from typing import Any


LIFECYCLE_MODES = [
    ("A", "control", "all"),
    ("B", "pack-before-prefill", "all"),
    ("C", "pack-after-prefill", "all"),
    ("D", "pack-before-prefill-reevict", "all"),
    ("E", "duplicate-packed", "all"),
]

SELECTIVE_MODES = [
    ("F", "pack-before-prefill-reevict", "gate-up"),
    ("G", "pack-before-prefill-reevict", "mlp"),
    ("H", "pack-before-prefill-reevict", "attention"),
    ("I", "pack-before-prefill-reevict", "attention-gate-up"),
    ("J", "pack-before-prefill-reevict", "all"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/ember")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpus", help="taskset CPU list, for example 0-3")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1729,
        help="seed used to shuffle mode order independently in each repetition",
    )
    parser.add_argument(
        "--max-start-temperature-c",
        type=float,
        help="wait between trials until the coretemp package is at or below this value",
    )
    parser.add_argument(
        "--cooldown-timeout-seconds",
        type=float,
        default=120.0,
        help="maximum wait for --max-start-temperature-c",
    )
    parser.add_argument(
        "--output-dir",
        help="default: data/benchmarks/packed-lifecycle-<UTC timestamp>",
    )
    parser.add_argument(
        "--unacceptable-peak-overhead-percent",
        type=float,
        default=25.0,
        help="stop before selective runs if delayed packing exceeds D by this percentage",
    )
    parser.add_argument(
        "--measurement-overhead-percent",
        type=float,
        default=1.0,
        help="stop if procfs hooks exceed this share of internal process time",
    )
    parser.add_argument(
        "--force-selective",
        action="store_true",
        help="run F-J even if an A-E stop condition fires",
    )
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coretemp_package_c() -> float | None:
    hwmon_root = pathlib.Path("/sys/class/hwmon")
    for hwmon in hwmon_root.glob("hwmon*"):
        try:
            if (hwmon / "name").read_text().strip() != "coretemp":
                continue
        except OSError:
            continue
        for label_path in hwmon.glob("temp*_label"):
            try:
                if label_path.read_text().strip() != "Package id 0":
                    continue
                input_path = label_path.with_name(
                    label_path.name.replace("_label", "_input")
                )
                return int(input_path.read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
    return None


def snapshot(result: dict[str, Any], phase: str) -> dict[str, Any]:
    for item in result["residency_snapshots"]:
        if item["phase"] == phase:
            return item
    raise KeyError(f"missing snapshot {phase}")


def median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def nested(result: dict[str, Any], *keys: str) -> Any:
    current: Any = result
    for key in keys:
        current = current[key]
    return current


def run_trial(
    args: argparse.Namespace,
    label: str,
    lifecycle: str,
    selection: str,
    repetition: int,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    command = [
        str(pathlib.Path(args.binary).resolve()),
        "bench-lifecycle",
        "--model",
        str(pathlib.Path(args.model).resolve()),
        "--tokenizer",
        str(pathlib.Path(args.tokenizer).resolve()),
        "--prompt",
        args.prompt,
        "--tokens",
        str(args.tokens),
        "--lifecycle",
        lifecycle,
        "--selection",
        selection,
    ]
    if args.max_seq_len is not None:
        command.extend(["--max-seq-len", str(args.max_seq_len)])
    if args.cpus:
        command = ["taskset", "-c", args.cpus, *command]

    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.threads)
    # The lifecycle constructor ignores automatic packing, but pinning the
    # rollback variable documents that no ambient setting controls a trial.
    environment["EMBER_LLAMA_PACKED_Q8"] = "0"

    cooldown_started = time.perf_counter()
    if args.max_start_temperature_c is not None:
        while True:
            current_temperature = coretemp_package_c()
            if (
                current_temperature is None
                or current_temperature <= args.max_start_temperature_c
            ):
                break
            if time.perf_counter() - cooldown_started > args.cooldown_timeout_seconds:
                raise RuntimeError(
                    "timed out waiting for package temperature to fall below "
                    f"{args.max_start_temperature_c} C (last {current_temperature} C)"
                )
            time.sleep(2.0)
    cooldown_seconds = time.perf_counter() - cooldown_started
    start_temperature_c = coretemp_package_c()
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    external_wall_ns = time.perf_counter_ns() - started_ns
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"mode {label} repetition {repetition} exited {completed.returncode}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise
    result["ablation_label"] = label
    result["repetition"] = repetition
    result["external_whole_process_ns"] = external_wall_ns
    result["start_temperature_c"] = start_temperature_c
    result["end_temperature_c"] = coretemp_package_c()
    result["pretrial_cooldown_seconds"] = cooldown_seconds
    result["reproduction_command"] = command
    result["stderr"] = completed.stderr
    raw_path = output_dir / f"{label.lower()}-r{repetition}.json"
    raw_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_mode_group(
    args: argparse.Namespace,
    modes: list[tuple[str, str, str]],
    output_dir: pathlib.Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[list[str]]]:
    trials = {label: [] for label, _, _ in modes}
    orders: list[list[str]] = []
    for repetition in range(1, args.repetitions + 1):
        ordered_modes = list(modes)
        random.Random(args.random_seed + repetition).shuffle(ordered_modes)
        orders.append([label for label, _, _ in ordered_modes])
        for label, lifecycle, selection in ordered_modes:
            trials[label].append(
                run_trial(
                    args,
                    label,
                    lifecycle,
                    selection,
                    repetition,
                    output_dir,
                )
            )
    return trials, orders


def summarize_group(label: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "packing_ns": ("timings_ns", "packing"),
        "prefill_ns": ("timings_ns", "prefill"),
        "decode_tokens_per_second": ("decode_tokens_per_second",),
        "time_to_first_token_ns": ("timings_ns", "time_to_first_token_work"),
        "predecode_work_ns": ("timings_ns", "predecode_work"),
        "external_whole_process_ns": ("external_whole_process_ns",),
        "peak_rss_kib": ("peak_rss_kib",),
        "post_prefill_rss_kib": ("post_prefill", "rss_kib"),
        "post_prefill_anonymous_pss_kib": (
            "post_prefill",
            "anonymous_pss_kib",
        ),
        "post_prefill_file_pss_kib": ("post_prefill", "file_pss_kib"),
        "minor_faults": ("faults", "minor_total"),
        "major_faults": ("faults", "major_total"),
        "measurement_overhead_fraction": ("measurement_overhead_fraction",),
        "packed_bytes": ("packed_bytes",),
        "start_temperature_c": ("start_temperature_c",),
        "end_temperature_c": ("end_temperature_c",),
    }
    summary: dict[str, Any] = {
        "label": label,
        "lifecycle": trials[0]["lifecycle"],
        "selection": trials[0]["selection"],
        "repetitions": len(trials),
        "output_hashes": sorted({trial["output_hash"] for trial in trials}),
    }
    for name, path in paths.items():
        values = [nested(trial, *path) for trial in trials]
        if any(value is None for value in values):
            summary[name] = None
        else:
            summary[name] = median(values)
    for phase_name, summary_prefix in (
        ("post_pack_eviction_complete", "post_pack"),
        ("post_prefill_reeviction_complete", "post_reeviction"),
        ("decode_complete", "post_decode"),
    ):
        for field in ("rss_kib", "anonymous_pss_kib", "file_pss_kib"):
            summary[f"{summary_prefix}_{field}"] = median(
                [snapshot(trial, phase_name)[field] for trial in trials]
            )
    return summary


def add_break_even(
    summaries: dict[str, dict[str, Any]],
    control_label: str = "A",
) -> None:
    control = summaries[control_label]
    control_decode_ns = 1_000_000_000.0 / control["decode_tokens_per_second"]
    for label, item in summaries.items():
        if label == control_label:
            item["break_even_decode_evaluations"] = 0
            item["break_even_generated_tokens"] = 1
            continue
        variant_tps = item["decode_tokens_per_second"]
        if not variant_tps:
            item["break_even_decode_evaluations"] = None
            item["break_even_generated_tokens"] = None
            continue
        saving_ns = control_decode_ns - 1_000_000_000.0 / variant_tps
        extra_ns = item["predecode_work_ns"] - control["predecode_work_ns"]
        item["incremental_predecode_ns"] = extra_ns
        item["decode_saving_ns_per_evaluation"] = saving_ns
        if saving_ns <= 0:
            item["break_even_decode_evaluations"] = None
            item["break_even_generated_tokens"] = None
        else:
            evaluations = max(0, math.ceil(extra_ns / saving_ns))
            item["break_even_decode_evaluations"] = evaluations
            # Generic prefill produces token one; decode savings begin while
            # producing token two.
            item["break_even_generated_tokens"] = evaluations + 1


def evaluate_lifecycle_stops(
    args: argparse.Namespace,
    trials: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
) -> list[str]:
    stops: list[str] = []
    d_trial = trials["D"][0]
    d_prefill_file = snapshot(d_trial, "prefill_complete")["file_pss_kib"]
    d_reevict_file = snapshot(
        d_trial, "post_prefill_reeviction_complete"
    )["file_pss_kib"]
    if d_reevict_file >= d_prefill_file:
        stops.append(
            "post-prefill re-eviction did not reduce file-backed PSS "
            f"({d_prefill_file} -> {d_reevict_file} KiB)"
        )

    delayed_peak = summaries["C"]["peak_rss_kib"]
    durable_peak = summaries["D"]["peak_rss_kib"]
    peak_limit = durable_peak * (
        1.0 + args.unacceptable_peak_overhead_percent / 100.0
    )
    if delayed_peak > peak_limit:
        stops.append(
            "delayed packing exceeded the configured peak-RSS threshold "
            f"({delayed_peak:.0f} vs {durable_peak:.0f} KiB)"
        )

    d_after_reevict_file = d_reevict_file
    d_decode_file = snapshot(d_trial, "decode_complete")["file_pss_kib"]
    permitted_refault_kib = max(4096, int(d_trial["packed_bytes"] * 0.01 / 1024))
    if d_decode_file - d_after_reevict_file > permitted_refault_kib:
        stops.append(
            "packed decode substantially re-faulted original file-backed pages "
            f"({d_after_reevict_file} -> {d_decode_file} KiB)"
        )

    overhead_limit = args.measurement_overhead_percent / 100.0
    for label, item in summaries.items():
        if item["measurement_overhead_fraction"] > overhead_limit:
            stops.append(
                f"mode {label} measurement hooks consumed "
                f"{item['measurement_overhead_fraction'] * 100:.2f}% of internal time"
            )
    return stops


def evaluate_selective_stops(
    summaries: dict[str, dict[str, Any]],
) -> list[str]:
    stops: list[str] = []
    control_tps = summaries["A"]["decode_tokens_per_second"]
    full_gain = summaries["J"]["decode_tokens_per_second"] - control_tps
    if full_gain <= 0:
        return ["all-projection packed decode did not improve whole-model throughput"]
    for label in ("F", "G", "H", "I"):
        retained = (
            summaries[label]["decode_tokens_per_second"] - control_tps
        ) / full_gain
        summaries[label]["fraction_of_full_decode_gain"] = retained
        if retained < 0.5:
            stops.append(
                f"mode {label} retained only {retained * 100:.1f}% of the full decode gain"
            )
    return stops


def markdown_report(
    summaries: dict[str, dict[str, Any]],
    parity: dict[str, bool],
    stops: list[str],
) -> str:
    lines = [
        "# Packed lifecycle benchmark",
        "",
        "| Mode | Lifecycle | Selection | Pack ms | Prefill ms | Decode tok/s | "
        "TTFT ms | Process ms | Peak MiB | Post-decode MiB | Anon PSS MiB | "
        "File PSS MiB | Start °C | minflt | majflt | Break-even generated | Parity |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in sorted(summaries):
        item = summaries[label]
        break_even = item.get("break_even_generated_tokens")
        lines.append(
            f"| {label} | {item['lifecycle']} | {item['selection']} | "
            f"{item['packing_ns'] / 1e6:.1f} | "
            f"{item['prefill_ns'] / 1e6:.1f} | "
            f"{item['decode_tokens_per_second']:.3f} | "
            f"{item['time_to_first_token_ns'] / 1e6:.1f} | "
            f"{item['external_whole_process_ns'] / 1e6:.1f} | "
            f"{item['peak_rss_kib'] / 1024:.1f} | "
            f"{item['post_decode_rss_kib'] / 1024:.1f} | "
            f"{item['post_decode_anonymous_pss_kib'] / 1024:.1f} | "
            f"{item['post_decode_file_pss_kib'] / 1024:.1f} | "
            f"{item['start_temperature_c'] if item['start_temperature_c'] is not None else 'n/a'} | "
            f"{item['minor_faults']:.0f} | {item['major_faults']:.0f} | "
            f"{break_even if break_even is not None else 'never'} | "
            f"{'yes' if parity[label] else 'NO'} |"
        )
    lines.extend(["", "## Stop-condition audit", ""])
    if stops:
        lines.extend(f"- {stop}" for stop in stops)
    else:
        lines.append("- No configured stop condition fired.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = pathlib.Path(
        args.output_dir or f"data/benchmarks/packed-lifecycle-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "created_utc": timestamp,
        "command": vars(args),
        "rustc": subprocess.run(
            ["rustc", "-Vv"], check=True, capture_output=True, text=True
        ).stdout,
        "lscpu": subprocess.run(
            ["lscpu"], check=True, capture_output=True, text=True
        ).stdout,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if args.force_selective:
        trials, orders = run_mode_group(
            args, LIFECYCLE_MODES + SELECTIVE_MODES, output_dir
        )
        summaries = {
            label: summarize_group(label, label_trials)
            for label, label_trials in trials.items()
        }
        lifecycle_stops = evaluate_lifecycle_stops(args, trials, summaries)
        stops = [*lifecycle_stops, *evaluate_selective_stops(summaries)]
    else:
        trials, lifecycle_orders = run_mode_group(
            args, LIFECYCLE_MODES, output_dir
        )
        orders = lifecycle_orders
        summaries = {
            label: summarize_group(label, label_trials)
            for label, label_trials in trials.items()
        }
        lifecycle_stops = evaluate_lifecycle_stops(args, trials, summaries)
        stops = list(lifecycle_stops)
        if not lifecycle_stops:
            selective_trials, selective_orders = run_mode_group(
                args, SELECTIVE_MODES, output_dir
            )
            orders.extend(selective_orders)
            trials.update(selective_trials)
            summaries.update(
                {
                    label: summarize_group(label, label_trials)
                    for label, label_trials in selective_trials.items()
                }
            )
            stops.extend(evaluate_selective_stops(summaries))

    add_break_even(summaries)
    control_hashes = set(summaries["A"]["output_hashes"])
    parity = {
        label: len(control_hashes) == 1
        and set(item["output_hashes"]) == control_hashes
        for label, item in summaries.items()
    }
    if not all(parity.values()):
        stops.append("deterministic generated-token parity failed")

    # Checksums run after all measured processes so reading the complete files
    # cannot warm the page cache before a lifecycle trial.
    metadata["model_sha256"] = sha256_file(pathlib.Path(args.model))
    metadata["tokenizer_sha256"] = sha256_file(pathlib.Path(args.tokenizer))
    metadata["trial_orders"] = orders
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary_document = {
        "metadata": metadata,
        "summaries": summaries,
        "deterministic_parity": parity,
        "stop_conditions": stops,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_document, indent=2) + "\n"
    )
    report = markdown_report(summaries, parity, stops)
    (output_dir / "report.md").write_text(report)
    print(report)
    print(f"Artifacts: {output_dir}")
    return 2 if stops else 0


if __name__ == "__main__":
    raise SystemExit(main())
