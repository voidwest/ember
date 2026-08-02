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
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
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
    parser.add_argument("--trial-timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2 to measure decode evaluations")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if not args.prompt:
        parser.error("--prompt must not be empty")
    if args.max_seq_len is not None and args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")
    for name in (
        "cooldown_timeout_seconds",
        "unacceptable_peak_overhead_percent",
        "measurement_overhead_percent",
        "trial_timeout_seconds",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.trial_timeout_seconds <= 0.0:
        parser.error("--trial-timeout-seconds must be positive")
    if args.max_start_temperature_c is not None and not math.isfinite(
        args.max_start_temperature_c
    ):
        parser.error("--max-start-temperature-c must be finite")
    if (
        args.max_start_temperature_c is not None
        and args.max_start_temperature_c < -273.15
    ):
        parser.error("--max-start-temperature-c cannot be below absolute zero")
    if args.cpus:
        if not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", args.cpus):
            parser.error("--cpus must be a comma-separated CPU/range list such as 0-3,8")
        if shutil.which("taskset") is None:
            parser.error("--cpus requires the taskset executable")
    for name in ("binary", "model", "tokenizer"):
        if not pathlib.Path(getattr(args, name)).is_file():
            parser.error(f"--{name} file does not exist: {getattr(args, name)}")
    if not os.access(args.binary, os.X_OK):
        parser.error(f"--binary is not executable: {args.binary}")
    return args


def sha256_file(path: pathlib.Path) -> str:
    before = file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if file_identity(path) != before:
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest()


def file_identity(path: pathlib.Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_trial_result(
    result: Any,
    lifecycle: str,
    selection: str,
    *,
    expected_tokens: int | None = None,
    expected_threads: int | None = None,
    expected_model: str | None = None,
    expected_tokenizer: str | None = None,
    expected_prompt: str | None = None,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("lifecycle benchmark output must be a JSON object")
    if result.get("schema_version") != 2 or result.get("benchmark") != "packed_lifecycle":
        raise ValueError("unsupported lifecycle benchmark schema")
    expected_lifecycle = lifecycle.replace("-", "_")
    expected_selection = selection.replace("-", "_")
    if (
        result.get("lifecycle") != expected_lifecycle
        or result.get("selection") != expected_selection
    ):
        raise ValueError("lifecycle benchmark echoed the wrong mode configuration")
    required_paths = [
        ("timings_ns", "packing"),
        ("timings_ns", "prefill"),
        ("timings_ns", "decode"),
        ("timings_ns", "time_to_first_token_work"),
        ("timings_ns", "predecode_work"),
        ("timings_ns", "whole_process_until_exit_snapshot"),
        ("decode_evaluations_per_second",),
        ("decode_evaluations",),
        ("peak_rss_kib",),
        ("faults", "minor_total"),
        ("faults", "major_total"),
        ("measurement_overhead_fraction",),
        ("packed_bytes",),
        ("output_hash",),
    ]
    integer_paths = {
        path
        for path in required_paths
        if path[0] == "timings_ns"
        or path
        in {
            ("decode_evaluations",),
            ("peak_rss_kib",),
            ("faults", "minor_total"),
            ("faults", "major_total"),
            ("packed_bytes",),
        }
    }
    for path in required_paths:
        try:
            value = nested(result, *path)
        except (KeyError, TypeError) as error:
            raise ValueError(f"lifecycle result is missing {'.'.join(path)}") from error
        if path == ("output_hash",):
            if not isinstance(value, str) or not re.fullmatch(r"fnv1a64:[0-9a-f]{16}", value):
                raise ValueError("lifecycle output_hash must be a canonical FNV-1a digest")
        elif path in integer_paths and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"lifecycle integer metric {'.'.join(path)} is invalid: {value!r}")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"lifecycle metric {'.'.join(path)} is invalid: {value!r}")
    if result["decode_evaluations_per_second"] <= 0.0:
        raise ValueError("decode_evaluations_per_second must be positive")
    if result["measurement_overhead_fraction"] > 1.0:
        raise ValueError("measurement_overhead_fraction cannot exceed 1")
    if expected_tokens is not None:
        if result.get("requested_generated_tokens") != expected_tokens:
            raise ValueError("lifecycle benchmark echoed the wrong generated-token count")
        if result.get("decode_evaluations") != expected_tokens - 1:
            raise ValueError("lifecycle decode-evaluation count is inconsistent")
        generated = result.get("generated_tokens")
        if not isinstance(generated, list) or len(generated) != expected_tokens:
            raise ValueError("lifecycle generated-token payload has the wrong length")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in generated):
            raise ValueError("lifecycle generated-token payload is invalid")
    if expected_threads is not None and result.get("threads") != expected_threads:
        raise ValueError("lifecycle benchmark used the wrong Rayon thread count")
    for name, expected in (
        ("model", expected_model),
        ("tokenizer", expected_tokenizer),
        ("prompt", expected_prompt),
    ):
        if expected is not None and result.get(name) != expected:
            raise ValueError(f"lifecycle benchmark echoed the wrong {name}")
    prompt_tokens = result.get("prompt_tokens")
    if (
        not isinstance(prompt_tokens, list)
        or len(prompt_tokens) < 2
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in prompt_tokens
        )
    ):
        raise ValueError("lifecycle prompt-token audit is invalid")
    if result.get("residency_measurement_enabled") is not True:
        raise ValueError("lifecycle benchmark did not enable residency measurement")

    decode_ns = nested(result, "timings_ns", "decode")
    if decode_ns <= 0:
        raise ValueError("decode timing must be positive")
    expected_rate = result["decode_evaluations"] * 1_000_000_000.0 / decode_ns
    if not math.isclose(
        result["decode_evaluations_per_second"],
        expected_rate,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("decode evaluation rate is inconsistent with its timing")

    snapshots = result.get("residency_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("residency_snapshots must be a non-empty array")
    phases = set()
    prior_elapsed_ns = -1
    for index, item in enumerate(snapshots):
        if not isinstance(item, dict) or not isinstance(item.get("phase"), str):
            raise ValueError(f"invalid residency snapshot {index}")
        if item["phase"] in phases:
            raise ValueError(f"duplicate residency snapshot phase {item['phase']!r}")
        phases.add(item["phase"])
        elapsed_ns = item.get("elapsed_ns")
        if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, int) or elapsed_ns < prior_elapsed_ns:
            raise ValueError(f"snapshot {item['phase']!r} has invalid/non-monotonic elapsed_ns")
        prior_elapsed_ns = elapsed_ns
        for field in ("rss_kib", "anonymous_pss_kib", "file_pss_kib"):
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"snapshot {item['phase']!r} has invalid {field}")
    for phase in (
        "prefill_complete",
        "post_pack_eviction_complete",
        "post_prefill_reeviction_complete",
        "decode_complete",
    ):
        if phase not in phases:
            raise ValueError(f"lifecycle result is missing snapshot {phase!r}")
    post_prefill = result.get("post_prefill")
    expected_post_prefill = snapshot(result, "prefill_complete")
    if not isinstance(post_prefill, dict) or any(
        post_prefill.get(field) != expected_post_prefill.get(field)
        for field in ("rss_kib", "anonymous_pss_kib", "file_pss_kib")
    ):
        raise ValueError("post_prefill summary does not match its residency snapshot")
    return result


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
    environment["LC_ALL"] = "C"
    # The lifecycle constructor ignores automatic packing, but pinning the
    # rollback variable documents that no ambient setting controls a trial.
    environment["EMBER_LLAMA_PACKED_Q8"] = "0"

    cooldown_started = time.perf_counter()
    if args.max_start_temperature_c is not None:
        while True:
            current_temperature = coretemp_package_c()
            if current_temperature is None:
                raise RuntimeError(
                    "--max-start-temperature-c was requested but coretemp package data is unavailable"
                )
            if current_temperature <= args.max_start_temperature_c:
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
        timeout=args.trial_timeout_seconds,
    )
    external_wall_ns = time.perf_counter_ns() - started_ns
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"mode {label} repetition {repetition} exited {completed.returncode}"
        )
    try:
        result = json.loads(
            completed.stdout,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value!r} in benchmark output")
            ),
        )
    except (json.JSONDecodeError, ValueError):
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise
    result = validate_trial_result(
        result,
        lifecycle,
        selection,
        expected_tokens=args.tokens,
        expected_threads=args.threads,
        expected_model=str(pathlib.Path(args.model).resolve()),
        expected_tokenizer=str(pathlib.Path(args.tokenizer).resolve()),
        expected_prompt=args.prompt,
    )
    internal_wall_ns = nested(
        result, "timings_ns", "whole_process_until_exit_snapshot"
    )
    if external_wall_ns < internal_wall_ns:
        raise ValueError(
            "external process wall time is shorter than the internal measurement"
        )
    result["ablation_label"] = label
    result["repetition"] = repetition
    result["external_whole_process_ns"] = external_wall_ns
    result["start_temperature_c"] = start_temperature_c
    result["end_temperature_c"] = coretemp_package_c()
    result["pretrial_cooldown_seconds"] = cooldown_seconds
    result["reproduction_command"] = command
    result["stderr"] = completed.stderr
    raw_path = output_dir / f"{label.lower()}-r{repetition}.json"
    atomic_write(raw_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
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
        atomic_write(
            output_dir / "progress.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "running",
                    "mode_group": [label for label, _, _ in modes],
                    "trial_orders": orders,
                    "completed": {
                        trial_label: len(label_trials)
                        for trial_label, label_trials in trials.items()
                    },
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
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
            atomic_write(
                output_dir / "progress.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "running",
                        "mode_group": [item[0] for item in modes],
                        "trial_orders": orders,
                        "completed": {
                            trial_label: len(label_trials)
                            for trial_label, label_trials in trials.items()
                        },
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
            )
    return trials, orders


def summarize_group(label: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "packing_ns": ("timings_ns", "packing"),
        "prefill_ns": ("timings_ns", "prefill"),
        "decode_evaluations_per_second": ("decode_evaluations_per_second",),
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
    control_decode_ns = 1_000_000_000.0 / control[
        "decode_evaluations_per_second"
    ]
    for label, item in summaries.items():
        if label == control_label:
            item["break_even_decode_evaluations"] = 0
            item["break_even_generated_tokens"] = 1
            continue
        variant_tps = item["decode_evaluations_per_second"]
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
    summaries: dict[str, dict[str, Any]],
) -> list[str]:
    stops: list[str] = []
    d_prefill_file = summaries["D"]["post_prefill_file_pss_kib"]
    d_reevict_file = summaries["D"]["post_reeviction_file_pss_kib"]
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
    d_decode_file = summaries["D"]["post_decode_file_pss_kib"]
    permitted_refault_kib = max(
        4096, int(summaries["D"]["packed_bytes"] * 0.01 / 1024)
    )
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
    control_tps = summaries["A"]["decode_evaluations_per_second"]
    full_gain = summaries["J"]["decode_evaluations_per_second"] - control_tps
    if full_gain <= 0:
        return ["all-projection packed decode did not improve whole-model throughput"]
    for label in ("F", "G", "H", "I"):
        retained = (
            summaries[label]["decode_evaluations_per_second"] - control_tps
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
        "| Mode | Lifecycle | Selection | Pack ms | Prefill ms | Decode eval/s | "
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
            f"{item['decode_evaluations_per_second']:.3f} | "
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
    input_paths = [
        pathlib.Path(args.binary).resolve(),
        pathlib.Path(args.model).resolve(),
        pathlib.Path(args.tokenizer).resolve(),
    ]
    if output_dir.resolve() in set(input_paths):
        raise ValueError("output directory must not replace an input file")
    initial_identities = {
        str(path): file_identity(path)
        for path in input_paths
    }
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": timestamp,
        "command": vars(args),
        "rustc": subprocess.run(
            ["rustc", "-Vv"], check=True, capture_output=True, text=True
        ).stdout,
        "lscpu": subprocess.run(
            ["lscpu"], check=True, capture_output=True, text=True
        ).stdout,
    }
    atomic_write(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
    )

    if args.force_selective:
        trials, orders = run_mode_group(
            args, LIFECYCLE_MODES + SELECTIVE_MODES, output_dir
        )
        summaries = {
            label: summarize_group(label, label_trials)
            for label, label_trials in trials.items()
        }
        lifecycle_stops = evaluate_lifecycle_stops(args, summaries)
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
        lifecycle_stops = evaluate_lifecycle_stops(args, summaries)
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
    for path in input_paths:
        if file_identity(path) != initial_identities[str(path)]:
            raise RuntimeError(f"benchmark input changed while trials were running: {path}")
    metadata["binary_sha256"] = sha256_file(pathlib.Path(args.binary))
    metadata["model_sha256"] = sha256_file(pathlib.Path(args.model))
    metadata["tokenizer_sha256"] = sha256_file(pathlib.Path(args.tokenizer))
    metadata["trial_orders"] = orders
    atomic_write(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
    )
    summary_document = {
        "schema_version": 2,
        "metadata": metadata,
        "summaries": summaries,
        "deterministic_parity": parity,
        "stop_conditions": stops,
    }
    atomic_write(
        output_dir / "summary.json",
        json.dumps(summary_document, indent=2, allow_nan=False) + "\n",
    )
    report = markdown_report(summaries, parity, stops)
    atomic_write(output_dir / "report.md", report)
    atomic_write(
        output_dir / "progress.json",
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "trial_orders": orders,
                "completed": {
                    label: len(label_trials) for label, label_trials in trials.items()
                },
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    print(report)
    print(f"Artifacts: {output_dir}")
    return 2 if stops else 0


if __name__ == "__main__":
    raise SystemExit(main())
