#!/usr/bin/env python3
"""Thread-scaling benchmark harness for Ember's CPU inference path.

Runs a grid search over threads × calibrated prompt cases, collecting:
  - Fresh-process wall-clock latency
  - Per-operation breakdown (from trace JSON)
  - Hardware counters (from perf stat)
  - Scaling efficiency

Usage:
    python3 scripts/bench.py \
        --model Qwen3-0.6B-Q8_0.gguf \
        --arch qwen3 \
        --threads 1,2,4,8 \
        --prompt-lengths 1,8,32 \
        --decode-tokens 16 \
        --warmup 3 \
        --runs 10
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ── perf stat parsing ────────────────────────────────────────────────────────

PERF_METRICS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "LLC-loads",
    "LLC-load-misses",
    "branch-misses",
    "context-switches",
    "page-faults",
    "cpu-clock",
    "task-clock",
]

PERF_RE = re.compile(
    r"^\s*([\d,]+(?:\.\d+)?)\s+(" + "|".join(PERF_METRICS) + r")\b",
    re.MULTILINE,
)

PERF_TIME_RE = re.compile(
    r"^\s*([\d.]+)\s+seconds time elapsed",
    re.MULTILINE,
)


def parse_perf_stat(output: str) -> dict[str, float]:
    """Extract hardware-counter values from `perf stat` output."""
    result: dict[str, float] = {}
    for match in PERF_RE.finditer(output):
        value = float(match.group(1).replace(",", ""))
        key = match.group(2)
        result[key] = value

    # seconds elapsed
    time_match = PERF_TIME_RE.search(output)
    if time_match:
        result["seconds_elapsed"] = float(time_match.group(1))

    # Derived metrics
    if result.get("instructions") and result.get("cycles"):
        result["IPC"] = result["instructions"] / max(result["cycles"], 1)
    if result.get("instructions") and result.get("cpu-clock"):
        # perf reports cpu-clock in milliseconds under the forced C locale.
        result["instructions_per_cpu_ms"] = result["instructions"] / max(
            result["cpu-clock"], 1
        )
    if result.get("cache-misses") and result.get("cache-references"):
        miss_rate = result["cache-misses"] / max(result["cache-references"], 1)
        result["cache_miss_pct"] = miss_rate * 100
    if result.get("LLC-load-misses") and result.get("LLC-loads"):
        llc_rate = result["LLC-load-misses"] / max(result["LLC-loads"], 1)
        result["LLC_miss_pct"] = llc_rate * 100
    return result


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    before = file_identity(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if file_identity(source) != before:
        raise RuntimeError(f"file changed while hashing it: {source}")
    return digest.hexdigest()


def file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# ── trace JSON parsing ───────────────────────────────────────────────────────


def parse_trace_json(path: str) -> dict:
    """Load trace JSON artifact."""
    source = Path(path)

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {source}")

    with source.open(encoding="utf-8") as f:
        report = json.load(f, parse_constant=reject_constant)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError(f"unsupported trace envelope in {source}")
    return report


def trace_summary(
    envelope: dict, decode_tokens: int = 0, expected_threads: int | None = None
) -> dict:
    """Compute metrics from the versioned trace envelope's decode report."""
    decode = envelope.get("decode")
    if not isinstance(decode, dict) or decode.get("phase") != "decode":
        raise ValueError("trace envelope has no decode report")
    events = decode.get("events")
    total_ns = decode.get("total_duration_ns")
    if (
        not isinstance(events, list)
        or not events
        or isinstance(total_ns, bool)
        or not isinstance(total_ns, int)
        or total_ns <= 0
    ):
        raise ValueError("decode trace must contain events and a positive total_duration_ns")
    total_ms = total_ns / 1_000_000
    run_metadata = decode.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise ValueError("decode trace is missing requested run metadata")
    if expected_threads is not None and run_metadata.get("thread_count") != expected_threads:
        raise ValueError(
            "decode trace reports the wrong thread count: "
            f"{run_metadata.get('thread_count')!r} != {expected_threads}"
        )

    # Count unique token indices for decode phase
    token_indices: set[int] = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"decode trace event {index} is not an object")
        duration = event.get("duration_ns")
        token_index = event.get("token_index")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 0
            or isinstance(token_index, bool)
            or not isinstance(token_index, int)
            or token_index < 0
        ):
            raise ValueError(f"decode trace event {index} has invalid timing/index fields")
        if event.get("phase") != "decode":
            raise ValueError(f"decode trace event {index} has the wrong phase")
        token_indices.add(token_index)
    n_tokens = len(token_indices)
    if n_tokens < 1:
        raise ValueError("decode trace contains no token indices")
    ordered_indices = sorted(token_indices)
    if ordered_indices != list(range(n_tokens)):
        raise ValueError(
            f"decode trace token indices are not contiguous from zero: {ordered_indices}"
        )
    eval_s = n_tokens / (total_ns / 1_000_000_000)

    by_kind: dict[str, float] = {}
    by_name: dict[str, float] = {}
    for e in events:
        kind = e.get("op_kind")
        name = e.get("name")
        if not isinstance(kind, str) or not kind or not isinstance(name, str) or not name:
            raise ValueError("trace events require non-empty op_kind and name")
        by_kind[kind] = by_kind.get(kind, 0) + e["duration_ns"]
        by_name[name] = by_name.get(name, 0) + e["duration_ns"]

    for k in by_kind:
        by_kind[k] = by_kind[k] / total_ns * 100
    for k in by_name:
        by_name[k] = by_name[k] / total_ns * 100

    return {
        "total_ms": total_ms,
        "decode_eval_s": eval_s,
        "decode_evaluations": n_tokens,
        "requested_max_generated_tokens": decode_tokens,
        "prefill_ms": validate_prefill_trace(
            envelope.get("prefill"), expected_threads=expected_threads
        ),
        "run_metadata": run_metadata,
        "by_kind": by_kind,
        "by_name": by_name,
    }


def validate_prefill_trace(report: object, expected_threads: int | None) -> float:
    if not isinstance(report, dict) or report.get("phase") != "prefill":
        raise ValueError("trace envelope has no prefill report")
    duration = report.get("total_duration_ns")
    if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
        raise ValueError("prefill trace duration must be a positive integer")
    metadata = report.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("prefill trace is missing requested run metadata")
    if expected_threads is not None and metadata.get("thread_count") != expected_threads:
        raise ValueError("prefill trace reports the wrong thread count")
    return duration / 1_000_000


# ── run helpers ──────────────────────────────────────────────────────────────


def build_ember_cmd(
    binary: str,
    model: str,
    arch: str,
    prompt: str,
    decode_tokens: int,
    temperature: float,
    trace_values: str,
    trace_run_metadata: bool,
    tokenizer: str | None = None,
) -> list[str]:
    """Build the `ember` command-line arguments."""
    cmd = [
        binary,
        "--model",
        model,
        "--arch",
        arch,
        "--prompt",
        prompt,
        "--max-tokens",
        str(decode_tokens),
        "--temperature",
        str(temperature),
        "--trace",
        "ops",
        "--trace-values",
        trace_values,
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    if trace_run_metadata:
        cmd.append("--trace-run-metadata")
    return cmd


def run_one(
    cmd: list[str],
    threads: int,
    trace_out: str,
    use_perf: bool = False,
    timeout: float = 300.0,
) -> tuple[float, dict | None, dict | None, str]:
    """Run one invocation and return wall time, artifacts, and output digest."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    env["LC_ALL"] = "C"

    full_cmd: list[str] = []
    if use_perf:
        full_cmd = [
            "perf",
            "stat",
            "-e",
            ",".join(PERF_METRICS),
        ] + cmd
    else:
        full_cmd = cmd + ["--trace-out", trace_out]

    trace_path = Path(trace_out)
    if not use_perf:
        trace_path.unlink(missing_ok=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    wall = time.perf_counter() - t0

    trace_data = None
    perf_data = None

    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark command failed with exit code {result.returncode}: "
            f"{result.stderr[-4000:]}"
        )

    if not use_perf:
        if not trace_path.is_file():
            raise RuntimeError(f"benchmark command did not produce trace: {trace_path}")
        trace_data = parse_trace_json(trace_out)

    if use_perf:
        perf_data = parse_perf_stat(result.stderr)
        required = {"cycles", "instructions", "seconds_elapsed"}
        missing = sorted(required - perf_data.keys())
        if missing:
            raise RuntimeError(
                "perf did not report required counters "
                f"{missing}; check perf permissions and hardware support"
            )

    output_sha256 = hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()
    return wall, trace_data, perf_data, output_sha256


def run_batch(
    cmd: list[str],
    threads: int,
    warmup: int,
    runs: int,
    trace_dir: str,
    decode_tokens: int = 0,
    use_perf: bool = False,
    timeout: float = 300.0,
) -> dict:
    """Run warmup + measured iterations, return aggregated stats."""
    os.makedirs(trace_dir, exist_ok=True)

    # Warmup
    for i in range(warmup):
        trace_path = os.path.join(trace_dir, f"warmup_{i}.json")
        run_one(cmd, threads, trace_path, use_perf=False, timeout=timeout)

    # Measured runs
    latencies: list[float] = []
    eval_s_list: list[float] = []
    perf_list: list[dict] = []
    by_kind_agg: dict[str, list[float]] = {}
    output_hashes: list[str] = []

    for i in range(runs):
        trace_path = os.path.join(trace_dir, f"run_{i}.json")
        wall, trace_data, _, output_sha256 = run_one(
            cmd, threads, trace_path, use_perf=False, timeout=timeout
        )
        latencies.append(wall)
        output_hashes.append(output_sha256)

        if trace_data:
            summary = trace_summary(
                trace_data,
                decode_tokens=decode_tokens,
                expected_threads=threads,
            )
            eval_s_list.append(summary["decode_eval_s"])
            for kind, pct in summary.get("by_kind", {}).items():
                by_kind_agg.setdefault(kind, []).append(pct)

        # Perf run (separate, no trace JSON)
        if use_perf:
            _, _, perf, perf_output_sha256 = run_one(
                cmd, threads, trace_path, use_perf=True, timeout=timeout
            )
            if perf_output_sha256 != output_sha256:
                raise RuntimeError("perf and trace passes generated different output")
            if perf:
                perf_list.append(perf)

    def stats(values: list[float]) -> dict:
        if not values:
            return {}
        s = sorted(values)
        n = len(s)
        return {
            "median": statistics.median(s),
            "mean": statistics.mean(s),
            "stdev": statistics.stdev(s) if n > 1 else 0,
            "p50": statistics.median(s),
            "p95_nearest_rank": s[max(0, math.ceil(n * 0.95) - 1)],
            "min": min(s),
            "max": max(s),
        }

    result = {
        "threads": threads,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "process_wall_seconds": stats(latencies),
        "throughput_decode_eval_s": stats(eval_s_list),
        "by_kind": {k: stats(v) for k, v in by_kind_agg.items()},
        "generated_output_sha256s": sorted(set(output_hashes)),
        "deterministic_output": len(set(output_hashes)) == 1,
    }
    if not result["deterministic_output"]:
        raise RuntimeError("greedy measured runs generated different output")

    if perf_list:
        # Average perf counters across runs
        avg_perf: dict[str, float] = {}
        for key in PERF_METRICS + [
            "seconds_elapsed",
            "IPC",
            "instructions_per_cpu_ms",
            "cache_miss_pct",
            "LLC_miss_pct",
        ]:
            vals = [p[key] for p in perf_list if key in p]
            if vals:
                avg_perf[key] = statistics.mean(vals)
        result["perf"] = avg_perf

    return result


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Ember thread-scaling benchmark harness"
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--arch", default="qwen3", help="Model architecture")
    parser.add_argument(
        "--prompt",
        default=None,
        help="replace the calibrated prompt (requires one --prompt-lengths case)",
    )
    parser.add_argument("--tokenizer", help="optional external tokenizer.json")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "target/release/ember",
        help="release Ember executable",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="do not build the default release executable before benchmarking",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="per-process timeout in seconds")
    parser.add_argument(
        "--threads",
        default="1,2,4,8",
        help="Comma-separated thread counts",
    )
    parser.add_argument(
        "--prompt-lengths",
        default="1,8,32",
        help="comma-separated calibrated prompt case names: 1,8,32",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=16,
        help="Number of tokens to generate per run",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of measured runs",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Also collect hardware counters via perf stat",
    )
    parser.add_argument(
        "--trace-values",
        default="none",
        choices=["none", "summary"],
        help="Trace values collection level",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/bench_results",
        help="Output directory for trace JSON artifacts",
    )
    args = parser.parse_args()

    try:
        threads = [int(t.strip()) for t in args.threads.split(",")]
        prompt_lengths = [int(p.strip()) for p in args.prompt_lengths.split(",")]
    except ValueError as error:
        parser.error(f"thread and prompt-length lists must contain integers: {error}")
    if not threads or len(threads) != len(set(threads)) or any(value < 1 for value in threads):
        parser.error("--threads must contain unique positive integers")
    if not prompt_lengths or len(prompt_lengths) != len(set(prompt_lengths)):
        parser.error("--prompt-lengths must contain unique configured prompt cases")
    if any(value not in {1, 8, 32} for value in prompt_lengths):
        parser.error("--prompt-lengths currently supports only calibrated cases 1,8,32")
    if args.decode_tokens < 2 or args.warmup < 0 or args.runs < 1:
        parser.error(
            "decode tokens must be >= 2, runs positive, and warmup non-negative"
        )
    if not math.isfinite(args.temperature) or args.temperature != 0.0:
        parser.error("thread-scaling comparisons require deterministic --temperature 0")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    if args.arch not in {"gpt2", "llama", "qwen3", "gemma4"}:
        parser.error("unsupported --arch")
    if not Path(args.model).is_file():
        parser.error(f"model file does not exist: {args.model}")
    if args.tokenizer and not Path(args.tokenizer).is_file():
        parser.error(f"tokenizer file does not exist: {args.tokenizer}")
    if args.prompt is not None and len(prompt_lengths) != 1:
        parser.error("--prompt requires exactly one --prompt-lengths case")
    if args.prompt == "":
        parser.error("--prompt must not be empty")
    if args.perf and shutil.which("perf") is None:
        parser.error("--perf requested but the perf executable is unavailable")
    repo_root = Path(__file__).resolve().parents[1]
    if not args.skip_build:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(repo_root / "Cargo.toml")],
            check=True,
        )
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error(f"Ember executable does not exist or is not executable: {args.binary}")
    binary = str(args.binary.resolve())
    input_paths = [Path(args.model).resolve(), Path(binary)]
    if args.tokenizer:
        input_paths.append(Path(args.tokenizer).resolve())
    output = Path(args.output_dir) / "results.json"
    if output.resolve() in set(input_paths):
        parser.error("--output-dir/results.json must not overwrite an input")
    initial_identities = {str(path): file_identity(path) for path in input_paths}

    # Prompts by approximate token count (space-separated words)
    # These names are historical workload labels, not verified token counts.
    PROMPTS = {
        1: "Riyadh",
        8: "The capital of Saudi Arabia is Riyadh",
        32: "The capital of Saudi Arabia is Riyadh, a modern metropolis located in the heart of the Arabian Peninsula known for its skyscrapers",
    }

    results: list[dict] = []

    for n_threads in threads:
        for plen in prompt_lengths:
            prompt = args.prompt if args.prompt is not None else PROMPTS[plen]
            case_label = f"prompt_case_{plen}"

            run_dir = os.path.join(
                args.output_dir,
                f"t{n_threads}_p{plen}",
            )

            print(f"\n{'='*60}")
            print(f"Threads={n_threads}  prompt_case={plen}")
            print(f"{'='*60}")

            cmd = build_ember_cmd(
                binary=binary,
                model=args.model,
                arch=args.arch,
                prompt=prompt,
                decode_tokens=args.decode_tokens,
                temperature=args.temperature,
                trace_values=args.trace_values,
                trace_run_metadata=True,
                tokenizer=args.tokenizer,
            )

            batch = run_batch(
                cmd=cmd,
                threads=n_threads,
                warmup=args.warmup,
                runs=args.runs,
                trace_dir=run_dir,
                decode_tokens=args.decode_tokens,
                use_perf=args.perf,
                timeout=args.timeout,
            )
            batch["prompt_len"] = plen
            batch["prompt_case"] = case_label
            batch["prompt_text"] = prompt
            results.append(batch)

            # Make long matrices recoverable without hashing model inputs and
            # perturbing the remaining page-cache state.
            atomic_json(
                output,
                {
                    "schema_version": 3,
                    "status": "running",
                    "architecture": args.arch,
                    "results": results,
                },
            )

            # Print per-batch summary
            lat = batch["process_wall_seconds"]
            tp = batch["throughput_decode_eval_s"]
            print(
                f"  latency: {lat['median']:.1f}s median, "
                f"{lat['p95_nearest_rank']:.1f}s p95 (±{lat.get('stdev', 0):.1f}s)"
            )
            print(
                f"  throughput: {tp['median']:.2f} decode eval/s median"
            )
            if "perf" in batch:
                p = batch["perf"]
                print(
                    f"  perf: IPC={p.get('IPC', 0):.2f}  "
                    f"cache_miss={p.get('cache_miss_pct', 0):.1f}%  "
                    f"LLC_miss={p.get('LLC_miss_pct', 0):.1f}%"
                )

    # ── Scaling efficiency table ──────────────────────────────────────────────
    for path in input_paths:
        if file_identity(path) != initial_identities[str(path)]:
            raise RuntimeError(f"benchmark input changed while trials were running: {path}")
    for prompt_case in prompt_lengths:
        hashes = {
            digest
            for result in results
            if result["prompt_len"] == prompt_case
            for digest in result["generated_output_sha256s"]
        }
        if len(hashes) != 1:
            raise RuntimeError(
                f"thread counts generated different greedy output for prompt case {prompt_case}"
            )

    print("\n" + "=" * 80)
    print("SCALING EFFICIENCY")
    print("=" * 80)

    # Group by prompt length, compute relative to 1-thread baseline
    for plen in prompt_lengths:
        baseline = None
        for r in results:
            if r["prompt_len"] == plen and r["threads"] == 1:
                baseline = r
                break

        if not baseline:
            print(f"\n  prompt_case={plen}: no 1-thread baseline")
            continue

        t1_tps = baseline["throughput_decode_eval_s"]["median"]

        print(f"\n  prompt_case={plen}")
        print(
            f"  {'Threads':>8} {'Eval/s':>10} {'Speedup':>8} {'Efficiency':>11} "
            f"{'IPC':>6} {'LLCmiss%':>9} {'MatMul%':>8}"
        )

        for r in sorted(results, key=lambda x: x["threads"]):
            if r["prompt_len"] != plen:
                continue
            n = r["threads"]
            tps = r["throughput_decode_eval_s"]["median"]
            speedup = tps / t1_tps if t1_tps > 0 else 1
            efficiency = speedup / n * 100

            ipc_str = ""
            llc_str = ""
            mm_str = ""
            if "perf" in r:
                p = r["perf"]
                ipc_str = f"{p.get('IPC', 0):.2f}"
                llc_str = f"{p.get('LLC_miss_pct', 0):.1f}"
            if "by_kind" in r and "MatMulQ8_0" in r["by_kind"]:
                mm_str = f"{r['by_kind']['MatMulQ8_0']['median']:.1f}"

            print(
                f"  {n:>8} {tps:>10.2f} {speedup:>8.2f}x {efficiency:>10.1f}% "
                f"{ipc_str:>6} {llc_str:>9} {mm_str:>8}"
            )

    # ── Save aggregate results ────────────────────────────────────────────────
    agg_path = os.path.join(args.output_dir, "results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "schema_version": 3,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_sha256": sha256_path(args.model),
        "binary": binary,
        "binary_sha256": sha256_path(binary),
        "architecture": args.arch,
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": sha256_path(args.tokenizer) if args.tokenizer else None,
        "temperature": args.temperature,
        "trace_values": args.trace_values,
        "host": platform.platform(),
        "python": sys.version,
        "cache_state": "uncontrolled; fresh processes run sequentially",
        "metric_note": "decode throughput counts model decode evaluations, not emitted tokens",
        "prompt_case_note": "prompt-length values are named cases, not tokenizer-verified token counts",
        "results": results,
    }
    atomic_json(Path(agg_path), payload)
    print(f"\nResults saved to {agg_path}")


if __name__ == "__main__":
    main()
