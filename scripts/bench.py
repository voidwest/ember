#!/usr/bin/env python3
"""Thread-scaling benchmark harness for Ember's CPU inference path.

Runs a grid search over threads × prompt lengths × phases, collecting:
  - Wall-clock latency (from trace JSON)
  - Per-operation breakdown (from trace JSON)
  - Hardware counters (from perf stat)
  - Scaling efficiency

Usage:
    python3 scripts/bench.py \
        --model Qwen3-0.6B-Q8_0.gguf \
        --arch qwen3 \
        --prompt "The capital of Saudi Arabia is" \
        --threads 1,2,4,8 \
        --prompt-lengths 1,8,32 \
        --decode-tokens 16 \
        --warmup 3 \
        --runs 10
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
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
        result["IPC_clock"] = result["instructions"] / max(
            result["cpu-clock"], 1
        )
    if result.get("cache-misses") and result.get("cache-references"):
        miss_rate = result["cache-misses"] / max(result["cache-references"], 1)
        result["cache_miss_pct"] = miss_rate * 100
    if result.get("LLC-load-misses") and result.get("LLC-loads"):
        llc_rate = result["LLC-load-misses"] / max(result["LLC-loads"], 1)
        result["LLC_miss_pct"] = llc_rate * 100
    return result


# ── trace JSON parsing ───────────────────────────────────────────────────────


def parse_trace_json(path: str) -> dict:
    """Load trace JSON artifact."""
    with open(path) as f:
        events = json.load(f)
    return events


def trace_summary(events: list[dict], decode_tokens: int = 0) -> dict:
    """Compute aggregate metrics from trace events."""
    if not events:
        return {"total_ms": 0, "tok_s": 0, "by_kind": {}, "by_name": {}}

    total_ns = sum(e["duration_ns"] for e in events)
    total_ms = total_ns / 1_000_000

    # Count unique token indices for decode phase
    token_indices: set[int] = set()
    for e in events:
        ti = e.get("token_index", 0)
        if ti > 0 or e.get("phase") == "decode":
            token_indices.add(ti)
    n_tokens = len(token_indices) if token_indices else (decode_tokens or 1)
    tok_s = n_tokens / (total_ns / 1_000_000_000) if total_ns > 0 else 0

    by_kind: dict[str, float] = {}
    by_name: dict[str, float] = {}
    for e in events:
        kind = e["op_kind"]
        name = e["name"]
        by_kind[kind] = by_kind.get(kind, 0) + e["duration_ns"]
        by_name[name] = by_name.get(name, 0) + e["duration_ns"]

    for k in by_kind:
        by_kind[k] = by_kind[k] / total_ns * 100
    for k in by_name:
        by_name[k] = by_name[k] / total_ns * 100

    return {
        "total_ms": total_ms,
        "tok_s": tok_s,
        "by_kind": by_kind,
        "by_name": by_name,
    }


# ── run helpers ──────────────────────────────────────────────────────────────


def build_cargo_cmd(
    model: str,
    arch: str,
    prompt: str,
    decode_tokens: int,
    temperature: float,
    trace_values: str,
    trace_run_metadata: bool,
) -> list[str]:
    """Build the `ember` command-line arguments."""
    cmd = [
        "cargo",
        "run",
        "--release",
        "--",
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
    if trace_run_metadata:
        cmd.append("--trace-run-metadata")
    return cmd


def run_one(
    cmd: list[str],
    threads: int,
    trace_out: str,
    use_perf: bool = False,
) -> tuple[float, dict | None, dict | None]:
    """Run a single ember invocation. Returns (wall_seconds, trace, perf)."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)

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

    t0 = time.perf_counter()
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    wall = time.perf_counter() - t0

    trace_data = None
    perf_data = None

    if not use_perf and os.path.exists(trace_out):
        try:
            trace_data = parse_trace_json(trace_out)
        except (json.JSONDecodeError, OSError):
            pass

    if use_perf:
        perf_data = parse_perf_stat(result.stderr)

    # Fallback: try to parse trace from the JSON file even with perf
    if trace_data is None and os.path.exists(trace_out):
        try:
            trace_data = parse_trace_json(trace_out)
        except (json.JSONDecodeError, OSError):
            pass

    return wall, trace_data, perf_data


def run_batch(
    cmd: list[str],
    threads: int,
    warmup: int,
    runs: int,
    trace_dir: str,
    decode_tokens: int = 0,
    use_perf: bool = False,
) -> dict:
    """Run warmup + measured iterations, return aggregated stats."""
    os.makedirs(trace_dir, exist_ok=True)

    # Warmup
    for i in range(warmup):
        trace_path = os.path.join(trace_dir, f"warmup_{i}.json")
        run_one(cmd, threads, trace_path, use_perf=False)

    # Measured runs
    latencies: list[float] = []
    tok_s_list: list[float] = []
    perf_list: list[dict] = []
    by_kind_agg: dict[str, list[float]] = {}

    for i in range(runs):
        trace_path = os.path.join(trace_dir, f"run_{i}.json")
        wall, trace_data, perf = run_one(cmd, threads, trace_path, use_perf=False)
        latencies.append(wall)

        if trace_data:
            # Decode phase: events are in the report which is a list of OpTrace
            summary = trace_summary(
                trace_data if isinstance(trace_data, list) else [],
                decode_tokens=decode_tokens,
            )
            tok_s_list.append(summary["tok_s"])
            for kind, pct in summary.get("by_kind", {}).items():
                by_kind_agg.setdefault(kind, []).append(pct)

        # Perf run (separate, no trace JSON)
        if use_perf:
            _, _, perf = run_one(cmd, threads, trace_path, use_perf=True)
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
            "p50": s[n // 2],
            "p95": s[int(n * 0.95)] if n > 1 else s[0],
            "min": min(s),
            "max": max(s),
        }

    result = {
        "threads": threads,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "latency": stats(latencies),
        "throughput_tok_s": stats(tok_s_list),
        "by_kind": {k: stats(v) for k, v in by_kind_agg.items()},
    }

    if perf_list:
        # Average perf counters across runs
        avg_perf: dict[str, float] = {}
        for key in PERF_METRICS + ["IPC", "IPC_clock", "cache_miss_pct", "LLC_miss_pct"]:
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
        default="The capital of Saudi Arabia is",
        help="Prompt text",
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,8",
        help="Comma-separated thread counts",
    )
    parser.add_argument(
        "--prompt-lengths",
        default="1,8,32",
        help="Comma-separated prompt lengths (token count)",
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

    threads = [int(t.strip()) for t in args.threads.split(",")]
    prompt_lengths = [int(p.strip()) for p in args.prompt_lengths.split(",")]

    # Prompts by approximate token count (space-separated words)
    # These are calibrated for the default GPT-2/Llama tokenizer
    PROMPTS = {
        1: "Riyadh",
        8: "The capital of Saudi Arabia is Riyadh",
        32: "The capital of Saudi Arabia is Riyadh, a modern metropolis located in the heart of the Arabian Peninsula known for its skyscrapers",
    }

    results: list[dict] = []

    for n_threads in threads:
        for plen in prompt_lengths:
            prompt = PROMPTS.get(plen, args.prompt)
            phase_label = "prefill" if plen > 1 else "decode"

            run_dir = os.path.join(
                args.output_dir,
                f"t{n_threads}_p{plen}",
            )

            print(f"\n{'='*60}")
            print(f"Threads={n_threads}  prompt_len={plen}  phase={phase_label}")
            print(f"{'='*60}")

            cmd = build_cargo_cmd(
                model=args.model,
                arch=args.arch,
                prompt=prompt,
                decode_tokens=args.decode_tokens,
                temperature=args.temperature,
                trace_values=args.trace_values,
                trace_run_metadata=True,
            )

            batch = run_batch(
                cmd=cmd,
                threads=n_threads,
                warmup=args.warmup,
                runs=args.runs,
                trace_dir=run_dir,
                decode_tokens=args.decode_tokens,
                use_perf=args.perf,
            )
            batch["prompt_len"] = plen
            batch["phase"] = phase_label
            results.append(batch)

            # Print per-batch summary
            lat = batch["latency"]
            tp = batch["throughput_tok_s"]
            print(
                f"  latency: {lat['median']:.1f}s median, "
                f"{lat['p95']:.1f}s p95 (±{lat.get('stdev', 0):.1f}s)"
            )
            print(
                f"  throughput: {tp['median']:.2f} tok/s median"
            )
            if "perf" in batch:
                p = batch["perf"]
                print(
                    f"  perf: IPC={p.get('IPC', 0):.2f}  "
                    f"cache_miss={p.get('cache_miss_pct', 0):.1f}%  "
                    f"LLC_miss={p.get('LLC_miss_pct', 0):.1f}%"
                )

    # ── Scaling efficiency table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SCALING EFFICIENCY")
    print("=" * 80)

    # Group by prompt length, compute relative to 1-thread baseline
    for plen in prompt_lengths:
        phase = "prefill" if plen > 1 else "decode"
        baseline = None
        for r in results:
            if r["prompt_len"] == plen and r["threads"] == 1:
                baseline = r
                break

        if not baseline:
            print(f"\n  prompt_len={plen} ({phase}): no 1-thread baseline")
            continue

        t1_lat = baseline["latency"]["median"]
        t1_tps = baseline["throughput_tok_s"]["median"]

        print(f"\n  prompt_len={plen} ({phase})")
        print(
            f"  {'Threads':>8} {'Tok/s':>10} {'Speedup':>8} {'Efficiency':>11} "
            f"{'IPC':>6} {'LLCmiss%':>9} {'MatMul%':>8}"
        )

        for r in sorted(results, key=lambda x: x["threads"]):
            if r["prompt_len"] != plen:
                continue
            n = r["threads"]
            tps = r["throughput_tok_s"]["median"]
            speedup = tps / t1_tps if t1_tps > 0 else 1
            efficiency = speedup / n * 100

            perf_str = ""
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
    with open(agg_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {agg_path}")


if __name__ == "__main__":
    main()
