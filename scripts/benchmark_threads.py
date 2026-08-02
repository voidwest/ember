#!/usr/bin/env python3
"""Benchmark Ember throughput across Rayon thread counts.

This is intentionally a thin wrapper around the release binary. It measures the
existing `--benchmark` output for one or more local GGUFs while varying
`RAYON_NUM_THREADS`, then writes a JSON report.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


BENCH_RE = re.compile(
    r"^(prefill|decode):\s+(\d+)\s+(tokens|evals) in "
    r"([0-9.]+)ms -> ([0-9.]+) (tok|eval)/s$"
)


def _sha256(path: Path) -> str:
    before = _identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if _identity(path) != before:
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest()


def parse_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    label, path = value.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    return label, path


def parse_threads(value: str) -> list[int]:
    try:
        threads = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("threads must contain integers") from error
    if not threads or any(t < 1 for t in threads) or len(threads) != len(set(threads)):
        raise argparse.ArgumentTypeError("threads must be comma-separated positive integers")
    return threads


def parse_benchmark(stderr: str) -> dict:
    parsed = {}
    for line in stderr.splitlines():
        match = BENCH_RE.match(line.strip())
        if not match:
            continue
        phase, count_text, count_unit, ms_text, rate_text, rate_unit = match.groups()
        if phase in parsed:
            raise ValueError(f"benchmark output contained duplicate {phase} measurements")
        expected_units = {
            "prefill": ("tokens", "tok"),
            "decode": ("evals", "eval"),
        }
        if (count_unit, rate_unit) != expected_units[phase]:
            raise ValueError(f"benchmark output used inconsistent units for {phase}")
        count = int(count_text)
        ms = float(ms_text)
        rate = float(rate_text)
        parsed[phase] = {
            "count": count,
            "unit": "tokens" if phase == "prefill" else "decode_evaluations",
            "ms": ms,
            "rate_per_second": rate,
        }
        # Check that the two rounded values could come from the same duration.
        # The Rust CLI currently prints three decimals, while this also accepts
        # older, more coarsely rounded artifacts without inventing precision.
        ms_decimals = len(ms_text.partition(".")[2]) if "." in ms_text else 0
        rate_decimals = len(rate_text.partition(".")[2]) if "." in rate_text else 0
        ms_error = 0.5 * (10.0 ** -ms_decimals)
        rate_error = 0.5 * (10.0 ** -rate_decimals)
        duration_low_ms = max(ms - ms_error, float.fromhex("0x1p-1022"))
        duration_high_ms = ms + ms_error
        compatible_rate_low = count * 1000.0 / duration_high_ms
        compatible_rate_high = count * 1000.0 / duration_low_ms
        if rate + rate_error < compatible_rate_low or rate - rate_error > compatible_rate_high:
            raise ValueError(
                f"{phase} rate {rate_text}/s is inconsistent with "
                f"{count_text} work items in {ms_text}ms"
            )
    if set(parsed) != {"prefill", "decode"}:
        raise ValueError("benchmark output did not contain both prefill and decode measurements")
    for phase, values in parsed.items():
        if (
            values["count"] < 1
            or not math.isfinite(values["ms"])
            or values["ms"] <= 0.0
            or not math.isfinite(values["rate_per_second"])
            or values["rate_per_second"] <= 0.0
        ):
            raise ValueError(f"invalid {phase} benchmark metrics: {values}")
    return parsed


def _identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _summaries(results: list[dict]) -> list[dict]:
    summaries = []
    keys = sorted({(item["label"], item["threads"]) for item in results})
    for label, threads in keys:
        group = [
            item
            for item in results
            if item["label"] == label and item["threads"] == threads
        ]
        phase_summary = {}
        for phase in ("prefill", "decode"):
            rates = [item["benchmark"][phase]["rate_per_second"] for item in group]
            milliseconds = [item["benchmark"][phase]["ms"] for item in group]
            phase_summary[phase] = {
                "unit": group[0]["benchmark"][phase]["unit"],
                "median_rate_per_second": statistics.median(rates),
                "mean_rate_per_second": statistics.mean(rates),
                "sample_stdev_rate_per_second": (
                    statistics.stdev(rates) if len(rates) > 1 else 0.0
                ),
                "median_ms": statistics.median(milliseconds),
            }
        summaries.append(
            {
                "label": label,
                "threads": threads,
                "repetitions": len(group),
                "process_wall_median_s": statistics.median(
                    item["elapsed_s"] for item in group
                ),
                "phases": phase_summary,
            }
        )
    return summaries


def _atomic_json(path: Path, payload: dict) -> None:
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


def run_once(args, label: str, model_path: str, threads: int, repeat: int) -> dict:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    env["LC_ALL"] = "C"
    cmd = [
        str(args.binary),
        "--arch",
        args.arch,
        "--model",
        model_path,
        "--prompt",
        args.prompt,
        "-n",
        str(args.tokens),
        "--temperature",
        "0",
        "--benchmark",
    ]
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    if args.max_seq_len:
        cmd.extend(["--max-seq-len", str(args.max_seq_len)])

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchmark failed for {label}, threads={threads}, repeat={repeat}: "
            f"{proc.stderr[-4000:]}"
        )
    benchmark = parse_benchmark(proc.stderr)
    return {
        "label": label,
        "model": model_path,
        "threads": threads,
        "repeat": repeat,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "benchmark": benchmark,
        "command": cmd,
        "generated_stdout_sha256": hashlib.sha256(
            proc.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr_tail": proc.stderr.splitlines()[-20:],
    }


def main():
    parser = argparse.ArgumentParser(description="benchmark Ember with different Rayon thread counts")
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        metavar="LABEL:PATH",
        help="model label and GGUF path; may be repeated",
    )
    parser.add_argument("--arch", default="qwen3", choices=["gpt2", "llama", "qwen3", "gemma4"])
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--threads", type=parse_threads, default=parse_threads("1,2,4,8"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "target/release/ember",
    )
    parser.add_argument("--output", type=Path, default=Path("data/thread_benchmarks.json"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.tokens < 2:
        parser.error("--tokens must be >= 2 to measure decode throughput")
    if args.max_seq_len is not None and args.max_seq_len < 1:
        parser.error("--max-seq-len must be >= 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    if not args.prompt:
        parser.error("--prompt must not be empty")
    labels = [label for label, _ in args.model]
    if len(labels) != len(set(labels)):
        parser.error("model labels must be unique")
    for label, model_path in args.model:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
            parser.error(f"unsafe model label: {label!r}")
        if not Path(model_path).is_file():
            parser.error(f"model file does not exist: {model_path}")
    if args.tokenizer and not Path(args.tokenizer).is_file():
        parser.error(f"tokenizer file does not exist: {args.tokenizer}")
    if not args.skip_build:
        repo_root = Path(__file__).resolve().parents[1]
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(repo_root / "Cargo.toml"),
            ],
            check=True,
        )
    if not args.binary.is_file():
        parser.error(f"release binary does not exist: {args.binary}")
    if not os.access(args.binary, os.X_OK):
        parser.error(f"release binary is not executable: {args.binary}")

    resolved_inputs = [args.binary.resolve()]
    resolved_inputs.extend(Path(path).resolve() for _, path in args.model)
    if args.tokenizer:
        resolved_inputs.append(Path(args.tokenizer).resolve())
    if args.output.resolve() in set(resolved_inputs):
        parser.error("--output must not overwrite a benchmark input")
    initial_identities = {str(path): _identity(path) for path in resolved_inputs}

    results = []
    for label, model_path in args.model:
        for threads in args.threads:
            for repeat in range(args.repeats):
                result = run_once(args, label, model_path, threads, repeat)
                results.append(result)
                _atomic_json(
                    args.output,
                    {
                        "schema_version": 3,
                        "status": "running",
                        "arch": args.arch,
                        "results": results,
                    },
                )
                bench = result.get("benchmark", {})
                prefill = bench.get("prefill", {}).get("rate_per_second")
                decode = bench.get("decode", {}).get("rate_per_second")
                print(
                    f"{label:>12} threads={threads:<2} repeat={repeat:<2} "
                    f"prefill={prefill!s:>8} tok/s decode={decode!s:>8} eval/s "
                    f"rc={result['returncode']}"
                )

    for path in resolved_inputs:
        if _identity(path) != initial_identities[str(path)]:
            raise RuntimeError(f"benchmark input changed while trials were running: {path}")
    for label in labels:
        output_hashes = {
            result["generated_stdout_sha256"]
            for result in results
            if result["label"] == label
        }
        if len(output_hashes) != 1:
            raise RuntimeError(
                f"greedy output for model {label!r} changed across repetitions/threads"
            )

    payload = {
        "schema_version": 3,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arch": args.arch,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "tokens": args.tokens,
        "max_seq_len": args.max_seq_len,
        "threads": args.threads,
        "repeats": args.repeats,
        "binary_sha256": _sha256(args.binary),
        "binary": str(args.binary.resolve()),
        "tokenizer_sha256": _sha256(Path(args.tokenizer)) if args.tokenizer else None,
        "models": [
            {"label": label, "path": path, "sha256": _sha256(Path(path))}
            for label, path in args.model
        ],
        "metric_note": (
            "decode rate is whole decode-loop time per model evaluation; "
            "it is not emitted-token throughput"
        ),
        "summaries": _summaries(results),
        "results": results,
    }
    _atomic_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
