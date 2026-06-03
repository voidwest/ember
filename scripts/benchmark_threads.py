#!/usr/bin/env python3
"""Benchmark Ember throughput across Rayon thread counts.

This is intentionally a thin wrapper around the release binary. It measures the
existing `--benchmark` output for one or more local GGUFs while varying
`RAYON_NUM_THREADS`, then writes a JSON report.
"""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


BENCH_RE = re.compile(
    r"^(prefill|decode):\s+(\d+)\s+tokens in ([0-9.]+)ms -> ([0-9.]+) tok/s$"
)


def parse_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    label, path = value.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    return label, path


def parse_threads(value: str) -> list[int]:
    threads = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not threads or any(t < 1 for t in threads):
        raise argparse.ArgumentTypeError("threads must be comma-separated positive integers")
    return threads


def parse_benchmark(stderr: str) -> dict:
    parsed = {}
    for line in stderr.splitlines():
        match = BENCH_RE.match(line.strip())
        if not match:
            continue
        phase, tokens, ms, tok_s = match.groups()
        parsed[phase] = {
            "tokens": int(tokens),
            "ms": float(ms),
            "tok_s": float(tok_s),
        }
    return parsed


def run_once(args, label: str, model_path: str, threads: int, repeat: int) -> dict:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
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
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "model": model_path,
        "threads": threads,
        "repeat": repeat,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "benchmark": parse_benchmark(proc.stderr),
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
    parser.add_argument("--binary", type=Path, default=Path("target/release/ember"))
    parser.add_argument("--output", type=Path, default=Path("data/thread_benchmarks.json"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not args.skip_build:
        subprocess.run(["cargo", "build", "--release"], check=True)

    results = []
    for label, model_path in args.model:
        for threads in args.threads:
            for repeat in range(args.repeats):
                result = run_once(args, label, model_path, threads, repeat)
                results.append(result)
                bench = result.get("benchmark", {})
                prefill = bench.get("prefill", {}).get("tok_s")
                decode = bench.get("decode", {}).get("tok_s")
                print(
                    f"{label:>12} threads={threads:<2} repeat={repeat:<2} "
                    f"prefill={prefill!s:>8} tok/s decode={decode!s:>8} tok/s "
                    f"rc={result['returncode']}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arch": args.arch,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "tokens": args.tokens,
        "max_seq_len": args.max_seq_len,
        "threads": args.threads,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
