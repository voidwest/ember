#!/usr/bin/env python3
"""Run local Ember smoke tests and write auditable logs.

Smoke status is structural: command exit, basic timing parse, and optional
degenerate-output warning. Raw generation text is not a quality benchmark.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from benchmark_threads import parse_benchmark
except ModuleNotFoundError:  # imported as scripts.run_smoke in tests/tools
    from scripts.benchmark_threads import parse_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "scripts" / "smoke_models.json"

PROMPT_PRESETS = {
    "raw_france": "The capital of France is",
    "llama3_chat_france": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "qwen_chat_france": (
        "<|im_start|>user\nThe capital of France is<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="run Ember GGUF smoke tests")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--model", help="configured model label to run")
    selector.add_argument("--all", action="store_true", help="run all configured available models")
    parser.add_argument("--tokens", type=int, default=32, help="generated token count")
    parser.add_argument("--prompt", default="The capital of France is", help="raw prompt text")
    parser.add_argument(
        "--prompt-preset",
        choices=sorted(PROMPT_PRESETS),
        help="use a built-in raw/chat-template prompt preset",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--out-dir", default="logs", help="directory for logs and summaries")
    parser.add_argument("--dry-run", action="store_true", help="print and summarize commands without running")
    parser.add_argument("--continue-on-fail", action="store_true", help="continue --all after failures")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="model config JSON path")
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def load_config(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {source}")

    with source.open(encoding="utf-8") as f:
        config = json.load(f, parse_constant=reject_constant)
    if not isinstance(config, dict):
        raise ValueError("smoke config must be a JSON object keyed by label")
    for label, entry in config.items():
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
            raise ValueError(f"invalid smoke model label: {label!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"smoke config entry {label!r} must be an object")
        for field in ("arch", "model", "tokenizer"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"smoke config entry {label!r} requires {field}")
        if entry["arch"] not in {"gpt2", "llama", "qwen3", "gemma4"}:
            raise ValueError(f"unsupported architecture for {label!r}: {entry['arch']}")
        if "experimental" in entry and not isinstance(entry["experimental"], bool):
            raise ValueError(f"experimental flag for {label!r} must be boolean")
        if "note" in entry and not isinstance(entry["note"], str):
            raise ValueError(f"note for {label!r} must be a string")
        if "notes" in entry and (
            not isinstance(entry["notes"], list)
            or any(not isinstance(note, str) for note in entry["notes"])
        ):
            raise ValueError(f"notes for {label!r} must be a string array")
    return config


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_int(value):
    if value is None:
        return None
    match = re.search(r"\d+", value.replace(",", ""))
    return int(match.group(0)) if match else None


def machine_info():
    info = {
        "host": socket.gethostname(),
        "architecture": None,
        "cpu_model": None,
        "cpu_cores": None,
        "cpu_threads": None,
    }
    result = subprocess.run(
        ["lscpu"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0:
        return info

    values = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    threads_per_core = parse_int(values.get("Thread(s) per core"))
    cores_per_socket = parse_int(values.get("Core(s) per socket"))
    sockets = parse_int(values.get("Socket(s)"))
    cpu_threads = parse_int(values.get("CPU(s)"))
    cpu_cores = None
    if cores_per_socket is not None and sockets is not None:
        cpu_cores = cores_per_socket * sockets
    elif cpu_threads is not None and threads_per_core:
        cpu_cores = cpu_threads // threads_per_core

    info.update(
        {
            "architecture": values.get("Architecture"),
            "cpu_model": values.get("Model name"),
            "cpu_cores": cpu_cores,
            "cpu_threads": cpu_threads,
        }
    )
    return info


def ember_base_command():
    binary = REPO_ROOT / "target" / "release" / "ember"
    return [str(binary)]


def resolve_prompt(args):
    if args.prompt_preset:
        return PROMPT_PRESETS[args.prompt_preset]
    return args.prompt


def run_command(command, timeout):
    timed = ["/usr/bin/time", "-v", *command]
    process = subprocess.Popen(
        timed,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "LC_ALL": "C"},
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(timed, process.returncode, stdout, stderr), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(timed, 124, stdout, stderr), True


def parse_time_output(stderr):
    max_rss = None
    elapsed = None
    prompt_tokens = None
    decode_evaluations = None
    prefill_tps = None
    decode_tps = None

    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if rss_match:
        max_rss = int(rss_match.group(1))

    elapsed_seconds = None
    elapsed_match = re.search(
        r"Elapsed \(wall clock\) time[^\n]*\):\s*"
        r"(\d+(?::\d+){1,2}(?:\.\d+)?)\s*$",
        stderr,
        re.MULTILINE,
    )
    if elapsed_match:
        elapsed = elapsed_match.group(1).strip()
        parts = elapsed.split(":")
        try:
            if len(parts) == 2:
                elapsed_seconds = int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                elapsed_seconds = (
                    int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                )
        except ValueError:
            elapsed_seconds = None

    try:
        benchmark = parse_benchmark(stderr)
    except ValueError:
        benchmark = None
    if benchmark is not None:
        prompt_tokens = benchmark["prefill"]["count"]
        prefill_tps = benchmark["prefill"]["rate_per_second"]
        decode_evaluations = benchmark["decode"]["count"]
        decode_tps = benchmark["decode"]["rate_per_second"]

    return (
        max_rss,
        elapsed,
        elapsed_seconds,
        prompt_tokens,
        decode_evaluations,
        prefill_tps,
        decode_tps,
    )


def generation_warning(text):
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 8:
        return None
    most_common = max(tokens.count(token) for token in set(tokens))
    if most_common / len(tokens) >= 0.6:
        return "degenerate/repetitive output heuristic triggered"
    for n in range(1, min(5, len(tokens) // 2 + 1)):
        chunk = tokens[:n]
        repeats = 0
        for i in range(0, len(tokens) - n + 1, n):
            if tokens[i : i + n] == chunk:
                repeats += 1
        if repeats >= 4:
            return "degenerate/repetitive output heuristic triggered"
    return None


def config_notes(entry):
    notes = []
    if entry.get("note"):
        notes.append(entry["note"])
    for note in entry.get("notes", []):
        notes.append(note)
    if entry.get("experimental"):
        notes.append("experimental model config")
    if entry.get("generation_warning"):
        notes.append(entry["generation_warning"])
    return notes


def sha256_file(path: Path) -> str:
    before = file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if file_identity(path) != before:
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest()


def file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_log(path, metadata, stdout, stderr):
    lines = [
        "# ember smoke run",
        "",
        "## metadata",
        json.dumps(metadata, indent=2, sort_keys=True),
        "",
        "## raw smoke output",
        stdout,
        "",
        "## stderr and /usr/bin/time -v",
        stderr,
    ]
    atomic_write(path, "\n".join(lines))


def summarize_skip(label, entry, args, reason, machine, commit):
    now = datetime.now(timezone.utc).isoformat()
    notes = config_notes(entry)
    notes.append(reason)
    return {
        "schema_version": 2,
        "label": label,
        "arch": entry.get("arch"),
        "model": entry.get("model"),
        "tokenizer": entry.get("tokenizer"),
        "command": None,
        "exit_status": None,
        "status": "smoke_skipped",
        "pass_fail": "skip",
        "generated_text": None,
        "prompt_token_count": None,
        "decode_evaluation_count": None,
        "prefill_tps": None,
        "decode_evaluations_per_second": None,
        "max_rss_kb": None,
        "elapsed_time": None,
        "notes": notes,
        "requested_max_generated_tokens": args.tokens,
        "commit_hash": commit,
        "host": machine["host"],
        "machine": machine,
        "date": now,
    }


def run_one(label, entry, args, out_dir, commit, machine):
    prompt = resolve_prompt(args)
    model_path = (REPO_ROOT / entry["model"]).resolve()
    tokenizer_path = (REPO_ROOT / entry["tokenizer"]).resolve()
    notes = config_notes(entry)

    missing = []
    if not model_path.is_file():
        missing.append(f"missing model file: {entry['model']}")
    if not tokenizer_path.is_file():
        missing.append(f"missing tokenizer file: {entry['tokenizer']}")
    if missing:
        summary = summarize_skip(
            label, entry, args, "; ".join(missing), machine, commit
        )
        if args.model:
            summary["status"] = "smoke_fail"
            summary["pass_fail"] = "fail"
        return summary

    command = [
        *ember_base_command(),
        "--arch",
        entry["arch"],
        "--model",
        str(model_path),
        "--tokenizer",
        str(tokenizer_path),
        "--prompt",
        prompt,
        "--max-tokens",
        str(args.tokens),
        "--temperature",
        str(args.temperature),
        "--benchmark",
    ]
    ember_command_string = " ".join(shlex.quote(part) for part in command)
    command_string = " ".join(shlex.quote(part) for part in ["/usr/bin/time", "-v", *command])
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = out_dir / f"{stamp}_{label}.log"
    summary_path = out_dir / f"{stamp}_{label}_summary.json"

    metadata = {
        "schema_version": 2,
        "label": label,
        "arch": entry["arch"],
        "model": entry["model"],
        "resolved_model": str(model_path),
        "tokenizer": entry["tokenizer"],
        "resolved_tokenizer": str(tokenizer_path),
        "command": command_string,
        "command_argv": ["/usr/bin/time", "-v", *command],
        "ember_command": ember_command_string,
        "requested_max_generated_tokens": args.tokens,
        "commit_hash": commit,
        "host": machine["host"],
        "machine": machine,
        "date": now.isoformat(),
        "prompt": prompt,
        "raw_smoke_output_note": "generation text is raw smoke output, not quality validation",
    }

    if args.dry_run:
        metadata["model_sha256"] = sha256_file(model_path)
        metadata["tokenizer_sha256"] = sha256_file(tokenizer_path)
        print(command_string)
        summary = {
            **metadata,
            "exit_status": None,
            "status": "dry_run",
            "pass_fail": "skip",
            "generated_text": None,
            "prompt_token_count": None,
            "decode_evaluation_count": None,
            "prefill_tps": None,
            "decode_evaluations_per_second": None,
            "max_rss_kb": None,
            "elapsed_time": None,
            "notes": notes,
            "log_path": str(log_path),
        }
        atomic_write(
            summary_path,
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        return summary

    binary_path = Path(command[0])
    input_paths = [binary_path, model_path, tokenizer_path]
    initial_identities = {str(path): file_identity(path) for path in input_paths}
    result, timed_out = run_command(command, args.timeout)
    (
        max_rss,
        elapsed,
        elapsed_seconds,
        prompt_tokens,
        decode_evaluations,
        prefill_tps,
        decode_eval_s,
    ) = parse_time_output(
        result.stderr or ""
    )
    for path in input_paths:
        if file_identity(path) != initial_identities[str(path)]:
            raise RuntimeError(f"smoke input changed while the command was running: {path}")
    metadata["binary_sha256"] = sha256_file(binary_path)
    metadata["model_sha256"] = sha256_file(model_path)
    metadata["tokenizer_sha256"] = sha256_file(tokenizer_path)
    metadata["generated_stdout_sha256"] = hashlib.sha256(
        (result.stdout or "").encode("utf-8")
    ).hexdigest()
    warning = entry.get("generation_warning") or generation_warning(result.stdout or "")
    if warning and warning not in notes:
        notes.append(warning)
    output_exists = bool((result.stdout or "").strip())

    timings_complete = all(
        value is not None
        for value in (
            prompt_tokens,
            decode_evaluations,
            prefill_tps,
            decode_eval_s,
            max_rss,
            elapsed_seconds,
        )
    )
    if result.returncode == 0 and output_exists and timings_complete and warning:
        status = "smoke_pass_generation_warning"
        pass_fail = "pass"
    elif result.returncode == 0 and output_exists and timings_complete:
        status = "smoke_pass"
        pass_fail = "pass"
    else:
        status = "smoke_fail"
        pass_fail = "fail"
        if timed_out:
            notes.append(f"timed out after {args.timeout} seconds")
        elif result.returncode != 0:
            notes.append(f"command exited with status {result.returncode}")
        if result.returncode == 0 and not output_exists:
            notes.append("missing generated output")
        if result.returncode == 0 and not timings_complete:
            notes.append("missing required benchmark or RSS metrics")

    summary = {
        **metadata,
        "exit_status": result.returncode,
        "status": status,
        "pass_fail": pass_fail,
        "generated_text": (result.stdout or "").strip(),
        "prompt_token_count": prompt_tokens,
        "decode_evaluation_count": decode_evaluations,
        "prefill_tps": prefill_tps,
        "decode_evaluations_per_second": decode_eval_s,
        "max_rss_kb": max_rss,
        "elapsed_time": elapsed,
        "elapsed_seconds": elapsed_seconds,
        "timed_out": timed_out,
        "notes": notes,
        "log_path": str(log_path),
    }
    write_log(log_path, metadata, result.stdout or "", result.stderr or "")
    atomic_write(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def main():
    args = parse_args()
    if args.tokens < 2:
        raise SystemExit("--tokens must be >= 2 to exercise decode")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        raise SystemExit("--temperature must be finite and non-negative")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise SystemExit("--timeout must be finite and positive")
    if not resolve_prompt(args):
        raise SystemExit("the selected prompt must not be empty")
    if not Path("/usr/bin/time").is_file():
        raise SystemExit("/usr/bin/time is required for smoke resource metrics")
    config = load_config(args.config)
    if args.model and args.model not in config:
        raise SystemExit(f"unknown model label: {args.model}")

    binary = Path(ember_base_command()[0])
    if not args.dry_run and not binary.is_file():
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(REPO_ROOT / "Cargo.toml"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    if not args.dry_run and (
        not binary.is_file() or not os.access(binary, os.X_OK)
    ):
        raise SystemExit(f"release Ember executable is unavailable: {binary}")

    labels = list(config) if args.all else [args.model]
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    machine = machine_info()

    summaries = []
    failed = False
    for label in labels:
        summary = run_one(label, config[label], args, out_dir, commit, machine)
        summaries.append(summary)
        print(f"{label}: {summary['status']}")
        if summary["pass_fail"] == "fail":
            failed = True
            if not args.continue_on_fail:
                break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    aggregate = out_dir / f"{stamp}_smoke_summary.json"
    atomic_write(
        aggregate,
        json.dumps(
            {
                "schema_version": 2,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_path": str(Path(args.config).resolve()),
                "config_sha256": sha256_file(Path(args.config)),
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    print(f"summary: {aggregate}")

    if failed or (args.all and not args.dry_run and all(item["pass_fail"] == "skip" for item in summaries)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
