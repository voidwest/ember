#!/usr/bin/env python3
"""Run local Ember smoke tests and write auditable logs.

Smoke status is structural: command exit, basic timing parse, and optional
degenerate-output warning. Raw generation text is not a quality benchmark.
"""

import argparse
import json
import re
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("smoke config must be a JSON object keyed by label")
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


def ember_base_command():
    binary = REPO_ROOT / "target" / "release" / "ember"
    if binary.exists():
        return [str(binary)]
    return ["cargo", "run", "--release", "--"]


def resolve_prompt(args):
    if args.prompt_preset:
        return PROMPT_PRESETS[args.prompt_preset]
    return args.prompt


def run_command(command):
    timed = ["/usr/bin/time", "-v", *command]
    return subprocess.run(
        timed,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_time_output(stderr):
    max_rss = None
    elapsed = None
    prompt_tokens = None
    decode_tokens = None
    prefill_tps = None
    decode_tps = None

    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if rss_match:
        max_rss = int(rss_match.group(1))

    elapsed_match = re.search(r"Elapsed \(wall clock\) time.*:\s*(.+)", stderr)
    if elapsed_match:
        elapsed = elapsed_match.group(1).strip()

    prefill_match = re.search(r"prefill:\s+(\d+) tokens in [\d.]+ms ->\s+([\d.]+) tok/s", stderr)
    if prefill_match:
        prompt_tokens = int(prefill_match.group(1))
        prefill_tps = float(prefill_match.group(2))

    decode_match = re.search(r"decode:\s+(\d+) tokens in [\d.]+ms ->\s+([\d.]+) tok/s", stderr)
    if decode_match:
        decode_tokens = int(decode_match.group(1))
        decode_tps = float(decode_match.group(2))

    return max_rss, elapsed, prompt_tokens, decode_tokens, prefill_tps, decode_tps


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
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_skip(label, entry, args, reason):
    now = datetime.now(timezone.utc).isoformat()
    notes = config_notes(entry)
    notes.append(reason)
    return {
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
        "decode_token_count": None,
        "prefill_tps": None,
        "decode_tps": None,
        "max_rss_kb": None,
        "elapsed_time": None,
        "notes": notes,
        "generated_token_count": args.tokens,
        "commit_hash": git_commit(),
        "host": socket.gethostname(),
        "date": now,
    }


def run_one(label, entry, args, out_dir, commit):
    prompt = resolve_prompt(args)
    model_path = REPO_ROOT / entry["model"]
    tokenizer_path = REPO_ROOT / entry["tokenizer"]
    notes = config_notes(entry)

    missing = []
    if not model_path.exists():
        missing.append(f"missing model file: {entry['model']}")
    if not tokenizer_path.exists():
        missing.append(f"missing tokenizer file: {entry['tokenizer']}")
    if missing:
        summary = summarize_skip(label, entry, args, "; ".join(missing))
        if args.model:
            summary["status"] = "smoke_fail"
            summary["pass_fail"] = "fail"
        return summary

    command = [
        *ember_base_command(),
        "--arch",
        entry["arch"],
        "--model",
        entry["model"],
        "--tokenizer",
        entry["tokenizer"],
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
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    log_path = out_dir / f"{stamp}_{label}.log"
    summary_path = out_dir / f"{stamp}_{label}_summary.json"

    metadata = {
        "label": label,
        "arch": entry["arch"],
        "model": entry["model"],
        "tokenizer": entry["tokenizer"],
        "command": command_string,
        "ember_command": ember_command_string,
        "generated_token_count": args.tokens,
        "commit_hash": commit,
        "host": socket.gethostname(),
        "date": now.isoformat(),
        "prompt": prompt,
        "raw_smoke_output_note": "generation text is raw smoke output, not quality validation",
    }

    if args.dry_run:
        print(command_string)
        summary = {
            **metadata,
            "exit_status": None,
            "status": "dry_run",
            "pass_fail": "skip",
            "generated_text": None,
            "prompt_token_count": None,
            "decode_token_count": None,
            "prefill_tps": None,
            "decode_tps": None,
            "max_rss_kb": None,
            "elapsed_time": None,
            "notes": notes,
            "log_path": str(log_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    result = run_command(command)
    max_rss, elapsed, prompt_tokens, decode_tokens, prefill_tps, decode_tps = parse_time_output(
        result.stderr
    )
    warning = entry.get("generation_warning") or generation_warning(result.stdout)
    if warning and warning not in notes:
        notes.append(warning)
    output_exists = bool(result.stdout.strip())

    if result.returncode == 0 and output_exists and warning:
        status = "smoke_pass_generation_warning"
        pass_fail = "pass"
    elif result.returncode == 0 and output_exists:
        status = "smoke_pass"
        pass_fail = "pass"
    else:
        status = "smoke_fail"
        pass_fail = "fail"
        if result.returncode == 0 and not output_exists:
            notes.append("missing generated output")

    summary = {
        **metadata,
        "exit_status": result.returncode,
        "status": status,
        "pass_fail": pass_fail,
        "generated_text": result.stdout.strip(),
        "prompt_token_count": prompt_tokens,
        "decode_token_count": decode_tokens,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "max_rss_kb": max_rss,
        "elapsed_time": elapsed,
        "notes": notes,
        "log_path": str(log_path),
    }
    write_log(log_path, metadata, result.stdout, result.stderr)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    if args.tokens < 0:
        raise SystemExit("--tokens must be >= 0")
    config = load_config(args.config)
    if args.model and args.model not in config:
        raise SystemExit(f"unknown model label: {args.model}")

    labels = list(config) if args.all else [args.model]
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    summaries = []
    failed = False
    for label in labels:
        summary = run_one(label, config[label], args, out_dir, commit)
        summaries.append(summary)
        print(f"{label}: {summary['status']}")
        if summary["pass_fail"] == "fail":
            failed = True
            if not args.continue_on_fail:
                break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    aggregate = out_dir / f"{stamp}_smoke_summary.json"
    aggregate.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary: {aggregate}")

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
