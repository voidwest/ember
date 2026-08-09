#!/usr/bin/env python3
"""Run the real-GGUF, process-isolated KV snapshot replay matrix.

Large prompt-derived snapshots and full-logit NPY files stay under ignored
``runs/``.  Compact, mechanically-derived JSON evidence can optionally be
published under ``artifacts/benchmark-kv-v1/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACE_SCHEMA = "ember.kv-replay-matrix.v1"
PROMPTS = (
    ("en_france", "English", "The capital of France is"),
    ("ar_france", "Arabic", "ما هي عاصمة فرنسا؟"),
)
MODEL_ROWS = (
    ("llama-3.2-1b", "llama", "q8_0", "llama-3.2-1b-q8_0.gguf"),
    ("llama-3.2-1b", "llama", "q6_k", "llama-3.2-1b-q6_k.gguf"),
    ("llama-3.2-1b", "llama", "q4_k_m", "llama-3.2-1b-q4_k_m.gguf"),
    ("qwen2.5-1.5b", "qwen3", "q8_0", "qwen2.5-1.5b-q8_0.gguf"),
    ("qwen2.5-1.5b", "qwen3", "q6_k", "qwen2.5-1.5b-q6_k.gguf"),
    ("qwen2.5-1.5b", "qwen3", "q4_k_m", "qwen2.5-1.5b-q4_k_m.gguf"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def relative_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return f"external:{path.name}"


def sanitize_commands(
    commands: list[dict[str, Any]],
    *,
    binary: Path,
    model_paths: Iterable[Path],
    tokenizer_paths: Iterable[Path],
    raw_root: Path,
) -> list[dict[str, Any]]:
    replacements = {str(binary): "$EMBER"}
    replacements.update({str(path): f"$MODEL/{path.name}" for path in model_paths})
    replacements.update({str(path): f"$TOKENIZER/{path.name}" for path in tokenizer_paths})
    raw_prefix = str(raw_root) + os.sep
    sanitized = []
    for source in commands:
        record = dict(source)
        argv = []
        for argument in source["argv"]:
            if argument in replacements:
                argv.append(replacements[argument])
            elif argument.startswith(raw_prefix):
                argv.append("$RAW/" + argument[len(raw_prefix):])
            else:
                argv.append(argument)
        record["argv"] = argv
        record["display"] = shlex.join(argv)
        sanitized.append(record)
    return sanitized


def parse_time_v(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key] = value
    integer_keys = {
        "Maximum resident set size (kbytes)": "max_rss_kib",
        "Major (requiring I/O) page faults": "major_page_faults",
        "Minor (reclaiming a frame) page faults": "minor_page_faults",
        "File system inputs": "filesystem_inputs",
        "File system outputs": "filesystem_outputs",
    }
    result: dict[str, Any] = {}
    for source, target in integer_keys.items():
        try:
            result[target] = int(values[source])
        except (KeyError, ValueError):
            pass
    for source, target in (
        ("User time (seconds)", "user_seconds"),
        ("System time (seconds)", "system_seconds"),
    ):
        try:
            result[target] = float(values[source])
        except (KeyError, ValueError):
            pass
    return result


def best_effort_dontneed(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        record: dict[str, Any] = {"path": relative_label(path), "advised": False}
        if not hasattr(os, "posix_fadvise"):
            record["reason"] = "os.posix_fadvise unavailable"
            records.append(record)
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            record["advised"] = True
        except OSError as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return records


def run_process(
    *,
    name: str,
    argv: list[str],
    output_dir: Path,
    environment: dict[str, str],
    timeout: int,
    commands: list[dict[str, Any]],
    advice_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    advice = best_effort_dontneed(advice_paths)
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    time_path = output_dir / "time.txt"
    wrapped = ["/usr/bin/time", "-v", "-o", str(time_path), *argv]
    started_ns = time.perf_counter_ns()
    timed_out = False
    process = subprocess.Popen(
        wrapped,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        returncode = 124
        stderr += f"\nTIMEOUT after {timeout}s\n".encode()
    wall_ns = time.perf_counter_ns() - started_ns
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    record = {
        "ordinal": len(commands),
        "name": name,
        "argv": argv,
        "display": shlex.join(argv),
        "cwd": ".",
        "returncode": returncode,
        "timed_out": timed_out,
        "process_wall_ns": wall_ns,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "time_v": parse_time_v(time_path),
        "cache_advice": advice,
    }
    commands.append(record)
    return record


def load_trace(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("trace_schema") != "ember.kv-replay-trace.v1":
        raise ValueError(f"unexpected trace schema in {path}")
    return value


def load_logits(path: Path, rows: int | None = None) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.dtype.str != "<f4" or not array.flags.c_contiguous:
        raise ValueError(f"{path} is not a C-order 2-D little-endian f32 array")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{path} has {array.shape[0]} rows, expected {rows}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite logits")
    return array


def array_digest(array: np.ndarray) -> str:
    return sha256_bytes(array.tobytes(order="C"))


def compare_exact(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "exact": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "mismatch_count": None,
        }
    mismatch = actual.view(np.uint32) != expected.view(np.uint32)
    mismatch_count = int(np.count_nonzero(mismatch))
    result: dict[str, Any] = {
        "exact": mismatch_count == 0,
        "shape": list(actual.shape),
        "mismatch_count": mismatch_count,
        "actual_payload_sha256": array_digest(actual),
        "expected_payload_sha256": array_digest(expected),
    }
    if mismatch_count:
        first = tuple(int(index) for index in np.argwhere(mismatch)[0])
        left = np.float32(actual[first])
        right = np.float32(expected[first])
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        result.update({
            "first_mismatch": list(first),
            "actual_f32": float(left),
            "expected_f32": float(right),
            "actual_u32": int(left.view(np.uint32)),
            "expected_u32": int(right.view(np.uint32)),
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
        })
    else:
        result.update({"max_abs_difference": 0.0, "mean_abs_difference": 0.0})
    return result


def phase_summary(native: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    nt = native["timings_ms"]
    rt = replay["timings_ms"]
    native_inference = nt["prefill_inference"] + nt["decode_inference"]
    replay_setup = rt["snapshot_load_and_verify"] + rt["snapshot_import"]
    replay_inference = rt["decode_inference"]
    return {
        "native_prefill_ms": nt["prefill_inference"],
        "native_decode_ms": nt["decode_inference"],
        "native_prefill_plus_decode_ms": native_inference,
        "replay_load_verify_ms": rt["snapshot_load_and_verify"],
        "replay_import_ms": rt["snapshot_import"],
        "replay_decode_ms": replay_inference,
        "replay_setup_plus_decode_ms": replay_setup + replay_inference,
        "avoided_prefill_minus_replay_setup_ms": nt["prefill_inference"] - replay_setup,
        "interpretation": (
            "Named in-process phases only; excludes model/tokenizer hashing and load. "
            "Resume-token emission is not a forward and is not called first-token inference. "
            "Trace rows stream between forwards; import includes repeated in-memory verification."
        ),
    }


def inspect_environment(binary: Path, threads: int) -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    def output(argv: list[str]) -> str:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
        return (result.stdout or result.stderr).strip()
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "rayon_num_threads": threads,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rustc": output(["rustc", "--version"]),
        "binary_sha256": sha256_file(binary),
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": output(["git", "status", "--porcelain"]),
        "cache_state": "unverified",
        "cache_note": (
            "Observation 0 receives best-effort POSIX_FADV_DONTNEED before each process. "
            "The CLI then hashes the full GGUF before inference, so neither observation is a "
            "verified cold/resident-cache measurement. Observation labels are chronological only. "
            "Trace rows are streamed between timed forwards and can perturb later cache state; "
            "snapshot_import includes a second in-memory integrity verification."
        ),
    }


def require_fresh_directory(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing existing output directory: {path}")
    path.mkdir(parents=True, exist_ok=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh ignored raw run directory")
    parser.add_argument("--evidence-output", type=Path,
                        help="fresh directory for compact JSON evidence")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/ember")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/v03-ladder")
    parser.add_argument("--ladder-manifest", type=Path,
                        default=ROOT / "models/v03-ladder/ladder-manifest.json")
    parser.add_argument("--llama-tokenizer", type=Path, default=ROOT / "tokenizer.json")
    parser.add_argument(
        "--qwen-tokenizer", type=Path,
        default=Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/"
                "snapshots/8faed761d45a263340a0528343f099c05c9a4323/tokenizer.json",
    )
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--observations", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--family", action="append", choices=["llama-3.2-1b", "qwen2.5-1.5b"])
    parser.add_argument("--rung", action="append", choices=["q8_0", "q6_k", "q4_k_m"])
    parser.add_argument("--prompt-id", action="append", choices=[item[0] for item in PROMPTS])
    args = parser.parse_args()

    if args.tokens < 2:
        parser.error("--tokens must be at least 2 so replay evaluates a continuation row")
    if args.max_seq_len < args.tokens:
        parser.error("--max-seq-len is implausibly small")
    if args.observations < 1:
        parser.error("--observations must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    selected_models = [row for row in MODEL_ROWS
                       if (not args.family or row[0] in args.family)
                       and (not args.rung or row[2] in args.rung)]
    selected_prompts = [row for row in PROMPTS if not args.prompt_id or row[0] in args.prompt_id]
    if not selected_models or not selected_prompts:
        parser.error("filters selected an empty matrix")
    expected_cases = len(selected_models) * len(selected_prompts)
    partial = expected_cases != len(MODEL_ROWS) * len(PROMPTS)

    output = args.output.resolve()
    evidence_output = args.evidence_output.resolve() if args.evidence_output else None
    runs_root = (ROOT / "runs").resolve()
    try:
        output.relative_to(runs_root)
    except ValueError:
        parser.error(f"--output must stay under ignored raw root {runs_root}")
    if output == runs_root:
        parser.error("--output must be a new child directory below runs/")
    require_fresh_directory(output)
    raw_dir = output / "raw"
    raw_dir.mkdir()

    environment = os.environ.copy()
    environment.update({"RAYON_NUM_THREADS": str(args.threads), "LC_ALL": "C", "TZ": "UTC"})
    commands: list[dict[str, Any]] = []
    if not args.skip_build:
        build = run_process(
            name="cargo-build-release",
            argv=["cargo", "build", "--release"],
            output_dir=raw_dir / "build",
            environment=environment,
            timeout=args.timeout,
            commands=commands,
        )
        if build["returncode"] != 0:
            atomic_json(output / "commands.json", commands)
            return 1
    binary = args.binary.resolve()
    selected_families = {row[0] for row in selected_models}
    required = [binary, args.ladder_manifest.resolve()]
    if "llama-3.2-1b" in selected_families:
        required.append(args.llama_tokenizer.resolve())
    if "qwen2.5-1.5b" in selected_families:
        required.append(args.qwen_tokenizer.resolve())
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")

    ladder = json.loads(args.ladder_manifest.read_text(encoding="utf-8"))
    ladder_by_key = {(row["family"], row["rung"]): row for row in ladder}

    input_records: dict[str, Any] = {}
    model_paths: dict[tuple[str, str], Path] = {}
    for family, _arch, rung, filename in selected_models:
        model = (args.models_dir / filename).resolve()
        if not model.is_file():
            raise SystemExit(f"matrix model is missing: {model}")
        ladder_row = ladder_by_key[(family, rung)]
        actual_hash = sha256_file(model)
        expected_hash = ladder_row["target"]["sha256"]
        if actual_hash != expected_hash:
            raise SystemExit(f"model hash mismatch for {family}/{rung}: {actual_hash} != {expected_hash}")
        model_paths[(family, rung)] = model
        input_records[f"{family}/{rung}"] = {
            "label": filename,
            "sha256": actual_hash,
            "bytes": model.stat().st_size,
            "identity_before": file_identity(model),
            "ladder_manifest_sha256": expected_hash,
        }
    tokenizer_paths = {
        "llama-3.2-1b": args.llama_tokenizer.resolve(),
        "qwen2.5-1.5b": args.qwen_tokenizer.resolve(),
    }
    tokenizer_records = {
        family: {
            "label": relative_label(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "identity_before": file_identity(path),
        }
        for family, path in tokenizer_paths.items()
        if any(model[0] == family for model in selected_models)
    }

    environment_record = inspect_environment(binary, args.threads)
    tested_source_paths = [
        ROOT / "src/atomic_file.rs",
        ROOT / "src/cli_kv.rs",
        ROOT / "src/kv_cache.rs",
        ROOT / "src/kv_snapshot.rs",
        ROOT / "src/kv_transfer/mod.rs",
        ROOT / "src/kv_transfer/rope.rs",
        ROOT / "src/llama.rs",
        ROOT / "src/main.rs",
        ROOT / "src/npy.rs",
        ROOT / "src/plan.rs",
        Path(__file__).resolve(),
    ]
    manifest = {
        "schema": TRACE_SCHEMA,
        "status": "running",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_run_directory": relative_label(output),
        "harness": {
            "path": "scripts/validate_kv_replay_matrix.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "partial_matrix": partial,
        "invocation": {
            "skip_build": args.skip_build,
            "tokens": args.tokens,
            "max_seq_len": args.max_seq_len,
            "observations": args.observations,
            "threads": args.threads,
            "timeout_seconds": args.timeout,
            "family_filters": args.family or [],
            "rung_filters": args.rung or [],
            "prompt_filters": args.prompt_id or [],
            "raw_output": relative_label(output),
            "evidence_output": relative_label(evidence_output) if evidence_output else None,
        },
        "tested_source_files": {
            relative_label(path): sha256_file(path) for path in tested_source_paths
        },
        "protocol": {
            "families": list(dict.fromkeys(row[0] for row in selected_models)),
            "rungs": list(dict.fromkeys(row[2] for row in selected_models)),
            "prompts": [{"id": key, "language": language, "text": prompt,
                         "utf8_sha256": sha256_bytes(prompt.encode())}
                        for key, language, prompt in selected_prompts],
            "execution": "planned",
            "k_strategy": "auto",
            "continuation_tokens": args.tokens,
            "common_cache_capacity": args.max_seq_len,
            "observations": args.observations,
            "comparison_gate": "f32 payload bit equality; no tolerance",
            "row_contract": (
                "native[N,V] == concatenate(export_boundary[1,V], "
                "replay_continuation[N-1,V])"
            ),
            "eos_policy": "ignore-fixed-count",
            "sampling": "greedy full-f32 argmax, lowest token id wins ties",
        },
        "environment": environment_record,
        "models": input_records,
        "tokenizers": tokenizer_records,
        "ladder_manifest": {
            "label": relative_label(args.ladder_manifest),
            "sha256": sha256_file(args.ladder_manifest),
        },
        "command_templates": {
            "native": "$EMBER --k-strategy auto kv trace-native ...",
            "export": "$EMBER --k-strategy auto kv export ...",
            "verify": "$EMBER kv verify $SNAPSHOT",
            "replay": "$EMBER --k-strategy auto kv replay ...",
        },
        "non_claims": [
            "No cross-model transfer or mapper is exercised.",
            "No logits are compared across model families or quantization rungs.",
            "This is Ember-internal replay, not llama.cpp or quality parity.",
            "Arabic is a multilingual smoke prompt, not a morphology result.",
            "Timing observations do not establish controlled cold/warm performance or a speedup.",
            "Q8 uses Ember's frozen native Q8 fast path even though provenance mode is planned.",
        ],
    }
    atomic_json(output / "benchmark_manifest.json", manifest)
    summary: dict[str, Any] = {
        "schema": TRACE_SCHEMA,
        "status": "running",
        "expected_cases": expected_cases,
        "completed_cases": 0,
        "passed_cases": 0,
        "failed_cases": 0,
        "partial_matrix": partial,
        "cases": [],
        "failures": [],
    }
    atomic_json(output / "benchmark_summary.json", summary)

    stop = False
    for family, arch, rung, filename in selected_models:
        if stop:
            break
        model = model_paths[(family, rung)]
        tokenizer = tokenizer_paths[family]
        model_label = f"{family}/{rung}"
        for prompt_id, language, prompt in selected_prompts:
            case_id = f"{family}__{rung}__{prompt_id}"
            case_root = raw_dir / "cases" / family / rung / prompt_id
            case_root.mkdir(parents=True, exist_ok=False)
            setup = case_root / "setup"
            setup.mkdir()
            snapshot = setup / "snapshot"
            boundary_npy = setup / "boundary.npy"
            export_json = setup / "export.json"
            common = [
                "--model", str(model), "--tokenizer", str(tokenizer), "--arch", arch,
                "--execution", "planned", "--max-seq-len", str(args.max_seq_len),
            ]
            export_command = [
                str(binary), "--k-strategy", "auto", "kv", "export", *common,
                "--prompt", prompt, "--output", str(snapshot),
                "--boundary-logits-output", str(boundary_npy),
                "--metrics-output", str(export_json),
            ]
            export_record = run_process(
                name=f"{case_id}:export",
                argv=export_command,
                output_dir=setup / "export-process",
                environment=environment,
                timeout=args.timeout,
                commands=commands,
            )
            verify_record: dict[str, Any] = {"returncode": -1}
            if export_record["returncode"] == 0:
                verify_record = run_process(
                    name=f"{case_id}:verify",
                    argv=[str(binary), "kv", "verify", str(snapshot)],
                    output_dir=setup / "verify-process",
                    environment=environment,
                    timeout=args.timeout,
                    commands=commands,
                )
            observations: list[dict[str, Any]] = []
            case_errors: list[str] = []
            if export_record["returncode"] != 0:
                case_errors.append("export process failed")
            if verify_record["returncode"] != 0:
                case_errors.append("verify process failed")

            for observation in range(args.observations):
                observation_dir = case_root / f"observation-{observation:02d}"
                observation_dir.mkdir()
                native_npy = observation_dir / "native.npy"
                native_json = observation_dir / "native.json"
                replay_npy = observation_dir / "replay.npy"
                replay_json = observation_dir / "replay.json"
                native_command = [
                    str(binary), "--k-strategy", "auto", "kv", "trace-native", *common,
                    "--prompt", prompt, "--max-tokens", str(args.tokens),
                    "--logits-output", str(native_npy), "--metrics-output", str(native_json),
                ]
                advice = [model, tokenizer, binary] if observation == 0 else []
                native_record = run_process(
                    name=f"{case_id}:observation-{observation}:native",
                    argv=native_command,
                    output_dir=observation_dir / "native-process",
                    environment=environment,
                    timeout=args.timeout,
                    commands=commands,
                    advice_paths=advice,
                )
                replay_command = [
                    str(binary), "--k-strategy", "auto", "kv", "replay", *common,
                    "--snapshot", str(snapshot), "--max-tokens", str(args.tokens),
                    "--logits-output", str(replay_npy), "--metrics-output", str(replay_json),
                ]
                replay_advice = ([model, tokenizer, binary, snapshot / "manifest.json",
                                  snapshot / "keys.f16le", snapshot / "values.f16le"]
                                 if observation == 0 and snapshot.exists() else [])
                replay_record = run_process(
                    name=f"{case_id}:observation-{observation}:replay",
                    argv=replay_command,
                    output_dir=observation_dir / "replay-process",
                    environment=environment,
                    timeout=args.timeout,
                    commands=commands,
                    advice_paths=replay_advice,
                )
                observation_result: dict[str, Any] = {
                    "index": observation,
                    "access_class": (
                        "post-export-first-observed-with-best-effort-dontneed"
                        if observation == 0 else "post-export-subsequent-observed"
                    ),
                    "cache_state": "unverified",
                    "native_process_wall_ns": native_record["process_wall_ns"],
                    "replay_process_wall_ns": replay_record["process_wall_ns"],
                    "native_time_v": native_record["time_v"],
                    "replay_time_v": replay_record["time_v"],
                }
                if native_record["returncode"] != 0:
                    case_errors.append(f"observation {observation} native process failed")
                if replay_record["returncode"] != 0:
                    case_errors.append(f"observation {observation} replay process failed")
                if not case_errors or (native_record["returncode"] == 0
                                       and replay_record["returncode"] == 0
                                       and export_record["returncode"] == 0):
                    try:
                        native_trace = load_trace(native_json)
                        export_trace = load_trace(export_json)
                        replay_trace = load_trace(replay_json)
                        native_logits = load_logits(native_npy, args.tokens)
                        boundary_logits = load_logits(boundary_npy, 1)
                        replay_logits = load_logits(replay_npy, args.tokens - 1)
                        if native_logits.shape[1] != boundary_logits.shape[1] \
                                or native_logits.shape[1] != replay_logits.shape[1]:
                            raise ValueError("vocabulary widths differ")
                        boundary_comparison = compare_exact(native_logits[:1], boundary_logits)
                        continuation_comparison = compare_exact(native_logits[1:], replay_logits)
                        native_tokens = native_trace["generated_token_ids"]
                        replay_tokens = replay_trace["generated_token_ids"]
                        snapshot_manifest = json.loads(
                            (snapshot / "manifest.json").read_text(encoding="utf-8"))
                        inventory = sorted(path.name for path in snapshot.iterdir())
                        common_hashes = {
                            trace["model_sha256"] for trace in
                            (native_trace, export_trace, replay_trace)
                        }
                        common_tokenizers = {
                            trace["tokenizer_sha256"] for trace in
                            (native_trace, export_trace, replay_trace)
                        }
                        fingerprints = {
                            trace["execution_fingerprint"] for trace in
                            (native_trace, export_trace, replay_trace)
                        }
                        prefix_hashes = {
                            trace["prefix_token_ids_sha256"] for trace in
                            (native_trace, export_trace, replay_trace)
                        }
                        plan_hashes = {
                            trace["execution_plan_hash"] for trace in
                            (native_trace, export_trace, replay_trace)
                        }
                        native_file_sha256 = sha256_file(native_npy)
                        boundary_file_sha256 = sha256_file(boundary_npy)
                        replay_file_sha256 = sha256_file(replay_npy)
                        prompt_sha256 = sha256_bytes(prompt.encode())
                        prefix_length = snapshot_manifest["sequence_length"]
                        metadata_checks = {
                            "model_hashes_match": common_hashes == {input_records[model_label]["sha256"]},
                            "tokenizer_hashes_match": common_tokenizers == {tokenizer_records[family]["sha256"]},
                            "execution_fingerprints_match": len(fingerprints) == 1,
                            "execution_plan_hashes_match": len(plan_hashes) == 1,
                            "prefix_token_hashes_match": (
                                len(prefix_hashes) == 1
                                and snapshot_manifest["provenance"]["prefix_token_ids_sha256"]
                                    in prefix_hashes
                            ),
                            "prompt_hashes_match": (
                                native_trace["prompt_sha256"]
                                == export_trace["prompt_sha256"]
                                == prompt_sha256
                                and replay_trace["prompt_sha256"] is None
                            ),
                            "trace_file_hashes_match": (
                                native_trace["logits_sha256"] == native_file_sha256
                                and export_trace["logits_sha256"] == boundary_file_sha256
                                and replay_trace["logits_sha256"] == replay_file_sha256
                            ),
                            "trace_kinds_and_rows_match": (
                                native_trace["kind"] == "native-uninterrupted"
                                and export_trace["kind"] == "snapshot-export-boundary"
                                and replay_trace["kind"] == "snapshot-replay-continuation"
                                and native_trace["logits_rows"] == args.tokens
                                and export_trace["logits_rows"] == 1
                                and replay_trace["logits_rows"] == args.tokens - 1
                                and native_trace["logits_global_row_start"] == 0
                                and export_trace["logits_global_row_start"] == 0
                                and replay_trace["logits_global_row_start"] == 1
                            ),
                            "trace_positions_match": (
                                native_trace["prefix_length"] == prefix_length
                                and export_trace["prefix_length"] == prefix_length
                                and replay_trace["prefix_length"] == prefix_length
                                and native_trace["predicted_absolute_position_start"] == prefix_length
                                and export_trace["predicted_absolute_position_start"] == prefix_length
                                and replay_trace["predicted_absolute_position_start"]
                                    == prefix_length + 1
                            ),
                            "trace_modes_and_counts_match": (
                                all(trace["execution_mode"] == "planned" for trace in
                                    (native_trace, export_trace, replay_trace))
                                and native_trace["max_tokens"] == args.tokens
                                and export_trace["max_tokens"] == 1
                                and replay_trace["max_tokens"] == args.tokens
                                and native_trace["forward_evaluations"] == args.tokens
                                and export_trace["forward_evaluations"] == 1
                                and replay_trace["forward_evaluations"] == args.tokens - 1
                            ),
                            "cache_capacities_match_requested": all(
                                trace["cache_capacity"] == args.max_seq_len
                                for trace in (native_trace, export_trace, replay_trace)
                            ),
                            "snapshot_source_capacity_matches": snapshot_manifest["max_seq"] == args.max_seq_len,
                            "snapshot_hashes_match": (
                                export_trace["snapshot_hash"]
                                == replay_trace["snapshot_hash"]
                                == snapshot_manifest["snapshot_hash"]
                            ),
                            "snapshot_inventory_exact": inventory
                                == ["keys.f16le", "manifest.json", "values.f16le"],
                            "resume_token_matches": (
                                snapshot_manifest["provenance"]["resume_token_id"]
                                == native_tokens[0]
                                == export_trace["generated_token_ids"][0]
                                == replay_trace["effective_resume_token_id"]
                            ),
                            "generated_tokens_match": native_tokens == replay_tokens,
                            "native_argmax_matches_tokens": (
                                np.argmax(native_logits, axis=1).astype(np.uint32).tolist()
                                == native_tokens
                            ),
                            "boundary_argmax_matches_resume": int(np.argmax(boundary_logits[0]))
                                == native_tokens[0],
                            "replay_argmax_matches_tokens_1_plus": (
                                np.argmax(replay_logits, axis=1).astype(np.uint32).tolist()
                                == native_tokens[1:]
                            ),
                        }
                        checks_pass = (
                            boundary_comparison["exact"]
                            and continuation_comparison["exact"]
                            and all(metadata_checks.values())
                        )
                        observation_result.update({
                            "exact": checks_pass,
                            "native_logits_file_sha256": native_file_sha256,
                            "boundary_logits_file_sha256": boundary_file_sha256,
                            "replay_logits_file_sha256": replay_file_sha256,
                            "boundary_comparison": boundary_comparison,
                            "continuation_comparison": continuation_comparison,
                            "metadata_checks": metadata_checks,
                            "generated_token_ids": native_tokens,
                            "phase_timing": phase_summary(native_trace, replay_trace),
                        })
                        if not checks_pass:
                            case_errors.append(f"observation {observation} exact gate failed")
                    except Exception as error:  # retain the rest of the matrix
                        observation_result.update({"exact": False, "comparison_error": repr(error)})
                        case_errors.append(f"observation {observation} comparison error: {error}")
                else:
                    observation_result["exact"] = False
                atomic_json(observation_dir / "comparison.json", observation_result)
                observations.append(observation_result)

            export_trace_summary = None
            if export_json.exists():
                try:
                    export_trace_summary = load_trace(export_json)
                except Exception as error:
                    case_errors.append(f"cannot summarize export metadata: {error}")
            case_pass = not case_errors and len(observations) == args.observations \
                and all(item.get("exact") for item in observations)
            case_summary = {
                "case_id": case_id,
                "family": family,
                "rung": rung,
                "architecture_argument": arch,
                "model_label": filename,
                "prompt_id": prompt_id,
                "language": language,
                "prompt_utf8_sha256": sha256_bytes(prompt.encode()),
                "status": "pass" if case_pass else "fail",
                "snapshot_hash": (
                    export_trace_summary.get("snapshot_hash")
                    if export_trace_summary else None
                ),
                "export_process_wall_ns": export_record["process_wall_ns"],
                "verify_process_wall_ns": verify_record.get("process_wall_ns"),
                "export_phase_timing": (
                    export_trace_summary.get("timings_ms")
                    if export_trace_summary else None
                ),
                "observations": observations,
                "errors": case_errors,
            }
            summary["cases"].append(case_summary)
            summary["completed_cases"] += 1
            if case_pass:
                summary["passed_cases"] += 1
            else:
                summary["failed_cases"] += 1
                summary["failures"].append({"case_id": case_id, "errors": case_errors})
                if args.fail_fast:
                    stop = True
            atomic_json(output / "commands.json", commands)
            atomic_json(output / "benchmark_summary.json", summary)
            if stop:
                break

    identities_unchanged = True
    for key, record in input_records.items():
        family, rung = key.split("/")
        try:
            after = file_identity(model_paths[(family, rung)])
            record["identity_after"] = after
            record["identity_unchanged"] = after == record["identity_before"]
        except OSError as error:
            record["identity_after_error"] = repr(error)
            record["identity_unchanged"] = False
        identities_unchanged &= record["identity_unchanged"]
    for family, record in tokenizer_records.items():
        try:
            after = file_identity(tokenizer_paths[family])
            record["identity_after"] = after
            record["identity_unchanged"] = after == record["identity_before"]
        except OSError as error:
            record["identity_after_error"] = repr(error)
            record["identity_unchanged"] = False
        identities_unchanged &= record["identity_unchanged"]
    complete = summary["completed_cases"] == expected_cases
    passed = complete and summary["failed_cases"] == 0 and identities_unchanged
    summary.update({
        "status": "complete-pass" if passed else ("complete-fail" if complete else "incomplete"),
        "inputs_unchanged": identities_unchanged,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commands_executed": len(commands),
        "raw_run_directory": relative_label(output),
        "timing_verdict": "observational-only-no-controlled-cold-warm-claim",
    })
    manifest["status"] = summary["status"]
    manifest["models"] = input_records
    manifest["tokenizers"] = tokenizer_records
    atomic_json(output / "commands.json", commands)
    atomic_json(output / "benchmark_manifest.json", manifest)
    atomic_json(output / "benchmark_summary.json", summary)

    if evidence_output is not None:
        require_fresh_directory(evidence_output)
        atomic_json(evidence_output / "benchmark_manifest.json", manifest)
        atomic_json(evidence_output / "benchmark_summary.json", summary)
        atomic_json(
            evidence_output / "commands.json",
            sanitize_commands(
                commands,
                binary=binary,
                model_paths=model_paths.values(),
                tokenizer_paths=tokenizer_paths.values(),
                raw_root=output,
            ),
        )

    print(json.dumps({
        "status": summary["status"],
        "passed_cases": summary["passed_cases"],
        "expected_cases": expected_cases,
        "raw": relative_label(output),
        "evidence": relative_label(evidence_output) if evidence_output else None,
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
