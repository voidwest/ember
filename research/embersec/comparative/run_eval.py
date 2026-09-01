#!/usr/bin/env python3
"""EmberSEC comparative evaluation runner.

Process-isolated execution of hostile-input corpus cases against any
configured target. Each case runs in its own subprocess with a fixed
timeout, stdout/stderr capture, exit-code capture, and peak RSS via
os.wait4. Child panics/crashes never kill the runner.

    python research/embersec/comparative/run_eval.py \
        --target ember-current --out results/ember-current.json
    python research/embersec/comparative/run_eval.py --target ember-baseline --out ...
    python research/embersec/comparative/run_eval.py --target llama-cpp --out ...
    python research/embersec/comparative/run_eval.py --target candle --out ...

Modes:
    --perf   measure wall time + peak RSS for the perf case set (3 runs each)
    --case ID  restrict to one case (repeatable)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "research" / "embersec" / "comparative"
DEFAULT_TIMEOUT = 30.0

OUTCOMES = [
    "ACCEPT", "STRUCTURED_REJECT", "PANIC", "PROCESS_CRASH", "TIMEOUT",
    "RESOURCE_LIMIT", "SEMANTIC_MISINTERPRETATION", "UNSUPPORTED",
    "HARNESS_ERROR",
]


def load_json(path):
    return json.loads(Path(path).read_text())


def run_with_rusage(cmd, env, timeout, cwd=None, out_path=None, err_path=None):
    """Run a command with timeout + per-child peak RSS (Linux wait4)."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as fo, \
         tempfile.NamedTemporaryFile(delete=False) as fe:
        out_name, err_name = fo.name, fe.name
    try:
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd, stdout=open(out_name, "wb"), stderr=open(err_name, "wb"),
            env=env, cwd=cwd)
        timed_out = False
        rusage = None
        while True:
            try:
                pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    rusage = ru
                    break
            except ChildProcessError:
                break
            if time.monotonic() - start > timeout:
                timed_out = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                os.wait4(proc.pid, 0)
                break
            time.sleep(0.01)
        wall_ms = (time.monotonic() - start) * 1000.0
        if rusage is None:
            try:
                _, _, rusage = os.wait4(proc.pid, 0)
            except ChildProcessError:
                pass
        exit_code = os.waitstatus_to_exitcode(status) if not timed_out else -9
        out = Path(out_name).read_bytes().decode(errors="replace")
        err = Path(err_name).read_bytes().decode(errors="replace")
        peak_rss_kb = rusage.ru_maxrss if rusage else None
        return {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "wall_ms": round(wall_ms, 1),
            "peak_rss_kb": peak_rss_kb,
            "stdout": out,
            "stderr": err,
        }
    finally:
        for n in (out_name, err_name):
            try:
                os.unlink(n)
            except OSError:
                pass


def classify_ember(res):
    if res["timed_out"]:
        return "TIMEOUT"
    code = res["exit_code"]
    if code == 0:
        return "ACCEPT"
    if code == 1:
        return "STRUCTURED_REJECT"
    if code == 101:
        return "PANIC"
    if code == -9:
        return "RESOURCE_LIMIT"
    if code < 0:
        return "PROCESS_CRASH"
    return "HARNESS_ERROR"


def classify_llamacpp(res):
    if res["timed_out"]:
        return "TIMEOUT"
    code = res["exit_code"]
    err = res["stderr"]
    if code == 0:
        return "ACCEPT"
    if code == 1:
        if "GGML_ASSERT" in err or "llama_assert" in err or "assert" in err.lower() and "error" in err.lower():
            return "PANIC"
        return "STRUCTURED_REJECT"
    if code in (101,):
        return "PANIC"
    if code == -9:
        return "RESOURCE_LIMIT"
    if code < 0:
        return "PROCESS_CRASH"
    return "HARNESS_ERROR"


def classify_candle(res):
    # Rust bin: 0 accept, 1 structured reject, 101 panic, -6 abort...
    return classify_ember(res)


def classify_llama_loader(res):
    # Loader harness: 0 = load+free OK, 1 = structured load error,
    # 101 = panic, negative signal = crash, timeout handled by caller.
    if res["timed_out"]:
        return "TIMEOUT"
    code = res["exit_code"]
    if code == 0:
        return "ACCEPT"
    if code == 1:
        return "STRUCTURED_REJECT"
    if code == 101:
        return "PANIC"
    if code < 0:
        return "PROCESS_CRASH"
    return "HARNESS_ERROR"


CLASSIFIERS = {
    "ember-current": classify_ember,
    "ember-baseline": classify_ember,
    "llama-cpp": classify_llamacpp,
    "llama-cpp-loader": classify_llama_loader,
    "candle": classify_candle,
}

STAGE_FOR_CASE = {}


def stage_for_case(case):
    if case["input_type"] == "TOKENIZER_JSON":
        return "tokenizer_check"
    cov = case.get("coverage", [])
    if "model construction" in cov or "architecture metadata" in cov:
        return "gguf_model_check"
    return "gguf_load_check"


def resolve_harness_binary(tgt):
    """Locate the compiled harness test binary in the target worktree."""
    if tgt.get("harness_binary"):
        return tgt["harness_binary"]
    hits = sorted(
        p for p in Path(tgt["worktree"]).glob("target/release/deps/_embersec_harness-*")
        if p.is_file() and not p.name.endswith(".d") and os.access(p, os.X_OK)
    )
    if not hits:
        sys.exit(f"harness binary not built in {tgt['worktree']} "
                 f"(run: cargo test --release --test _embersec_harness --no-run)")
    return str(hits[-1])


def build_command(target_cfg, case, fixture_abs):
    kind = target_cfg["kind"]
    if kind == "ember":
        stage = stage_for_case(case)
        binary = resolve_harness_binary(target_cfg)
        return [binary, stage, "--exact", "--nocapture"], {
            "EMBERSEC_FIXTURE": str(fixture_abs),
        }
    if kind == "llama-cpp":
        binary = target_cfg["binary"]
        args = target_cfg.get("args", [])
        cmd = [binary] + [a.replace("{fixture}", str(fixture_abs)) for a in args]
        return cmd, {}
    if kind == "llama-cpp-loader":
        return [target_cfg["binary"], str(fixture_abs)], {}
    if kind == "candle":
        binary = target_cfg["binary"]
        return [binary, str(fixture_abs)], {}
    raise ValueError(f"unknown target kind {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--perf-runs", type=int, default=3)
    args = ap.parse_args()

    envs = load_json(HERE / "environments.json")
    if args.target not in envs["targets"]:
        sys.exit(f"unknown target {args.target}; have {list(envs['targets'])}")
    tgt = envs["targets"][args.target]
    corpus = load_json(HERE / "corpus.json")

    if args.perf:
        run_perf(args, envs, tgt, corpus)
        return

    classifier = CLASSIFIERS[args.target]
    results = []
    # llama.cpp is exercised through the loader-harness kind below; it also
    # cannot consume tokenizer.json, so keep tokenizer-only inputs out of its
    # GGUF outcome totals just like the Candle parser target.
    skip_tokenizer = tgt["kind"] in ("llama-cpp", "llama-cpp-loader", "candle")
    for case in corpus["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        if skip_tokenizer and case["input_type"] == "TOKENIZER_JSON":
            results.append({
                "case": case["id"],
                "name": case["name"],
                "stage": "not-run",
                "exit_code": None,
                "timed_out": False,
                "outcome": "NOT_COMPARABLE",
                "wall_ms": 0.0,
                "peak_rss_kb": None,
                "stderr_category": "tokenizer-only-input",
                "stderr_tail": "",
            })
            print(f"{case['id']:10s} NOT_COMPARABLE (tokenizer-only)", flush=True)
            continue
        fixture_abs = HERE / case["fixture"]
        cmd, extra_env = build_command(tgt, case, fixture_abs)
        env = dict(os.environ)
        env.update(extra_env)
        res = run_with_rusage(cmd, env, args.timeout)
        outcome = classifier(res)
        # llama.cpp stderr category
        stderr_category = None
        if tgt["kind"] == "llama-cpp-loader":
            e = res["stderr"]
            if "GGML_ASSERT" in e or "fatal error" in e:
                stderr_category = "assert"
            elif "HARNESS: LOAD_OK" in e:
                stderr_category = "none"
            else:
                stderr_category = "load-error"
        elif tgt["kind"] == "llama-cpp":
            e = res["stderr"]
            if "not a multiple of block size" in e:
                stderr_category = "block-alignment-reject"
            elif "GGML_ASSERT" in e or "llama_assert" in e:
                stderr_category = "assert"
            elif "unknown model architecture" in e:
                stderr_category = "arch-dispatch-reject-parser-accepted"
            elif "data is not within the file bounds" in e:
                stderr_category = "bounds-reject"
            elif "key not found in model" in e:
                stderr_category = "missing-hparam"
            elif "failed to read header" in e:
                stderr_category = "header-reject"
            elif "failed to read tensor info" in e or "failed to read tensor data" in e:
                stderr_category = "tensor-info-reject"
            elif "out of memory" in e or "failed to allocate" in e or "allocation failed" in e:
                stderr_category = "allocation-failure"
            elif outcome == "ACCEPT":
                stderr_category = "none"
            else:
                stderr_category = "other"
        results.append({
            "case": case["id"],
            "name": case["name"],
            "stage": stage_for_case(case),
            "exit_code": res["exit_code"],
            "timed_out": res["timed_out"],
            "outcome": outcome,
            "wall_ms": res["wall_ms"],
            "peak_rss_kb": res["peak_rss_kb"],
            "stderr_category": stderr_category,
            "stderr_tail": res["stderr"][-400:],
        })
        print(f"{case['id']:10s} {outcome:24s} exit={res['exit_code']:>4} "
              f"rss={res['peak_rss_kb']}KB wall={res['wall_ms']:.0f}ms", flush=True)

    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "target": args.target,
        "commit": tgt.get("commit"),
        "cases_run": len(results),
        "results": results,
    }, indent=2) + "\n")
    from collections import Counter
    print("summary:", dict(Counter(r["outcome"] for r in results)))


PERF_CASES = [
    # (id, label, stage)
    ("gguf-041", "valid-tiny-llama", "gguf_model_check"),
    ("gguf-010", "bad-magic-early-reject", "gguf_load_check"),
    ("gguf-042", "hostile-context-late-reject", "gguf_model_check"),
]


def run_perf(args, envs, tgt, corpus):
    """Wall + peak RSS for small fixtures and (optionally) a real model."""
    by_id = {c["id"]: c for c in corpus["cases"]}
    rows = []
    for cid, label, stage in PERF_CASES:
        case = by_id[cid]
        fixture_abs = HERE / case["fixture"]
        env = dict(os.environ)
        if tgt["kind"] == "ember":
            env["EMBERSEC_FIXTURE"] = str(fixture_abs)
            cmd = [resolve_harness_binary(tgt), stage, "--exact", "--nocapture"]
        elif tgt["kind"] == "llama-cpp":
            cmd = [tgt["binary"]] + [a.replace("{fixture}", str(fixture_abs)) for a in tgt.get("args", [])]
        else:
            cmd = [tgt["binary"], str(fixture_abs)]
        samples = []
        for _ in range(args.perf_runs):
            res = run_with_rusage(cmd, env, args.timeout)
            samples.append((res["wall_ms"], res["peak_rss_kb"], res["exit_code"]))
        rows.append({
            "case": cid, "label": label, "file_size_bytes": case["size_bytes"],
            "runs": samples,
        })
        print(f"{label}: " + ", ".join(f"{w:.1f}ms/{r}KB" for w, r, _ in samples))

    real_model = os.environ.get("EMBERSEC_REAL_MODEL")
    if real_model and tgt["kind"] == "ember":
        import hashlib
        path = Path(real_model)
        digest = None
        digest_path = HERE / "results" / "real_model.sha256"
        if digest_path.exists():
            digest = digest_path.read_text().strip().split()[0]
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digest_path.parent.mkdir(parents=True, exist_ok=True)
            digest_path.write_text(f"{digest}  {path.name}\n")
        env = dict(os.environ)
        env["EMBERSEC_FIXTURE"] = str(path)
        samples = []
        for _ in range(args.perf_runs):
            res = run_with_rusage(
                [resolve_harness_binary(tgt), "gguf_model_check", "--exact", "--nocapture"],
                env, 300.0)
            samples.append((res["wall_ms"], res["peak_rss_kb"], res["exit_code"]))
        rows.append({
            "case": "real-model", "label": path.name,
            "file_size_bytes": path.stat().st_size, "sha256": digest, "runs": samples,
        })
        print("real model:", ", ".join(f"{w:.0f}ms/{r}KB" for w, r, _ in samples))

    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "target": args.target,
        "commit": tgt.get("commit"),
        "note": "wall ms / peak RSS KB per run; 3 runs",
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
