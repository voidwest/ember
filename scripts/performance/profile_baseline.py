#!/usr/bin/env python3
"""Ember performance-baseline driver.

Runs the bench-decode matrix (models x executions x thread counts), the
generate-path observability matrix (baseline / capture / intervention), and
the trace-based generic-path op breakdown. Raw JSON lands in
artifacts/performance-baseline-<date>/raw/. Resumable: completed runs are
skipped (a run is complete when its output file exists and is non-empty).

Usage:
  python3 scripts/performance/profile_baseline.py [--only <model>] [--quick]
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, MODELS, bench_decode, generate, commit

OUT = REPO / "artifacts" / "performance-baseline" / time.strftime("%Y-%m-%d") / "raw"
CAPTURE_TOML = Path("/tmp/ember-perf-capture.toml")
ZERO = "8:attention"
_capture_counter = [0]

def write_capture_toml():
    """Rewrite the capture TOML with a fresh output dir (capture refuses to
    overwrite an existing artifact, so every run needs its own dir)."""
    _capture_counter[0] += 1
    out_dir = f"{REPO}/runs/perf-capture-{os.getpid()}-{_capture_counter[0]}"
    CAPTURE_TOML.write_text(f"""schema_version = 1
output_dir = "{out_dir}"
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
stages = ["after-attention", "after-mlp"]
phase = "decode"
token_positions = []
max_records = 512
omit_prompt_text = true
""")
    return str(CAPTURE_TOML)

def done(name):
    p = OUT / name
    return p.exists() and p.stat().st_size > 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="only this model key")
    ap.add_argument("--quick", action="store_true", help="fewer tokens/reps for a fast pass")
    args = ap.parse_args()
    write_capture_toml()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"commit: {commit()}\noutput: {OUT}")

    models = [args.only] if args.only else list(MODELS)
    tok = 64 if args.quick else 128
    reps = 3 if args.quick else 5
    warm = 1 if args.quick else 2
    # K-quant decode is 10-40x slower than Q8 on this host; keep its
    # matrix smaller so the whole baseline finishes in reasonable time.
    kquant_tok = 64
    kquant_reps = 3
    kquant_warm = 1
    meta = {"commit": commit(), "tokens": tok, "reps": reps, "warmups": warm,
            "prompt": "Arabic morphology prompt", "model_keys": models,
            "capture_toml": CAPTURE_TOML.read_text(), "zero_layer": ZERO}
    (OUT / "_matrix.json").write_text(json.dumps(meta, indent=1))

    # ---------- 1. bench-decode matrix ----------
    for mk in models:
        if mk == "llama-1b-q8":
            execs = ["reference"]           # Q8 fast path
        elif mk == "qwen-1.5b-q8":
            execs = ["reference"]           # generic path
        else:
            execs = ["reference", "planned", "planned-fused"]
        for ex in execs:
            if mk in ("llama-1b-q4km", "llama-1b-q6k"):
                threads = [1, 2, 4, 8]
                use_tok, use_reps, use_warm = kquant_tok, kquant_reps, kquant_warm
            else:
                threads = [1, 2, 4, 8] if args.quick else [1, 2, 3, 4, 6, 8]
                use_tok, use_reps, use_warm = tok, reps, warm
            for t in threads:
                name = f"bench_{mk}_{ex}_t{t}.json"
                if done(name):
                    continue
                print(f"[bench] {mk} {ex} threads={t} ...", flush=True)
                bench_decode(mk, OUT / name, tokens=use_tok, warmups=use_warm, reps=use_reps, threads=t, execution=ex)

    # ---------- 2. profile-operators (per-op breakdown) ----------
    for mk, ex, ts in [("llama-1b-q8", "reference", [1, 2, 4, 8]),
                       ("llama-1b-q4km", "planned", [1, 2, 4, 8])]:
        for t in ts:
            name = f"profile_{mk}_{ex}_t{t}.json"
            if done(name):
                continue
            print(f"[profile] {mk} {ex} threads={t} ...", flush=True)
            bench_decode(mk, OUT / name, tokens=64, warmups=1, reps=3, threads=t, execution=ex, profile=True)

    # ---------- 3. allocation accounting ----------
    for mk, ex, ts in [("llama-1b-q8", "reference", [8, 1]),
                       ("llama-1b-q4km", "planned", [8]),
                       ("qwen-1.5b-q8", "reference", [8])]:
        for t in ts:
            name = f"alloc_{mk}_{ex}_t{t}.json"
            if done(name):
                continue
            print(f"[alloc] {mk} {ex} threads={t} ...", flush=True)
            bench_decode(mk, OUT / name, tokens=64, warmups=1, reps=3, threads=t, execution=ex, allocations=True)

    # ---------- 4. generate-path observability ----------
    for mk, ts in [("llama-1b-q8", [1, 4, 8]), ("qwen-1.5b-q8", [8])]:
        for t in ts:
            for mode, extra in [("baseline", {}), ("capture", {"capture": write_capture_toml()}),
                                ("intervene", {"zero_layer": ZERO})]:
                name = f"gen_{mk}_{mode}_t{t}.json"
                if done(name):
                    continue
                print(f"[gen] {mk} {mode} threads={t} ...", flush=True)
                rc, out = generate(mk, None, n_tokens=64, threads=t, **extra)
                if rc != 0:
                    print(f"  !! generate FAILED ({rc}): {out[-500:]}")
                    continue
                # extract the --- benchmark --- block
                lines = out.splitlines()
                bench = [l for l in lines if l.startswith(("prefill:", "decode:"))]
                (OUT / name).write_text(json.dumps({
                    "model": mk, "mode": mode, "threads": t, "benchmark_lines": bench,
                }, indent=1))

    # ---------- 5. trace-based op breakdown (generic path) ----------
    for mk, t in [("llama-1b-q8", 8), ("qwen-1.5b-q8", 8)]:
        for mode, extra in [("baseline", {}), ("capture", {"capture": write_capture_toml()}),
                            ("intervene", {"zero_layer": ZERO})]:
            name = f"trace_{mk}_{mode}_t{t}.json"
            if done(name):
                continue
            print(f"[trace] {mk} {mode} threads={t} ...", flush=True)
            trace_file = OUT / f"trace_{mk}_{mode}_t{t}.trace.json"
            rc, out = generate(mk, None, n_tokens=64, threads=t, trace_out=str(trace_file), **extra)
            if rc != 0 or not trace_file.exists():
                print(f"  !! trace generate FAILED ({rc})")
                continue
            (OUT / name).write_text(json.dumps({"model": mk, "mode": mode, "threads": t,
                                                "trace_file": trace_file.name}, indent=1))

    print("\nmatrix complete")

if __name__ == "__main__":
    main()
