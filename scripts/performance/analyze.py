#!/usr/bin/env python3
"""Aggregate raw baseline JSON into summary tables (JSON + markdown)."""
import json, sys, statistics
from pathlib import Path
from collections import defaultdict

def median(xs):
    return statistics.median(xs) if xs else None

def load_all(raw_dir):
    out = {}
    for p in sorted(Path(raw_dir).glob("*.json")):
        if p.name == "_matrix.json":
            continue
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception:
            pass
    return out

def bench_summary(d):
    return {
        "median_ns": d.get("median_ns"),
        "median_tps": d.get("median_tokens_per_second"),
        "samples_ns": d.get("samples_ns"),
        "threads": d.get("threads"),
        "tokens": d.get("tokens"),
        "reps": d.get("repetitions"),
    }

def op_aggregate(d):
    """Aggregate operator_profile by operator name across all layers."""
    agg = defaultdict(lambda: {"total_ns": 0, "samples": 0, "medians": [], "dims": set()})
    for o in d.get("operator_profile", []) or []:
        a = agg[o["operator"]]
        a["total_ns"] += o["total_elapsed_ns"]
        a["samples"] += o["samples"]
        a["medians"].extend([o["median_elapsed_ns"]] * o["samples"])
        a["dims"].add((o["input_dimension"], o["output_dimension"]))
    tot = sum(a["total_ns"] for a in agg.values())
    rows = []
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["total_ns"]):
        rows.append({
            "operator": name,
            "share_pct": 100 * a["total_ns"] / tot if tot else 0,
            "total_ms": a["total_ns"] / 1e6,
            "median_us": median(a["medians"]) / 1e3 if a["medians"] else None,
            "samples": a["samples"],
            "dims": sorted(a["dims"])[:4],
        })
    return {"total_op_ms": tot / 1e6, "rows": rows}

def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/performance-baseline/2026-08-10/raw"
    data = load_all(raw_dir)
    report = {}

    # 1. decode throughput table
    decode_rows = []
    for name, d in data.items():
        if name.startswith("bench_") and "median_tokens_per_second" in d:
            parts = name.split("_")  # bench_<model>_<exec>_t<threads>
            decode_rows.append({
                "run": name, "model": parts[1], "exec": parts[2],
                "threads": d["threads"], "median_tps": d["median_tokens_per_second"],
                "median_ns": d["median_ns"],
            })
    decode_rows.sort(key=lambda r: (r["model"], r["exec"], r["threads"]))
    report["decode_throughput"] = decode_rows

    # 2. operator breakdowns
    report["operator_breakdowns"] = {}
    for name, d in data.items():
        if name.startswith("profile_"):
            report["operator_breakdowns"][name] = op_aggregate(d)

    # 3. allocations
    report["allocations"] = {}
    for name, d in data.items():
        if name.startswith("alloc_"):
            report["allocations"][name] = {
                "caller_events_per_token": d["allocation_report"]["caller_thread_alloc_events_per_token"],
                "caller_bytes_per_token": d["allocation_report"]["caller_thread_alloc_bytes_per_token"],
                "caller_median_events": d["allocation_report"]["caller_thread_alloc_events_median"],
                "caller_median_bytes": d["allocation_report"]["caller_thread_alloc_bytes_median"],
                "caller_max_events": d["allocation_report"]["caller_thread_alloc_events_max"],
                "global_events_per_token": d["allocation_report"]["global_alloc_events_per_token"],
                "global_bytes_per_token": d["allocation_report"]["global_alloc_bytes_per_token"],
                "per_token_bytes": d["allocation_report"]["per_token_alloc_bytes"][:8],
                "per_token_events": d["allocation_report"]["per_token_alloc_events"][:8],
            }

    # 4. generate observability
    report["generate"] = {}
    for name, d in data.items():
        if name.startswith("gen_"):
            report["generate"][name] = d
        if name.startswith("trace_"):
            report["generate"][name] = d

    print(json.dumps(report, indent=1, default=str))

if __name__ == "__main__":
    main()
