#!/usr/bin/env python3
"""Phase V Step 2 analysis: per-bit x d-magnitude fault outcomes."""
import json
from collections import defaultdict
import statistics as st

rows = [json.loads(l) for l in open("/tmp/opencode/phase5/sweep.jsonl")]

GROUPS = {
    "Q4_K": {"small": [1.14441e-5, 3e-5, 5.4121e-5, 7.1228e-5],
             "median": [8.738e-5, 9.8646e-5, 1.15514e-4, 1.7941e-4],
             "large": [2.63929e-4, 5e-4, 1.93119e-3],
             "control": [1.0]},
    "Q6_K": {"small": [-2.915859e-4, -2.98023e-5, -2.11e-5, -1.42455e-5],
             "median": [-8.16584e-6, 8.16584e-6, 1.40071e-5, 2.0504e-5],
             "large": [2.95043e-5, 2.59876e-4, 1.0],
             "control": [-1.0]},
    # note: +1.0 control for Q6_K lives in "large" (sorted order); split it out
    "Q8_0": {"small": [2.76566e-5, 1.5378e-4, 2.0957e-4, 2.78711e-4],
             "median": [3.34501e-4, 4.07457e-4, 5.84602e-4, 8.3828e-4],
             "large": [2e-3, 5e-3, 9.41467e-3],
             "control": [1.0]},
}
# move Q6_K +1.0 into control for a clean real-vs-synthetic split
GROUPS["Q6_K"]["large"] = [2.95043e-5, 2.59876e-4]
GROUPS["Q6_K"]["control"] = [1.0, -1.0]


def group_of(dtype, d_req):
    for g, vs in GROUPS[dtype].items():
        if any(abs(d_req - v) / max(abs(v), 1e-30) < 1e-6 for v in vs):
            return g
    raise ValueError((dtype, d_req))


print("=== per (dtype, group, bit): n_nonfinite/n, finite rel_l2 median/max, top1 flips ===")
table = defaultdict(list)
for r in rows:
    table[(r["dtype"], group_of(r["dtype"], r["d_requested"]), r["bit"])].append(r)

print(f"{'dtype':5} {'group':7} {'bit':>3} {'nf/n':>7} | {'rel_med':>10} {'rel_max':>10} {'top1':>4}")
for dtype in ["Q4_K", "Q6_K", "Q8_0"]:
    for g in ["small", "median", "large", "control"]:
        for bit in range(16):
            rs = table[(dtype, g, bit)]
            nf = sum(1 for r in rs if not r["finite"])
            fin = [r for r in rs if r["finite"]]
            rels = sorted(float(r["rel_l2"]) for r in fin)
            med = st.median(rels) if rels else float("nan")
            mx = max(rels) if rels else float("nan")
            tf = sum(1 for r in fin if r["top1_flipped"])
            print(f"{dtype:5} {g:7} {bit:>3} {nf}/{len(rs)}   | {med:10.3e} {mx:10.3e} {tf:>4}")

print()
print("=== faulted-d value classes at real d (excluding controls) ===")
for r in rows:
    if group_of(r["dtype"], r["d_requested"]) == "control":
        continue
    if not r["finite"] or r["bit"] in (10, 11, 12, 13, 14, 15):
        pass
print("(see faulted_d column per row for exact post-fault scale)")
# distribution of faulted_d classes for exponent-bit flips at real d
cls = defaultdict(int)
for r in rows:
    if group_of(r["dtype"], r["d_requested"]) == "control":
        continue
    if r["bit"] >= 10:
        fd = r["faulted_d"]
        if fd in ("Inf", "-Inf", "NaN"):
            c = "nonfinite"
        else:
            c = "finite"
        cls[(r["dtype"], r["bit"], c)] += 1
for k in sorted(cls):
    print(k, cls[k])

print()
print("=== max finite rel_l2 per bit over REAL d only (all 3 dtypes pooled by bit) ===")
for bit in range(16):
    rels = [float(r["rel_l2"]) for r in rows
            if r["bit"] == bit and r["finite"]
            and group_of(r["dtype"], r["d_requested"]) != "control"]
    print(f"bit {bit:2}: n={len(rels)} med={st.median(rels):.3e} max={max(rels):.3e}")
