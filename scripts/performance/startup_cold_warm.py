#!/usr/bin/env python3
"""Cold vs warm startup measurement.

Cold = model file evicted from page cache via posix_fadvise(DONTNEED).
Warm = file freshly read (page cache populated).

For each model: 3 cold runs, then 3 warm runs. Each run is
`/usr/bin/time -v ember bench-decode --tokens 8 ...` (decode kept tiny so
load dominates). Also runs `bench-lifecycle control --timing-only` for
phase-level timing (model init / tokenizer init / prefill / decode).

Output: artifacts/performance-baseline/<date>/startup.json
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, BIN, MODELS, TOKENIZER, commit

OUT = REPO / "artifacts" / "performance-baseline" / time.strftime("%Y-%m-%d")
OUT.mkdir(parents=True, exist_ok=True)

def evict(path):
    """Drop the file's pages from the page cache (no root required)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

def time_v(cmd, env=None, timeout=1800):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(["/usr/bin/time", "-v"] + cmd, cwd=REPO, capture_output=True,
                       text=True, env=e, timeout=timeout)
    stderr = p.stderr
    fields = {}
    for pat, key in [
        (r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", "wall"),
        (r"User time \(seconds\): ([\d.]+)", "user"),
        (r"System time \(seconds\): ([\d.]+)", "sys"),
        (r"Percent of CPU this job got: ([\d]+)%", "cpu_pct"),
        (r"Maximum resident set size \(kbytes\): ([\d]+)", "max_rss_kb"),
        (r"Minor \(reclaiming a frame\) page faults: ([\d]+)", "minor_faults"),
        (r"Major \(requiring I/O\) page faults: ([\d]+)", "major_faults"),
        (r"Voluntary context switches: ([\d]+)", "vol_ctx"),
        (r"Involuntary context switches: ([\d]+)", "invol_ctx"),
        (r"File system inputs: ([\d]+)", "fs_inputs"),
        (r"File system outputs: ([\d]+)", "fs_outputs"),
    ]:
        m = re.search(pat, stderr)
        if m:
            fields[key] = m.group(1)
    return fields, p.returncode, p.stdout[:200], stderr[:600]

def run_trials(model_key, trials=3):
    m = MODELS[model_key]
    results = {"cold": [], "warm": []}
    cmd = [str(BIN), "bench-decode", "--model", m["path"], "--arch", m["arch"],
           "--tokens", "8", "--warmups", "1", "--repetitions", "3", "--execution", "reference"]
    for label, evict_first in [("cold", True), ("warm", False)]:
        for i in range(trials):
            if evict_first:
                evict(m["path"])
                evict(TOKENIZER)
                time.sleep(0.2)
            f, rc, out, err = time_v(cmd)
            results[label].append({"trial": i, **f, "rc": rc})
            print(f"  {model_key} {label} trial {i}: wall={f.get('wall')} maj={f.get('major_faults')} min={f.get('minor_faults')} rss={f.get('max_rss_kb')}")
    return results

def lifecycle(model_key, trials=2):
    m = MODELS[model_key]
    results = []
    for i in range(trials):
        cmd = [str(BIN), "bench-lifecycle", "--model", m["path"], "--tokenizer", str(TOKENIZER),
               "--lifecycle", "control", "-n", "8", "--timing-only"]
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800,
                           env=dict(os.environ, RAYON_NUM_THREADS="8"))
        # parse the printed JSON-ish report
        try:
            start = p.stdout.index("{")
            data = json.loads(p.stdout[start:])
            results.append(data)
        except Exception as ex:
            print(f"  !! lifecycle parse failed: {ex}\n{p.stdout[-800:]}\n{p.stderr[-400:]}")
    return results

def main():
    models = sys.argv[1:] or ["llama-1b-q8", "qwen-1.5b-q8"]
    report = {"commit": commit(), "cpu": "11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz",
              "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for mk in models:
        print(f"[startup] {mk}")
        report[mk] = {"time_v": run_trials(mk)}
        if mk == "llama-1b-q8":
            print(f"[lifecycle] {mk}")
            report[mk]["lifecycle"] = lifecycle(mk)
    (OUT / "startup.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "startup.json")

if __name__ == "__main__":
    main()
