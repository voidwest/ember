#!/usr/bin/env python3
"""Generate research-ready tables from corpus.json + results/*.json.

Outputs Markdown + CSV into tables/:
  A. corpus_taxonomy
  B. baseline_vs_current (machine delta)
  C. comparators (llama-cpp, candle)
  D. coverage (semantic matrix)
  E. perf
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"


def load(name):
    return json.loads((HERE / name).read_text())


def write_md(name, title, header, rows):
    (OUT / f"{name}.md").write_text(f"# {title}\n\n| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n")
    with (OUT / f"{name}.md").open("a") as f:
        for r in rows:
            f.write("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |\n")


def write_csv(name, header, rows):
    with (OUT / f"{name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    corpus = load("corpus.json")
    cur = {r["case"]: r for r in load("results/ember-current.json")["results"]}
    base = {r["case"]: r for r in load("results/ember-baseline.json")["results"]}
    llama = {r["case"]: r for r in load("results/llama-cpp.json")["results"]}
    candle = {r["case"]: r for r in load("results/candle.json")["results"]}

    cases = corpus["cases"]

    # ---- A. corpus taxonomy -------------------------------------------
    h = ["id", "name", "input_type", "origin", "bug_class", "format_status",
         "comparability", "coverage", "size_bytes", "sha256"]
    rows = [[c["id"], c["name"], c["input_type"], c["origin"], c["bug_class"],
             c["format_status"], c["semantic_comparability"],
             "+".join(c["coverage"]), c["size_bytes"], c["sha256"][:12]]
            for c in cases]
    write_md("A_corpus_taxonomy", "Corpus taxonomy", h, rows)
    write_csv("A_corpus_taxonomy", h, rows)

    # ---- B. baseline vs current ---------------------------------------
    h = ["case", "name", "bug_class", "baseline", "current", "delta"]
    rows = []
    for c in cases:
        b = base[c["id"]]["outcome"]
        k = cur[c["id"]]["outcome"]
        delta = "same" if b == k else f"{b}->{k}"
        rows.append([c["id"], c["name"], c["bug_class"], b, k, delta])
    write_md("B_baseline_vs_current", "Ember baseline vs current outcomes", h, rows)
    write_csv("B_baseline_vs_current", h, rows)

    changed = [r for r in rows if r[5] != "same"]
    print(f"delta: {len(changed)} changed of {len(rows)}")
    for r in changed:
        print("  ", r[0], r[2], r[5])

    # summary counts
    print("baseline:", dict(Counter(r[3] for r in rows)))
    print("current :", dict(Counter(r[4] for r in rows)))

    # ---- C. comparators ------------------------------------------------
    h = ["case", "name", "bug_class", "ember_current", "llama_cpp", "llama_stderr_category", "candle"]
    rows = []
    for c in cases:
        if c["input_type"] == "TOKENIZER_JSON":
            continue
        l = llama[c["id"]]
        rows.append([c["id"], c["name"], c["bug_class"], cur[c["id"]]["outcome"],
                     l["outcome"], l.get("stderr_category") or "", candle[c["id"]]["outcome"]])
    write_md("C_comparators", "Comparator outcomes (GGUF cases)", h, rows)
    write_csv("C_comparators", h, rows)
    print("llama.cpp:", dict(Counter(r[4] for r in rows)))
    print("candle   :", dict(Counter(r[6] for r in rows)))

    # llama.cpp crash detail
    for c in cases:
        l = llama[c["id"]]
        if l["outcome"] in ("PROCESS_CRASH", "PANIC"):
            print(f"  llama.cpp crash: {c['id']} {c['name']} exit={l['exit_code']}")
            print("    stderr:", (l["stderr_tail"] or "").strip()[-160:])

    # ---- D. semantic coverage ------------------------------------------
    areas = ["header/count", "metadata", "strings/arrays", "tensor descriptors",
             "extent arithmetic", "quantization layout", "overlap/range",
             "architecture metadata", "model construction", "tokenizer JSON"]
    h = ["area", "cases", "current_reject", "current_accept", "baseline_failures"]
    rows = []
    for a in areas:
        ids = [c["id"] for c in cases if a in c["coverage"]]
        kr = sum(1 for i in ids if cur[i]["outcome"] == "STRUCTURED_REJECT")
        ka = sum(1 for i in ids if cur[i]["outcome"] == "ACCEPT")
        bf = sum(1 for i in ids if base[i]["outcome"] not in ("ACCEPT", "STRUCTURED_REJECT"))
        rows.append([a, len(ids), kr, ka, bf])
    write_md("D_coverage", "Semantic coverage of the hostile corpus", h, rows)
    write_csv("D_coverage", h, rows)

    # ---- F. findings cross-runtime (class-level figure table) ----------
    # Each row = one failure class with its representative corpus case(s);
    # outcomes are machine-read from the result files.
    classes = [
        ("semantic config panic", ["gguf-045", "gguf-046"], "E",
         "odd head_dim / 1-D linear weight reach asserts in model construction"),
        ("layout misinterpretation", ["gguf-025"], "D",
         "q4_k contiguous dim not 256-aligned: baseline eager-dequantizes; llama.cpp rejects by name"),
        ("resource amplification", ["gguf-042", "gguf-047"], "G",
         "metadata-driven multi-TB rope allocation (ctx u32::MAX; 16M x 4096 product)"),
        ("zero-dim crash", ["gguf-021"], "D",
         "zero tensor dimension: llama.cpp SIGFPE in gguf_init_from_file_impl"),
        ("empty metadata key", ["gguf-035"], "A",
         "llama.cpp GGML_ASSERT(!key.empty()) abort at gguf.cpp:132"),
        ("alignment zero", ["gguf-053"], "A",
         "candle div_ceil(0) panic; Ember and llama.cpp reject"),
    ]
    h = ["class", "cases", "bug_class", "baseline", "EmberSEC", "llama.cpp", "candle", "note"]
    rows = []
    for name, ids, bc, note in classes:
        def outcome(t, i):
            return cur[i]["outcome"] if t == "current" else base[i]["outcome"] if t == "baseline" else llama[i]["outcome"] if t == "llama" else candle[i]["outcome"]
        rows.append([
            name, "+".join(ids), bc,
            "/".join(outcome("baseline", i) for i in ids),
            "/".join(outcome("current", i) for i in ids),
            "/".join(outcome("llama", i) for i in ids),
            "/".join(outcome("candle", i) for i in ids),
            note,
        ])
    write_md("F_findings_cross_runtime", "Failure classes across runtimes", h, rows)
    write_csv("F_findings_cross_runtime", h, rows)

    # ---- G. summary outcome counts --------------------------------------
    h = ["target", "ACCEPT", "STRUCTURED_REJECT", "PANIC", "PROCESS_CRASH", "TIMEOUT", "NOT_COMPARABLE"]
    rows = []
    for t, resmap in (("ember-baseline", base), ("ember-current", cur), ("llama-cpp", llama), ("candle", candle)):
        c = Counter(r["outcome"] for r in resmap.values())
        rows.append([t] + [c.get(k, 0) for k in h[1:]])
    write_md("G_outcome_summary", "Outcome summary (62-case corpus)", h, rows)
    write_csv("G_outcome_summary", h, rows)

    # ---- E. perf --------------------------------------------------------
    h = ["case", "label", "target", "wall_ms_median", "peak_rss_kb_median", "file_size_bytes"]
    rows = []
    for fname, tgt in (("results/perf-current.json", "ember-current"),
                       ("results/perf-baseline.json", "ember-baseline")):
        perf = load(fname)
        for row in perf["rows"]:
            walls = sorted(r[0] for r in row["runs"])
            rss = sorted(r[1] for r in row["runs"] if r[1] is not None)
            rows.append([row["case"], row["label"], tgt,
                         round(walls[len(walls) // 2], 1),
                         rss[len(rss) // 2] if rss else None,
                         row["file_size_bytes"]])
    write_md("E_perf", "Rejection cost (wall ms / peak RSS)", h, rows)
    write_csv("E_perf", h, rows)
    for r in rows:
        print("  perf:", r[1], r[2], f"{r[3]}ms", r[4])


if __name__ == "__main__":
    main()
