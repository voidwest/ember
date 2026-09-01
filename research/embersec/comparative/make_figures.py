#!/usr/bin/env python3
"""Generate the EmberSEC result figures (PNG) from the frozen results.

    python research/embersec/comparative/make_figures.py

Figures:
  fig1_outcomes.png    outcome distribution per runtime (62-case corpus)
  fig2_delta.png       baseline vs current per-class delta (failure -> reject)
  fig3_diff_fuzz.png   mutation-run failure rates (10k differential fuzz)
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

LABELS = ["ACCEPT", "STRUCTURED_REJECT", "PANIC", "PROCESS_CRASH", "TIMEOUT",
          "NOT_COMPARABLE"]
COLORS = ["#4caf50", "#2196f3", "#ff9800", "#f44336", "#9e9e9e", "#bdbdbd"]
SHORT = {"ember-baseline": "Ember baseline", "ember-current": "EmberSEC",
         "llama-cpp": "llama.cpp b7999", "candle": "Candle 0.11"}


def outcome_counts(results):
    c = Counter(r["outcome"] for r in results["results"])
    return [c.get(k, 0) for k in LABELS]


def main():
    corpus = json.loads((HERE / "corpus.json").read_text())
    results = {}
    for name in ("ember-baseline", "ember-current", "llama-cpp", "candle"):
        results[name] = json.loads((HERE / "results" / f"{name}.json").read_text())

    # fig1: outcome distribution per runtime
    targets = ["ember-baseline", "ember-current", "llama-cpp", "candle"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(targets))
    bottom = [0] * len(targets)
    for label, color in zip(LABELS, COLORS):
        vals = [outcome_counts(results[t])[LABELS.index(label)] for t in targets]
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([SHORT[t] for t in targets], fontsize=9)
    ax.set_ylabel("cases (62-case corpus)")
    ax.set_title("Hostile-input corpus outcomes per runtime")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_outcomes.png", dpi=150)
    plt.close(fig)

    # fig2: baseline vs current failure classes (delta)
    classes = [
        ("semantic config\npanic (E)", 2),      # gguf-045/046
        ("layout\nmisinterp (D)", 1),           # gguf-025
        ("resource\namplification (G)", 2),     # gguf-042/047
        ("tokenizer\nupstream panic (F)", 3),   # tok-002/003/004
    ]
    baseline_fail = [2, 1, 2, 3]
    current_fail = [0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xx = range(len(classes))
    ax.bar([i - 0.18 for i in xx], baseline_fail, width=0.36, label="baseline (failures)", color="#f44336")
    ax.bar([i + 0.18 for i in xx], current_fail, width=0.36, label="EmberSEC (failures)", color="#4caf50")
    ax.set_xticks(list(xx))
    ax.set_xticklabels([c for c, _ in classes], fontsize=8)
    ax.set_ylabel("corpus cases failing")
    ax.set_title("Failure classes: baseline vs EmberSEC (all converted to structured rejection)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 3.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_delta.png", dpi=150)
    plt.close(fig)

    # fig3: differential fuzz failure rates (10k raw mutations).  Use the
    # run-specific summary so a later construction campaign cannot silently
    # replace the data plotted here.
    summary = json.loads((HERE / "results" / "diff_fuzz" /
                          "summary_raw-10000-7.json").read_text())
    pt = summary["per_target"]
    rows = ["ember-baseline", "ember-current", "llama-cpp", "candle"]
    fail = [pt[t].get("PANIC", 0) + pt[t].get("PROCESS_CRASH", 0) + pt[t].get("TIMEOUT", 0)
            for t in rows]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(rows)), [f / summary["n_mutations"] * 100 for f in fail],
           color=["#ff9800", "#4caf50", "#f44336", "#ff9800"])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([SHORT[t] for t in rows], fontsize=9)
    ax.set_ylabel("% of 10,000 mutations failing")
    ax.set_title("Differential fuzzing: failure rate per runtime (10k raw mutations)")
    for i, (f, n) in enumerate(zip(fail, rows)):
        ax.text(i, f / summary["n_mutations"] * 100 + 0.05, f"{f}", ha="center", fontsize=9)
    ax.set_ylim(0, 3.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_diff_fuzz.png", dpi=150)
    plt.close(fig)
    print("figures written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
