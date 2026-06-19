"""Compare probe results from two model runs.

Reads baseline_probe_summary.json, baseline_control_report.json,
and heldout_probe_results.json from two directories and produces
a Markdown comparison table + JSON summary.
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def get_heldout_lift(heldout, task, strategy="lemma-heldout"):
    """Extract probe-char lift for a task under a given heldout strategy."""
    if heldout is None:
        return None, None, None
    tr = heldout.get(task, {}).get("strategies", {}).get(strategy, {})
    if not tr or "probe_best_accuracy" not in tr:
        return None, None, None
    return (
        tr.get("probe_best_accuracy"),
        tr.get("char_ngram_accuracy"),
        tr.get("probe_minus_char"),
    )


def get_baseline_acc(baseline, task):
    """Extract best accuracy from baseline summary."""
    if baseline is None:
        return None, None
    tr = baseline.get("tasks", {}).get(task, {})
    return tr.get("best_accuracy"), tr.get("best_layer")


def get_token_stats(token_diag):
    """Extract token distribution stats."""
    if token_diag is None:
        return None, None
    td = token_diag.get("token_distribution", {})
    return td.get("single_token_pct"), td.get("mean_tokens")


def main():
    parser = argparse.ArgumentParser(description="compare two model probe runs")
    parser.add_argument("dir_a", help="first model result directory")
    parser.add_argument("dir_b", help="second model result directory")
    parser.add_argument("--label-a", default="Model A", help="label for first model")
    parser.add_argument("--label-b", default="Model B", help="label for second model")
    parser.add_argument("--output", default=None, help="output Markdown file (prints to stdout if omitted)")
    args = parser.parse_args()

    a = Path(args.dir_a)
    b = Path(args.dir_b)

    # Load results
    a_baseline = load_json(a / "baseline_probe_summary.json")
    b_baseline = load_json(b / "baseline_probe_summary.json")
    a_heldout = load_json(a / "heldout_probe_results.json")
    b_heldout = load_json(b / "heldout_probe_results.json")
    a_token = load_json(a / "token_diagnostics.json")
    b_token = load_json(b / "token_diagnostics.json")

    lines = []
    def w(s=""):
        lines.append(s)

    w(f"# Model Comparison: {args.label_a} vs {args.label_b}")
    w()
    w(f"**{args.label_a}**: `{args.dir_a}`")
    w(f"**{args.label_b}**: `{args.dir_b}`")
    w()

    # ── Heldout probe comparison ──
    w("## Heldout Probe Accuracy (probe / char / probe−char)")
    w()
    w(f"| Task | Strategy | {args.label_a} | {args.label_b} | Δ lift |")
    w("|------|----------|------------|------------|--------|")

    for task in ["pos", "features.gender", "features.number"]:
        for strategy in ["surface-heldout", "lemma-heldout", "root-heldout"]:
            pa, ca, la = get_heldout_lift(a_heldout, task, strategy)
            pb, cb, lb = get_heldout_lift(b_heldout, task, strategy)
            if pa is None and pb is None:
                continue
            a_str = f"{pa:.3f}/{ca:.3f}/{la:+.3f}" if pa is not None else "—"
            b_str = f"{pb:.3f}/{cb:.3f}/{lb:+.3f}" if pb is not None else "—"
            delta = ""
            if la is not None and lb is not None:
                delta = f"{lb - la:+.3f}"
            display_task = {"pos": "POS", "features.gender": "gender", "features.number": "number"}.get(task, task)
            w(f"| {display_task} | {strategy} | {a_str} | {b_str} | {delta} |")

    w()

    # ── Random CV vs heldout gap ──
    w("## Random CV → Heldout Gap")
    w()
    w(f"| Task | Strategy | {args.label_a} gap | {args.label_b} gap |")
    w("|------|----------|---------------|---------------|")

    for task in ["pos", "features.gender", "features.number"]:
        a_rand_acc, _ = get_baseline_acc(a_baseline, task)
        b_rand_acc, _ = get_baseline_acc(b_baseline, task)
        for strategy in ["lemma-heldout"]:
            pa, _, _ = get_heldout_lift(a_heldout, task, strategy)
            pb, _, _ = get_heldout_lift(b_heldout, task, strategy)
            if pa is None and pb is None:
                continue
            a_gap = f"{a_rand_acc - pa:+.3f}" if a_rand_acc and pa else "—"
            b_gap = f"{b_rand_acc - pb:+.3f}" if b_rand_acc and pb else "—"
            display_task = {"pos": "POS", "features.gender": "gender", "features.number": "number"}.get(task, task)
            w(f"| {display_task} | {strategy} | {a_gap} | {b_gap} |")

    w()

    # ── Tokenization comparison ──
    w("## Tokenization Statistics")
    w()
    a_st, a_mt = get_token_stats(a_token)
    b_st, b_mt = get_token_stats(b_token)
    w(f"| Metric | {args.label_a} | {args.label_b} |")
    w("|--------|------------|------------|")
    w(f"| % single-token | {a_st}% | {b_st}% |" if a_st else f"| % single-token | — | — |")
    w(f"| Mean tokens/word | {a_mt} | {b_mt} |" if a_mt else f"| Mean tokens/word | — | — |")

    w()

    # ── Baseline accuracy comparison ──
    w("## Baseline (Random CV) Comparison")
    w()
    w(f"| Task | {args.label_a} best L | {args.label_a} acc | {args.label_b} best L | {args.label_b} acc |")
    w("|------|-------------------|---------------|-------------------|---------------|")
    for task in ["root", "lemma", "pos", "abstract_pattern", "concrete_pattern", "features.gender", "features.number"]:
        aa, al = get_baseline_acc(a_baseline, task)
        ba, bl = get_baseline_acc(b_baseline, task)
        display = {"pos": "POS", "abstract_pattern": "abs pat", "concrete_pattern": "conc pat",
                   "features.gender": "gender", "features.number": "number"}.get(task, task)
        a_str = f"L{al} {aa:.3f}" if aa else "—"
        b_str = f"L{bl} {ba:.3f}" if ba else "—"
        w(f"| {display} | {a_str} | {b_str} |")

    w()
    output = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(f"Saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
