"""Control / baseline analysis for probe runs.

Extends run_baseline_probes with:
  - Descriptive label statistics (entropy, min/median/max per class)
  - Shuffled-label probe control (Hewitt & Liang 2019)
  - Multi-seed stability (mean ± std across seeds)
  - Selectivity scoring
  - Char n-gram surface baseline (optional)
"""

import argparse
import json
import os
import sys
import re
import math
from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# reuse from baseline script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_probes import (
    load_activations,
    load_stimuli,
    extract_labels,
    train_layer_probes,
    make_probe,
    safe_key,
    get_field,
    DEFAULT_TASKS,
    TASK_DISPLAY,
)

ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")


def dediac(s: str) -> str:
    return ARABIC_DIACRITICS.sub("", s)


# ── descriptive statistics ────────────────────────────────────────


def label_entropy(class_counts: dict) -> float:
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in class_counts.values():
        if cnt > 0:
            p = cnt / total
            ent -= p * math.log2(p)
    n = len(class_counts)
    if n <= 1:
        return 0.0
    return ent / math.log2(n)  # normalized


def descriptive_stats(rows, task, min_examples_per_label=3):
    """Compute full descriptive stats for a task's label distribution."""
    indices, labels, info = extract_labels(rows, task, min_examples_per_label)
    cc = info["class_counts"]
    counts = list(cc.values())
    info["min_examples_per_class"] = int(min(counts))
    info["median_examples_per_class"] = float(np.median(counts))
    info["max_examples_per_class"] = int(max(counts))
    info["label_entropy"] = round(label_entropy(cc), 4)
    return indices, labels, info


# ── shuffled-label control ────────────────────────────────────────


def _control_probe():
    """Fast closed-form probe for control/stability analysis."""
    return Pipeline([
        ("standardscaler", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=1.0)),
    ])


def train_control_probes(
    activations, labels, n_folds=5, n_shuffles=5, seed=42, layer_stride=4,
):
    """Train probes on shuffled labels using fast RidgeClassifier.

    layer_stride: only probe every Nth layer (control accuracy is ~chance everywhere).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    min_per_class = int(np.bincount(y).min())
    effective_folds = min(n_folds, min_per_class)

    splits = None
    if effective_folds >= 2:
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(y)), y))

    n_layers = activations.shape[1]
    probe_layers = list(range(0, n_layers, layer_stride))
    all_acc = np.zeros((n_shuffles, len(probe_layers)))

    for shuffle_i in range(n_shuffles):
        rng = np.random.RandomState(seed * 31 + shuffle_i * 7 + 1)
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)

        for li, layer in enumerate(probe_layers):
            X = activations[:, layer, :]
            probe = _control_probe()

            if splits is None:
                probe.fit(X, y_shuffled)
                all_acc[shuffle_i, li] = float(probe.score(X, y_shuffled))
            else:
                scores = []
                for train_idx, test_idx in splits:
                    clone = _control_probe()
                    clone.fit(X[train_idx], y_shuffled[train_idx])
                    scores.append(clone.score(X[test_idx], y_shuffled[test_idx]))
                all_acc[shuffle_i, li] = float(np.mean(scores))

    return all_acc.mean(axis=0), all_acc.std(axis=0), probe_layers


# ── multi-seed stability ──────────────────────────────────────────


def train_multiseed_probes(
    activations, labels, seeds=(42, 123, 456, 789, 1024),
    n_folds=5, layer_stride=2,
):
    """Train Ridge probes across multiple seeds. Returns (mean_acc, std_acc, probe_layers)."""
    n_seeds = len(seeds)
    n_layers = activations.shape[1]
    probe_layers = list(range(0, n_layers, layer_stride))
    all_acc = np.zeros((n_seeds, len(probe_layers)))

    le = LabelEncoder()
    y = le.fit_transform(labels)
    min_per_class = int(np.bincount(y).min())
    effective_folds = min(n_folds, min_per_class)

    for i, seed in enumerate(seeds):
        splits = None
        if effective_folds >= 2:
            skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
            splits = list(skf.split(np.zeros(len(y)), y))

        for li, layer in enumerate(probe_layers):
            X = activations[:, layer, :]
            probe = _control_probe()

            if splits is None:
                probe.fit(X, y)
                all_acc[i, li] = float(probe.score(X, y))
            else:
                scores = []
                for train_idx, test_idx in splits:
                    clone = _control_probe()
                    clone.fit(X[train_idx], y[train_idx])
                    scores.append(clone.score(X[test_idx], y[test_idx]))
                all_acc[i, li] = float(np.mean(scores))

    return all_acc.mean(axis=0), all_acc.std(axis=0), probe_layers


# ── selectivity ───────────────────────────────────────────────────


def selectivity(real_acc, control_acc, chance):
    """Hewitt & Liang (2019) selectivity: (real - control) / (1 - max(control, chance))."""
    denom = 1.0 - np.maximum(control_acc, chance)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    sel = (real_acc - control_acc) / denom
    return np.clip(sel, 0.0, 1.0)


# ── char n-gram surface baseline ──────────────────────────────────


def char_ngram_baseline(rows, task, min_examples_per_label=3, ngram_range=(1, 4), max_iter=2000, seed=42):
    """Train a char n-gram logistic regression on surface forms."""
    indices, labels, info = extract_labels(rows, task, min_examples_per_label)

    surfaces = []
    for idx in indices:
        row = rows[idx]
        surf = row.get("surface") or row.get("expected_surface") or ""
        surfaces.append(dediac(surf))

    le = LabelEncoder()
    y = le.fit_transform(labels)

    vec = CountVectorizer(analyzer="char", ngram_range=ngram_range, binary=True)
    X = vec.fit_transform(surfaces)

    # stratified CV
    min_per_class = int(np.bincount(y).min())
    effective_folds = min(5, min_per_class)
    scores = []
    if effective_folds >= 2:
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            clf = LogisticRegression(max_iter=max_iter)
            clf.fit(X[train_idx], y[train_idx])
            scores.append(clf.score(X[test_idx], y[test_idx]))
        acc = float(np.mean(scores))
    else:
        clf = LogisticRegression(max_iter=max_iter)
        clf.fit(X, y)
        acc = float(clf.score(X, y))

    return acc, info


def try_char_ngram_baselines(rows, tasks, min_examples_per_label=3, ngram_range=(1, 4), max_iter=2000, seed=42):
    """Run char n-gram baselines for all tasks."""
    results = {}
    for task in tasks:
        try:
            acc, info = char_ngram_baseline(
                rows, task, min_examples_per_label, ngram_range, max_iter, seed
            )
            results[task] = {
                "char_ngram_accuracy": round(acc, 4),
                "char_ngram_lift": round(acc - info["majority_baseline_accuracy"], 4),
                "num_examples": info["num_examples"],
                "num_classes": info["num_classes"],
                "majority_baseline": info["majority_baseline_accuracy"],
            }
            print(f"  {task:<18s} char-ngram acc={acc:.4f}  lift={acc - info['majority_baseline_accuracy']:+.4f}")
        except ValueError as e:
            print(f"  {task:<18s} SKIP: {e}")
    return results


# ── report printing ───────────────────────────────────────────────


def print_control_summary(report: dict):
    """Print a formatted terminal summary."""
    tasks = [t for t in DEFAULT_TASKS if t in report.get("tasks", {})]

    print()
    print("=" * 96)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 96)
    header = f"{'task':<16s} {'ex':>4s} {'cls':>3s} {'min':>4s} {'med':>4s} {'max':>4s} {'ent':>6s} {'maj%':>6s}"
    print(header)
    print("-" * len(header))
    for t in tasks:
        s = report["tasks"][t]["descriptive"]
        display = TASK_DISPLAY.get(t, t)
        print(
            f"{display:<16s} {s['num_examples']:>4d} {s['num_classes']:>3d} "
            f"{s['min_examples_per_class']:>4d} {s['median_examples_per_class']:>4.0f} "
            f"{s['max_examples_per_class']:>4d} {s['label_entropy']:>6.4f} "
            f"{s['majority_baseline_accuracy']*100:>5.1f}%"
        )

    print()
    print("=" * 96)
    print("PROBE PERFORMANCE (real vs control vs selectivity)")
    print("=" * 96)
    header = (
        f"{'task':<16s} {'best L':>6s} {'real':>7s} {'control':>7s} "
        f"{'±':>5s} {'select':>7s} {'lift':>7s} {'surface':>8s}"
    )
    print(header)
    print("-" * len(header))
    for t in tasks:
        display = TASK_DISPLAY.get(t, t)
        s = report["tasks"][t]
        best = s["best_layer"]
        real = s["best_accuracy"]
        ctrl = s["control_best_accuracy"]
        ctrl_std = s["control_best_std"]
        sel = s["best_selectivity"]
        lift = s["best_accuracy_minus_majority"]
        surf = s.get("char_ngram_accuracy")
        surf_str = f"{surf:.4f}" if surf is not None else "  —"
        print(
            f"{display:<16s} {best:>5d}  {real:>6.4f}  {ctrl:>6.4f} "
            f"±{ctrl_std:.4f} {sel:>6.4f}  {lift:>6.4f}  {surf_str:>8s}"
        )

    print()
    print("=" * 96)
    print("MULTI-SEED STABILITY (mean ± std over 5 seeds)")
    print("=" * 96)
    header = f"{'task':<16s} {'best L':>6s} {'mean':>7s} {'±std':>6s} {'range':>8s}"
    print(header)
    print("-" * len(header))
    for t in tasks:
        display = TASK_DISPLAY.get(t, t)
        s = report["tasks"][t]["multiseed"]
        print(
            f"{display:<16s} {s['best_layer']:>5d}  {s['mean_best_accuracy']:>6.4f} "
            f"±{s['std_best_accuracy']:.4f} [{s['min_best_accuracy']:.4f}-{s['max_best_accuracy']:.4f}]"
        )

    print()
    print("=" * 96)
    print("SELECTIVITY METRICS (layerwise)")
    print("=" * 96)
    header = f"{'task':<16s} {'best L':>6s} {'best sel':>9s} {'mean sel':>9s} {'>0.5':>6s}"
    print(header)
    print("-" * len(header))
    for t in tasks:
        display = TASK_DISPLAY.get(t, t)
        s = report["tasks"][t]
        sel_arr = np.array([x for x in s["layerwise_selectivity"] if x is not None])
        best_sel = s["best_selectivity"]
        mean_sel = sel_arr.mean() if len(sel_arr) > 0 else float('nan')
        n_strong = int((sel_arr > 0.5).sum()) if len(sel_arr) > 0 else 0
        n_probed = len(sel_arr)
        print(f"{display:<16s} {s['best_selectivity_layer']:>5d}  {best_sel:>8.4f}  {mean_sel:>8.4f}  {n_strong:>3d}/{n_probed}")

    # char n-gram summary
    if any(report["tasks"].get(t, {}).get("char_ngram_accuracy") is not None for t in tasks):
        print()
        print("=" * 60)
        print("CHAR N-GRAM SURFACE BASELINE (1-4 grams)")
        print("=" * 60)
        header = f"{'task':<16s} {'surface':>8s} {'probe':>8s} {'probe-surf':>11s}"
        print(header)
        print("-" * len(header))
        for t in tasks:
            s = report["tasks"][t]
            surf = s.get("char_ngram_accuracy")
            if surf is None:
                continue
            display = TASK_DISPLAY.get(t, t)
            probe_best = s["best_accuracy"]
            diff = probe_best - surf
            print(f"{display:<16s} {surf:>8.4f} {probe_best:>8.4f} {diff:>+10.4f}")
    print()


# ── main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="control / baseline analysis for probes")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--min-examples-per-label", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--n-shuffles", type=int, default=5, help="shuffled-label repeats")
    parser.add_argument("--n-seeds", type=int, default=5, help="multi-seed repeats")
    parser.add_argument("--no-control", action="store_true", help="skip shuffled-label control")
    parser.add_argument("--no-multiseed", action="store_true", help="skip multi-seed probes")
    parser.add_argument("--no-surface", action="store_true", help="skip char n-gram baseline")
    parser.add_argument("--skip-real-probes", action="store_true",
                        help="skip real probe training (use existing summary)")
    parser.add_argument("--control-layer-stride", type=int, default=4,
                        help="only probe every Nth layer for shuffled-label control (default: 4)")
    parser.add_argument("--multiseed-layer-stride", type=int, default=2,
                        help="only probe every Nth layer for multi-seed stability (default: 2)")
    parser.add_argument("--control-folds", type=int, default=3,
                        help="CV folds for shuffled-label control (default: 3)")
    parser.add_argument("--multinomial-threshold", type=int, default=10,
                        help="use multinomial LR for tasks with >= this many classes (default: 10)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    activations = load_activations(args.activations)
    print(f"  activations: {activations.shape}")
    rows = load_stimuli(args.stimuli)
    print(f"  stimuli: {len(rows)} rows")

    multiseed_seeds = [args.seed + i * 100 for i in range(args.n_seeds)]

    report = {"config": vars(args), "tasks": {}}

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"  {task}")
        print(f"{'='*60}")

        # ── descriptive stats ──
        print("  descriptive stats...")
        try:
            indices, labels, info = descriptive_stats(
                rows, task, args.min_examples_per_label
            )
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        task_acts = activations[indices]
        chance = 1.0 / info["num_classes"]

        task_report = {
            "descriptive": info,
            "chance": chance,
        }
        print(f"    examples={info['num_examples']}  classes={info['num_classes']}  "
              f"entropy={info['label_entropy']:.4f}  maj={info['majority_baseline_accuracy']:.1%}")

        # ── real probes (or load existing) ──
        if args.skip_real_probes:
            existing_summary = out_dir / "baseline_probe_summary.json"
            if existing_summary.exists():
                existing = json.loads(existing_summary.read_text())
                et = existing["tasks"].get(task, {})
                task_report["layerwise_accuracy"] = et.get("layerwise_accuracy", [])
                task_report["best_layer"] = et.get("best_layer", 0)
                task_report["best_accuracy"] = et.get("best_accuracy", 0)
                task_report["best_accuracy_minus_majority"] = et.get("best_accuracy_minus_majority", 0)
                print(f"    loaded existing: best L{task_report['best_layer']} = {task_report['best_accuracy']:.4f}")
            else:
                print("    no existing summary, training real probes...")
                args.skip_real_probes = False

        if not args.skip_real_probes:
            print("  training real probes...")
            acc, probes, le, _ = train_layer_probes(
                task_acts, labels,
                n_folds=args.folds, max_iter=args.max_iter,
                solver=args.solver, tol=args.tol,
                n_jobs=args.n_jobs, seed=args.seed,
            )
            best_idx = int(np.argmax(acc))
            task_report["layerwise_accuracy"] = [float(a) for a in acc]
            task_report["best_layer"] = best_idx
            task_report["best_accuracy"] = float(acc[best_idx])
            task_report["best_accuracy_minus_majority"] = float(
                acc[best_idx] - info["majority_baseline_accuracy"]
            )
            print(f"    best L{best_idx} = {acc[best_idx]:.4f}  lift={task_report['best_accuracy_minus_majority']:.4f}")

        # ── shuffled-label control ──
        if not args.no_control:
            print(f"  shuffled-label control ({args.n_shuffles} shuffles)...")
            n_cls = info["num_classes"]
            ctrl_mean, ctrl_std, ctrl_layers = train_control_probes(
                task_acts, labels,
                n_folds=args.control_folds, n_shuffles=args.n_shuffles,
                seed=args.seed,
                layer_stride=args.control_layer_stride,
            )
            best_ctrl_idx = int(np.argmax(ctrl_mean))
            task_report["control_layerwise_mean"] = [float(x) for x in ctrl_mean]
            task_report["control_layerwise_std"] = [float(x) for x in ctrl_std]
            task_report["control_best_accuracy"] = float(ctrl_mean[best_ctrl_idx])
            task_report["control_best_std"] = float(ctrl_std[best_ctrl_idx])
            task_report["control_best_layer"] = ctrl_layers[best_ctrl_idx]

            # selectivity (computed at probed layers only)
            task_report["control_layers"] = ctrl_layers
            real_at_ctrl = np.array(task_report["layerwise_accuracy"])[ctrl_layers]
            sel = selectivity(real_at_ctrl, ctrl_mean, chance)
            best_sel_local = int(np.argmax(sel))
            best_sel_layer = ctrl_layers[best_sel_local]
            # interpolate selectivity to all layers for reporting
            sel_full = np.full(len(task_report["layerwise_accuracy"]), np.nan)
            for li, l in enumerate(ctrl_layers):
                sel_full[l] = sel[li]
            task_report["layerwise_selectivity"] = [float(x) if not np.isnan(x) else None for x in sel_full]
            task_report["best_selectivity"] = float(sel[best_sel_local])
            task_report["best_selectivity_layer"] = best_sel_layer
            print(f"    control best: L{ctrl_layers[best_ctrl_idx]} = {ctrl_mean[best_ctrl_idx]:.4f} ±{ctrl_std[best_ctrl_idx]:.4f}")
            print(f"    selectivity best: L{best_sel_layer} = {sel[best_sel_local]:.4f}")

        # ── multi-seed ──
        if not args.no_multiseed:
            print(f"  multi-seed probes ({args.n_seeds} seeds)...")
            n_cls = info["num_classes"]
            ms_mean, ms_std, ms_layers = train_multiseed_probes(
                task_acts, labels,
                seeds=multiseed_seeds,
                n_folds=args.folds,
                layer_stride=args.multiseed_layer_stride,
            )
            ms_best_idx = int(np.argmax(ms_mean))
            ms_best_layer = ms_layers[ms_best_idx]
            task_report["multiseed"] = {
                "mean_best_accuracy": float(ms_mean[ms_best_idx]),
                "std_best_accuracy": float(ms_std[ms_best_idx]),
                "best_layer": ms_best_layer,
                "probe_layers": ms_layers,
                "layerwise_mean": [float(x) for x in ms_mean],
                "layerwise_std": [float(x) for x in ms_std],
                "min_best_accuracy": float(ms_mean[ms_best_idx] - 2 * ms_std[ms_best_idx]),
                "max_best_accuracy": float(ms_mean[ms_best_idx] + 2 * ms_std[ms_best_idx]),
            }
            print(f"    mean best: L{ms_best_layer} = {ms_mean[ms_best_idx]:.4f} ±{ms_std[ms_best_idx]:.4f}")

        report["tasks"][task] = task_report

    # ── char n-gram surface baseline ──
    if not args.no_surface:
        print(f"\n{'='*60}")
        print("  char n-gram surface baseline (1-4 grams)")
        print(f"{'='*60}")
        for task in args.tasks:
            if task not in report["tasks"]:
                continue
            try:
                surf_acc, surf_info = char_ngram_baseline(
                    rows, task, args.min_examples_per_label, (1, 4), args.max_iter, args.seed
                )
                report["tasks"][task]["char_ngram_accuracy"] = round(surf_acc, 4)
                report["tasks"][task]["char_ngram_lift"] = round(
                    surf_acc - surf_info["majority_baseline_accuracy"], 4
                )
                print(f"  {task:<18s} acc={surf_acc:.4f}  lift={surf_acc - surf_info['majority_baseline_accuracy']:+.4f}")
            except ValueError as e:
                print(f"  {task:<18s} SKIP: {e}")

    # ── save ──
    report_path = out_dir / "baseline_control_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved control report to {report_path}")

    # ── print terminal summary ──
    print_control_summary(report)


if __name__ == "__main__":
    main()
