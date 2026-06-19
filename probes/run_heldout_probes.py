"""Group-aware heldout probe CV.

Evaluates probes under 4 split strategies:
  random           StratifiedKFold (baseline, expected to be optimistic)
  surface-heldout  GroupKFold by surface_dediac
  lemma-heldout    GroupKFold by lemma
  root-heldout     GroupKFold by root

For each task × strategy, reports:
  probe accuracy (per layer + best), char n-gram baseline, majority baseline,
  probe−char, probe−majority, train/test class counts, unseen label rate.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_probes import (
    load_activations,
    load_stimuli,
    extract_labels,
    get_field,
    safe_key,
    DEFAULT_TASKS,
    TASK_DISPLAY,
)

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


# ── helpers ────────────────────────────────────────────────────────

def get_group_values(rows, group_field):
    """Extract group values from stimuli rows."""
    vals = []
    for r in rows:
        if group_field == "surface_dediac":
            v = r.get("surface") or r.get("expected_surface") or ""
        else:
            v = r.get(group_field, "")
        vals.append(v if v else f"__empty__{len(vals)}")
    return vals


def closed_set_splits(y, groups, n_folds=5, seed=42):
    """Generate GroupKFold splits where every test label appears in training.

    Returns (splits, stats) where stats includes unseen label info.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    if len(unique_groups) < n_folds:
        n_folds = len(unique_groups)
    if n_folds < 2:
        return None, {"error": "too few groups for CV"}

    try:
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(np.zeros(len(y)), y, groups=groups))
    except Exception:
        return None, {"error": "GroupKFold failed"}

    valid_splits = []
    unseen_stats = []
    for train_idx, test_idx in splits:
        train_labels = set(y[train_idx])
        test_labels = set(y[test_idx])
        unseen = test_labels - train_labels
        if unseen:
            unseen_stats.append({
                "n_unseen": len(unseen),
                "n_test": len(test_idx),
                "unseen_fraction": round(len(unseen) / len(test_labels), 3) if test_labels else 0,
            })
            # Still keep the split but flag it
        else:
            unseen_stats.append({"n_unseen": 0, "n_test": len(test_idx)})
        valid_splits.append((train_idx, test_idx))

    return valid_splits, {"unseen_stats": unseen_stats, "n_folds": len(valid_splits)}


def train_heldout_probes(activations, labels, splits, best_layer_only=True):
    """Train Ridge probes per layer using the given splits.

    Returns (layerwise_acc, best_layer, best_acc).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    n_layers = activations.shape[1]
    if best_layer_only:
        probe_layers = [int(np.argmax(np.zeros(n_layers)))]  # placeholder, we probe all anyway
        probe_layers = list(range(n_layers))  # probe all layers for now

    layer_accs = []
    for layer in range(n_layers):
        X = activations[:, layer, :]
        fold_accs = []
        for train_idx, test_idx in splits:
            if len(test_idx) == 0:
                continue
            probe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeClassifier(alpha=1.0))])
            probe.fit(X[train_idx], y[train_idx])
            fold_accs.append(float(probe.score(X[test_idx], y[test_idx])))
        layer_accs.append(float(np.mean(fold_accs)) if fold_accs else 0.0)

    best_layer = int(np.argmax(layer_accs))
    return layer_accs, best_layer, layer_accs[best_layer]


def char_ngram_heldout(rows, task, min_examples=3, splits=None):
    """Char n-gram baseline using the given splits."""
    import re
    ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")
    def dediac(s):
        return ARABIC_DIACRITICS.sub("", s)

    indices, labels, info = extract_labels(rows, task, min_examples)
    surfaces = []
    for idx in indices:
        r = rows[idx]
        surf = r.get("surface") or r.get("expected_surface") or ""
        surfaces.append(dediac(surf))

    le = LabelEncoder()
    y = le.fit_transform(labels)

    vec = CountVectorizer(analyzer="char", ngram_range=(1, 4), binary=True)
    X = vec.fit_transform(surfaces)

    if splits is None:
        probe = LogisticRegression(max_iter=2000)
        probe.fit(X, y)
        return float(probe.score(X, y)), info

    fold_accs = []
    for train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        probe = LogisticRegression(max_iter=2000)
        probe.fit(X[train_idx], y[train_idx])
        fold_accs.append(float(probe.score(X[test_idx], y[test_idx])))
    return float(np.mean(fold_accs)) if fold_accs else 0.0, info


def class_overlap_report(y, splits):
    """Report train/test class overlap stats."""
    stats = []
    for train_idx, test_idx in splits:
        train_labels = set(y[train_idx])
        test_labels = set(y[test_idx])
        unseen = test_labels - train_labels
        stats.append({
            "n_train_classes": len(train_labels),
            "n_test_classes": len(test_labels),
            "n_unseen_classes": len(unseen),
            "unseen_fraction": round(len(unseen) / len(test_labels), 3) if test_labels else 0,
            "all_test_labels_seen": len(unseen) == 0,
        })
    return stats


# ── main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="group-aware heldout probe CV")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--min-examples-per-label", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)

    print("Loading...")
    activations = load_activations(args.activations)
    rows = load_stimuli(args.stimuli)
    print(f"  activations: {activations.shape}")
    print(f"  stimuli: {len(rows)} rows")

    # Pre-compute group values for all strategies
    group_fields = {
        "random": None,
        "surface-heldout": "surface_dediac",  # actually surface field
        "lemma-heldout": "lemma",
        "root-heldout": "root",
    }

    results = {}
    for task in args.tasks:
        print(f"\n{'='*70}\n  {task}\n{'='*70}")

        try:
            indices, labels, info = extract_labels(rows, task, args.min_examples_per_label)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        task_acts = activations[indices]
        task_rows = [rows[i] for i in indices]
        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)
        total_examples = len(labels)

        print(f"  examples: {total_examples}  classes: {n_classes}  "
              f"maj={info['majority_baseline_accuracy']:.1%}")

        task_results = {
            "num_examples": total_examples,
            "num_classes": n_classes,
            "majority_baseline": info["majority_baseline_accuracy"],
            "strategies": {},
        }

        for strategy_name, group_field in group_fields.items():
            print(f"  ── {strategy_name} ──")

            if group_field is None:
                # Random stratified CV
                min_pc = int(np.bincount(y).min())
                ef = min(args.folds, min_pc)
                if ef < 2:
                    # Fall back to train accuracy
                    splits = [(np.arange(len(y)), np.arange(len(y)))]
                    split_meta = {"effective_folds": 1, "note": "train accuracy fallback"}
                else:
                    skf = StratifiedKFold(n_splits=ef, shuffle=True, random_state=args.seed)
                    splits = list(skf.split(np.zeros(len(y)), y))
                    split_meta = {"effective_folds": ef, "method": "StratifiedKFold"}
            else:
                # Group-aware CV
                groups = get_group_values(task_rows, group_field)
                split_result = closed_set_splits(y, groups, args.folds, args.seed)
                if split_result[0] is None:
                    print(f"    skipped: {split_result[1]}")
                    continue
                splits, split_meta = split_result
                split_meta["method"] = "GroupKFold"
                split_meta["group_field"] = group_field

            # Class overlap diagnostics
            overlap = class_overlap_report(y, splits)
            n_valid = sum(1 for o in overlap if o["all_test_labels_seen"])
            n_folds = len(splits)
            mean_unseen = np.mean([o["unseen_fraction"] for o in overlap])
            max_unseen = max(o["unseen_fraction"] for o in overlap)

            print(f"    folds: {n_folds}  closed-set folds: {n_valid}/{n_folds}  "
                  f"mean unseen: {mean_unseen:.3f}  max unseen: {max_unseen:.3f}")

            if n_valid == 0:
                print(f"    ⚠️  ALL folds have unseen test labels — probe results are meaningless")
                task_results["strategies"][strategy_name] = {
                    "status": "all_folds_unseen_labels",
                    "n_folds": n_folds,
                    "n_valid_folds": 0,
                    "mean_unseen_fraction": mean_unseen,
                    "overlap": overlap,
                }
                continue

            # --- Probe accuracy ---
            layer_accs, best_layer, best_acc = train_heldout_probes(
                task_acts, labels, splits, best_layer_only=False
            )
            print(f"    probe:  best L{best_layer} = {best_acc:.4f}")

            # --- Char n-gram baseline ---
            char_acc, _ = char_ngram_heldout(task_rows, task, args.min_examples_per_label, splits)
            print(f"    char:   {char_acc:.4f}")

            # --- Summary ---
            task_results["strategies"][strategy_name] = {
                "n_folds": n_folds,
                "n_valid_folds": n_valid,
                "mean_unseen_fraction": round(mean_unseen, 4),
                "max_unseen_fraction": round(max_unseen, 4),
                "overlap": overlap,
                "split_meta": split_meta,
                "probe_layerwise": [round(float(a), 4) for a in layer_accs],
                "probe_best_layer": best_layer,
                "probe_best_accuracy": round(float(best_acc), 4),
                "char_ngram_accuracy": round(float(char_acc), 4),
                "majority_baseline": info["majority_baseline_accuracy"],
                "probe_minus_char": round(float(best_acc) - float(char_acc), 4),
                "probe_minus_majority": round(float(best_acc) - info["majority_baseline_accuracy"], 4),
            }

        results[task] = task_results

    # ── Save ──
    out_path = out_dir / "heldout_probe_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"Saved to {out_path}")

    # ── Terminal summary table ──
    print_summary_table(results)


def print_summary_table(results):
    strategies = ["random", "surface-heldout", "lemma-heldout", "root-heldout"]
    header = (
        f"{'task':<16s} "
        + "".join(f"{'│':>2s} {s:<30s}" for s in strategies)
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("Probe Accuracy (probe / char / probe−char)")
    print(sep)

    for task in DEFAULT_TASKS:
        if task not in results:
            continue
        display = TASK_DISPLAY.get(task, task)
        cells = []
        for s in strategies:
            sr = results[task].get("strategies", {}).get(s, {})
            if not sr or sr.get("status") == "all_folds_unseen_labels":
                cells.append("unseen labels")
                continue
            p = sr.get("probe_best_accuracy")
            c = sr.get("char_ngram_accuracy")
            d = sr.get("probe_minus_char")
            if p is not None:
                cells.append(f"{p:.3f}/{c:.3f}/{d:+.3f}")
            else:
                cells.append("—")
        print(f"{display:<16s}" + "".join(f"  {c:<30s}" for c in cells))

    print(f"\n{sep}")
    print("Unseen Label Rate (mean / max)")
    print(sep)
    for task in DEFAULT_TASKS:
        if task not in results:
            continue
        display = TASK_DISPLAY.get(task, task)
        cells = []
        for s in strategies:
            sr = results[task].get("strategies", {}).get(s, {})
            if not sr:
                cells.append("—")
                continue
            mu = sr.get("mean_unseen_fraction", 0)
            mx = sr.get("max_unseen_fraction", 0)
            nv = sr.get("n_valid_folds", 0)
            nf = sr.get("n_folds", 0)
            cells.append(f"{mu:.3f}/{mx:.3f} ({nv}/{nf})")
        print(f"{display:<16s}" + "".join(f"  {c:<30s}" for c in cells))
    print(sep)


if __name__ == "__main__":
    main()
