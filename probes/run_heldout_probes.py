"""Group-aware heldout probe CV.

Evaluates probes under 4 split strategies:
  random           StratifiedKFold (baseline, expected to be optimistic)
  surface-heldout  closed-set grouped CV by surface_dediac
  lemma-heldout    closed-set grouped CV by lemma
  root-heldout     closed-set grouped CV by root

For each task × strategy, reports:
  probe accuracy (per layer + best), char n-gram baseline, majority baseline,
  probe−char, probe−majority, train/test class counts, unseen label rate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .run_baseline_probes import (
        DEFAULT_TASKS,
        TASK_DISPLAY,
        extract_labels,
        get_field,
        load_activations,
        load_stimuli,
        safe_key,
    )
    from .train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        make_splits,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from run_baseline_probes import (
        DEFAULT_TASKS,
        TASK_DISPLAY,
        extract_labels,
        get_field,
        load_activations,
        load_stimuli,
        safe_key,
    )
    from train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        make_splits,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


# ── helpers ────────────────────────────────────────────────────────

def get_group_values(rows, group_field):
    """Extract group values from stimuli rows."""
    vals = []
    missing = []
    for index, r in enumerate(rows):
        if group_field == "surface_dediac":
            v = r.get("surface_dediac") or r.get("surface") or r.get("expected_surface")
        else:
            v = get_field(r, group_field)
        if not isinstance(v, (str, int, float)) or isinstance(v, bool) or not str(v).strip():
            missing.append(index)
        else:
            vals.append(str(v))
    if missing:
        raise ValueError(
            f"group field {group_field!r} is missing or non-scalar for "
            f"{len(missing)} row(s); first index {missing[0]}"
        )
    return vals


def closed_set_splits(y, groups, n_folds=5, seed=42):
    """Generate grouped splits where every test label appears in training.

    Returns (splits, stats) where stats includes unseen label info.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    try:
        splits = make_splits(
            y,
            n_folds=n_folds,
            groups=groups,
            split_name="heldout",
            random_state=seed,
        )
    except ValueError as error:
        return None, {"error": str(error)}
    unseen_stats = [
        {"n_unseen": 0, "n_test": int(len(test_idx)), "unseen_fraction": 0.0}
        for _, test_idx in splits
    ]
    return splits, {"unseen_stats": unseen_stats, "n_folds": len(splits)}


def train_heldout_probes(activations, labels, splits):
    """Train logistic probes per layer using the given splits.

    Returns (layerwise_acc, best_layer, best_acc).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    n_layers = activations.shape[1]
    layer_accs = []
    for layer in range(n_layers):
        X = activations[:, layer, :]
        pred = np.full_like(y, -1)
        for train_idx, test_idx in splits:
            probe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=2000)),
                ]
            )
            probe.fit(X[train_idx], y[train_idx])
            pred[test_idx] = probe.predict(X[test_idx])
        if np.any(pred < 0):
            raise RuntimeError("heldout folds did not predict every sample exactly once")
        layer_accs.append(float(np.mean(pred == y)))

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

    if splits is None:
        raise ValueError("character baseline requires held-out splits")

    pred = np.full_like(y, -1)
    for train_idx, test_idx in splits:
        pipeline = Pipeline(
            [
                ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 4), binary=True)),
                ("probe", LogisticRegression(max_iter=2000)),
            ]
        )
        train_surfaces = [surfaces[index] for index in train_idx]
        test_surfaces = [surfaces[index] for index in test_idx]
        pipeline.fit(train_surfaces, y[train_idx])
        pred[test_idx] = pipeline.predict(test_surfaces)
    if np.any(pred < 0):
        raise RuntimeError("character baseline did not predict every sample exactly once")
    return float(np.mean(pred == y)), info


def majority_baseline_for_splits(y, splits):
    predictions = np.full_like(y, -1)
    for train_idx, test_idx in splits:
        counts = np.bincount(y[train_idx])
        predictions[test_idx] = int(np.argmax(counts))
    if np.any(predictions < 0):
        raise RuntimeError("majority baseline did not predict every sample exactly once")
    return float(np.mean(predictions == y))


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
    parser.add_argument("--require-activation-provenance", action="store_true")
    parser.add_argument("--allow-label-revealed-prompts", action="store_true")
    parser.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    args = parser.parse_args()

    if args.min_examples_per_label < 2:
        parser.error("--min-examples-per-label must be at least 2")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("--tasks must not contain duplicates")

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading...")
    activations = load_activations(args.activations)
    rows = load_stimuli(args.stimuli)
    activation_metadata = load_activation_metadata(args.activations)
    validate_activation_provenance(
        args.activations,
        tuple(activations.shape),
        args.stimuli,
        activation_metadata,
        require=args.require_activation_provenance,
    )
    activations = validate_activation_tensor(
        activations,
        args.activations,
        expected_rows=len(rows),
    )
    leakage_report = enforce_prompt_contract(
        rows,
        args.tasks,
        activation_metadata,
        allow_label_revealed=args.allow_label_revealed_prompts,
        allow_unverifiable=args.allow_unverifiable_prompt_contract,
        context="heldout probe",
    )
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
                    print("    skipped: fewer than 2 examples in the smallest class")
                    continue
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
                split_meta["method"] = "closed-set grouped cross-validation"
                split_meta["group_field"] = group_field

            # Class overlap diagnostics
            overlap = class_overlap_report(y, splits)
            n_valid = sum(1 for o in overlap if o["all_test_labels_seen"])
            n_folds = len(splits)
            mean_unseen = np.mean([o["unseen_fraction"] for o in overlap])
            max_unseen = max(o["unseen_fraction"] for o in overlap)

            print(f"    folds: {n_folds}  closed-set folds: {n_valid}/{n_folds}  "
                  f"mean unseen: {mean_unseen:.3f}  max unseen: {max_unseen:.3f}")

            if n_valid != n_folds:
                raise RuntimeError("closed-set split construction returned unseen test labels")

            # --- Probe accuracy ---
            layer_accs, best_layer, best_acc = train_heldout_probes(
                task_acts, labels, splits
            )
            print(f"    probe:  best L{best_layer} = {best_acc:.4f}")

            # --- Char n-gram baseline ---
            char_acc, _ = char_ngram_heldout(task_rows, task, args.min_examples_per_label, splits)
            majority_acc = majority_baseline_for_splits(y, splits)
            print(f"    char:   {char_acc:.4f}")

            # --- Summary ---
            task_results["strategies"][strategy_name] = {
                "n_folds": n_folds,
                "n_valid_folds": n_valid,
                "mean_unseen_fraction": round(mean_unseen, 4),
                "max_unseen_fraction": round(max_unseen, 4),
                "overlap": overlap,
                "split_meta": split_meta,
                "probe_layerwise": [float(a) for a in layer_accs],
                "probe_best_layer": best_layer,
                "probe_best_accuracy": float(best_acc),
                "char_ngram_accuracy": float(char_acc),
                "majority_baseline": majority_acc,
                "probe_minus_char": float(best_acc) - float(char_acc),
                "probe_minus_majority": float(best_acc) - majority_acc,
                "best_layer_selection": "descriptive_same_cv",
            }

        results[task] = task_results

    # ── Save ──
    out_path = out_dir / "heldout_probe_results.json"
    if not results:
        raise ValueError("no heldout probe task could be evaluated")
    atomic_write_text(
        out_path,
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    provenance_path = out_path.with_name(f"{out_path.stem}_provenance.json")
    atomic_write_text(
        provenance_path,
        json.dumps(
            {
                "schema_version": 2,
                "results": out_path.name,
                "results_sha256": sha256_file(out_path),
                "activations_sha256": sha256_file(args.activations),
                "stimuli_sha256": sha256_file(args.stimuli),
                "activation_shape": list(activations.shape),
                "seed": args.seed,
                "folds": args.folds,
                "min_examples_per_label": args.min_examples_per_label,
                "prompt_leakage_audit": leakage_report,
                "label_revealed_prompt_allowed": args.allow_label_revealed_prompts,
                "unverifiable_prompt_contract_allowed": (
                    args.allow_unverifiable_prompt_contract
                ),
                "implementation_sha256": sha256_file(__file__),
                "probe_classifier": "standardized_logistic_regression",
                "character_classifier": "binary_count_char_1_4gram_logistic_regression",
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    print(f"\n{'='*70}")
    print(f"Saved to {out_path}")
    print(f"Provenance: {provenance_path}")

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

    for task in results:
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
    for task in results:
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
