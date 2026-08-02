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
import sys
import re
import math
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# reuse from baseline script
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .run_baseline_probes import (
        DEFAULT_TASKS,
        TASK_DISPLAY,
        extract_labels,
        load_activations,
        load_stimuli,
        make_probe,
        safe_key,
        train_layer_probes,
    )
    from .train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from run_baseline_probes import (
        DEFAULT_TASKS,
        TASK_DISPLAY,
        extract_labels,
        load_activations,
        load_stimuli,
        make_probe,
        safe_key,
        train_layer_probes,
    )
    from train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
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


def _control_probe(max_iter=2000, solver="lbfgs", tol=1e-4, n_jobs=None):
    """Use the same probe family and preprocessing as the real probe."""
    return make_probe(
        max_iter=max_iter,
        scale=True,
        solver=solver,
        tol=tol,
        n_jobs=n_jobs,
        classifier="logistic",
    )


def train_control_probes(
    activations, labels, n_folds=5, n_shuffles=5, seed=42, layer_stride=4,
    max_iter=2000, solver="lbfgs", tol=1e-4, n_jobs=None,
):
    """Train matched logistic probes on shuffled labels.

    layer_stride: only probe every Nth layer (control accuracy is ~chance everywhere).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    min_per_class = int(np.bincount(y).min())
    effective_folds = min(n_folds, min_per_class)
    if effective_folds < 2:
        raise ValueError(
            f"shuffled-label control requires at least 2 examples per class; minimum is {min_per_class}"
        )
    if n_shuffles < 2:
        raise ValueError("at least 2 shuffles are required to estimate control variance")
    if layer_stride < 1:
        raise ValueError("layer_stride must be at least 1")
    splits = list(
        StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )

    n_layers = activations.shape[1]
    probe_layers = list(range(0, n_layers, layer_stride))
    all_acc = np.zeros((n_shuffles, len(probe_layers)))

    for shuffle_i in range(n_shuffles):
        rng = np.random.RandomState(seed * 31 + shuffle_i * 7 + 1)
        y_shuffled = y.copy()
        for _ in range(100):
            rng.shuffle(y_shuffled)
            if all(len(np.unique(y_shuffled[train_idx])) >= 2 for train_idx, _ in splits):
                break
        else:
            raise ValueError("could not construct a valid shuffled-label assignment")

        for li, layer in enumerate(probe_layers):
            X = activations[:, layer, :]
            predictions = np.full_like(y_shuffled, -1)
            for train_idx, test_idx in splits:
                clone = _control_probe(max_iter, solver, tol, n_jobs)
                clone.fit(X[train_idx], y_shuffled[train_idx])
                predictions[test_idx] = clone.predict(X[test_idx])
            if np.any(predictions < 0):
                raise RuntimeError("control CV did not predict every sample")
            all_acc[shuffle_i, li] = float(np.mean(predictions == y_shuffled))

    return all_acc.mean(axis=0), all_acc.std(axis=0, ddof=1), probe_layers


# ── multi-seed stability ──────────────────────────────────────────


def train_multiseed_probes(
    activations, labels, seeds=(42, 123, 456, 789, 1024),
    n_folds=5, layer_stride=2, max_iter=2000, solver="lbfgs", tol=1e-4,
    n_jobs=None,
):
    """Train matched logistic probes across CV seeds."""
    n_seeds = len(seeds)
    n_layers = activations.shape[1]
    probe_layers = list(range(0, n_layers, layer_stride))
    all_acc = np.zeros((n_seeds, len(probe_layers)))

    le = LabelEncoder()
    y = le.fit_transform(labels)
    min_per_class = int(np.bincount(y).min())
    effective_folds = min(n_folds, min_per_class)
    if effective_folds < 2:
        raise ValueError(
            f"multi-seed CV requires at least 2 examples per class; minimum is {min_per_class}"
        )
    if len(seeds) < 2:
        raise ValueError("at least 2 seeds are required to estimate stability")
    if layer_stride < 1:
        raise ValueError("layer_stride must be at least 1")

    for i, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(y)), y))

        for li, layer in enumerate(probe_layers):
            X = activations[:, layer, :]
            predictions = np.full_like(y, -1)
            for train_idx, test_idx in splits:
                clone = _control_probe(max_iter, solver, tol, n_jobs)
                clone.fit(X[train_idx], y[train_idx])
                predictions[test_idx] = clone.predict(X[test_idx])
            if np.any(predictions < 0):
                raise RuntimeError("multi-seed CV did not predict every sample")
            all_acc[i, li] = float(np.mean(predictions == y))

    return (
        all_acc.mean(axis=0),
        all_acc.std(axis=0, ddof=1),
        probe_layers,
        all_acc,
    )


# ── selectivity ───────────────────────────────────────────────────


def selectivity(real_acc, control_acc, chance):
    """Hewitt & Liang (2019) selectivity: (real - control) / (1 - max(control, chance))."""
    denom = 1.0 - np.maximum(control_acc, chance)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    sel = (real_acc - control_acc) / denom
    return np.clip(sel, 0.0, 1.0)


# ── char n-gram surface baseline ──────────────────────────────────


def char_ngram_baseline(
    rows,
    task,
    min_examples_per_label=3,
    ngram_range=(1, 4),
    max_iter=2000,
    seed=42,
    n_folds=5,
):
    """Train a char n-gram logistic regression on surface forms."""
    indices, labels, info = extract_labels(rows, task, min_examples_per_label)

    surfaces = []
    for idx in indices:
        row = rows[idx]
        surf = row.get("surface") or row.get("expected_surface") or ""
        surfaces.append(dediac(surf))

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # stratified CV
    min_per_class = int(np.bincount(y).min())
    effective_folds = min(n_folds, min_per_class)
    if effective_folds < 2:
        raise ValueError(
            f"character baseline requires at least 2 examples per class; minimum is {min_per_class}"
        )
    predictions = np.full_like(y, -1)
    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        pipeline = Pipeline(
            [
                ("vectorizer", CountVectorizer(analyzer="char", ngram_range=ngram_range, binary=True)),
                ("classifier", LogisticRegression(max_iter=max_iter)),
            ]
        )
        pipeline.fit([surfaces[index] for index in train_idx], y[train_idx])
        predictions[test_idx] = pipeline.predict([surfaces[index] for index in test_idx])
    if np.any(predictions < 0):
        raise RuntimeError("character CV did not predict every sample")
    acc = float(np.mean(predictions == y))

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
    tasks = list(report.get("tasks", {}))

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
        if "best_accuracy" not in s:
            continue
        best = s["best_layer"]
        real = s["best_accuracy"]
        ctrl = s.get("control_best_accuracy")
        ctrl_std = s.get("control_best_std")
        sel = s.get("best_selectivity")
        lift = s["best_accuracy_minus_majority"]
        surf = s.get("char_ngram_accuracy")
        surf_str = f"{surf:.4f}" if surf is not None else "  —"
        control_str = f"{ctrl:>6.4f}" if ctrl is not None else "     —"
        std_str = f"±{ctrl_std:.4f}" if ctrl_std is not None else "     —"
        selectivity_str = f"{sel:>6.4f}" if sel is not None else "     —"
        print(
            f"{display:<16s} {best:>5d}  {real:>6.4f}  {control_str} "
            f"{std_str} {selectivity_str}  {lift:>6.4f}  {surf_str:>8s}"
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
        s = report["tasks"][t].get("multiseed")
        if s is None:
            continue
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
        if "layerwise_selectivity" not in s:
            continue
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
    parser.add_argument(
        "--solver",
        default="lbfgs",
        choices=["lbfgs", "saga", "liblinear", "newton-cg", "newton-cholesky", "sag"],
    )
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
    parser.add_argument(
        "--control-folds",
        type=int,
        default=5,
        help="CV folds for shuffled-label control; must match --folds",
    )
    parser.add_argument("--require-activation-provenance", action="store_true")
    parser.add_argument("--allow-label-revealed-prompts", action="store_true")
    parser.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    args = parser.parse_args()

    if args.min_examples_per_label < 2:
        parser.error("--min-examples-per-label must be at least 2")
    if args.folds < 2 or args.control_folds < 2:
        parser.error("--folds and --control-folds must be at least 2")
    if args.control_folds != args.folds:
        parser.error(
            "--control-folds must equal --folds so shuffled and real probes use matched train sizes"
        )
    if args.max_iter < 1:
        parser.error("--max-iter must be at least 1")
    if not math.isfinite(args.tol) or args.tol <= 0.0:
        parser.error("--tol must be finite and greater than 0")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be 0")
    if not args.no_control and args.n_shuffles < 2:
        parser.error("--n-shuffles must be at least 2")
    if not args.no_multiseed and args.n_seeds < 2:
        parser.error("--n-seeds must be at least 2")
    if args.control_layer_stride < 1 or args.multiseed_layer_stride < 1:
        parser.error("layer strides must be at least 1")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("--tasks must not contain duplicates")

    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    activations = load_activations(args.activations)
    print(f"  activations: {activations.shape}")
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
        context="control analysis",
    )
    print(f"  stimuli: {len(rows)} rows")

    multiseed_seeds = [args.seed + i * 100 for i in range(args.n_seeds)]

    report = {
        "schema_version": 2,
        "config": vars(args),
        "prompt_leakage_audit": leakage_report,
        "activations_sha256": sha256_file(args.activations),
        "stimuli_sha256": sha256_file(args.stimuli),
        "activation_shape": list(activations.shape),
        "tasks": {},
    }

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
            if not existing_summary.exists():
                raise ValueError(f"--skip-real-probes requires {existing_summary}")
            existing = json.loads(
                existing_summary.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(
                        f"non-standard JSON constant {value!r} in {existing_summary}"
                    )
                ),
            )
            if existing.get("activation_shape") != list(activations.shape):
                raise ValueError("existing baseline summary activation shape does not match")
            if existing.get("activations_sha256") != sha256_file(args.activations):
                raise ValueError("existing baseline summary does not pin these activations by SHA-256")
            if existing.get("stimuli_sha256") != sha256_file(args.stimuli):
                raise ValueError("existing baseline summary does not pin these stimuli by SHA-256")
            existing_config = existing.get("config")
            expected_config = {
                "min_examples_per_label": args.min_examples_per_label,
                "cv_folds": args.folds,
                "max_iter": args.max_iter,
                "solver": args.solver,
                "classifier": "logistic",
                "tol": args.tol,
                "seed": args.seed,
                "n_jobs": args.n_jobs,
            }
            if not isinstance(existing_config, dict) or any(
                existing_config.get(key) != value
                for key, value in expected_config.items()
            ):
                raise ValueError(
                    "existing baseline summary probe configuration does not match this run"
                )
            et = existing.get("tasks", {}).get(task)
            if not isinstance(et, dict):
                raise ValueError(f"existing baseline summary has no task {task!r}")
            layerwise = et.get("layerwise_accuracy")
            if (
                not isinstance(layerwise, list)
                or len(layerwise) != activations.shape[1]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0.0 <= value <= 1.0
                    for value in layerwise
                )
            ):
                raise ValueError(f"existing baseline task {task!r} has invalid layerwise accuracy")
            best_layer = int(et["best_layer"])
            best_accuracy = float(et["best_accuracy"])
            if (
                best_layer != int(np.argmax(layerwise))
                or not math.isclose(best_accuracy, layerwise[best_layer], abs_tol=1e-12)
            ):
                raise ValueError(f"existing baseline task {task!r} has inconsistent best metrics")
            task_report["layerwise_accuracy"] = layerwise
            task_report["best_layer"] = best_layer
            task_report["best_accuracy"] = best_accuracy
            task_report["best_accuracy_minus_majority"] = float(
                et["best_accuracy_minus_majority"]
            )
            print(f"    loaded existing: best L{task_report['best_layer']} = {task_report['best_accuracy']:.4f}")
        else:
            print("  training real probes...")
            acc, _, _, _ = train_layer_probes(
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
            task_report["best_layer_selection"] = "descriptive_same_cross_validation"
            print(f"    best L{best_idx} = {acc[best_idx]:.4f}  lift={task_report['best_accuracy_minus_majority']:.4f}")

        # ── shuffled-label control ──
        if not args.no_control:
            print(f"  shuffled-label control ({args.n_shuffles} shuffles)...")
            ctrl_mean, ctrl_std, ctrl_layers = train_control_probes(
                task_acts, labels,
                n_folds=args.control_folds, n_shuffles=args.n_shuffles,
                seed=args.seed,
                layer_stride=args.control_layer_stride,
                max_iter=args.max_iter,
                solver=args.solver,
                tol=args.tol,
                n_jobs=args.n_jobs,
            )
            best_ctrl_idx = int(np.argmax(ctrl_mean))
            task_report["control_layerwise_mean"] = [float(x) for x in ctrl_mean]
            task_report["control_layerwise_std"] = [float(x) for x in ctrl_std]
            task_report["control_best_accuracy"] = float(ctrl_mean[best_ctrl_idx])
            task_report["control_best_std"] = float(ctrl_std[best_ctrl_idx])
            task_report["control_best_layer"] = ctrl_layers[best_ctrl_idx]
            task_report["control_best_layer_selection"] = "descriptive_same_repeats"
            task_report["control_repeat_dispersion"] = "sample_standard_deviation"

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
            ms_mean, ms_std, ms_layers, ms_all = train_multiseed_probes(
                task_acts, labels,
                seeds=multiseed_seeds,
                n_folds=args.folds,
                layer_stride=args.multiseed_layer_stride,
                max_iter=args.max_iter,
                solver=args.solver,
                tol=args.tol,
                n_jobs=args.n_jobs,
            )
            ms_best_idx = int(np.argmax(ms_mean))
            ms_best_layer = ms_layers[ms_best_idx]
            selected_seed_scores = ms_all[:, ms_best_idx]
            task_report["multiseed"] = {
                "mean_best_accuracy": float(ms_mean[ms_best_idx]),
                "std_best_accuracy": float(ms_std[ms_best_idx]),
                "best_layer": ms_best_layer,
                "probe_layers": ms_layers,
                "layerwise_mean": [float(x) for x in ms_mean],
                "layerwise_std": [float(x) for x in ms_std],
                "selected_layer_seed_accuracies": [
                    float(value) for value in selected_seed_scores
                ],
                "min_best_accuracy": float(np.min(selected_seed_scores)),
                "max_best_accuracy": float(np.max(selected_seed_scores)),
                "dispersion_interpretation": "cross_validation_split_sensitivity",
                "best_layer_selection": "descriptive_across_same_seeds",
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
                    rows,
                    task,
                    args.min_examples_per_label,
                    (1, 4),
                    args.max_iter,
                    args.seed,
                    args.folds,
                )
                report["tasks"][task]["char_ngram_accuracy"] = float(surf_acc)
                report["tasks"][task]["char_ngram_lift"] = float(
                    surf_acc - surf_info["majority_baseline_accuracy"]
                )
                print(f"  {task:<18s} acc={surf_acc:.4f}  lift={surf_acc - surf_info['majority_baseline_accuracy']:+.4f}")
            except ValueError as e:
                print(f"  {task:<18s} SKIP: {e}")

    # ── save ──
    report_path = out_dir / "baseline_control_report.json"
    if not report["tasks"]:
        raise ValueError("no control-analysis task could be evaluated")
    atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(f"\nSaved control report to {report_path}")

    # ── print terminal summary ──
    print_control_summary(report)


if __name__ == "__main__":
    main()
