"""Multi-task baseline probe runner.

Loads activations and stimuli, trains linear probes per layer for each
candidate label task, saves NPZ + JSON summary + plots, and prints a
terminal summary table.

Target tasks: root, lemma, pos, abstract_pattern, concrete_pattern,
features.gender, features.number.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from voidwest_theme import DARK, DARK_CYCLE, apply_matplotlib_theme  # noqa: E402

# ── palette ───────────────────────────────────────────────────────
CM_COLORS = DARK_CYCLE

DEFAULT_TASKS = [
    "root",
    "lemma",
    "pos",
    "abstract_pattern",
    "concrete_pattern",
    "features.gender",
    "features.number",
]

TASK_DISPLAY = {
    "root": "root",
    "lemma": "lemma",
    "pos": "POS",
    "abstract_pattern": "abs pat",
    "concrete_pattern": "conc pat",
    "features.gender": "gender",
    "features.number": "number",
}


# ── helpers ────────────────────────────────────────────────────────


def safe_key(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in value)


def get_field(row: dict, field: str, default=None):
    cur = row
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def load_activations(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        return np.load(path)["activations"]
    return np.load(path)


def load_stimuli(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("stimuli must be a JSON array")
    return data


# ── label extraction with filtering ────────────────────────────────


def extract_labels(
    rows: list[dict],
    field: str,
    min_examples_per_label: int = 3,
) -> tuple[list[int], list[str], dict]:
    """Return (indices, labels, info) for a task field.

    Filters out labels that appear fewer than min_examples_per_label times.
    Returns info with class_counts, num_classes, majority_class, majority_baseline.
    """
    indices = []
    labels = []
    for i, row in enumerate(rows):
        val = get_field(row, field)
        if val is None or val == "":
            continue
        indices.append(i)
        labels.append(str(val))

    if not labels:
        raise ValueError(f"field {field}: no non-empty labels")

    # count and filter
    unique, counts = np.unique(labels, return_counts=True)
    keep = set(unique[counts >= min_examples_per_label])

    filtered_indices = []
    filtered_labels = []
    for idx, lab in zip(indices, labels):
        if lab in keep:
            filtered_indices.append(idx)
            filtered_labels.append(lab)

    if len(set(filtered_labels)) < 2:
        raise ValueError(
            f"field {field}: fewer than 2 classes after filtering "
            f"(min_examples_per_label={min_examples_per_label})"
        )

    # recompute class stats
    class_values, class_counts = np.unique(filtered_labels, return_counts=True)
    majority_idx = int(np.argmax(class_counts))
    majority_class = str(class_values[majority_idx])
    majority_baseline = float(class_counts[majority_idx] / len(filtered_labels))

    info = {
        "num_examples": len(filtered_labels),
        "num_classes": len(class_values),
        "class_counts": {str(v): int(c) for v, c in zip(class_values, class_counts)},
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_baseline,
    }
    return filtered_indices, filtered_labels, info


# ── probe training ─────────────────────────────────────────────────


def make_probe(max_iter=2000, scale=True, solver="lbfgs", tol=1e-4, n_jobs=None):
    steps = []
    if scale:
        steps.append(("standardscaler", StandardScaler()))
    steps.append(
        (
            "logisticregression",
            LogisticRegression(
                max_iter=max_iter,
                solver=solver,
                tol=tol,
                n_jobs=n_jobs,
            ),
        )
    )
    return Pipeline(steps)


def train_layer_probes(
    activations: np.ndarray,
    labels: list[str],
    n_folds: int = 5,
    max_iter: int = 2000,
    scale: bool = True,
    solver: str = "lbfgs",
    tol: float = 1e-4,
    n_jobs: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list, LabelEncoder, np.ndarray | None]:
    """Train a probe per layer with cross-validation. Returns (accuracies, probes, le, confusions)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_layers = activations.shape[1]

    # set up CV
    min_per_class = int(np.bincount(y).min())
    effective_folds = min(n_folds, min_per_class)
    splits = None
    if effective_folds >= 2:
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))

    accuracies = []
    probes = []
    class_ids = np.arange(len(le.classes_))
    confusions = []

    for layer in range(n_layers):
        X = activations[:, layer, :]
        probe = make_probe(max_iter=max_iter, scale=scale, solver=solver, tol=tol, n_jobs=n_jobs)

        if splits is None:
            # fallback: train accuracy
            probe.fit(X, y)
            pred = probe.predict(X)
            acc = float(probe.score(X, y))
        else:
            scores = []
            pred = np.full_like(y, fill_value=-1)
            for train_idx, test_idx in splits:
                clone = make_probe(max_iter=max_iter, scale=scale, solver=solver, tol=tol, n_jobs=n_jobs)
                clone.fit(X[train_idx], y[train_idx])
                scores.append(clone.score(X[test_idx], y[test_idx]))
                pred[test_idx] = clone.predict(X[test_idx])
            acc = float(np.mean(scores))

        confusions.append(confusion_matrix(y, pred, labels=class_ids))
        accuracies.append(acc)
        probe.fit(X, y)  # refit on all data for export
        probes.append(probe)

    conf_arr = np.array(confusions) if confusions else None
    return np.array(accuracies), probes, le, conf_arr


# ── summary printing ───────────────────────────────────────────────


def print_summary_table(task_results: dict):
    """Print a terminal summary table."""
    header = f"{'task':<16s} {'ex':>4s} {'cls':>3s} {'maj%':>6s} {'best L':>6s} {'best acc':>8s} {'acc-maj':>8s}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for task_key in DEFAULT_TASKS:
        if task_key not in task_results:
            print(f"{TASK_DISPLAY.get(task_key, task_key):<16s}  SKIPPED (insufficient classes)")
            continue
        r = task_results[task_key]
        display = TASK_DISPLAY.get(task_key, task_key)
        num_ex = r["num_examples"]
        num_cls = r["num_classes"]
        maj_pct = r["majority_baseline_accuracy"] * 100
        best_layer = r["best_layer"]
        best_acc = r["best_accuracy"] * 100
        best_lift = r["best_accuracy_minus_majority"] * 100
        print(
            f"{display:<16s} {num_ex:>4d} {num_cls:>3d} "
            f"{maj_pct:>5.1f}% {best_layer:>5d}  {best_acc:>7.2f}% {best_lift:>7.2f}%"
        )
    print(sep)


# ── plotting ───────────────────────────────────────────────────────


def plot_baseline(task_results: dict, out_dir: str, dark: bool = True):
    """Generate probe_accuracy.png and probe_lift_over_majority.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dark:
        apply_matplotlib_theme(dark=True)

    tasks_in_order = [t for t in DEFAULT_TASKS if t in task_results]

    # ── accuracy plot ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, task_key in enumerate(tasks_in_order):
        r = task_results[task_key]
        acc = np.array(r["layerwise_accuracy"])
        layers = np.arange(len(acc))
        color = CM_COLORS[i % len(CM_COLORS)]
        display = TASK_DISPLAY.get(task_key, task_key)
        ax.plot(layers, acc, color=color, marker="o", markersize=3, linewidth=1.3, label=display)
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy")
    ax.set_title("per-layer probe accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "probe_accuracy.png", dpi=160, facecolor=DARK.bg)
    plt.close(fig)

    # ── lift over majority plot ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, task_key in enumerate(tasks_in_order):
        r = task_results[task_key]
        lift = np.array(r["layerwise_accuracy_minus_majority"])
        layers = np.arange(len(lift))
        color = CM_COLORS[i % len(CM_COLORS)]
        display = TASK_DISPLAY.get(task_key, task_key)
        ax.plot(layers, lift, color=color, marker="o", markersize=3, linewidth=1.3, label=display)
    ax.axhline(0.0, color=DARK.subtle, linestyle="--", alpha=0.7)
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy − majority baseline")
    ax.set_title("per-layer lift over majority baseline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "probe_lift_over_majority.png", dpi=160, facecolor=DARK.bg)
    plt.close(fig)

    print(f"\nsaved plots to {out_dir}/probe_accuracy.png and probe_lift_over_majority.png")


# ── main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="multi-task baseline probe runner")
    parser.add_argument("--activations", required=True, help="path to .npy activations")
    parser.add_argument("--stimuli", required=True, help="path to stimuli JSON")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="label tasks to probe (default: all 7)",
    )
    parser.add_argument(
        "--min-examples-per-label",
        type=int,
        default=3,
        help="drop labels with fewer than this many examples (default: 3)",
    )
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--max-iter", type=int, default=2000, help="LR max_iter (default: 2000)")
    parser.add_argument("--solver", default="lbfgs", help="LR solver (default: lbfgs)")
    parser.add_argument("--tol", type=float, default=1e-4, help="LR tolerance (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=None, help="LR n_jobs")
    parser.add_argument("--no-plot", action="store_true", help="skip plot generation")
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load
    print(f"loading activations from {args.activations}")
    activations = load_activations(args.activations)
    print(f"  shape: {activations.shape}")

    print(f"loading stimuli from {args.stimuli}")
    rows = load_stimuli(args.stimuli)
    print(f"  rows: {len(rows)}")

    task_results = {}
    npz_data = {}

    for task in args.tasks:
        print(f"\n── {task} ──")
        try:
            indices, labels, info = extract_labels(
                rows, task, min_examples_per_label=args.min_examples_per_label
            )
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  examples: {info['num_examples']}  classes: {info['num_classes']}")
        print(f"  majority: {info['majority_class']} ({info['majority_baseline_accuracy']:.1%})")
        class_counts = info["class_counts"]
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {cnt}")

        task_acts = activations[indices]
        acc, probes, le, confusions = train_layer_probes(
            task_acts,
            labels,
            n_folds=args.folds,
            max_iter=args.max_iter,
            solver=args.solver,
            tol=args.tol,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )

        best_idx = int(np.argmax(acc))
        info["layerwise_accuracy"] = [float(a) for a in acc]
        info["layerwise_accuracy_minus_majority"] = [
            float(a) - info["majority_baseline_accuracy"] for a in acc
        ]
        info["best_layer"] = best_idx
        info["best_accuracy"] = float(acc[best_idx])
        info["best_accuracy_minus_majority"] = float(
            acc[best_idx] - info["majority_baseline_accuracy"]
        )

        for layer_i, a in enumerate(acc):
            marker = " <-- best" if layer_i == best_idx else ""
            print(f"  layer {layer_i:2d}: {a:.4f}{marker}")

        task_results[task] = info

        # store in npz format
        key = safe_key(task)
        npz_data[f"{key}_accuracy"] = acc
        npz_data[f"{key}_classes"] = np.array(le.classes_, dtype=object)
        npz_data[f"{key}_class_counts"] = np.array(
            [class_counts.get(str(cls), 0) for cls in le.classes_], dtype=np.int64
        )
        npz_data[f"{key}_majority_baseline"] = np.array(info["majority_baseline_accuracy"])
        npz_data[f"{key}_accuracy_minus_majority"] = acc - info["majority_baseline_accuracy"]
        npz_data[f"{key}_confusion_matrices"] = confusions.astype(np.int64)
        npz_data[f"{key}_probe_weights"] = [
            p.named_steps["logisticregression"].coef_ for p in probes
        ]

    # save NPZ
    npz_path = output_dir / "qwen3_06b_last_probes.npz"
    save_data = {
        **npz_data,
        "probe_kind": "linear",
        "split_policy": "random-stratified",
        "tasks": np.array(list(task_results.keys()), dtype=object),
        "min_examples_per_label": args.min_examples_per_label,
    }
    np.savez(npz_path, **save_data)
    print(f"\nsaved probes to {npz_path}")

    # save JSON summary
    summary_path = output_dir / "baseline_probe_summary.json"
    summary = {
        "activation_shape": list(activations.shape),
        "activations": str(Path(args.activations).resolve()),
        "stimuli": str(Path(args.stimuli).resolve()),
        "probes": str(npz_path.resolve()),
        "plot": str((output_dir / "probe_accuracy.png").resolve()),
        "plot_lift": str((output_dir / "probe_lift_over_majority.png").resolve()),
        "config": {
            "min_examples_per_label": args.min_examples_per_label,
            "cv_folds": args.folds,
            "max_iter": args.max_iter,
            "solver": args.solver,
            "tol": args.tol,
            "seed": args.seed,
        },
        "tasks": task_results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved summary to {summary_path}")

    # print terminal summary
    print_summary_table(task_results)

    # plot
    if not args.no_plot:
        plot_baseline(task_results, str(output_dir))

    print("\ndone.")


if __name__ == "__main__":
    main()
