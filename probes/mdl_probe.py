"""data-efficiency / MDL-style probing curves.

This is a lightweight practical proxy for full online coding MDL: for each
layer and task, train the same probe on increasing fractions of the training
data and report held-out accuracy plus area under the data-efficiency curve.
Features that are easily extractable should reach high accuracy with less data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from train_linear_probe import get_field, load_activations, load_rows, make_probe, safe_key


DEFAULT_FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.8]


def load_labels(rows: list[dict], field: str) -> list[str]:
    labels = []
    for i, row in enumerate(rows):
        value = get_field(row, field)
        if value is None or value == "":
            raise ValueError(f"missing label field '{field}' at row {i}")
        labels.append(str(value))
    return labels


def train_size_for_fraction(y_train: np.ndarray, fraction: float) -> int:
    classes = len(set(y_train))
    return max(classes, int(round(len(y_train) * fraction)))


def stratified_subset(y_train: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size >= len(y_train):
        return np.arange(len(y_train))
    if np.bincount(y_train).min() < 2:
        rng = np.random.RandomState(seed)
        required = []
        for label in sorted(set(y_train)):
            required.append(int(np.flatnonzero(y_train == label)[0]))
        remaining = [i for i in range(len(y_train)) if i not in required]
        extra_size = max(0, min(size, len(y_train)) - len(required))
        extra = (
            rng.choice(remaining, size=extra_size, replace=False).tolist()
            if extra_size > 0
            else []
        )
        return np.sort(np.array(required + extra))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
    idx, _ = next(splitter.split(np.zeros(len(y_train)), y_train))
    return idx


def run_task(
    activations: np.ndarray,
    labels: list[str],
    fractions: list[float],
    probe_kind: str,
    seed: int,
) -> dict:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if np.bincount(y).min() < 2:
        train_idx = np.arange(len(y))
        test_idx = np.arange(len(y))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    y_train = y[train_idx]
    y_test = y[test_idx]

    n_layers = activations.shape[1]
    curves = np.zeros((n_layers, len(fractions)), dtype=np.float32)

    for li in range(n_layers):
        X_train = activations[train_idx, li, :]
        X_test = activations[test_idx, li, :]
        for fi, fraction in enumerate(fractions):
            size = train_size_for_fraction(y_train, fraction)
            subset = stratified_subset(y_train, size, seed + li * 997 + fi)
            probe = make_probe(probe_kind)
            probe.fit(X_train[subset], y_train[subset])
            curves[li, fi] = probe.score(X_test, y_test)

    auc = np.trapezoid(curves, x=np.asarray(fractions), axis=1) / (
        fractions[-1] - fractions[0]
    )
    return {
        "classes": le.classes_.tolist(),
        "fractions": fractions,
        "accuracy_curve": curves,
        "data_efficiency_auc": auc.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="run MDL-style data-efficiency probes")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--tasks", nargs="+", default=["root", "pattern"])
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    activations = load_activations(args.activations)
    rows = load_rows(args.stimuli)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
        activations = activations[: args.max_rows]
    save = {
        "tasks": np.array(args.tasks, dtype=object),
        "fractions": np.array(args.fractions, dtype=np.float32),
        "probe_kind": args.probe_kind,
    }

    for task in args.tasks:
        result = run_task(
            activations,
            load_labels(rows, task),
            args.fractions,
            args.probe_kind,
            args.seed,
        )
        key = safe_key(task)
        save[f"{key}_accuracy_curve"] = result["accuracy_curve"]
        save[f"{key}_data_efficiency_auc"] = result["data_efficiency_auc"]
        save[f"{key}_classes"] = np.array(result["classes"], dtype=object)
        best_layer = int(np.argmax(result["data_efficiency_auc"]))
        best_auc = float(result["data_efficiency_auc"][best_layer])
        print(f"{task}: best AUC={best_auc:.3f} at layer {best_layer}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **save)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
