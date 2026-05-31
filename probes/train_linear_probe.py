"""train linear classifiers on per-layer activations
to predict arabic root and pattern from hidden states.

supports:
  - standard cross-validated linear probing
  - random-label control tasks (selectivity, Hewitt & Liang 2019)
  - task-specific held-out splits
  - selectivity score reporting (real accuracy / max(control, chance))
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline


def load_activations(path: str) -> np.ndarray:
    """load activation tensor.

    supports .npz (key: "activations") and .npy (raw 3d array).
    shape: (n_stimuli, n_layers, hidden_dim).
    """
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(path)
        return data["activations"]
    elif p.suffix == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"unsupported activation format: {p.suffix}")


def get_field(row: dict, field: str, default=None):
    """read dotted fields from a stimulus/benchmark row."""
    cur = row
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def load_rows(stimuli_path: str) -> list[dict]:
    with open(stimuli_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("stimuli/benchmark file must be a JSON list")
    return rows


def load_labels(rows: list[dict], field: str) -> list[str]:
    """load labels from a dotted field path."""
    labels = []
    missing = []
    for i, row in enumerate(rows):
        value = get_field(row, field)
        if value is None or value == "":
            missing.append(i)
        labels.append(str(value))
    if missing:
        raise ValueError(
            f"label field '{field}' missing for {len(missing)} rows; "
            f"first missing index: {missing[0]}"
        )
    return labels


def encode_groups(values):
    """encode string group labels as integers for sklearn splitters."""
    le = LabelEncoder()
    return le.fit_transform(values)


def make_splits(y, n_folds=5, groups=None, split_name="random"):
    """make valid closed-set splits for a classification probe.

    Grouped splits are valid only when every test label also appears in the
    corresponding training fold. This prevents impossible setups such as
    predicting root identity while holding out entire roots.
    """
    y = np.asarray(y)
    min_per_class = int(np.bincount(y).min())

    if groups is None:
        effective_folds = min(n_folds, min_per_class)
        if effective_folds < 2:
            return None
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=0)
        return list(splitter.split(np.zeros(len(y)), y))

    groups = np.asarray(groups)
    n_groups = len(set(groups))
    for effective_folds in range(min(n_folds, n_groups), 1, -1):
        splitter = GroupKFold(n_splits=effective_folds)
        splits = list(splitter.split(np.zeros(len(y)), y, groups=groups))
        valid = True
        for train_idx, test_idx in splits:
            train_labels = set(y[train_idx])
            test_labels = set(y[test_idx])
            if not test_labels.issubset(train_labels):
                valid = False
                break
        if valid:
            return splits

    raise ValueError(
        f"{split_name} creates test labels that are absent from training. "
        "Choose a split whose groups are independent of the target label."
    )


def make_probe(probe_kind: str):
    """build the requested probe model."""
    if probe_kind == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, solver="lbfgs"),
        )
    if probe_kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-3,
                max_iter=500,
                random_state=0,
            ),
        )
    raise ValueError(f"unknown probe kind: {probe_kind}")


def train_probes(
    activations,
    labels,
    n_folds=5,
    groups=None,
    split_name="random",
    probe_kind="linear",
):
    """train linear probes on each layer's activations.

    returns per-layer accuracy and trained models.
    if groups is provided, uses GroupKFold (groups define
    disjoint sets like roots or patterns that must not span folds).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    splits = make_splits(y, n_folds=n_folds, groups=groups, split_name=split_name)

    if splits is None:
        min_per_class = min(np.bincount(y))
        print(
            f"  warning: only {min_per_class} samples per class, "
            "skipping cross-validation (using train accuracy)"
        )

    n_layers = activations.shape[1]
    accuracies = []
    probes = []

    for layer in range(n_layers):
        X = activations[:, layer, :]
        probe = make_probe(probe_kind)
        if splits is None:
            probe.fit(X, y)
            acc = probe.score(X, y)  # train accuracy (optimistic)
        else:
            scores = []
            for train_idx, test_idx in splits:
                probe_clone = make_probe(probe_kind)
                probe_clone.fit(X[train_idx], y[train_idx])
                scores.append(probe_clone.score(X[test_idx], y[test_idx]))
            acc = np.mean(scores)
        accuracies.append(acc)
        probe.fit(X, y)  # refit on all data for export
        probes.append(probe)

    return np.array(accuracies), probes, le


def run_control(
    activations,
    labels,
    n_folds=5,
    groups=None,
    n_repeats=5,
    probe_kind="linear",
):
    """run random-label control: shuffle labels, train probes, report accuracy.

    repeats n_repeats times and returns mean + std across repeats.
    a good probe should score far above the control accuracy.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_layers = activations.shape[1]
    all_acc = np.zeros((n_repeats, n_layers))

    for repeat in range(n_repeats):
        # shuffle labels to break any real signal
        y_shuffled = y.copy()
        rng = np.random.RandomState(repeat * 31 + 7)
        rng.shuffle(y_shuffled)

        acc, _, _ = train_probes(
            activations, le.inverse_transform(y_shuffled),
            n_folds=n_folds,
            groups=groups,
            split_name="control",
            probe_kind=probe_kind,
        )
        all_acc[repeat] = acc

    return all_acc.mean(axis=0), all_acc.std(axis=0)


def compute_selectivity(real_acc, control_acc_mean, chance):
    """compute selectivity score.

    selectivity = (real - control_mean) / (1 - max(control_mean, chance))
    clamps negative values to 0.

    this follows the spirit of Hewitt & Liang (2019): a good probe
    should do well on the real task and poorly on control tasks.
    """
    denominator = 1.0 - np.maximum(control_acc_mean, chance)
    # avoid division by zero
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    selectivity = (real_acc - control_acc_mean) / denominator
    return np.maximum(selectivity, 0.0)


def groups_for_split(split, roots, patterns):
    """return group ids for a split policy."""
    if split == "random":
        return None
    if split == "root":
        return encode_groups(roots)
    if split == "pattern":
        return encode_groups(patterns)
    raise ValueError(f"unknown split policy: {split}")


def groups_for_task(task, split, rows, group_field=None):
    """return group ids for a task/split policy."""
    if group_field:
        return encode_groups([str(get_field(row, group_field, "")) for row in rows])
    if task == "root":
        roots = load_labels(rows, "root")
        patterns = load_labels(rows, "pattern")
        return groups_for_split(split, roots, patterns)
    if task == "pattern":
        roots = load_labels(rows, "root")
        patterns = load_labels(rows, "pattern")
        return groups_for_split(split, roots, patterns)
    if split != "random":
        raise ValueError(
            f"task '{task}' needs --group-field for grouped split policy '{split}'"
        )
    return None


def safe_key(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in value)


def main():
    parser = argparse.ArgumentParser(
        description="train linear probes on llm activations"
    )
    parser.add_argument(
        "--activations", required=True, help="path to .npy or .npz with activations"
    )
    parser.add_argument(
        "--stimuli", required=True, help="path to stimuli json"
    )
    parser.add_argument(
        "--output", default=None, help="path to save probe weights (.npz)"
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="cv folds"
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="run random-label control tasks and report selectivity",
    )
    parser.add_argument(
        "--control-repeats",
        type=int,
        default=5,
        help="number of random-label repeats for control (default: 5)",
    )
    parser.add_argument(
        "--probe-kind",
        choices=["linear", "mlp"],
        default="linear",
        help="probe model: linear logistic regression or one-hidden-layer MLP",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["root", "pattern"],
        help="label fields to probe, e.g. root pattern labels.upos labels.Gender",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="limit rows for fast benchmark smoke tests",
    )
    parser.add_argument(
        "--group-field",
        default=None,
        help="dotted field used for grouped CV on generic benchmark tasks",
    )
    parser.add_argument(
        "--split-root",
        action="store_true",
        help="deprecated: use root-held-out CV for pattern probes only",
    )
    parser.add_argument(
        "--root-split",
        choices=["pattern", "random"],
        default="pattern",
        help="CV split for root probes. 'pattern' tests roots on held-out patterns.",
    )
    parser.add_argument(
        "--pattern-split",
        choices=["root", "random"],
        default="root",
        help="CV split for pattern probes. 'root' tests patterns on held-out roots.",
    )
    args = parser.parse_args()

    if args.split_root:
        print(
            "warning: --split-root is deprecated; using --pattern-split root "
            "and leaving --root-split unchanged"
        )
        args.pattern_split = "root"

    activations = load_activations(args.activations)
    rows = load_rows(args.stimuli)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
        activations = activations[: args.max_rows]

    print(f"activations shape: {activations.shape}")
    print(f"stimuli/benchmark rows: {len(rows)}")
    print(f"tasks: {', '.join(args.tasks)}")
    print(f"root probe split: {args.root_split}")
    print(f"pattern probe split: {args.pattern_split}")
    if args.group_field:
        print(f"group field: {args.group_field}")
    print(f"probe kind: {args.probe_kind}")
    if args.control:
        print(f"running random-label control ({args.control_repeats} repeats)")

    results = {}
    trained = {}
    for task in args.tasks:
        labels = load_labels(rows, task)
        split = args.root_split if task == "root" else args.pattern_split if task == "pattern" else "random"
        groups = groups_for_task(task, split, rows, args.group_field)
        print(f"\n--- {task} probes ---")
        print(f"  labels: {len(set(labels))} classes")
        acc, probes, le = train_probes(
            activations,
            labels,
            args.folds,
            groups=groups,
            split_name=f"{task}-split={split}",
            probe_kind=args.probe_kind,
        )
        for i, layer_acc in enumerate(acc):
            print(f"  layer {i:2d}: {layer_acc:.3f}")
        key = safe_key(task)
        results[f"{key}_accuracy"] = acc
        results[f"{key}_classes"] = np.array(le.classes_, dtype=object)
        trained[key] = probes

        if args.control:
            print(f"\n--- {task}: random-label control ---")
            control_mean, control_std = run_control(
                activations,
                labels,
                args.folds,
                groups=groups,
                n_repeats=args.control_repeats,
                probe_kind=args.probe_kind,
            )
            chance = 1.0 / len(set(labels))
            selectivity = compute_selectivity(acc, control_mean, chance)
            for i, (real, ctrl, sel) in enumerate(zip(acc, control_mean, selectivity)):
                print(
                    f"  layer {i:2d}: real={real:.3f}  control={ctrl:.3f}  "
                    f"selectivity={sel:.3f}"
                )
            print(
                f"  mean selectivity: {selectivity.mean():.3f} "
                f"(max: {selectivity.max():.3f} at layer {selectivity.argmax()})"
            )
            results[f"{key}_control_mean"] = control_mean
            results[f"{key}_control_std"] = control_std
            results[f"{key}_selectivity"] = selectivity

    # save
    if args.output:
        save_dict = {
            **results,
            "probe_kind": args.probe_kind,
            "root_split": args.root_split,
            "pattern_split": args.pattern_split,
            "tasks": np.array(args.tasks, dtype=object),
        }
        if args.probe_kind == "linear":
            for key, probes in trained.items():
                save_dict[f"{key}_probe_weights"] = [
                    p.named_steps["logisticregression"].coef_
                    for p in probes
                ]
        np.savez(args.output, **save_dict)
        print(f"\nsaved probe weights to {args.output}")
        if args.control:
            print("  (includes control and selectivity arrays)")


if __name__ == "__main__":
    main()
