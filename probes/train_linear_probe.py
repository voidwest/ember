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
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline


SPLIT_ALIASES = {
    "random": "random",
    "random-stratified": "random",
    "stratified": "random",
    "root": "root-heldout",
    "root-heldout": "root-heldout",
    "pattern": "pattern-heldout",
    "pattern-heldout": "pattern-heldout",
    "combination": "combination-heldout",
    "combination-heldout": "combination-heldout",
    "root-pattern": "combination-heldout",
    "root-pattern-heldout": "combination-heldout",
    "template": "template-heldout",
    "template-heldout": "template-heldout",
}
SPLIT_CHOICES = sorted(SPLIT_ALIASES)
TEMPLATE_FIELDS = [
    "prompt_template",
    "probe_template",
    "template",
    "metadata.prompt_template",
    "metadata.probe_template",
]


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


def metadata_path_for_activations(path: str) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}_metadata.json")


def load_activation_metadata(path: str) -> dict:
    metadata_path = metadata_path_for_activations(path)
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"activation metadata must be a JSON object: {metadata_path}")
    return metadata


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


def load_available_labels(rows: list[dict], field: str) -> tuple[list[int], list[str]]:
    """load labels and row indices for rows where a sparse field exists."""
    indices = []
    labels = []
    for i, row in enumerate(rows):
        value = get_field(row, field)
        if value is None or value == "":
            continue
        indices.append(i)
        labels.append(str(value))
    if not labels:
        raise ValueError(f"label field '{field}' missing for all rows")
    return indices, labels


def require_field_values(rows: list[dict], field: str, policy: str) -> list[str]:
    values = []
    missing = []
    for i, row in enumerate(rows):
        value = get_field(row, field)
        if value is None or value == "":
            missing.append(i)
        else:
            values.append(str(value))
    if missing:
        raise ValueError(
            f"split policy '{policy}' requires field '{field}', missing for "
            f"{len(missing)} rows; first missing index: {missing[0]}"
        )
    return values


def template_values(
    rows: list[dict],
    activation_metadata: dict,
    policy: str,
) -> tuple[list[str], str]:
    for field in TEMPLATE_FIELDS:
        if any(get_field(row, field) not in (None, "") for row in rows):
            return require_field_values(rows, field, policy), field
    for field in ("probe_template", "prompt_template", "template"):
        value = activation_metadata.get(field)
        if value not in (None, ""):
            return [str(value)] * len(rows), f"activation_metadata.{field}"
    raise ValueError(
        "split policy 'template-heldout' requires prompt template metadata. "
        f"Expected one of row fields {TEMPLATE_FIELDS} or an activation metadata sidecar."
    )


def encode_groups(values):
    """encode string group labels as integers for sklearn splitters."""
    le = LabelEncoder()
    return le.fit_transform(values)


def normalize_split_policy(split: str) -> str:
    try:
        return SPLIT_ALIASES[split]
    except KeyError as exc:
        raise ValueError(
            f"unknown split policy: {split}. Choices: {', '.join(SPLIT_CHOICES)}"
        ) from exc


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
    if n_groups < 2:
        raise ValueError(f"{split_name} requires at least 2 groups; found {n_groups}")
    for effective_folds in range(min(n_folds, n_groups), 1, -1):
        splitter = GroupKFold(n_splits=effective_folds)
        splits = list(splitter.split(np.zeros(len(y)), y, groups=groups))
        valid = True
        for train_idx, test_idx in splits:
            group_overlap = set(groups[train_idx]) & set(groups[test_idx])
            if group_overlap:
                raise ValueError(
                    f"{split_name} produced train/test group overlap: "
                    f"{sorted(group_overlap)[:5]}"
                )
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


def prepare_splits(
    labels,
    n_folds=5,
    groups=None,
    group_values=None,
    split_name="random",
):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    splits = make_splits(y, n_folds=n_folds, groups=groups, split_name=split_name)
    metadata = {
        "split_name": split_name,
        "requested_folds": int(n_folds),
        "effective_folds": len(splits) if splits is not None else None,
        "n_samples": int(len(labels)),
        "n_classes": int(len(le.classes_)),
        "classes": [str(value) for value in le.classes_],
        "uses_train_accuracy": splits is None,
    }
    if groups is not None:
        values = (
            [str(value) for value in group_values]
            if group_values is not None
            else [str(v) for v in groups]
        )
        metadata["n_groups"] = len(set(values))
        metadata["groups"] = sorted(set(values))
    if splits is not None:
        metadata["folds"] = [
            {
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
            }
            for train_idx, test_idx in splits
        ]
    return splits, metadata


def make_probe(
    probe_kind: str,
    max_iter: int = 2000,
    scale: bool = True,
    solver: str = "lbfgs",
    tol: float = 1e-4,
    n_jobs: int | None = None,
):
    """build the requested probe model."""
    steps = []
    if scale:
        steps.append(StandardScaler())
    if probe_kind == "linear":
        steps.append(
            LogisticRegression(
                max_iter=max_iter,
                solver=solver,
                tol=tol,
                n_jobs=n_jobs,
            )
        )
        return make_pipeline(*steps)
    if probe_kind == "sgd":
        steps.append(
            SGDClassifier(
                loss="log_loss",
                max_iter=max_iter,
                tol=tol,
                random_state=0,
                n_jobs=n_jobs,
            )
        )
        return make_pipeline(*steps)
    if probe_kind == "mlp":
        steps.append(
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-3,
                max_iter=500,
                random_state=0,
            )
        )
        return make_pipeline(*steps)
    raise ValueError(f"unknown probe kind: {probe_kind}")


def train_probes(
    activations,
    labels,
    n_folds=5,
    groups=None,
    split_name="random",
    probe_kind="linear",
    max_iter=2000,
    scale=True,
    solver="lbfgs",
    tol=1e-4,
    n_jobs=None,
    splits_override=None,
    collect_confusion=False,
):
    """train linear probes on each layer's activations.

    returns per-layer accuracy and trained models.
    if groups is provided, uses GroupKFold (groups define
    disjoint sets like roots or patterns that must not span folds).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    splits = (
        splits_override
        if splits_override is not None
        else make_splits(y, n_folds=n_folds, groups=groups, split_name=split_name)
    )

    if splits is None:
        min_per_class = min(np.bincount(y))
        print(
            f"  warning: only {min_per_class} samples per class, "
            "skipping cross-validation (using train accuracy)"
        )

    n_layers = activations.shape[1]
    accuracies = []
    confusion_matrices = []
    probes = []
    class_ids = np.arange(len(le.classes_))

    for layer in range(n_layers):
        X = activations[:, layer, :]
        probe = make_probe(
            probe_kind,
            max_iter=max_iter,
            scale=scale,
            solver=solver,
            tol=tol,
            n_jobs=n_jobs,
        )
        if splits is None:
            probe.fit(X, y)
            pred = probe.predict(X)
            acc = probe.score(X, y)  # train accuracy (optimistic)
        else:
            scores = []
            pred = np.full_like(y, fill_value=-1)
            for train_idx, test_idx in splits:
                probe_clone = make_probe(
                    probe_kind,
                    max_iter=max_iter,
                    scale=scale,
                    solver=solver,
                    tol=tol,
                    n_jobs=n_jobs,
                )
                probe_clone.fit(X[train_idx], y[train_idx])
                scores.append(probe_clone.score(X[test_idx], y[test_idx]))
                pred[test_idx] = probe_clone.predict(X[test_idx])
            acc = np.mean(scores)
        accuracies.append(acc)
        if collect_confusion:
            confusion_matrices.append(confusion_matrix(y, pred, labels=class_ids))
        probe.fit(X, y)  # refit on all data for export
        probes.append(probe)

    if collect_confusion:
        return np.array(accuracies), probes, le, np.array(confusion_matrices)
    return np.array(accuracies), probes, le


def run_control(
    activations,
    labels,
    n_folds=5,
    groups=None,
    n_repeats=5,
    probe_kind="linear",
    max_iter=2000,
    scale=True,
    solver="lbfgs",
    tol=1e-4,
    n_jobs=None,
    splits_override=None,
):
    """run random-label control: shuffle labels, train probes, report accuracy.

    repeats n_repeats times and returns mean + std across repeats.
    a good probe should score far above the control accuracy.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    control_splits = (
        splits_override
        if splits_override is not None
        else make_splits(
            y,
            n_folds=n_folds,
            groups=groups,
            split_name="control-real-label-split",
        )
    )
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
            max_iter=max_iter,
            scale=scale,
            solver=solver,
            tol=tol,
            n_jobs=n_jobs,
            splits_override=control_splits,
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
    return np.clip(selectivity, 0.0, 1.0)


def group_values_for_policy(policy, rows, activation_metadata=None):
    """return string group labels and metadata for a normalized split policy."""
    activation_metadata = activation_metadata or {}
    if policy == "random":
        return None, {"effective_policy": "random", "group_field": None}
    if policy == "root-heldout":
        values = require_field_values(rows, "root", policy)
        return values, {"effective_policy": policy, "group_field": "root"}
    if policy == "pattern-heldout":
        values = require_field_values(rows, "pattern", policy)
        return values, {"effective_policy": policy, "group_field": "pattern"}
    if policy == "combination-heldout":
        roots = require_field_values(rows, "root", policy)
        patterns = require_field_values(rows, "pattern", policy)
        values = [f"{root}::{pattern}" for root, pattern in zip(roots, patterns)]
        return values, {"effective_policy": policy, "group_field": "root+pattern"}
    if policy == "template-heldout":
        values, source = template_values(rows, activation_metadata, policy)
        return values, {"effective_policy": policy, "group_field": source}
    raise ValueError(f"unknown split policy: {policy}")


def groups_for_task(task, split, rows, group_field=None, activation_metadata=None):
    """return group ids and metadata for a task/split policy."""
    requested_policy = split
    normalized_policy = normalize_split_policy(split)
    if group_field:
        values = require_field_values(rows, group_field, "group-field")
        metadata = {
            "requested_policy": requested_policy,
            "normalized_policy": normalized_policy,
            "effective_policy": "group-field",
            "group_field": group_field,
            "n_groups": len(set(values)),
        }
        return encode_groups(values), values, metadata

    values, metadata = group_values_for_policy(
        normalized_policy,
        rows,
        activation_metadata=activation_metadata,
    )
    metadata = {
        "requested_policy": requested_policy,
        "normalized_policy": normalized_policy,
        **metadata,
    }
    if values is None:
        metadata["n_groups"] = None
        return None, None, metadata
    metadata["n_groups"] = len(set(values))
    return encode_groups(values), values, metadata


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
        choices=["linear", "sgd", "mlp"],
        default="linear",
        help="probe model: logistic linear, SGD linear, or one-hidden-layer MLP",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="maximum iterations for linear logistic regression",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "saga", "liblinear", "newton-cg", "newton-cholesky", "sag"],
        default="lbfgs",
        help="solver for linear logistic regression",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="tolerance for linear logistic regression convergence",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="parallel workers for LogisticRegression when supported",
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        action="store_true",
        default=True,
        help="standardize activations before fitting probes",
    )
    parser.add_argument(
        "--no-scale",
        dest="scale",
        action="store_false",
        help="fit probes without StandardScaler",
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
        help="dotted field used for grouped CV; overrides task-specific split grouping",
    )
    parser.add_argument(
        "--split-policy",
        choices=SPLIT_CHOICES,
        default="random",
        help=(
            "split policy for tasks without a task-specific default. "
            "Use root-heldout, pattern-heldout, combination-heldout, "
            "template-heldout, or random."
        ),
    )
    parser.add_argument(
        "--split-root",
        action="store_true",
        help="deprecated: use root-held-out CV for pattern probes only",
    )
    parser.add_argument(
        "--root-split",
        choices=SPLIT_CHOICES,
        default="pattern",
        help=(
            "CV split for root probes. Default 'pattern' means pattern-heldout, "
            "testing roots on held-out patterns."
        ),
    )
    parser.add_argument(
        "--pattern-split",
        choices=SPLIT_CHOICES,
        default="root",
        help=(
            "CV split for pattern probes. Default 'root' means root-heldout, "
            "testing patterns on held-out roots."
        ),
    )
    args = parser.parse_args()

    if args.split_root:
        print(
            "warning: --split-root is deprecated; using --pattern-split root "
            "and leaving --root-split unchanged"
        )
        args.pattern_split = "root"

    activations = load_activations(args.activations)
    activation_metadata = load_activation_metadata(args.activations)
    rows = load_rows(args.stimuli)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
        activations = activations[: args.max_rows]

    print(f"activations shape: {activations.shape}")
    print(f"stimuli/benchmark rows: {len(rows)}")
    print(f"tasks: {', '.join(args.tasks)}")
    print(f"root probe split: {args.root_split}")
    print(f"pattern probe split: {args.pattern_split}")
    print(f"default split policy: {args.split_policy}")
    if args.group_field:
        print(f"group field: {args.group_field}")
    print(f"probe kind: {args.probe_kind}")
    if args.probe_kind in {"linear", "sgd"}:
        print(f"linear max_iter: {args.max_iter}")
        if args.probe_kind == "linear":
            print(f"linear solver: {args.solver}")
        print(f"linear tol: {args.tol}")
        if args.n_jobs is not None:
            print(f"linear n_jobs: {args.n_jobs}")
    print(f"scale activations: {args.scale}")
    if args.control:
        print(f"running random-label control ({args.control_repeats} repeats)")

    results = {}
    trained = {}
    split_policy_records = []
    for task in args.tasks:
        task_indices, labels = load_available_labels(rows, task)
        task_rows = [rows[i] for i in task_indices]
        task_activations = activations[task_indices]
        split = (
            args.root_split
            if task == "root"
            else args.pattern_split
            if task == "pattern"
            else args.split_policy
        )
        groups, group_values, split_metadata = groups_for_task(
            task,
            split,
            task_rows,
            args.group_field,
            activation_metadata=activation_metadata,
        )
        splits, cv_metadata = prepare_splits(
            labels,
            args.folds,
            groups=groups,
            group_values=group_values,
            split_name=f"{task}-split={split_metadata['effective_policy']}",
        )
        split_metadata = {
            "task": task,
            "label_field": task,
            "row_count": len(task_rows),
            "usable_indices": task_indices,
            **split_metadata,
            **cv_metadata,
        }
        split_policy_records.append(split_metadata)
        print(f"\n--- {task} probes ---")
        if len(task_rows) != len(rows):
            print(f"  usable rows: {len(task_rows)} / {len(rows)}")
        print(f"  labels: {len(set(labels))} classes")
        print(f"  split policy: {split_metadata['effective_policy']}")
        if split_metadata.get("group_field"):
            print(
                f"  grouped by: {split_metadata['group_field']} "
                f"({split_metadata.get('n_groups')} groups)"
            )
        acc, probes, le, confusions = train_probes(
            task_activations,
            labels,
            args.folds,
            groups=groups,
            split_name=f"{task}-split={split_metadata['effective_policy']}",
            probe_kind=args.probe_kind,
            max_iter=args.max_iter,
            scale=args.scale,
            solver=args.solver,
            tol=args.tol,
            n_jobs=args.n_jobs,
            splits_override=splits,
            collect_confusion=True,
        )
        for i, layer_acc in enumerate(acc):
            print(f"  layer {i:2d}: {layer_acc:.3f}")
        key = safe_key(task)
        class_values, class_counts = np.unique(labels, return_counts=True)
        class_count_map = {
            str(value): int(count)
            for value, count in zip(class_values, class_counts)
        }
        results[f"{key}_accuracy"] = acc
        results[f"{key}_classes"] = np.array(le.classes_, dtype=object)
        results[f"{key}_class_counts"] = np.array(
            [class_count_map[str(cls)] for cls in le.classes_],
            dtype=np.int64,
        )
        results[f"{key}_chance"] = np.array(1.0 / len(le.classes_))
        results[f"{key}_confusion_matrices"] = confusions.astype(np.int64)
        trained[key] = probes

        if args.control:
            print(f"\n--- {task}: random-label control ---")
            control_mean, control_std = run_control(
                task_activations,
                labels,
                args.folds,
                groups=groups,
                n_repeats=args.control_repeats,
                probe_kind=args.probe_kind,
                max_iter=args.max_iter,
                scale=args.scale,
                solver=args.solver,
                tol=args.tol,
                n_jobs=args.n_jobs,
                splits_override=splits,
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
            "default_split_policy": args.split_policy,
            "split_policy": "task-specific",
            "task_split_policy_json": json.dumps(
                split_policy_records,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "split_policy_json": json.dumps(
                split_policy_records,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "tasks": np.array(args.tasks, dtype=object),
        }
        if args.probe_kind in {"linear", "sgd"}:
            for key, probes in trained.items():
                save_dict[f"{key}_probe_weights"] = [
                    (
                        p.named_steps["logisticregression"].coef_
                        if "logisticregression" in p.named_steps
                        else p.named_steps["sgdclassifier"].coef_
                    )
                    for p in probes
                ]
        np.savez(args.output, **save_dict)
        print(f"\nsaved probe weights to {args.output}")
        split_sidecar = Path(args.output).with_name(
            f"{Path(args.output).stem}_split_policy.json"
        )
        split_sidecar.write_text(
            json.dumps(split_policy_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved split policy metadata to {split_sidecar}")
        if args.control:
            print("  (includes control and selectivity arrays)")


if __name__ == "__main__":
    main()
