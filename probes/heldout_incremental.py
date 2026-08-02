"""Run held-out probe strategies incrementally with resumable provenance.

Each task/strategy pair is written atomically to a separate JSON document.
An existing result is reused only when its complete input/configuration
fingerprint matches the current invocation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .run_baseline_probes import (
        extract_labels,
        load_activations,
        load_stimuli,
        safe_key,
    )
    from .run_heldout_probes import (
        char_ngram_heldout,
        class_overlap_report,
        closed_set_splits,
        get_group_values,
        majority_baseline_for_splits,
        train_heldout_probes,
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
    from run_baseline_probes import extract_labels, load_activations, load_stimuli, safe_key
    from run_heldout_probes import (
        char_ngram_heldout,
        class_overlap_report,
        closed_set_splits,
        get_group_values,
        majority_baseline_for_splits,
        train_heldout_probes,
    )
    from train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )


GROUP_FIELDS = {
    "random": None,
    "surface-heldout": "surface_dediac",
    "lemma-heldout": "lemma",
    "root-heldout": "root",
}


def _load_existing(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"existing result is not a JSON object: {path}")
    return value


def _evaluate(task_acts, task_rows, labels, task, strategy, folds, seed, min_examples):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    group_field = GROUP_FIELDS[strategy]
    if group_field is None:
        effective = min(folds, int(np.bincount(y).min()))
        if effective < 2:
            raise ValueError("random CV requires at least two examples in every class")
        splitter = StratifiedKFold(n_splits=effective, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
        split_meta = {"method": "StratifiedKFold", "effective_folds": effective}
    else:
        groups = get_group_values(task_rows, group_field)
        splits, split_meta = closed_set_splits(y, groups, folds, seed)
        if splits is None:
            raise ValueError(split_meta.get("error", "could not construct closed-set folds"))
        split_meta = {
            **split_meta,
            "method": "closed-set grouped CV",
            "group_field": group_field,
        }

    overlap = class_overlap_report(y, splits)
    if not all(fold["all_test_labels_seen"] for fold in overlap):
        raise RuntimeError("split construction returned an unseen test label")
    layerwise, best_layer, best_accuracy = train_heldout_probes(
        task_acts, labels, splits
    )
    char_accuracy, _ = char_ngram_heldout(
        task_rows, task, min_examples=min_examples, splits=splits
    )
    majority_accuracy = majority_baseline_for_splits(y, splits)
    return {
        "n_folds": len(splits),
        "split_meta": split_meta,
        "overlap": overlap,
        "probe_layerwise": [float(value) for value in layerwise],
        "probe_best_layer": int(best_layer),
        "probe_best_accuracy": float(best_accuracy),
        "char_ngram_accuracy": float(char_accuracy),
        "majority_baseline": float(majority_accuracy),
        "probe_minus_char": float(best_accuracy - char_accuracy),
        "probe_minus_majority": float(best_accuracy - majority_accuracy),
        "best_layer_selection": "descriptive_same_cv",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["pos"])
    parser.add_argument(
        "--strategies", nargs="+", choices=sorted(GROUP_FIELDS), default=list(GROUP_FIELDS)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing results instead of resuming matching ones",
    )
    parser.add_argument("--require-activation-provenance", action="store_true")
    parser.add_argument("--allow-label-revealed-prompts", action="store_true")
    parser.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.min_examples < 2:
        parser.error("--min-examples must be at least 2")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("--tasks must not contain duplicates")
    if len(args.strategies) != len(set(args.strategies)):
        parser.error("--strategies must not contain duplicates")

    activations = load_activations(args.activations)
    stimuli = load_stimuli(args.stimuli)
    activation_metadata = load_activation_metadata(args.activations)
    validate_activation_provenance(
        args.activations,
        tuple(activations.shape),
        args.stimuli,
        activation_metadata,
        require=args.require_activation_provenance,
    )
    activations = validate_activation_tensor(
        activations, args.activations, expected_rows=len(stimuli)
    )
    leakage_report = enforce_prompt_contract(
        stimuli,
        args.tasks,
        activation_metadata,
        allow_label_revealed=args.allow_label_revealed_prompts,
        allow_unverifiable=args.allow_unverifiable_prompt_contract,
        context="incremental heldout probe",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "activations_sha256": sha256_file(args.activations),
        "stimuli_sha256": sha256_file(args.stimuli),
        "activation_shape": list(activations.shape),
        "folds": args.folds,
        "seed": args.seed,
        "min_examples": args.min_examples,
        "prompt_leakage_audit": leakage_report,
        "label_revealed_prompt_allowed": args.allow_label_revealed_prompts,
        "unverifiable_prompt_contract_allowed": args.allow_unverifiable_prompt_contract,
        "implementation_sha256": {
            "heldout_incremental.py": sha256_file(__file__),
            "run_heldout_probes.py": sha256_file(
                Path(__file__).with_name("run_heldout_probes.py")
            ),
            "run_baseline_probes.py": sha256_file(
                Path(__file__).with_name("run_baseline_probes.py")
            ),
        },
        "probe_classifier": "standardized_logistic_regression",
    }

    completed = 0
    for task in args.tasks:
        try:
            indices, labels, descriptive = extract_labels(
                stimuli, task, args.min_examples
            )
        except ValueError as error:
            print(f"SKIP {task}: {error}")
            continue

        task_acts = activations[indices]
        task_rows = [stimuli[index] for index in indices]
        print(f"\n=== {task}: {len(labels)} examples, {len(set(labels))} classes ===")
        for strategy in args.strategies:
            key = f"heldout_{safe_key(task)}_{safe_key(strategy)}.json"
            output_path = output_dir / key
            identity = {**fingerprint, "task": task, "strategy": strategy}
            if output_path.exists() and not args.overwrite:
                existing = _load_existing(output_path)
                if existing.get("input") != identity:
                    raise ValueError(
                        f"existing result has different provenance: {output_path}; "
                        "use --overwrite to replace it"
                    )
                print(f"  RESUME {strategy}: matching result already exists")
                completed += 1
                continue

            print(f"  Running {strategy}...")
            result = _evaluate(
                task_acts,
                task_rows,
                labels,
                task,
                strategy,
                args.folds,
                args.seed,
                args.min_examples,
            )
            payload = {
                "schema_version": 2,
                "input": identity,
                "descriptive": descriptive,
                "result": result,
            }
            atomic_write_text(
                output_path,
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )
            print(
                f"    L={result['probe_best_layer']} "
                f"probe={result['probe_best_accuracy'] * 100:.1f}% "
                f"char={result['char_ngram_accuracy'] * 100:.1f}%"
            )
            completed += 1

    if completed == 0:
        raise ValueError("no task/strategy result was produced or resumed")


if __name__ == "__main__":
    main()
