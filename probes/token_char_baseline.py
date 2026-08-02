"""Token-fragment and whole-word character baselines for morphology probes.

The vectorizer is fitted inside every cross-validation fold. Group-aware
evaluations are closed-set: every test label must occur in the training fold.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .run_baseline_probes import extract_labels as extract_probe_labels
    from .run_baseline_probes import load_stimuli
    from .train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        make_splits,
        sha256_file,
    )
except ImportError:  # direct script execution
    from run_baseline_probes import extract_labels as extract_probe_labels
    from run_baseline_probes import load_stimuli
    from train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        make_splits,
        sha256_file,
    )


def extract_labels(stimuli, field, min_examples=3):
    """Compatibility wrapper returning labels, indices, and diagnostics."""
    indices, labels, info = extract_probe_labels(stimuli, field, min_examples)
    return labels, indices, info


def get_last_tokens(prompts, tokenizer, targets=None):
    """Return the last subword overlapping each target occurrence.

    When targets are supplied, the exact final occurrence of the target in the
    prompt is used. This avoids accidentally measuring prompt punctuation or a
    template suffix instead of the surface-form fragment.
    """
    from tokenizers import Tokenizer

    tokenizer_path = Path(tokenizer)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer file does not exist: {tokenizer_path}")
    if targets is not None and len(targets) != len(prompts):
        raise ValueError("targets and prompts must have the same length")

    tok = Tokenizer.from_file(str(tokenizer_path))
    last_tokens = []
    for row_index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"prompt {row_index} must be a non-empty string")
        enc = tok.encode(prompt)
        if not (len(enc.ids) == len(enc.offsets) == len(enc.special_tokens_mask)):
            raise ValueError(f"tokenizer returned inconsistent fields for row {row_index}")

        if targets is None:
            candidates = [
                index
                for index, ((start, end), special) in enumerate(
                    zip(enc.offsets, enc.special_tokens_mask)
                )
                if not special and start < end
            ]
        else:
            target = targets[row_index]
            if not isinstance(target, str) or not target:
                raise ValueError(f"surface target {row_index} must be a non-empty string")
            start = prompt.rfind(target)
            if start < 0:
                raise ValueError(
                    f"surface target for row {row_index} does not occur verbatim in its prompt"
                )
            end = start + len(target)
            candidates = [
                index
                for index, ((token_start, token_end), special) in enumerate(
                    zip(enc.offsets, enc.special_tokens_mask)
                )
                if not special
                and token_start < token_end
                and token_start < end
                and token_end > start
            ]

        if not candidates:
            raise ValueError(f"no non-special token found for row {row_index}")
        fragment = tok.decode([enc.ids[candidates[-1]]], skip_special_tokens=True)
        if not fragment:
            # Some byte-level tokenizers only decode a fragment with context.
            fragment = enc.tokens[candidates[-1]]
        if not isinstance(fragment, str) or not fragment:
            raise ValueError(f"last target token for row {row_index} decoded to empty text")
        last_tokens.append(fragment)
    return last_tokens


def char_ngram_acc(tokens, labels, groups, n_folds, seed):
    """Return exact out-of-fold accuracy with fold-local feature fitting."""
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length")
    if not tokens:
        raise ValueError("cannot evaluate an empty task")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("all token strings must be non-empty")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    if len(encoder.classes_) < 2:
        raise ValueError("character baseline requires at least two classes")
    splits = make_splits(
        y,
        n_folds=n_folds,
        groups=groups,
        split_name="group" if groups is not None else "random",
        random_state=seed,
    )

    predictions = np.full(y.shape, -1, dtype=np.int64)
    texts = np.asarray(tokens, dtype=str)
    for train_idx, test_idx in splits:
        model = Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(analyzer="char", ngram_range=(1, 5), binary=True),
                ),
                ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
        model.fit(texts[train_idx].tolist(), y[train_idx])
        predictions[test_idx] = model.predict(texts[test_idx].tolist())
    if np.any(predictions < 0):
        raise RuntimeError("cross-validation did not predict every sample exactly once")
    return float(np.mean(predictions == y))


def _group_values(stimuli, indices, field):
    values = []
    for index in indices:
        value = stimuli[index].get(field)
        if field == "surface_dediac" and not value:
            value = stimuli[index].get("surface") or stimuli[index].get("expected_surface")
        if (
            not isinstance(value, (str, int, float))
            or isinstance(value, bool)
            or (isinstance(value, float) and not np.isfinite(value))
            or not str(value)
        ):
            raise ValueError(f"group field {field!r} is missing at stimulus row {index}")
        values.append(str(value))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--tasks", nargs="+", default=["pos", "features.gender", "features.number"]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples", type=int, default=3)
    parser.add_argument("--output", help="optional JSON report")
    parser.add_argument(
        "--allow-label-revealed-prompts",
        action="store_true",
        help="allow an explicitly labeled positive-control baseline",
    )
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.min_examples < 2:
        parser.error("--min-examples must be at least 2")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("--tasks must not contain duplicates")

    stimuli = load_stimuli(args.stimuli)
    leakage_report = enforce_prompt_contract(
        stimuli,
        args.tasks,
        {"probe_template": "morph_context", "probe_position": "last"},
        allow_label_revealed=args.allow_label_revealed_prompts,
        context="token/character baseline",
    )
    print(f"Loaded {len(stimuli)} stimuli")
    prompts = []
    targets = []
    for index, stimulus in enumerate(stimuli):
        prompt_map = stimulus.get("prompts")
        prompt = prompt_map.get("morph_context") if isinstance(prompt_map, dict) else None
        target = stimulus.get("surface") or stimulus.get("expected_surface")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"row {index} has no non-empty prompts.morph_context")
        if not isinstance(target, str) or not target:
            raise ValueError(f"row {index} has no non-empty surface target")
        prompts.append(prompt)
        targets.append(target)

    last_tokens = get_last_tokens(prompts, args.tokenizer, targets)
    print(f"Extracted {len(last_tokens)} target-final tokens")
    print(f"Sample: {last_tokens[:8]}")

    print(f"\n{'task':20s} {'split':20s} {'word-char':>8s} {'tok-char':>8s}")
    print("-" * 60)
    report = {
        "schema_version": 2,
        "stimuli_sha256": sha256_file(args.stimuli),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "folds_requested": args.folds,
        "seed": args.seed,
        "character_ngram_range": [1, 5],
        "prompt_leakage_audit": leakage_report,
        "label_revealed_positive_control": leakage_report["status"] == "label_revealed",
        "tasks": {},
    }
    evaluated_measurements = 0

    for task in args.tasks:
        try:
            labels, indices, info = extract_labels(stimuli, task, args.min_examples)
        except ValueError as error:
            print(f"{task:20s} SKIPPED ({error})")
            report["tasks"][task] = {"status": "skipped", "reason": str(error)}
            continue

        task_tokens = [last_tokens[index] for index in indices]
        surfaces = [targets[index] for index in indices]
        task_result = {"status": "evaluated", "descriptive": info, "strategies": {}}
        for split_name, group_field in [
            ("random", None),
            ("lemma-heldout", "lemma"),
            ("root-heldout", "root"),
        ]:
            groups = (
                _group_values(stimuli, indices, group_field) if group_field is not None else None
            )
            strategy_result = {}
            try:
                word_accuracy = char_ngram_acc(
                    surfaces, labels, groups, args.folds, args.seed
                )
            except ValueError as error:
                word_accuracy = None
                strategy_result["whole_word_error"] = str(error)
            try:
                token_accuracy = char_ngram_acc(
                    task_tokens, labels, groups, args.folds, args.seed
                )
            except ValueError as error:
                token_accuracy = None
                strategy_result["last_token_error"] = str(error)
            if word_accuracy is not None:
                evaluated_measurements += 1
            if token_accuracy is not None:
                evaluated_measurements += 1
            task_result["strategies"][split_name] = {
                "whole_word_char_accuracy": word_accuracy,
                "last_token_char_accuracy": token_accuracy,
                **strategy_result,
            }
            word_display = (
                f"{word_accuracy * 100:7.1f}%"
                if word_accuracy is not None
                else "     n/a"
            )
            token_display = (
                f"{token_accuracy * 100:7.1f}%"
                if token_accuracy is not None
                else "     n/a"
            )
            print(f"{task:20s} {split_name:20s} {word_display} {token_display}")
        print()
        report["tasks"][task] = task_result

    if not any(value.get("status") == "evaluated" for value in report["tasks"].values()):
        raise ValueError("no requested task could be evaluated")
    if evaluated_measurements == 0:
        raise ValueError("no character baseline measurement could be evaluated")
    if args.output:
        output = Path(args.output)
        if output.resolve() in {
            Path(args.stimuli).resolve(),
            Path(args.tokenizer).resolve(),
        }:
            parser.error("--output must not overwrite stimuli or tokenizer input")
        atomic_write_text(
            args.output,
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )


if __name__ == "__main__":
    main()
