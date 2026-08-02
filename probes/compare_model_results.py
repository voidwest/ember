"""Compare two model probe runs after verifying dataset provenance."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .train_linear_probe import atomic_write_text
except ImportError:  # direct script execution
    from train_linear_probe import atomic_write_text


def load_json(path, *, required=False):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _number(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric when present")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _accuracy(value, field):
    value = _number(value, field)
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return value


def get_heldout_lift(heldout, task, strategy="lemma-heldout"):
    if heldout is None:
        return None, None, None
    task_result = heldout.get(task, {})
    if not isinstance(task_result, dict):
        raise ValueError(f"heldout task {task!r} must be an object")
    strategies = task_result.get("strategies", {})
    if not isinstance(strategies, dict):
        raise ValueError(f"heldout strategies for {task!r} must be an object")
    result = strategies.get(strategy, {})
    if not result or "probe_best_accuracy" not in result:
        return None, None, None
    probe = _accuracy(result.get("probe_best_accuracy"), "probe_best_accuracy")
    char = _accuracy(result.get("char_ngram_accuracy"), "char_ngram_accuracy")
    lift = _number(result.get("probe_minus_char"), "probe_minus_char")
    if probe is None or char is None:
        raise ValueError("heldout result is missing probe or character accuracy")
    computed_lift = probe - char
    if lift is None:
        lift = computed_lift
    elif not math.isclose(lift, computed_lift, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("heldout probe_minus_char is inconsistent with its component metrics")
    return probe, char, lift


def get_baseline_acc(baseline, task):
    if baseline is None:
        return None, None
    tasks = baseline.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError("baseline tasks must be an object")
    result = tasks.get(task, {})
    if not isinstance(result, dict):
        raise ValueError(f"baseline task {task!r} must be an object")
    accuracy = _accuracy(result.get("best_accuracy"), "best_accuracy")
    layer = result.get("best_layer")
    if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or layer < 0):
        raise ValueError("best_layer must be a non-negative integer")
    return accuracy, layer


def get_token_stats(token_diag):
    if token_diag is None:
        return None, None
    distribution = token_diag.get("token_distribution", {})
    if not isinstance(distribution, dict):
        raise ValueError("token_distribution must be an object")
    single = _number(distribution.get("single_token_pct"), "single_token_pct")
    mean = _number(distribution.get("mean_tokens"), "mean_tokens")
    if single is not None and not 0.0 <= single <= 100.0:
        raise ValueError("single_token_pct must be in [0, 100]")
    if mean is not None and mean <= 0.0:
        raise ValueError("mean_tokens must be greater than zero")
    return single, mean


def load_run(directory: Path) -> dict:
    heldout_path = directory / "heldout_probe_results.json"
    heldout_provenance = load_json(
        heldout_path.with_name(f"{heldout_path.stem}_provenance.json")
    )
    run = {
        "directory": str(directory),
        "baseline": load_json(directory / "baseline_probe_summary.json"),
        "heldout": load_json(heldout_path),
        "heldout_provenance": heldout_provenance,
        "token": load_json(directory / "token_diagnostics.json"),
    }
    if all(run[key] is None for key in ("baseline", "heldout", "token")):
        raise ValueError(f"no recognized probe reports found in {directory}")
    heldout_verified = run["heldout"] is None
    if run["heldout"] is not None and heldout_provenance is not None:
        declared_results = heldout_provenance.get("results")
        declared_sha = heldout_provenance.get("results_sha256")
        actual_sha = hashlib.sha256(heldout_path.read_bytes()).hexdigest()
        heldout_verified = (
            declared_results == heldout_path.name and declared_sha == actual_sha
        )
    run["heldout_verified"] = heldout_verified
    hashes = set()
    for document in (run["baseline"], heldout_provenance, run["token"]):
        if document is None:
            continue
        value = document.get("stimuli_sha256")
        if value is not None:
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid stimuli_sha256 in {directory}")
            hashes.add(value.lower())
    if len(hashes) > 1:
        raise ValueError(f"reports within {directory} refer to different stimuli files")
    run["stimuli_sha256"] = next(iter(hashes), None)
    return run


def comparison_contract_mismatches(run_a: dict, run_b: dict) -> list[str]:
    """Find analysis choices that would make model deltas non-comparable."""
    mismatches = []
    baseline_a, baseline_b = run_a["baseline"], run_b["baseline"]
    if baseline_a is not None and baseline_b is not None:
        for field in ("config", "prompt_leakage_audit"):
            if baseline_a.get(field) != baseline_b.get(field):
                mismatches.append(f"baseline {field} differs")
        shape_a = baseline_a.get("activation_shape")
        shape_b = baseline_b.get("activation_shape")
        if (
            isinstance(shape_a, list)
            and isinstance(shape_b, list)
            and shape_a
            and shape_b
            and shape_a[0] != shape_b[0]
        ):
            mismatches.append("baseline activation row counts differ")

    heldout_a, heldout_b = run_a["heldout_provenance"], run_b["heldout_provenance"]
    if heldout_a is not None and heldout_b is not None:
        for field in (
            "seed",
            "folds",
            "min_examples_per_label",
            "prompt_leakage_audit",
            "probe_classifier",
            "character_classifier",
        ):
            if heldout_a.get(field) != heldout_b.get(field):
                mismatches.append(f"heldout {field} differs")
        shape_a = heldout_a.get("activation_shape")
        shape_b = heldout_b.get("activation_shape")
        if (
            isinstance(shape_a, list)
            and isinstance(shape_b, list)
            and shape_a
            and shape_b
            and shape_a[0] != shape_b[0]
        ):
            mismatches.append("heldout activation row counts differ")

    token_a, token_b = run_a["token"], run_b["token"]
    if token_a is not None and token_b is not None:
        for field in (
            "seed",
            "folds_requested",
            "prompt_leakage_audit",
            "label_revealed_prompt_allowed",
            "unverifiable_prompt_contract_allowed",
        ):
            if token_a.get(field) != token_b.get(field):
                mismatches.append(f"token diagnostics {field} differs")
    return mismatches


def _all_tasks(run_a, run_b):
    tasks = set()
    for run in (run_a, run_b):
        for key in ("baseline", "heldout"):
            document = run[key]
            if isinstance(document, dict) and isinstance(document.get("tasks"), dict):
                tasks.update(document["tasks"])
            elif key == "heldout" and isinstance(document, dict):
                tasks.update(document)
    preferred = [
        "root",
        "lemma",
        "pos",
        "abstract_pattern",
        "concrete_pattern",
        "features.gender",
        "features.number",
    ]
    return [task for task in preferred if task in tasks] + sorted(tasks - set(preferred))


def _display(task):
    return {
        "pos": "POS",
        "abstract_pattern": "abs pat",
        "concrete_pattern": "conc pat",
        "features.gender": "gender",
        "features.number": "number",
    }.get(task, task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a")
    parser.add_argument("dir_b")
    parser.add_argument("--label-a", default="Model A")
    parser.add_argument("--label-b", default="Model B")
    parser.add_argument("--output", help="output Markdown file (stdout if omitted)")
    parser.add_argument("--output-json", help="optional structured comparison JSON")
    parser.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help="compare legacy reports that lack matching stimuli SHA-256 provenance",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="compare metrics produced with different analysis/prompt configurations",
    )
    args = parser.parse_args()

    run_a = load_run(Path(args.dir_a))
    run_b = load_run(Path(args.dir_b))
    if (
        not run_a["heldout_verified"] or not run_b["heldout_verified"]
    ) and not args.allow_unverified_inputs:
        raise ValueError(
            "heldout results require matching results_sha256 provenance; use "
            "--allow-unverified-inputs only after external verification"
        )
    hash_a = run_a["stimuli_sha256"]
    hash_b = run_b["stimuli_sha256"]
    if hash_a is None or hash_b is None:
        if not args.allow_unverified_inputs:
            raise ValueError(
                "both model runs require stimuli_sha256 provenance; use "
                "--allow-unverified-inputs only after external verification"
            )
        alignment = "user_assumed_legacy_inputs"
    elif hash_a != hash_b:
        raise ValueError("model runs use different stimuli SHA-256 values")
    else:
        alignment = "matching_stimuli_sha256"

    config_mismatches = comparison_contract_mismatches(run_a, run_b)
    if config_mismatches and not args.allow_config_mismatch:
        raise ValueError(
            "model-run analysis configurations differ: " + "; ".join(config_mismatches)
        )

    tasks = _all_tasks(run_a, run_b)
    strategies = sorted(
        {
            strategy
            for run in (run_a, run_b)
            for task_result in (run["heldout"] or {}).values()
            if isinstance(task_result, dict)
            for strategy in task_result.get("strategies", {})
        },
        key=lambda value: (value != "random", value),
    )
    comparison = {
        "schema_version": 2,
        "labels": {"a": args.label_a, "b": args.label_b},
        "directories": {"a": args.dir_a, "b": args.dir_b},
        "alignment_evidence": alignment,
        "stimuli_sha256": hash_a if hash_a == hash_b else None,
        "configuration_mismatches": config_mismatches,
        "configuration_mismatch_allowed": args.allow_config_mismatch,
        "tasks": {},
    }

    lines = [
        f"# Model Comparison: {args.label_a} vs {args.label_b}",
        "",
        f"- {args.label_a}: `{args.dir_a}`",
        f"- {args.label_b}: `{args.dir_b}`",
        f"- Alignment: `{alignment}`",
        f"- Configuration: `{'mismatch explicitly allowed' if config_mismatches else 'matched'}`",
        "",
        "## Heldout Probe Accuracy (probe / char / probe−char)",
        "",
        f"| Task | Strategy | {args.label_a} | {args.label_b} | Δ lift (B−A) |",
        "|---|---|---:|---:|---:|",
    ]
    for task in tasks:
        task_json = {"strategies": {}}
        for strategy in strategies:
            values_a = get_heldout_lift(run_a["heldout"], task, strategy)
            values_b = get_heldout_lift(run_b["heldout"], task, strategy)
            if values_a[0] is None and values_b[0] is None:
                continue
            text_a = (
                f"{values_a[0]:.3f}/{values_a[1]:.3f}/{values_a[2]:+.3f}"
                if values_a[0] is not None
                else "—"
            )
            text_b = (
                f"{values_b[0]:.3f}/{values_b[1]:.3f}/{values_b[2]:+.3f}"
                if values_b[0] is not None
                else "—"
            )
            delta = (
                values_b[2] - values_a[2]
                if values_a[2] is not None and values_b[2] is not None
                else None
            )
            lines.append(
                f"| {_display(task)} | {strategy} | {text_a} | {text_b} | "
                f"{delta:+.3f} |" if delta is not None else
                f"| {_display(task)} | {strategy} | {text_a} | {text_b} | — |"
            )
            task_json["strategies"][strategy] = {
                "a": values_a,
                "b": values_b,
                "delta_lift_b_minus_a": delta,
            }
        comparison["tasks"][task] = task_json

    lines.extend(
        [
            "",
            "## Baseline (Random CV) Comparison",
            "",
            f"| Task | {args.label_a} best | {args.label_b} best | Δ accuracy (B−A) |",
            "|---|---:|---:|---:|",
        ]
    )
    for task in tasks:
        accuracy_a, layer_a = get_baseline_acc(run_a["baseline"], task)
        accuracy_b, layer_b = get_baseline_acc(run_b["baseline"], task)
        if accuracy_a is None and accuracy_b is None:
            continue
        text_a = f"L{layer_a} {accuracy_a:.3f}" if accuracy_a is not None else "—"
        text_b = f"L{layer_b} {accuracy_b:.3f}" if accuracy_b is not None else "—"
        delta = accuracy_b - accuracy_a if accuracy_a is not None and accuracy_b is not None else None
        lines.append(
            f"| {_display(task)} | {text_a} | {text_b} | "
            f"{delta:+.3f} |" if delta is not None else
            f"| {_display(task)} | {text_a} | {text_b} | — |"
        )
        comparison["tasks"][task]["baseline"] = {
            "a": {"accuracy": accuracy_a, "layer": layer_a},
            "b": {"accuracy": accuracy_b, "layer": layer_b},
            "delta_accuracy_b_minus_a": delta,
        }

    token_a = get_token_stats(run_a["token"])
    token_b = get_token_stats(run_b["token"])
    comparison["tokenization"] = {"a": token_a, "b": token_b}
    lines.extend(
        [
            "",
            "## Tokenization Statistics",
            "",
            f"| Metric | {args.label_a} | {args.label_b} |",
            "|---|---:|---:|",
            f"| % single-token | {token_a[0] if token_a[0] is not None else '—'} | "
            f"{token_b[0] if token_b[0] is not None else '—'} |",
            f"| Mean tokens/word | {token_a[1] if token_a[1] is not None else '—'} | "
            f"{token_b[1] if token_b[1] is not None else '—'} |",
            "",
        ]
    )
    markdown = "\n".join(lines)
    if args.output:
        atomic_write_text(args.output, markdown)
        print(f"Saved to {args.output}")
    else:
        print(markdown, end="")
    if args.output_json:
        atomic_write_text(
            args.output_json,
            json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )


if __name__ == "__main__":
    main()
