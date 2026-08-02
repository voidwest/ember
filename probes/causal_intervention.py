"""offline linear-direction interventions on cached activations.

This script exports a task/class direction from linear probe weights, removes
that direction from a chosen layer, and measures how much a freshly trained
probe's held-out accuracy drops on the intervened activations. Optional before
and after logits/continuation sidecars can be supplied for downstream deltas
computed outside this script.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .cca_analysis import validate_probe_artifact_contract
    from .train_linear_probe import (
        atomic_save_npy,
        atomic_savez,
        atomic_write_text,
        enforce_prompt_contract,
        export_linear_parameters,
        load_activation_metadata,
        load_activations,
        load_labels,
        load_rows,
        make_probe,
        make_splits,
        safe_key,
        sha256_file,
        train_probes,
        validate_activation_provenance,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from cca_analysis import validate_probe_artifact_contract
    from train_linear_probe import (
        atomic_save_npy,
        atomic_savez,
        atomic_write_text,
        enforce_prompt_contract,
        export_linear_parameters,
        load_activation_metadata,
        load_activations,
        load_labels,
        load_rows,
        make_probe,
        make_splits,
        safe_key,
        sha256_file,
        train_probes,
        validate_activation_provenance,
        validate_activation_tensor,
    )


def direction_from_weights(
    weights: np.ndarray,
    classes: list[str],
    class_label: str | None,
) -> dict:
    """Select and normalize one binary or named one-vs-rest class direction."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim not in {1, 2} or weights.size == 0 or not np.isfinite(weights).all():
        raise ValueError(f"invalid probe weight tensor: {weights.shape}")

    if weights.ndim == 1:
        direction = weights
        selected_class = class_label
    elif weights.shape[0] == 1:
        if len(classes) == 2:
            if class_label is not None and class_label not in classes:
                raise ValueError(f"class '{class_label}' not found in {classes}")
            selected_class = class_label or classes[1]
            direction = weights[0] if selected_class == classes[1] else -weights[0]
        else:
            direction = weights[0]
            selected_class = classes[0] if classes else class_label
    else:
        if class_label is None:
            raise ValueError(
                "multiclass probe directions require --class-label; silently selecting the "
                "first class would not represent the task subspace"
            )
        if class_label not in classes:
            raise ValueError(f"class '{class_label}' not found in {classes}")
        class_index = classes.index(class_label)
        selected = weights[class_index]
        others = np.delete(weights, class_index, axis=0)
        direction = selected - others.mean(axis=0)
        selected_class = classes[class_index] if classes else str(class_index)

    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("probe direction has near-zero norm")
    direction = direction / norm
    return {
        "direction": direction.astype(np.float32),
        "classes": classes,
        "selected_class": selected_class,
        "norm_before_normalization": norm,
        "weight_space": "raw_activation",
    }


def load_probe_direction(probe_path: str, task: str, layer: int, class_label: str | None) -> dict:
    key = safe_key(task)
    weight_key = f"{key}_probe_weights"
    class_key = f"{key}_classes"
    with np.load(probe_path, allow_pickle=False) as data:
        if "probe_weight_space" not in data:
            raise ValueError(
                "probe artifact does not declare probe_weight_space; legacy weights may be "
                "standardized-coordinate coefficients and are unsafe for raw activation intervention"
            )
        weight_space = str(data["probe_weight_space"])
        if weight_space != "raw_activation":
            raise ValueError(
                f"probe weights use unsupported coordinate space {weight_space!r}; "
                "raw_activation is required"
            )
        if weight_key not in data:
            raise ValueError(f"{probe_path} does not contain linear weights for task '{task}'")
        all_weights = np.asarray(data[weight_key], dtype=np.float32)
        if layer < 0 or layer >= len(all_weights):
            raise ValueError(f"probe layer {layer} is outside 0..{len(all_weights) - 1}")
        weights = np.asarray(all_weights[layer], dtype=np.float32)
        classes = [str(c) for c in data[class_key].tolist()] if class_key in data else []
    return direction_from_weights(weights, classes, class_label)


def remove_direction(activations: np.ndarray, layer: int, direction: np.ndarray) -> np.ndarray:
    activations = validate_activation_tensor(activations)
    direction = np.asarray(direction, dtype=np.float32)
    if layer < 0 or layer >= activations.shape[1]:
        raise ValueError(f"layer {layer} is outside 0..{activations.shape[1] - 1}")
    if direction.shape != (activations.shape[2],):
        raise ValueError(
            f"direction shape {direction.shape} does not match hidden width {activations.shape[2]}"
        )
    if not np.isfinite(direction).all():
        raise ValueError("direction contains non-finite values")
    direction_norm = float(np.linalg.norm(direction))
    if not np.isclose(direction_norm, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"direction must be unit-normalized, got norm {direction_norm}")
    intervened = activations.copy()
    X = intervened[:, layer, :]
    projection = X @ direction
    intervened[:, layer, :] = X - projection[:, None] * direction[None, :]
    return intervened


def single_layer_probe_score(
    activations: np.ndarray,
    labels: list[str],
    layer: int,
    probe_kind: str,
    folds: int,
) -> float:
    layer_acts = activations[:, layer : layer + 1, :]
    acc, _, _ = train_probes(
        layer_acts,
        labels,
        n_folds=folds,
        groups=None,
        split_name="intervention-random",
        probe_kind=probe_kind,
    )
    return float(acc[0])


def nested_direction_probe_scores(
    activations: np.ndarray,
    labels: list[str],
    layer: int,
    class_label: str | None,
    probe_kind: str,
    folds: int,
) -> dict:
    """Evaluate direction removal without using held-out labels to choose directions."""
    activations = validate_activation_tensor(
        activations,
        expected_rows=len(labels),
    )
    if layer < 0 or layer >= activations.shape[1]:
        raise ValueError(f"layer {layer} is outside 0..{activations.shape[1] - 1}")
    classes, y = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    splits = make_splits(
        y,
        n_folds=folds,
        groups=None,
        split_name="nested-intervention-random",
    )
    X = np.asarray(activations[:, layer, :], dtype=np.float32)
    predicted_before = np.full(len(y), -1, dtype=np.int64)
    predicted_after = np.full(len(y), -1, dtype=np.int64)
    fold_metadata = []

    for fold, (train, test) in enumerate(splits):
        direction_probe = make_probe("linear")
        direction_probe.fit(X[train], y[train])
        estimator = direction_probe.steps[-1][1]
        _, raw_weights, _ = export_linear_parameters(direction_probe)
        fold_classes = [str(classes[int(index)]) for index in estimator.classes_]
        direction_info = direction_from_weights(
            raw_weights,
            fold_classes,
            class_label,
        )
        direction = direction_info["direction"]
        train_projection = X[train] @ direction
        test_projection = X[test] @ direction
        X_train_after = X[train] - train_projection[:, None] * direction[None, :]
        X_test_after = X[test] - test_projection[:, None] * direction[None, :]

        before_probe = make_probe(probe_kind)
        before_probe.fit(X[train], y[train])
        predicted_before[test] = before_probe.predict(X[test])

        after_probe = make_probe(probe_kind)
        after_probe.fit(X_train_after, y[train])
        predicted_after[test] = after_probe.predict(X_test_after)
        fold_metadata.append(
            {
                "fold": fold,
                "train_size": int(len(train)),
                "test_size": int(len(test)),
                "selected_class": direction_info["selected_class"],
                "direction_norm_before_normalization": float(
                    direction_info["norm_before_normalization"]
                ),
            }
        )

    if np.any(predicted_before < 0) or np.any(predicted_after < 0):
        raise RuntimeError("nested intervention did not predict every held-out row exactly once")
    return {
        "before": float(np.mean(predicted_before == y)),
        "after": float(np.mean(predicted_after == y)),
        "evaluation": "nested_cross_validated_direction_selection_and_probe_scoring",
        "direction_fit_uses_heldout_labels": False,
        "effective_folds": len(splits),
        "folds": fold_metadata,
    }


def summarize_logits(before_path: str | None, after_path: str | None) -> dict | None:
    if not before_path or not after_path:
        return None
    before = np.load(before_path, allow_pickle=False).astype(np.float32)
    after = np.load(after_path, allow_pickle=False).astype(np.float32)
    if before.shape != after.shape:
        raise ValueError(f"logit shape mismatch: {before.shape} vs {after.shape}")
    if before.ndim not in {1, 2} or before.size == 0:
        raise ValueError(f"logits must be a non-empty rank-1 or rank-2 tensor, got {before.shape}")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("logits contain non-finite values")
    before_rows = before.reshape(1, -1) if before.ndim == 1 else before
    after_rows = after.reshape(1, -1) if after.ndim == 1 else after
    before_top = np.argmax(before_rows, axis=1)
    after_top = np.argmax(after_rows, axis=1)
    top_changed = before_top != after_top
    diff = after - before
    max_abs_shift = float(np.max(np.abs(diff)))
    return {
        "before": before_path,
        "after": after_path,
        "before_sha256": sha256_file(before_path),
        "after_sha256": sha256_file(after_path),
        "shape": list(before.shape),
        "mean_abs_shift": float(np.mean(np.abs(diff))),
        "max_abs_shift": max_abs_shift,
        "exact_bits_equal": bool(np.array_equal(before.view(np.uint32), after.view(np.uint32))),
        "top_token_changed": bool(np.any(top_changed)),
        "top_token_change_count": int(np.sum(top_changed)),
        "top_token_change_rate": float(np.mean(top_changed)),
        "changed": bool(max_abs_shift > 0.0),
    }


def summarize_continuations(before_path: str | None, after_path: str | None) -> dict | None:
    if not before_path or not after_path:
        return None
    def load_strict(path: str):
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value!r} in {path}")
            ),
        )

    before = load_strict(before_path)
    after = load_strict(after_path)
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("continuation artifacts must be JSON arrays")
    if len(before) != len(after):
        raise ValueError(f"continuation row count mismatch: {len(before)} vs {len(after)}")
    if not before:
        raise ValueError("continuation artifacts are empty")
    n = len(before)
    changes = 0
    for i in range(n):
        if isinstance(before[i], dict) and isinstance(after[i], dict):
            before_id = before[i].get("id", before[i].get("sample_id"))
            after_id = after[i].get("id", after[i].get("sample_id"))
            if before_id is not None or after_id is not None:
                if before_id != after_id:
                    raise ValueError(f"continuation identity mismatch at row {i}")
        b = before[i].get("generated") if isinstance(before[i], dict) else before[i]
        a = after[i].get("generated") if isinstance(after[i], dict) else after[i]
        changes += int(b != a)
    return {
        "before": before_path,
        "after": after_path,
        "before_sha256": sha256_file(before_path),
        "after_sha256": sha256_file(after_path),
        "compared": n,
        "changed": changes,
        "change_rate": float(changes / n),
        "changed_any": bool(changes > 0),
    }


def interpretation_text(
    target_probe_score_dropped: bool,
    logit_shift: dict | None,
    continuation_changes: dict | None,
) -> list[str]:
    lines = []
    if target_probe_score_dropped:
        lines.append("probe-direction removal affected decodability")
    else:
        lines.append("probe-direction removal did not reduce measured decodability")

    logits_changed = bool(logit_shift and logit_shift.get("changed"))
    continuations_changed = bool(continuation_changes and continuation_changes.get("changed_any"))
    if logits_changed:
        lines.append(
            "supplied downstream logits changed after intervention; interpret this as a logit shift, not behavioral causality by itself"
        )
    elif logit_shift is not None:
        lines.append("supplied downstream logits did not change")
    else:
        lines.append("no downstream logits were supplied")

    if continuations_changed:
        lines.append(
            "supplied continuations changed after intervention; behavioral interpretation still requires matched prompts and scoring"
        )
    elif continuation_changes is not None:
        lines.append("supplied continuations did not change")
    else:
        lines.append("no downstream continuations were supplied")

    if not (logits_changed or continuations_changed):
        lines.append("do not claim a downstream causal effect from this summary alone")
    return lines


def build_summary(
    *,
    activations_path: str,
    output_path: str,
    direction_output: str,
    task: str,
    layer: int,
    class_label: str | None,
    direction_info: dict,
    before_acc: float,
    after_acc: float,
    probe_evaluation: dict | None = None,
    logit_shift: dict | None = None,
    continuation_changes: dict | None = None,
) -> dict[str, Any]:
    accuracy_drop = float(before_acc - after_acc)
    target_probe_score_dropped = bool(accuracy_drop > 0.0)
    logits_changed = bool(logit_shift and logit_shift.get("changed"))
    continuations_changed = bool(continuation_changes and continuation_changes.get("changed_any"))
    summary = {
        "schema_version": 2,
        "inputs": {
            "activations": activations_path,
        },
        "outputs": {
            "intervened_activations": output_path,
            "direction": direction_output,
        },
        "intervention": {
            "type": "orthogonal_projection_removal",
            "task": task,
            "layer": int(layer),
            "class_label": class_label,
            "selected_class": direction_info.get("selected_class"),
            "classes": [str(value) for value in direction_info.get("classes", [])],
            "norm_before_normalization": float(direction_info["norm_before_normalization"]),
            "weight_space": direction_info.get("weight_space"),
        },
        "probe_accuracy": {
            "before": float(before_acc),
            "after": float(after_acc),
            "drop": accuracy_drop,
            "target_probe_score_dropped": target_probe_score_dropped,
            "evaluation": probe_evaluation,
        },
        "downstream": {
            "logit_shift": logit_shift,
            "continuation_changes": continuation_changes,
        },
        "claims": {
            "probe_direction_removal_affected_decodability": target_probe_score_dropped,
            "downstream_logits_changed": logits_changed,
            "downstream_continuations_changed": continuations_changed,
            "behavioral_causality_claimed": False,
            "direction_selection_uses_heldout_labels": bool(
                probe_evaluation
                and probe_evaluation.get("direction_fit_uses_heldout_labels")
            ),
        },
    }
    summary["interpretation"] = interpretation_text(
        target_probe_score_dropped,
        logit_shift,
        continuation_changes,
    )

    # Backwards-compatible top-level fields for older consumers.
    summary.update(
        {
            "activations": activations_path,
            "output": output_path,
            "direction_output": direction_output,
            "task": task,
            "layer": int(layer),
            "class_label": direction_info.get("selected_class"),
            "accuracy_before": float(before_acc),
            "accuracy_after": float(after_acc),
            "accuracy_drop": accuracy_drop,
            "target_probe_score_dropped": target_probe_score_dropped,
            "logit_shift": logit_shift,
            "continuation_changes": continuation_changes,
        }
    )
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_summary(summary: dict[str, Any]) -> str:
    probe = summary["probe_accuracy"]
    intervention = summary["intervention"]
    downstream = summary["downstream"]
    lines = [
        "# Causal Intervention Summary",
        "",
        "This report describes a probe-direction removal experiment. A probe-score drop is evidence about decodability, not behavioral causality.",
        "",
        "## Intervention",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| task | {_fmt(intervention.get('task'))} |",
        f"| layer | {_fmt(intervention.get('layer'))} |",
        f"| class label / direction | {_fmt(intervention.get('selected_class'))} |",
        f"| direction output | {_fmt(summary['outputs'].get('direction'))} |",
        "",
        "## Probe Decodability",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| accuracy before | {_fmt(probe.get('before'))} |",
        f"| accuracy after | {_fmt(probe.get('after'))} |",
        f"| accuracy drop | {_fmt(probe.get('drop'))} |",
        f"| target probe score dropped | {_fmt(probe.get('target_probe_score_dropped'))} |",
        "",
        "## Downstream Checks",
        "",
    ]
    logit_shift = downstream.get("logit_shift")
    if logit_shift is None:
        lines.append("Logit shift: missing")
    else:
        lines.extend(
            [
                "| logit metric | value |",
                "| --- | --- |",
                f"| mean abs shift | {_fmt(logit_shift.get('mean_abs_shift'))} |",
                f"| max abs shift | {_fmt(logit_shift.get('max_abs_shift'))} |",
                f"| top token changed | {_fmt(logit_shift.get('top_token_changed'))} |",
                f"| changed | {_fmt(logit_shift.get('changed'))} |",
            ]
        )
    lines.append("")
    continuation_changes = downstream.get("continuation_changes")
    if continuation_changes is None:
        lines.append("Continuation changes: missing")
    else:
        lines.extend(
            [
                "| continuation metric | value |",
                "| --- | --- |",
                f"| compared | {_fmt(continuation_changes.get('compared'))} |",
                f"| changed | {_fmt(continuation_changes.get('changed'))} |",
                f"| change rate | {_fmt(continuation_changes.get('change_rate'))} |",
                f"| changed any | {_fmt(continuation_changes.get('changed_any'))} |",
            ]
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {line}" for line in summary.get("interpretation", []))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="remove a linear probe direction from activations")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--probe-results", required=True, help=".npz from train_linear_probe.py")
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", required=True, help="intervened activations .npy")
    parser.add_argument("--direction-output", default=None, help="exported direction .npz")
    parser.add_argument("--class-label", default=None)
    parser.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--summary-md-output", default=None)
    parser.add_argument("--logits-before", default=None)
    parser.add_argument("--logits-after", default=None)
    parser.add_argument("--continuations-before", default=None)
    parser.add_argument("--continuations-after", default=None)
    parser.add_argument("--require-activation-provenance", action="store_true")
    parser.add_argument("--allow-label-revealed-prompts", action="store_true")
    parser.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be at least 1")
    for before, after, label in (
        (args.logits_before, args.logits_after, "logits"),
        (args.continuations_before, args.continuations_after, "continuations"),
    ):
        if bool(before) != bool(after):
            parser.error(f"--{label}-before and --{label}-after must be provided together")

    activations = load_activations(args.activations)
    rows = load_rows(args.stimuli)
    activation_metadata = load_activation_metadata(args.activations)
    validate_activation_provenance(
        args.activations,
        tuple(activations.shape),
        args.stimuli,
        activation_metadata,
        require=args.require_activation_provenance,
    )
    leakage_report = enforce_prompt_contract(
        rows,
        [args.task],
        activation_metadata,
        allow_label_revealed=args.allow_label_revealed_prompts,
        allow_unverifiable=args.allow_unverifiable_prompt_contract,
        context="offline direction intervention",
    )
    probe_contract = validate_probe_artifact_contract(
        args.probe_results,
        args.activations,
        tuple(activations.shape),
        allow_label_revealed=args.allow_label_revealed_prompts,
        allow_unverifiable=args.allow_unverifiable_prompt_contract,
    )
    if args.max_rows is not None:
        activations = activations[: args.max_rows]
        rows = rows[: args.max_rows]
    activations = validate_activation_tensor(
        activations,
        args.activations,
        expected_rows=len(rows),
    )
    if args.layer < 0 or args.layer >= activations.shape[1]:
        raise ValueError(f"layer {args.layer} out of range for {activations.shape[1]} layers")

    direction_info = load_probe_direction(
        args.probe_results,
        args.task,
        args.layer,
        args.class_label,
    )
    direction = direction_info["direction"]
    intervened = remove_direction(activations, args.layer, direction)
    atomic_save_npy(args.output, intervened.astype(np.float32))

    labels = load_labels(rows, args.task)
    probe_evaluation = nested_direction_probe_scores(
        activations,
        labels,
        args.layer,
        args.class_label,
        args.probe_kind,
        args.folds,
    )
    before_acc = probe_evaluation["before"]
    after_acc = probe_evaluation["after"]

    direction_output = args.direction_output or args.output.replace(".npy", "_direction.npz")
    atomic_savez(
        direction_output,
        task=args.task,
        layer=args.layer,
        classes=np.array(direction_info["classes"], dtype=str),
        selected_class=np.array(direction_info["selected_class"] or "", dtype=str),
        direction=direction,
        norm_before_normalization=direction_info["norm_before_normalization"],
        intervention="orthogonal_projection_removal",
        weight_space="raw_activation",
        evaluation=np.array(
            "exported_full_data_direction_for_descriptive_application; "
            "scores_use_nested_cross_validation"
        ),
        activations_sha256=np.array(sha256_file(args.activations)),
        probe_results_sha256=np.array(sha256_file(args.probe_results)),
        stimuli_sha256=np.array(sha256_file(args.stimuli)),
    )

    logit_shift = summarize_logits(args.logits_before, args.logits_after)
    continuation_changes = summarize_continuations(
        args.continuations_before,
        args.continuations_after,
    )
    summary = build_summary(
        activations_path=args.activations,
        output_path=args.output,
        direction_output=direction_output,
        task=args.task,
        layer=args.layer,
        class_label=args.class_label,
        direction_info=direction_info,
        before_acc=before_acc,
        after_acc=after_acc,
        probe_evaluation=probe_evaluation,
        logit_shift=logit_shift,
        continuation_changes=continuation_changes,
    )
    summary["provenance"] = {
        "activations_sha256": sha256_file(args.activations),
        "probe_results_sha256": sha256_file(args.probe_results),
        "stimuli_sha256": sha256_file(args.stimuli),
        "intervened_activations_sha256": sha256_file(args.output),
        "prompt_leakage_audit": leakage_report,
        "probe_contract": probe_contract,
    }
    summary["claims"]["downstream_artifacts_linked_to_intervention_by_this_script"] = False
    intervention_metadata = Path(args.output).with_name(
        f"{Path(args.output).stem}_metadata.json"
    )
    atomic_write_text(
        intervention_metadata,
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "offline_direction_removed_activations",
                "source_activations": args.activations,
                "source_activations_sha256": sha256_file(args.activations),
                "output": args.output,
                "output_sha256": sha256_file(args.output),
                "activation_shape": list(intervened.shape),
                "stimuli": args.stimuli,
                "stimuli_sha256": sha256_file(args.stimuli),
                "task": args.task,
                "layer": args.layer,
                "prompt_leakage_audit": leakage_report,
                "evaluation_scope": (
                    "the saved tensor uses the exported full-data direction; reported probe "
                    "scores use fold-specific directions selected without held-out labels"
                ),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    summary_output = args.summary_output or args.output.replace(".npy", "_summary.json")
    atomic_write_text(
        summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    summary_md_output = args.summary_md_output or args.output.replace(".npy", "_summary.md")
    atomic_write_text(
        summary_md_output,
        render_markdown_summary(summary),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"wrote {args.output}")
    print(f"wrote {direction_output}")
    print(f"wrote {summary_output}")
    print(f"wrote {summary_md_output}")


if __name__ == "__main__":
    main()
    enforce_prompt_contract,
    load_activation_metadata,
    validate_activation_provenance,
