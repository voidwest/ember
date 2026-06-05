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

from train_linear_probe import (
    load_activations,
    load_labels,
    load_rows,
    safe_key,
    train_probes,
)


def load_probe_direction(probe_path: str, task: str, layer: int, class_label: str | None) -> dict:
    data = np.load(probe_path, allow_pickle=True)
    key = safe_key(task)
    weight_key = f"{key}_probe_weights"
    class_key = f"{key}_classes"
    if weight_key not in data:
        raise ValueError(f"{probe_path} does not contain linear weights for task '{task}'")
    weights = np.asarray(data[weight_key][layer], dtype=np.float32)
    classes = [str(c) for c in data[class_key].tolist()] if class_key in data else []

    if weights.ndim == 1:
        direction = weights
        selected_class = class_label
    elif weights.shape[0] == 1:
        direction = weights[0]
        selected_class = classes[0] if classes else class_label
    else:
        class_index = 0
        if class_label is not None:
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
    }


def remove_direction(activations: np.ndarray, layer: int, direction: np.ndarray) -> np.ndarray:
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


def summarize_logits(before_path: str | None, after_path: str | None) -> dict | None:
    if not before_path or not after_path:
        return None
    before = np.load(before_path).astype(np.float32)
    after = np.load(after_path).astype(np.float32)
    if before.shape != after.shape:
        raise ValueError(f"logit shape mismatch: {before.shape} vs {after.shape}")
    diff = after - before
    max_abs_shift = float(np.max(np.abs(diff)))
    return {
        "before": before_path,
        "after": after_path,
        "shape": list(before.shape),
        "mean_abs_shift": float(np.mean(np.abs(diff))),
        "max_abs_shift": max_abs_shift,
        "top_token_changed": bool(np.argmax(before.reshape(-1)) != np.argmax(after.reshape(-1))),
        "changed": bool(max_abs_shift > 0.0),
    }


def summarize_continuations(before_path: str | None, after_path: str | None) -> dict | None:
    if not before_path or not after_path:
        return None
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))
    n = min(len(before), len(after))
    changes = 0
    for i in range(n):
        b = before[i].get("generated") if isinstance(before[i], dict) else before[i]
        a = after[i].get("generated") if isinstance(after[i], dict) else after[i]
        changes += int(b != a)
    return {
        "before": before_path,
        "after": after_path,
        "compared": n,
        "changed": changes,
        "change_rate": float(changes / max(n, 1)),
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
    logit_shift: dict | None = None,
    continuation_changes: dict | None = None,
) -> dict[str, Any]:
    accuracy_drop = float(before_acc - after_acc)
    target_probe_score_dropped = bool(accuracy_drop > 0.0)
    logits_changed = bool(logit_shift and logit_shift.get("changed"))
    continuations_changed = bool(continuation_changes and continuation_changes.get("changed_any"))
    summary = {
        "schema_version": 1,
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
        },
        "probe_accuracy": {
            "before": float(before_acc),
            "after": float(after_acc),
            "drop": accuracy_drop,
            "target_probe_score_dropped": target_probe_score_dropped,
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
    args = parser.parse_args()

    activations = load_activations(args.activations)
    rows = load_rows(args.stimuli)
    if args.max_rows is not None:
        activations = activations[: args.max_rows]
        rows = rows[: args.max_rows]
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, intervened.astype(np.float32))

    labels = load_labels(rows, args.task)
    before_acc = single_layer_probe_score(
        activations,
        labels,
        args.layer,
        args.probe_kind,
        args.folds,
    )
    after_acc = single_layer_probe_score(
        intervened,
        labels,
        args.layer,
        args.probe_kind,
        args.folds,
    )

    direction_output = args.direction_output or args.output.replace(".npy", "_direction.npz")
    np.savez(
        direction_output,
        task=args.task,
        layer=args.layer,
        classes=np.array(direction_info["classes"], dtype=object),
        selected_class=direction_info["selected_class"],
        direction=direction,
        norm_before_normalization=direction_info["norm_before_normalization"],
        intervention="orthogonal_projection_removal",
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
        logit_shift=logit_shift,
        continuation_changes=continuation_changes,
    )
    summary_output = args.summary_output or args.output.replace(".npy", "_summary.json")
    Path(summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_output = args.summary_md_output or args.output.replace(".npy", "_summary.md")
    Path(summary_md_output).write_text(
        render_markdown_summary(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    print(f"wrote {direction_output}")
    print(f"wrote {summary_output}")
    print(f"wrote {summary_md_output}")


if __name__ == "__main__":
    main()
