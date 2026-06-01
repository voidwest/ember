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
    return {
        "before": before_path,
        "after": after_path,
        "shape": list(before.shape),
        "mean_abs_shift": float(np.mean(np.abs(diff))),
        "max_abs_shift": float(np.max(np.abs(diff))),
        "top_token_changed": bool(np.argmax(before.reshape(-1)) != np.argmax(after.reshape(-1))),
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
    }


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

    summary = {
        "activations": args.activations,
        "output": args.output,
        "direction_output": direction_output,
        "task": args.task,
        "layer": args.layer,
        "class_label": direction_info["selected_class"],
        "accuracy_before": before_acc,
        "accuracy_after": after_acc,
        "accuracy_drop": before_acc - after_acc,
        "logit_shift": summarize_logits(args.logits_before, args.logits_after),
        "continuation_changes": summarize_continuations(
            args.continuations_before,
            args.continuations_after,
        ),
    }
    summary_output = args.summary_output or args.output.replace(".npy", "_summary.json")
    Path(summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    print(f"wrote {direction_output}")
    print(f"wrote {summary_output}")


if __name__ == "__main__":
    main()
