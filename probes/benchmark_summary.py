"""summarize benchmark artifacts into one JSON report.

The summary is intentionally conservative: it reports files that exist and
extracts stable aggregate metrics from NPZ/JSON artifacts without making
research claims. Missing optional artifacts are recorded as missing so dry runs
and partial benchmark runs remain inspectable.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .train_linear_probe import atomic_write_text
except ImportError:  # direct script execution
    from train_linear_probe import atomic_write_text


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_load_npz(path: str | None):
    if not path or not Path(path).is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except ValueError as error:
        raise ValueError(
            f"unsafe or invalid NPZ artifact {path}; object arrays are not accepted"
        ) from error


def _safe_load_json(path: str | None):
    if not path or not Path(path).is_file():
        return None
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
    )


def _strict_json_text(value: str, context: str):
    def reject_constant(constant):
        raise ValueError(f"non-standard JSON constant {constant!r} in {context}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {context}") from error


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data: dict, key: str):
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"artifact field {key!r} must be scalar")
    return value.reshape(-1)[0].item()


def _finite_vector(data: dict, key: str) -> np.ndarray:
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"artifact field {key!r} must be a non-empty finite vector")
    return value


def _finite_scalar(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" and >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{field} must be finite{qualifier}")
    return result


def _bounded_scalar(value: Any, field: str, low: float, high: float) -> float:
    result = _finite_scalar(value, field)
    if not low <= result <= high:
        raise ValueError(f"{field} must be in [{low}, {high}]")
    return result


def summarize_probe(path: str | None) -> dict:
    data = _safe_load_npz(path)
    if data is None:
        return {"path": path, "exists": False}

    if "tasks" not in data:
        raise ValueError("probe artifact is missing tasks")
    task_array = np.asarray(data["tasks"])
    if task_array.ndim != 1:
        raise ValueError("probe artifact tasks must be a vector")
    tasks = [str(t) for t in task_array.tolist()]
    if not tasks or any(not task for task in tasks) or len(tasks) != len(set(tasks)):
        raise ValueError("probe artifact tasks must be non-empty and unique")
    summary = {
        "path": path,
        "exists": True,
        "sha256": _sha256(path),
        "schema_version": _scalar(data, "schema_version"),
        "probe_kind": _scalar(data, "probe_kind"),
        "root_split": _scalar(data, "root_split"),
        "pattern_split": _scalar(data, "pattern_split"),
        "default_split_policy": (
            _scalar(data, "default_split_policy")
        ),
        "split_policy": _scalar(data, "split_policy"),
        "activations_sha256": _scalar(data, "activations_sha256"),
        "stimuli_sha256": _scalar(data, "stimuli_sha256"),
        "tasks": tasks,
        "task_metrics": {},
    }
    if "task_split_policy_json" in data:
        summary["split_policy_metadata"] = _strict_json_text(
            str(_scalar(data, "task_split_policy_json")), "task_split_policy_json"
        )
    elif "split_policy_json" in data:
        summary["split_policy_metadata"] = _strict_json_text(
            str(_scalar(data, "split_policy_json")), "split_policy_json"
        )
    if "prompt_leakage_audit_json" in data:
        leakage = _strict_json_text(
            str(_scalar(data, "prompt_leakage_audit_json")),
            "prompt_leakage_audit_json",
        )
        if not isinstance(leakage, dict) or not isinstance(leakage.get("status"), str):
            raise ValueError("probe prompt leakage audit must contain a string status")
        summary["prompt_leakage_audit"] = leakage
    for task in tasks:
        key = "".join(c if c.isalnum() or c in "_-" else "_" for c in task)
        acc_key = f"{key}_accuracy"
        if acc_key not in data:
            continue
        acc = _finite_vector(data, acc_key)
        if np.any((acc < 0.0) | (acc > 1.0)):
            raise ValueError(f"probe accuracies for {task!r} are outside [0, 1]")
        task_summary = {
            "best_layer": int(np.argmax(acc)),
            "best_accuracy": float(np.max(acc)),
            "mean_accuracy": float(np.mean(acc)),
            "final_layer_accuracy": float(acc[-1]),
            "n_layers": int(len(acc)),
        }
        class_key = f"{key}_classes"
        if class_key in data:
            class_array = np.asarray(data[class_key])
            if class_array.ndim != 1:
                raise ValueError(f"classes for task {task!r} must be a vector")
            classes = [str(value) for value in class_array.tolist()]
            if not classes or len(classes) != len(set(classes)):
                raise ValueError(f"classes for task {task!r} must be non-empty and unique")
            task_summary["n_classes"] = int(len(classes))
            task_summary["classes"] = classes
        count_key = f"{key}_class_counts"
        if class_key in data and count_key in data:
            raw_counts = np.asarray(data[count_key])
            if (
                raw_counts.ndim != 1
                or len(raw_counts) != len(task_summary["classes"])
                or raw_counts.dtype.kind not in "iu"
                or np.any(raw_counts <= 0)
            ):
                raise ValueError(f"class counts for task {task!r} are invalid")
            counts = [int(value) for value in raw_counts.tolist()]
            task_summary["class_counts"] = dict(zip(task_summary["classes"], counts))
            task_summary["min_class_count"] = int(min(counts)) if counts else None
            task_summary["max_class_count"] = int(max(counts)) if counts else None
        chance_key = f"{key}_chance"
        if chance_key in data:
            chance = float(_scalar(data, chance_key))
            if not math.isfinite(chance) or not 0.0 <= chance <= 1.0:
                raise ValueError(f"invalid chance value for task {task!r}")
            task_summary["chance"] = chance
        confusion_key = f"{key}_confusion_matrices"
        if confusion_key in data:
            confusions = np.asarray(data[confusion_key])
            expected_classes = task_summary.get("n_classes")
            if (
                confusions.ndim != 3
                or confusions.shape[0] != len(acc)
                or expected_classes is None
                or confusions.shape[1:] != (expected_classes, expected_classes)
                or confusions.dtype.kind not in "iu"
                or np.any(confusions < 0)
            ):
                raise ValueError(f"confusion matrices for task {task!r} are invalid")
            best_layer = task_summary["best_layer"]
            final_layer = int(len(acc) - 1)
            task_summary["confusion_matrices"] = {
                "best_layer": confusions[best_layer].astype(int).tolist(),
                "final_layer": confusions[final_layer].astype(int).tolist(),
            }
        sel_key = f"{key}_selectivity"
        if sel_key in data:
            sel = _finite_vector(data, sel_key)
            if len(sel) != len(acc):
                raise ValueError(
                    f"selectivity for task {task!r} does not match layer count"
                )
            task_summary["best_selectivity_layer"] = int(np.argmax(sel))
            task_summary["best_selectivity"] = float(np.max(sel))
            task_summary["mean_selectivity"] = float(np.mean(sel))
        summary["task_metrics"][task] = task_summary
    return summary


def summarize_mdl(path: str | None) -> dict:
    data = _safe_load_npz(path)
    if data is None:
        return {"path": path, "exists": False}
    if "tasks" not in data:
        raise ValueError("MDL artifact is missing tasks")
    task_array = np.asarray(data["tasks"])
    if task_array.ndim != 1:
        raise ValueError("MDL tasks must be a vector")
    tasks = [str(t) for t in task_array.tolist()]
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("MDL tasks must be non-empty and unique")
    summary = {
        "path": path,
        "exists": True,
        "sha256": _sha256(path),
        "schema_version": _scalar(data, "schema_version"),
        "evaluation": _scalar(data, "evaluation"),
        "probe_kind": _scalar(data, "probe_kind"),
        "activations_sha256": _scalar(data, "activations_sha256"),
        "stimuli_sha256": _scalar(data, "stimuli_sha256"),
        "task_metrics": {},
    }
    for task in tasks:
        key = "".join(c if c.isalnum() or c in "_-" else "_" for c in task)
        auc_key = f"{key}_data_efficiency_auc"
        if auc_key not in data:
            continue
        auc = _finite_vector(data, auc_key)
        if np.any((auc < 0.0) | (auc > 1.0)):
            raise ValueError(f"MDL AUC values for {task!r} are outside [0, 1]")
        summary["task_metrics"][task] = {
            "best_layer": int(np.argmax(auc)),
            "best_auc": float(np.max(auc)),
            "mean_auc": float(np.mean(auc)),
        }
    return summary


def summarize_matrix(path: str | None, matrix_key: str) -> dict:
    data = _safe_load_npz(path)
    if data is None:
        return {"path": path, "exists": False}
    if matrix_key not in data:
        raise ValueError(f"matrix artifact is missing required field {matrix_key!r}")
    mat = np.asarray(data[matrix_key], dtype=np.float64)
    if (
        mat.ndim != 2
        or mat.shape[0] == 0
        or mat.shape[0] != mat.shape[1]
        or not np.isfinite(mat).all()
    ):
        raise ValueError(f"matrix {matrix_key!r} must be finite, non-empty, and square")
    lower, upper = ((0.0, 1.0) if matrix_key.startswith("cca") else (-1.0, 1.0))
    tolerance = 1e-8
    if np.any(mat < lower - tolerance) or np.any(mat > upper + tolerance):
        raise ValueError(f"matrix {matrix_key!r} is outside [{lower}, {upper}]")
    summary = {
        "path": path,
        "exists": True,
        "sha256": _sha256(path),
        "schema_version": _scalar(data, "schema_version"),
        "evaluation": _scalar(data, "evaluation"),
        "metric": _scalar(data, "metric"),
        "shape": list(mat.shape),
        "mean": float(np.mean(mat)),
        "max": float(np.max(mat)),
        "min": float(np.min(mat)),
    }
    for key in ("n_components", "regularization", "cv_folds"):
        if key in data:
            summary[key] = _scalar(data, key)
    if "root_pattern_cca" in data:
        values = _finite_vector(data, "root_pattern_cca")
        if len(values) != mat.shape[0] or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("root-pattern CCA must match layer count and lie in [0, 1]")
        summary["root_pattern_cca"] = {
            "min_layer": int(np.argmin(values)),
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
        }
    return summary


def summarize_divergence(path: str | None) -> dict:
    data = _safe_load_npz(path)
    if data is None:
        return {"path": path, "exists": False}
    n_correct = int(_scalar(data, "n_correct")) if "n_correct" in data else None
    n_incorrect = int(_scalar(data, "n_incorrect")) if "n_incorrect" in data else None
    if n_correct is None or n_incorrect is None or n_correct < 1 or n_incorrect < 1:
        raise ValueError("divergence artifact requires positive correct and incorrect counts")
    summary = {
        "path": path,
        "exists": True,
        "sha256": _sha256(path),
        "schema_version": _scalar(data, "schema_version"),
        "evaluation": _scalar(data, "evaluation"),
        "activations_sha256": _scalar(data, "activations_sha256"),
        "correctness_sha256": _scalar(data, "correctness_sha256"),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
    }
    if "alignment_evidence_json" in data:
        evidence = _strict_json_text(
            str(_scalar(data, "alignment_evidence_json")),
            "alignment_evidence_json",
        )
        if not isinstance(evidence, dict):
            raise ValueError("divergence alignment evidence must be a JSON object")
        summary["alignment_evidence"] = evidence
    layer_count = None
    if "activation_shape" in data:
        shape = np.asarray(data["activation_shape"])
        if (
            shape.shape != (3,)
            or shape.dtype.kind not in "iu"
            or np.any(shape <= 0)
            or int(shape[0]) != n_correct + n_incorrect
        ):
            raise ValueError("divergence activation shape or row counts are inconsistent")
        layer_count = int(shape[1])
        summary["activation_shape"] = [int(value) for value in shape]
    if "cos_dist" in data:
        values = _finite_vector(data, "cos_dist")
        if layer_count is not None and len(values) != layer_count:
            raise ValueError("cosine divergence does not match activation layer count")
        if np.any((values < -1e-8) | (values > 2.0 + 1e-8)):
            raise ValueError("cosine divergence values are outside [0, 2]")
        summary["max_cos_layer"] = int(np.argmax(values))
        summary["max_cos_dist"] = float(np.max(values))
    if "eucl_dist" in data:
        values = _finite_vector(data, "eucl_dist")
        if layer_count is not None and len(values) != layer_count:
            raise ValueError("Euclidean divergence does not match activation layer count")
        if np.any(values < 0.0):
            raise ValueError("Euclidean divergence values must be non-negative")
        summary["max_eucl_layer"] = int(np.argmax(values))
        summary["max_eucl_dist"] = float(np.max(values))
    return summary


def summarize_fertility(path: str | None) -> dict:
    data = _safe_load_json(path)
    if data is None:
        return {"path": path, "exists": False}
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("fertility report must be a JSON array of objects")
    labels: set[str] = set()
    tokenizers = []
    for index, row in enumerate(data):
        label = row.get("label")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError(f"fertility row {index} has an invalid or duplicate label")
        labels.add(label)
        total_prompts = row.get("total_prompts")
        if isinstance(total_prompts, bool) or not isinstance(total_prompts, int) or total_prompts < 1:
            raise ValueError(f"fertility row {index} has invalid total_prompts")
        summarized = {"label": label, "total_prompts": total_prompts}
        for field in ("mean_fertility", "en_ar_ratio"):
            value = row.get(field)
            summarized[field] = (
                None
                if value is None
                else _finite_scalar(value, f"fertility[{index}].{field}", minimum=0.0)
            )
        for field in ("root_split_rate", "pattern_split_rate"):
            value = row.get(field)
            summarized[field] = (
                None
                if value is None
                else _bounded_scalar(value, f"fertility[{index}].{field}", 0.0, 1.0)
            )
        for field in ("tokenizer_path", "tokenizer_sha256", "stimuli_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"fertility row {index} is missing {field}")
            summarized[field] = value
        tokenizers.append(summarized)
    return {
        "path": path,
        "exists": True,
        "sha256": _sha256(path),
        "tokenizers": tokenizers,
    }


def summarize_run(
    *,
    config: dict,
    dry_run: bool,
    commands: list[dict],
    models: list[dict],
    fertility_path: str | None = None,
    plots: list[str] | None = None,
    config_path: str | None = None,
    config_sha256: str | None = None,
) -> dict:
    fertility_config = config.get("fertility", False)
    if isinstance(fertility_config, dict):
        fertility_enabled = fertility_config.get("enabled", False)
    else:
        fertility_enabled = fertility_config
    if not isinstance(fertility_enabled, bool):
        raise ValueError("fertility enabled flag must be boolean")

    def artifact(
        kind: str,
        path: str | None,
        matrix_key: str | None = None,
        enabled: bool = True,
    ):
        if not enabled:
            return {"path": path, "exists": False, "status": "disabled"}
        if dry_run:
            return {"path": path, "exists": False, "status": "planned_dry_run"}
        if kind == "probe":
            return summarize_probe(path)
        if kind == "mdl":
            return summarize_mdl(path)
        if kind == "matrix":
            return summarize_matrix(path, matrix_key)
        if kind == "divergence":
            return summarize_divergence(path)
        raise AssertionError(kind)

    stimuli = config.get("stimuli")
    stimuli_sha256 = (
        _sha256(stimuli)
        if isinstance(stimuli, str) and Path(stimuli).is_file()
        else None
    )
    return {
        "schema_version": 2,
        "name": config.get("name"),
        "dry_run": dry_run,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "stimuli": stimuli,
        "stimuli_sha256": stimuli_sha256,
        "tasks": config.get("tasks", ["root", "pattern"]),
        "split_policy": config.get("split_policy"),
        "analysis_configuration": {
            "probe_kind": config.get("probe_kind", "linear"),
            "folds": config.get("folds", 5),
            "cca_components": config.get("cca_components", 10),
            "cca_regularization": config.get("cca_reg", 1e-4),
            "cca_folds": config.get("cca_folds", 5),
            "rsa_metric": config.get("rsa_metric", "correlation"),
            "label_revealed_prompts_allowed": config.get(
                "allow_label_revealed_prompts", False
            ),
            "unverifiable_prompt_contract_allowed": config.get(
                "allow_unverifiable_prompt_contract", False
            ),
        },
        "command_count": len(commands),
        "models": [
            {
                "label": model["label"],
                "kind": model.get("kind"),
                "activations": model.get("activations"),
                "probe": artifact(
                    "probe",
                    model.get("probes"),
                    enabled=model.get("enabled", {}).get("probe", True),
                ),
                "mdl": artifact(
                    "mdl",
                    model.get("mdl"),
                    enabled=model.get("enabled", {}).get("mdl", True),
                ),
                "cca": artifact(
                    "matrix",
                    model.get("cca"),
                    "cca_layer_matrix",
                    model.get("enabled", {}).get("cca", True),
                ),
                "rsa": artifact(
                    "matrix",
                    model.get("rsa"),
                    "rsa_layer_matrix",
                    model.get("enabled", {}).get("rsa", True),
                ),
                "divergence": artifact(
                    "divergence",
                    model.get("divergence"),
                    enabled=model.get("enabled", {}).get("divergence", True),
                ),
            }
            for model in models
        ],
        "fertility": (
            {"path": fertility_path, "exists": False, "status": "planned_dry_run"}
            if dry_run
            else {"path": fertility_path, "exists": False, "status": "disabled"}
            if not fertility_enabled
            else summarize_fertility(fertility_path)
        ),
        "plots": [
            {"path": path, "exists": False if dry_run else Path(path).is_file()}
            for path in (plots or [])
        ],
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="summarize benchmark artifacts")
    parser.add_argument("--run-metadata", required=True, help="JSON from run_benchmark.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    meta = _safe_load_json(args.run_metadata)
    if not isinstance(meta, dict):
        raise ValueError("run metadata must be a JSON object")
    summary = summarize_run(
        config=meta["config"],
        dry_run=meta["dry_run"],
        commands=meta["commands"],
        models=meta.get("model_artifacts", []),
        fertility_path=meta.get("fertility_path"),
        plots=meta.get("plots", []),
        config_path=meta.get("config_path"),
        config_sha256=meta.get("config_sha256"),
    )
    atomic_write_text(
        args.output,
        json.dumps(
            summary,
            default=_jsonable,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
