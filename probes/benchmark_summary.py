"""summarize benchmark artifacts into one JSON report.

The summary is intentionally conservative: it reports files that exist and
extracts stable aggregate metrics from NPZ/JSON artifacts without making
research claims. Missing optional artifacts are recorded as missing so dry runs
and partial benchmark runs remain inspectable.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_load_npz(path: str | None):
    if not path or not Path(path).exists():
        return None
    return np.load(path, allow_pickle=True)


def _safe_load_json(path: str | None):
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_probe(path: str | None) -> dict:
    data = _safe_load_npz(path)
    if data is None:
        return {"path": path, "exists": False}

    tasks = [str(t) for t in data["tasks"].tolist()] if "tasks" in data else []
    summary = {
        "path": path,
        "exists": True,
        "probe_kind": str(data["probe_kind"]) if "probe_kind" in data else None,
        "root_split": str(data["root_split"]) if "root_split" in data else None,
        "pattern_split": str(data["pattern_split"]) if "pattern_split" in data else None,
        "default_split_policy": (
            str(data["default_split_policy"]) if "default_split_policy" in data else None
        ),
        "split_policy": str(data["split_policy"]) if "split_policy" in data else None,
        "tasks": tasks,
        "task_metrics": {},
    }
    if "task_split_policy_json" in data:
        summary["split_policy_metadata"] = json.loads(str(data["task_split_policy_json"]))
    elif "split_policy_json" in data:
        summary["split_policy_metadata"] = json.loads(str(data["split_policy_json"]))
    for task in tasks:
        key = "".join(c if c.isalnum() or c in "_-" else "_" for c in task)
        acc_key = f"{key}_accuracy"
        if acc_key not in data:
            continue
        acc = data[acc_key]
        task_summary = {
            "best_layer": int(np.argmax(acc)),
            "best_accuracy": float(np.max(acc)),
            "mean_accuracy": float(np.mean(acc)),
            "final_layer_accuracy": float(acc[-1]),
            "n_layers": int(len(acc)),
        }
        class_key = f"{key}_classes"
        if class_key in data:
            classes = [str(value) for value in data[class_key].tolist()]
            task_summary["n_classes"] = int(len(classes))
            task_summary["classes"] = classes
        count_key = f"{key}_class_counts"
        if class_key in data and count_key in data:
            counts = [int(value) for value in data[count_key].tolist()]
            task_summary["class_counts"] = dict(zip(task_summary["classes"], counts))
            task_summary["min_class_count"] = int(min(counts)) if counts else None
            task_summary["max_class_count"] = int(max(counts)) if counts else None
        chance_key = f"{key}_chance"
        if chance_key in data:
            task_summary["chance"] = float(data[chance_key])
        confusion_key = f"{key}_confusion_matrices"
        if confusion_key in data:
            confusions = data[confusion_key]
            best_layer = task_summary["best_layer"]
            final_layer = int(len(acc) - 1)
            task_summary["confusion_matrices"] = {
                "best_layer": confusions[best_layer].astype(int).tolist(),
                "final_layer": confusions[final_layer].astype(int).tolist(),
            }
        sel_key = f"{key}_selectivity"
        if sel_key in data:
            sel = data[sel_key]
            task_summary["best_selectivity_layer"] = int(np.argmax(sel))
            task_summary["best_selectivity"] = float(np.max(sel))
            task_summary["mean_selectivity"] = float(np.mean(sel))
        summary["task_metrics"][task] = task_summary
    return summary


def summarize_mdl(path: str | None) -> dict:
    data = _safe_load_npz(path)
    if data is None:
        return {"path": path, "exists": False}
    tasks = [str(t) for t in data["tasks"].tolist()] if "tasks" in data else []
    summary = {"path": path, "exists": True, "task_metrics": {}}
    for task in tasks:
        key = "".join(c if c.isalnum() or c in "_-" else "_" for c in task)
        auc_key = f"{key}_data_efficiency_auc"
        if auc_key not in data:
            continue
        auc = data[auc_key]
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
    summary = {"path": path, "exists": True}
    if matrix_key in data:
        mat = data[matrix_key]
        summary.update(
            {
                "shape": list(mat.shape),
                "mean": float(np.nanmean(mat)),
                "max": float(np.nanmax(mat)),
                "min": float(np.nanmin(mat)),
            }
        )
    if "root_pattern_cca" in data:
        values = data["root_pattern_cca"]
        summary["root_pattern_cca"] = {
            "min_layer": int(np.nanargmin(values)),
            "min": float(np.nanmin(values)),
            "mean": float(np.nanmean(values)),
        }
    return summary


def summarize_divergence(path: str | None) -> dict:
    data = _safe_load_npz(path)
    if data is None:
        return {"path": path, "exists": False}
    summary = {
        "path": path,
        "exists": True,
        "n_correct": int(data["n_correct"]) if "n_correct" in data else None,
        "n_incorrect": int(data["n_incorrect"]) if "n_incorrect" in data else None,
    }
    if "cos_dist" in data and not np.isnan(data["cos_dist"]).all():
        summary["max_cos_layer"] = int(np.nanargmax(data["cos_dist"]))
        summary["max_cos_dist"] = float(np.nanmax(data["cos_dist"]))
    if "eucl_dist" in data and not np.isnan(data["eucl_dist"]).all():
        summary["max_eucl_layer"] = int(np.nanargmax(data["eucl_dist"]))
        summary["max_eucl_dist"] = float(np.nanmax(data["eucl_dist"]))
    return summary


def summarize_fertility(path: str | None) -> dict:
    data = _safe_load_json(path)
    if data is None:
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "tokenizers": [
            {
                "label": row.get("label"),
                "mean_fertility": row.get("mean_fertility"),
                "en_ar_ratio": row.get("en_ar_ratio"),
                "root_split_rate": row.get("root_split_rate"),
                "pattern_split_rate": row.get("pattern_split_rate"),
            }
            for row in data
        ],
    }


def summarize_run(
    *,
    config: dict,
    dry_run: bool,
    commands: list[dict],
    models: list[dict],
    fertility_path: str | None = None,
    plots: list[str] | None = None,
) -> dict:
    return {
        "name": config.get("name"),
        "dry_run": dry_run,
        "stimuli": config.get("stimuli"),
        "tasks": config.get("tasks", ["root", "pattern"]),
        "split_policy": config.get("split_policy"),
        "command_count": len(commands),
        "models": [
            {
                "label": model["label"],
                "kind": model.get("kind"),
                "activations": model.get("activations"),
                "probe": summarize_probe(model.get("probes")),
                "mdl": summarize_mdl(model.get("mdl")),
                "cca": summarize_matrix(model.get("cca"), "cca_layer_matrix"),
                "rsa": summarize_matrix(model.get("rsa"), "rsa_layer_matrix"),
                "divergence": summarize_divergence(model.get("divergence")),
            }
            for model in models
        ],
        "fertility": summarize_fertility(fertility_path),
        "plots": [
            {"path": path, "exists": Path(path).exists()}
            for path in (plots or [])
        ],
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="summarize benchmark artifacts")
    parser.add_argument("--run-metadata", required=True, help="JSON from run_benchmark.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    meta = json.loads(Path(args.run_metadata).read_text(encoding="utf-8"))
    summary = summarize_run(
        config=meta["config"],
        dry_run=meta["dry_run"],
        commands=meta["commands"],
        models=meta.get("model_artifacts", []),
        fertility_path=meta.get("fertility_path"),
        plots=meta.get("plots", []),
    )
    Path(args.output).write_text(
        json.dumps(summary, default=_jsonable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
