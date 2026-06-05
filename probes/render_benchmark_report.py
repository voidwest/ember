#!/usr/bin/env python3
"""Render a conservative Markdown report from benchmark_summary.json."""

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


ARTIFACT_KEYS = ["probe", "mdl", "cca", "rsa", "divergence"]


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def fmt(value: Any) -> str:
    if _missing(value):
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, ensure_ascii=False, sort_keys=True) + "`"
    return str(value)


def md_escape(value: Any) -> str:
    return fmt(value).replace("|", "\\|").replace("\n", " ")


def boolish(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "missing"
    if value is None:
        return "missing"
    return fmt(value)


def artifact_exists(model: dict[str, Any], key: str) -> str:
    artifact = model.get(key)
    if not isinstance(artifact, dict):
        return "missing"
    return boolish(artifact.get("exists"))


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    if rows:
        for row in rows:
            lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    else:
        lines.append("| " + " | ".join(["missing"] * len(headers)) + " |")
    return lines


def command_text(command: Any) -> str:
    if isinstance(command, list):
        return shlex.join(str(part) for part in command)
    if isinstance(command, str):
        return command
    return json.dumps(command, ensure_ascii=False, sort_keys=True)


def model_sort_key(model: dict[str, Any]) -> tuple[str, str]:
    return (str(model.get("label") or ""), str(model.get("kind") or ""))


def model_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for model in sorted(summary.get("models") or [], key=model_sort_key):
        rows.append(
            [
                model.get("label"),
                model.get("kind"),
                model.get("activations"),
                artifact_exists(model, "probe"),
                artifact_exists(model, "mdl"),
                artifact_exists(model, "cca"),
                artifact_exists(model, "rsa"),
                artifact_exists(model, "divergence"),
            ]
        )
    return rows


def probe_metric_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for model in sorted(summary.get("models") or [], key=model_sort_key):
        label = model.get("label")
        probe = model.get("probe")
        if not isinstance(probe, dict) or not probe.get("exists"):
            rows.append([label, "missing", "missing", "missing", "missing", "missing", "missing"])
            continue
        metrics = probe.get("task_metrics")
        if not isinstance(metrics, dict) or not metrics:
            rows.append([label, "missing", "missing", "missing", "missing", "missing", "missing"])
            continue
        for task in sorted(metrics):
            task_metrics = metrics[task] if isinstance(metrics[task], dict) else {}
            rows.append(
                [
                    label,
                    task,
                    task_metrics.get("best_layer"),
                    task_metrics.get("best_accuracy"),
                    task_metrics.get("mean_accuracy"),
                    task_metrics.get("n_classes"),
                    task_metrics.get("best_selectivity"),
                ]
            )
    return rows


def mdl_metric_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for model in sorted(summary.get("models") or [], key=model_sort_key):
        label = model.get("label")
        mdl = model.get("mdl")
        if not isinstance(mdl, dict) or not mdl.get("exists"):
            continue
        metrics = mdl.get("task_metrics")
        if not isinstance(metrics, dict) or not metrics:
            rows.append([label, "missing", "missing", "missing", "missing"])
            continue
        for task in sorted(metrics):
            task_metrics = metrics[task] if isinstance(metrics[task], dict) else {}
            rows.append(
                [
                    label,
                    task,
                    task_metrics.get("best_layer"),
                    task_metrics.get("best_auc"),
                    task_metrics.get("mean_auc"),
                ]
            )
    return rows


def matrix_rows(summary: dict[str, Any], key: str) -> list[list[Any]]:
    rows = []
    for model in sorted(summary.get("models") or [], key=model_sort_key):
        artifact = model.get(key)
        if not isinstance(artifact, dict) or not artifact.get("exists"):
            continue
        root_pattern = artifact.get("root_pattern_cca")
        if not isinstance(root_pattern, dict):
            root_pattern = {}
        rows.append(
            [
                model.get("label"),
                key.upper(),
                artifact.get("path"),
                artifact.get("shape"),
                artifact.get("mean"),
                artifact.get("min"),
                artifact.get("max"),
                root_pattern.get("min_layer"),
                root_pattern.get("min"),
                root_pattern.get("mean"),
            ]
        )
    return rows


def fertility_rows(summary: dict[str, Any]) -> list[list[Any]]:
    fertility = summary.get("fertility")
    if not isinstance(fertility, dict) or not fertility.get("exists"):
        return []
    rows = []
    for row in sorted(fertility.get("tokenizers") or [], key=lambda item: str(item.get("label") or "")):
        rows.append(
            [
                row.get("label"),
                row.get("mean_fertility"),
                row.get("en_ar_ratio"),
                row.get("root_split_rate"),
                row.get("pattern_split_rate"),
            ]
        )
    return rows


def missing_artifact_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for model in sorted(summary.get("models") or [], key=model_sort_key):
        for key in ARTIFACT_KEYS:
            artifact = model.get(key)
            if not isinstance(artifact, dict):
                rows.append([model.get("label"), key, "missing", "artifact entry missing"])
            elif artifact.get("exists") is not True:
                rows.append([model.get("label"), key, artifact.get("path"), "exists false"])

    fertility = summary.get("fertility")
    if not isinstance(fertility, dict):
        rows.append(["benchmark", "fertility", "missing", "artifact entry missing"])
    elif fertility.get("exists") is not True:
        rows.append(["benchmark", "fertility", fertility.get("path"), "exists false"])

    for plot in summary.get("plots") or []:
        if isinstance(plot, dict) and plot.get("exists") is not True:
            rows.append(["benchmark", "plot", plot.get("path"), "exists false"])
    return rows


def command_lines(summary: dict[str, Any]) -> list[str]:
    commands = summary.get("commands") or []
    if not commands:
        return ["missing"]

    lines = []
    for index, entry in enumerate(commands, start=1):
        if isinstance(entry, dict):
            meta = []
            if "dry_run" in entry:
                meta.append(f"dry_run={fmt(entry.get('dry_run'))}")
            if "skipped" in entry:
                meta.append(f"skipped={fmt(entry.get('skipped'))}")
            if entry.get("reason"):
                meta.append(f"reason={fmt(entry.get('reason'))}")
            suffix = f" ({', '.join(meta)})" if meta else ""
            cmd = command_text(entry.get("cmd"))
        else:
            suffix = ""
            cmd = command_text(entry)
        lines.extend([f"Command {index}{suffix}:", "", "```bash", cmd, "```", ""])
    return lines


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Report: {fmt(summary.get('name'))}",
        "",
        "This report summarizes benchmark artifacts and probe decodability metrics only. It does not infer scientific conclusions.",
        "",
        "## Benchmark name",
        "",
        fmt(summary.get("name")),
        "",
        "## Dry run status",
        "",
        fmt(summary.get("dry_run")),
        "",
        "## Stimuli path",
        "",
        fmt(summary.get("stimuli")),
        "",
        "## Tasks",
        "",
    ]
    tasks = summary.get("tasks")
    if isinstance(tasks, list) and tasks:
        lines.extend(f"- {fmt(task)}" for task in tasks)
    else:
        lines.append("missing")
    lines.extend(
        [
            "",
            "## Split policy",
            "",
            fmt(summary.get("split_policy")),
            "",
            "## Models",
            "",
        ]
    )
    lines.extend(
        table(
            [
                "label",
                "kind",
                "activation path",
                "probe exists",
                "MDL exists",
                "CCA exists",
                "RSA exists",
                "divergence exists",
            ],
            model_rows(summary),
        )
    )
    lines.extend(["", "## Probe decodability metrics", ""])
    lines.extend(
        table(
            [
                "model",
                "task",
                "best layer",
                "best accuracy",
                "mean accuracy",
                "n classes",
                "best selectivity",
            ],
            probe_metric_rows(summary),
        )
    )
    lines.extend(["", "## MDL metrics", ""])
    mdl_rows = mdl_metric_rows(summary)
    if mdl_rows:
        lines.extend(table(["model", "task", "best layer", "best AUC", "mean AUC"], mdl_rows))
    else:
        lines.append("missing")

    lines.extend(["", "## CCA/RSA summaries", ""])
    matrix_summary_rows = matrix_rows(summary, "cca") + matrix_rows(summary, "rsa")
    if matrix_summary_rows:
        lines.extend(
            table(
                [
                    "model",
                    "artifact",
                    "path",
                    "shape",
                    "mean",
                    "min",
                    "max",
                    "root-pattern min layer",
                    "root-pattern min",
                    "root-pattern mean",
                ],
                matrix_summary_rows,
            )
        )
    else:
        lines.append("missing")

    lines.extend(["", "## Fertility summary", ""])
    fertility = summary.get("fertility")
    if isinstance(fertility, dict):
        lines.append(f"Path: {fmt(fertility.get('path'))}")
        lines.append("")
    fert_rows = fertility_rows(summary)
    if fert_rows:
        lines.extend(
            table(
                [
                    "tokenizer",
                    "mean fertility",
                    "EN/AR ratio",
                    "root split rate",
                    "pattern split rate",
                ],
                fert_rows,
            )
        )
    else:
        lines.append("missing")

    lines.extend(["", "## Missing artifacts", ""])
    missing_rows = missing_artifact_rows(summary)
    if missing_rows:
        lines.extend(table(["scope", "artifact", "path", "reason"], missing_rows))
    else:
        lines.append("No missing artifacts recorded in the summary.")

    lines.extend(["", "## Commands", ""])
    lines.extend(command_lines(summary))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="render benchmark_summary.json to Markdown")
    parser.add_argument("--summary", required=True, help="path to benchmark_summary.json")
    parser.add_argument("--output", required=True, help="Markdown report output path")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"expected JSON object in {summary_path}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(summary), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
