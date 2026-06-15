from __future__ import annotations

import json
from typing import Any

from .models import MorphRecord


DEFAULT_SFT_TASKS = ["analyze_form", "root_pattern", "feature_bundle"]


def morphology_payload(record: MorphRecord) -> dict[str, Any]:
    return {
        "lemma": record.lemma,
        "root": record.root,
        "abstract_pattern": record.abstract_pattern,
        "concrete_pattern": record.concrete_pattern,
        "pos": record.pos,
        "features": dict(sorted(record.features.items())),
    }


def make_sft_examples(records: list[MorphRecord], tasks: list[str] | None = None) -> list[dict[str, Any]]:
    tasks = tasks or DEFAULT_SFT_TASKS
    examples: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: (r.split or "", r.id)):
        for task in tasks:
            example = _sft_for_task(record, task)
            if example:
                examples.append(example)
    return examples


def _sft_for_task(record: MorphRecord, task: str) -> dict[str, Any] | None:
    if task == "analyze_form":
        user = f"حلّل الكلمة صرفيًا وأرجع الحقول بصيغة JSON فقط: {record.surface}"
        assistant = morphology_payload(record)
    elif task == "root_pattern":
        user = f"استخرج الجذر والوزن الصرفي بصيغة JSON فقط: {record.surface}"
        assistant = {
            "root": record.root,
            "abstract_pattern": record.abstract_pattern,
            "concrete_pattern": record.concrete_pattern,
        }
    elif task == "feature_bundle":
        user = f"استخرج نوع الكلمة والسمات الصرفية بصيغة JSON فقط: {record.surface}"
        assistant = {"pos": record.pos, "features": dict(sorted(record.features.items()))}
    elif task == "reinflect":
        if not record.lemma or not record.surface:
            return None
        user = (
            "صرّف ال lemma حسب السمات الهدف وأرجع JSON فقط: "
            + json.dumps({"lemma": record.lemma, "features": record.features}, ensure_ascii=False, sort_keys=True)
        )
        assistant = {"surface": record.surface}
    else:
        raise ValueError(f"Unknown SFT task {task}")

    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)},
        ],
        "metadata": {
            "task": task,
            "source_id": record.id,
            "split": record.split,
        },
    }


def make_probe_records(records: list[MorphRecord], split_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: (r.split or "", r.id)):
        rows.append(
            {
                "surface": record.surface,
                "surface_dediac": record.surface_dediac,
                "lemma": record.lemma,
                "root": record.root,
                "abstract_pattern": record.abstract_pattern,
                "concrete_pattern": record.concrete_pattern,
                "pos": record.pos,
                "features": dict(sorted(record.features.items())),
                "source": record.source,
                "split": record.split,
                "split_type": split_type,
            }
        )
    return rows
