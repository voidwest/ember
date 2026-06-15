from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .models import MorphRecord, REQUIRED_CANONICAL_FIELDS
from .split import leakage_report


LABEL_FIELDS = ["surface", "lemma", "root", "abstract_pattern", "concrete_pattern", "pos"]


def validate_canonical_rows(rows: list[dict[str, Any]], split_strategy: str | None = None) -> dict[str, Any]:
    missing_required = []
    for idx, row in enumerate(rows):
        row_id = row.get("id", f"<row:{idx}>")
        for field in REQUIRED_CANONICAL_FIELDS:
            if field not in row:
                missing_required.append({"id": row_id, "field": field})
    report = validate_canonical([MorphRecord.from_dict(row) for row in rows], split_strategy)
    report["missing_required"] = missing_required
    report["passed"] = bool(rows) and not missing_required and not report["duplicate_ids"] and not report["missing_labels"] and not (report["leakage"] and not report["leakage"].get("passed"))
    return report


def validate_canonical(records: list[MorphRecord], split_strategy: str | None = None) -> dict[str, Any]:
    ids = [r.id for r in records]
    duplicate_ids = sorted([item for item, count in Counter(ids).items() if count > 1])
    missing_required = []
    missing_labels = Counter()
    for record in records:
        for field in LABEL_FIELDS:
            value = getattr(record, field)
            if not value:
                missing_labels[field] += 1
        if not isinstance(record.features, dict):
            missing_required.append({"id": record.id, "field": "features"})
        if not isinstance(record.metadata, dict):
            missing_required.append({"id": record.id, "field": "metadata"})
    leakage = leakage_report(records, split_strategy) if split_strategy and any(r.split for r in records) else None
    passed = (
        bool(records)
        and not duplicate_ids
        and not missing_required
        and not missing_labels
        and not (leakage and not leakage.get("passed"))
    )
    return {
        "type": "canonical",
        "passed": passed,
        "num_records": len(records),
        "empty": not records,
        "duplicate_ids": duplicate_ids,
        "missing_required": missing_required,
        "missing_labels": dict(missing_labels),
        "leakage": leakage,
    }


def validate_sft_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    allowed_tasks = {"analyze_form", "root_pattern", "feature_bundle", "reinflect"}
    for idx, row in enumerate(rows):
        messages = row.get("messages")
        metadata = row.get("metadata") or {}
        if not isinstance(messages, list) or len(messages) != 2:
            errors.append({"index": idx, "error": "messages must contain user and assistant"})
            continue
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            errors.append({"index": idx, "error": "invalid message roles"})
        try:
            payload = json.loads(messages[1].get("content", ""))
        except json.JSONDecodeError:
            errors.append({"index": idx, "error": "assistant content is not JSON"})
            continue
        task = metadata.get("task")
        if task not in allowed_tasks:
            errors.append({"index": idx, "error": "invalid or missing task"})
            continue
        errors.extend(_validate_sft_payload(idx, task, payload))
    return {"type": "sft", "passed": bool(rows) and not errors, "num_records": len(rows), "empty": not rows, "errors": errors}


def _validate_sft_payload(idx: int, task: str, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return [{"index": idx, "error": "assistant JSON must be an object"}]
    required_by_task = {
        "analyze_form": {"lemma", "root", "abstract_pattern", "concrete_pattern", "pos", "features"},
        "root_pattern": {"root", "abstract_pattern", "concrete_pattern"},
        "feature_bundle": {"pos", "features"},
        "reinflect": {"surface"},
    }
    errors = []
    missing = sorted(required_by_task[task] - set(payload))
    if missing:
        errors.append({"index": idx, "error": f"assistant JSON missing keys for {task}: {missing}"})
    if "features" in payload and not isinstance(payload["features"], dict):
        errors.append({"index": idx, "error": "features must be an object"})
    return errors


def validate_probe_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = ["surface", "lemma", "root", "abstract_pattern", "concrete_pattern", "pos", "features", "source", "split", "split_type"]
    errors = []
    for idx, row in enumerate(rows):
        for field in required:
            if field not in row:
                errors.append({"index": idx, "error": f"missing {field}"})
            elif row[field] is None:
                errors.append({"index": idx, "error": f"null {field}"})
        if "features" in row and not isinstance(row["features"], dict):
            errors.append({"index": idx, "error": "features must be an object"})
    return {"type": "probes", "passed": bool(rows) and not errors, "num_records": len(rows), "empty": not rows, "errors": errors}
