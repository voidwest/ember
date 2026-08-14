from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .models import MorphRecord, REQUIRED_CANONICAL_FIELDS
from .split import SPLIT_STRATEGIES, leakage_report


LABEL_FIELDS = ["surface", "lemma", "root", "abstract_pattern", "concrete_pattern", "pos", "source"]


def validate_canonical_rows(rows: list[dict[str, Any]], split_strategy: str | None = None) -> dict[str, Any]:
    missing_required = []
    row_errors = []
    records = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            row_errors.append({"id": f"<row:{idx}>", "error": "row must be an object"})
            continue
        row_id = row.get("id", f"<row:{idx}>")
        for field in REQUIRED_CANONICAL_FIELDS:
            if field not in row:
                missing_required.append({"id": row_id, "field": field})
        try:
            records.append(MorphRecord.from_dict(row))
        except (TypeError, ValueError) as exc:
            row_errors.append({"id": row_id, "error": str(exc)})
    report = validate_canonical(records, split_strategy)
    report["missing_required"] = missing_required
    report["row_errors"] = row_errors
    report["passed"] = not missing_required and not row_errors and report["passed"]
    return report


def validate_canonical(records: list[MorphRecord], split_strategy: str | None = None) -> dict[str, Any]:
    if split_strategy is not None and split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split strategy {split_strategy!r}")
    ids = [r.id for r in records]
    duplicate_ids = sorted([item for item, count in Counter(ids).items() if count > 1])
    empty_ids = [index for index, record_id in enumerate(ids) if not record_id.strip()]
    invalid_splits = [
        {"id": record.id, "split": record.split}
        for record in records
        if (split_strategy is not None and record.split not in {"train", "dev", "test"})
        or (record.split is not None and record.split not in {"train", "dev", "test"})
    ]
    missing_labels = Counter()
    for record in records:
        for field in LABEL_FIELDS:
            value = getattr(record, field)
            if not value:
                missing_labels[field] += 1
    leakage = leakage_report(records, split_strategy) if split_strategy and any(r.split for r in records) else None
    passed = (
        bool(records)
        and not duplicate_ids
        and not empty_ids
        and not invalid_splits
        and not missing_labels
        and _leakage_allows_validation(leakage)
    )
    return {
        "type": "canonical",
        "passed": passed,
        "num_records": len(records),
        "empty": not records,
        "empty_ids": empty_ids,
        "duplicate_ids": duplicate_ids,
        "invalid_splits": invalid_splits,
        "missing_required": [],
        "missing_labels": dict(missing_labels),
        "leakage": leakage,
    }


def _leakage_allows_validation(leakage: dict[str, Any] | None) -> bool:
    if not leakage:
        return True
    if leakage.get("status") in {"not_applicable", "not_checked"}:
        return True
    return bool(leakage.get("passed"))


def validate_sft_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    seen_examples: set[tuple[str, str]] = set()
    allowed_tasks = {"analyze_form", "root_pattern", "feature_bundle", "reinflect"}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append({"index": idx, "error": "row must be an object"})
            continue
        messages = row.get("messages")
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            errors.append({"index": idx, "error": "metadata must be an object"})
            continue
        if not isinstance(messages, list) or len(messages) != 2:
            errors.append({"index": idx, "error": "messages must contain user and assistant"})
            continue
        if not isinstance(messages[0], dict) or not isinstance(messages[1], dict):
            errors.append({"index": idx, "error": "messages must contain objects"})
            continue
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            errors.append({"index": idx, "error": "invalid message roles"})
            continue
        if not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip():
            errors.append({"index": idx, "error": "user content must be a non-empty string"})
            continue
        assistant_content = messages[1].get("content")
        if not isinstance(assistant_content, str):
            errors.append({"index": idx, "error": "assistant content must be a JSON string"})
            continue
        try:
            payload = json.loads(assistant_content, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            errors.append({"index": idx, "error": "assistant content is not JSON"})
            continue
        task = metadata.get("task")
        if task not in allowed_tasks:
            errors.append({"index": idx, "error": "invalid or missing task"})
            continue
        source_id = metadata.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            errors.append({"index": idx, "error": "metadata.source_id must be non-empty"})
            continue
        split = metadata.get("split")
        if split not in {"train", "dev", "test"}:
            errors.append({"index": idx, "error": "metadata.split must be train/dev/test"})
            continue
        key = (source_id, task)
        if key in seen_examples:
            errors.append({"index": idx, "error": "duplicate source_id/task example"})
            continue
        seen_examples.add(key)
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
    for field in sorted(required_by_task[task] & set(payload) - {"features"}):
        if not isinstance(payload[field], str) or not payload[field].strip():
            errors.append({"index": idx, "error": f"{field} must be a non-empty string"})
    if "features" in payload and not isinstance(payload["features"], dict):
        errors.append({"index": idx, "error": "features must be an object"})
    return errors


def validate_probe_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = ["source_id", "surface", "lemma", "root", "abstract_pattern", "concrete_pattern", "pos", "features", "source", "split", "split_type"]
    required_scalars = ["source_id", "surface", "lemma", "root", "abstract_pattern", "concrete_pattern", "pos", "source", "split", "split_type"]
    errors = []
    seen_source_ids: set[str] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append({"index": idx, "error": "row must be an object"})
            continue
        for field in required:
            if field not in row:
                errors.append({"index": idx, "error": f"missing {field}"})
            elif row[field] is None:
                errors.append({"index": idx, "error": f"null {field}"})
            elif field in required_scalars and (
                not isinstance(row[field], str) or not row[field].strip()
            ):
                errors.append({"index": idx, "error": f"{field} must be a non-empty string"})
        if "features" in row and not isinstance(row["features"], dict):
            errors.append({"index": idx, "error": "features must be an object"})
        if row.get("split") not in {"train", "dev", "test"}:
            errors.append({"index": idx, "error": "split must be train/dev/test"})
        source_id = row.get("source_id")
        if isinstance(source_id, str):
            if source_id in seen_source_ids:
                errors.append({"index": idx, "error": "duplicate source_id"})
            seen_source_ids.add(source_id)
    return {"type": "probes", "passed": bool(rows) and not errors, "num_records": len(rows), "empty": not rows, "errors": errors}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")
