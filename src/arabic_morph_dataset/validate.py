from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .models import MorphRecord, REQUIRED_CANONICAL_FIELDS
from .split import leakage_report


def validate_canonical(records: list[MorphRecord], split_strategy: str | None = None) -> dict[str, Any]:
    ids = [r.id for r in records]
    duplicate_ids = sorted([item for item, count in Counter(ids).items() if count > 1])
    missing_required = []
    missing_labels = Counter()
    for record in records:
        data = record.to_dict()
        for field in REQUIRED_CANONICAL_FIELDS:
            if field not in data:
                missing_required.append({"id": record.id, "field": field})
        for field in ["surface", "lemma", "root", "abstract_pattern", "concrete_pattern", "pos"]:
            if not data.get(field):
                missing_labels[field] += 1
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
            json.loads(messages[1].get("content", ""))
        except json.JSONDecodeError:
            errors.append({"index": idx, "error": "assistant content is not JSON"})
        if metadata.get("task") not in allowed_tasks:
            errors.append({"index": idx, "error": "invalid or missing task"})
    return {"type": "sft", "passed": bool(rows) and not errors, "num_records": len(rows), "empty": not rows, "errors": errors}


def validate_probe_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = ["surface", "lemma", "root", "abstract_pattern", "concrete_pattern", "pos", "features", "source", "split", "split_type"]
    errors = []
    for idx, row in enumerate(rows):
        for field in required:
            if field not in row:
                errors.append({"index": idx, "error": f"missing {field}"})
    return {"type": "probes", "passed": bool(rows) and not errors, "num_records": len(rows), "empty": not rows, "errors": errors}
