from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .models import MorphRecord


def apply_filters(records: list[MorphRecord], filters: dict[str, Any] | None) -> tuple[list[MorphRecord], dict[str, Any]]:
    filters = filters or {}
    kept: list[MorphRecord] = []
    reasons: Counter[str] = Counter()

    pos_allowlist = {str(p).upper() for p in filters.get("pos_allowlist", [])}
    for record in records:
        dropped = False
        if filters.get("drop_missing_root", False) and not record.root:
            reasons["missing_root"] += 1
            dropped = True
        if filters.get("drop_missing_pattern", False) and not (record.abstract_pattern or record.concrete_pattern):
            reasons["missing_pattern"] += 1
            dropped = True
        if filters.get("drop_missing_lemma", False) and not record.lemma:
            reasons["missing_lemma"] += 1
            dropped = True
        if filters.get("drop_ambiguous", False) and record.is_ambiguous:
            reasons["ambiguous_analysis"] += 1
            dropped = True
        if pos_allowlist and record.pos.upper() not in pos_allowlist:
            reasons["pos_not_allowed"] += 1
            dropped = True
        if not dropped:
            kept.append(record)

    kept = _drop_below_min(kept, "root", int(filters.get("min_examples_per_root", 0) or 0), reasons)
    kept = _drop_below_min(kept, "abstract_pattern", int(filters.get("min_examples_per_pattern", 0) or 0), reasons)
    kept = _cap_group(kept, "root", int(filters.get("max_examples_per_root", 0) or 0), reasons)
    kept = _cap_group(kept, "abstract_pattern", int(filters.get("max_examples_per_pattern", 0) or 0), reasons)

    report = {
        "input_records": len(records),
        "output_records": len(kept),
        "dropped_records": len(records) - len(kept),
        "drop_reasons": dict(sorted(reasons.items())),
        "filters": filters,
    }
    return kept, report


def _value(record: MorphRecord, field: str) -> str:
    return str(getattr(record, field) or "")


def _drop_below_min(records: list[MorphRecord], field: str, minimum: int, reasons: Counter[str]) -> list[MorphRecord]:
    if minimum <= 1:
        return records
    counts = Counter(_value(record, field) for record in records if _value(record, field))
    kept = []
    for record in records:
        value = _value(record, field)
        if value and counts[value] < minimum:
            reasons[f"min_examples_per_{field}"] += 1
        else:
            kept.append(record)
    return kept


def _cap_group(records: list[MorphRecord], field: str, maximum: int, reasons: Counter[str]) -> list[MorphRecord]:
    if maximum <= 0:
        return records
    by_group: dict[str, list[MorphRecord]] = defaultdict(list)
    no_group: list[MorphRecord] = []
    for record in records:
        value = _value(record, field)
        if value:
            by_group[value].append(record)
        else:
            no_group.append(record)
    kept = list(no_group)
    for value in sorted(by_group):
        group = sorted(by_group[value], key=_content_sort_key)
        kept.extend(group[:maximum])
        reasons[f"max_examples_per_{field}"] += max(0, len(group) - maximum)
    return sorted(kept, key=_content_sort_key)


def _content_sort_key(record: MorphRecord) -> tuple[str, str, str, str, str, str]:
    return (
        record.lemma,
        record.root,
        record.abstract_pattern,
        record.concrete_pattern,
        record.surface,
        record.id,
    )
