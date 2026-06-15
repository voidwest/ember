from __future__ import annotations

from collections import Counter
from typing import Any

from .models import MorphRecord
from .split import split_records
from .stats import dataset_stats


def make_summary_report(
    records: list[MorphRecord],
    filter_report: dict[str, Any] | None = None,
    seed: int = 13,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    stats = dataset_stats(records)
    split_checks = {}
    for strategy in ["root_heldout", "abstract_pattern_heldout", "concrete_pattern_heldout"]:
        _, split_report = split_records(records, strategy=strategy, seed=seed, ratios=ratios)
        split_checks[strategy] = {
            "passed": split_report["leakage"]["passed"],
            "record_counts": split_report["record_counts"],
            "checks": split_report["leakage"]["checks"],
        }

    root_counts = Counter(r.root for r in records if r.root)
    abstract_pattern_counts = Counter(r.abstract_pattern for r in records if r.abstract_pattern)
    concrete_pattern_counts = Counter(r.concrete_pattern for r in records if r.concrete_pattern)

    return {
        "records": {
            "input": (filter_report or {}).get("input_records", len(records)),
            "kept": len(records),
            "dropped": (filter_report or {}).get("dropped_records", 0),
            "dropped_by_reason": (filter_report or {}).get("drop_reasons", {}),
        },
        "unique_roots": stats["unique_roots"],
        "unique_abstract_patterns": stats["unique_abstract_patterns"],
        "unique_concrete_patterns": stats["unique_concrete_patterns"],
        "split_leakage": {
            "root_heldout": split_checks["root_heldout"]["passed"],
            "abstract_pattern_heldout": split_checks["abstract_pattern_heldout"]["passed"],
            "concrete_pattern_heldout": split_checks["concrete_pattern_heldout"]["passed"],
            "details": split_checks,
        },
        "top_20_roots": _top_counts(root_counts, "root"),
        "top_20_abstract_patterns": _top_counts(abstract_pattern_counts, "pattern"),
        "top_20_concrete_patterns": _top_counts(concrete_pattern_counts, "pattern"),
    }


def _top_counts(counter: Counter[str], label: str) -> list[dict[str, int | str]]:
    return [{label: key, "count": count} for key, count in counter.most_common(20)]
