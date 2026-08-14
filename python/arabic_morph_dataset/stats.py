from __future__ import annotations

from collections import Counter
from typing import Any

from .models import MorphRecord


def dataset_stats(records: list[MorphRecord]) -> dict[str, Any]:
    feature_distribution: dict[str, Counter[str]] = {}
    for record in records:
        for key, value in record.features.items():
            feature_distribution.setdefault(str(key), Counter())[str(value)] += 1
    return {
        "num_records": len(records),
        "unique_surfaces": len({r.surface for r in records if r.surface}),
        "unique_lemmas": len({r.lemma for r in records if r.lemma}),
        "unique_roots": len({r.root for r in records if r.root}),
        "unique_abstract_patterns": len({r.abstract_pattern for r in records if r.abstract_pattern}),
        "unique_concrete_patterns": len({r.concrete_pattern for r in records if r.concrete_pattern}),
        "pos_distribution": dict(Counter(r.pos or "<missing>" for r in records)),
        "feature_distribution": {k: dict(v) for k, v in sorted(feature_distribution.items())},
        "examples_per_root": dict(Counter(r.root for r in records if r.root).most_common()),
        "examples_per_abstract_pattern": dict(Counter(r.abstract_pattern for r in records if r.abstract_pattern).most_common()),
        "examples_per_concrete_pattern": dict(Counter(r.concrete_pattern for r in records if r.concrete_pattern).most_common()),
        "split_counts": dict(Counter(r.split or "<unsplit>" for r in records)),
        "splits": _split_stats(records),
    }


def _split_stats(records: list[MorphRecord]) -> dict[str, dict[str, Any]]:
    result = {}
    split_names = ["train", "dev", "test", "<unsplit>"]
    extras = sorted({record.split or "<unsplit>" for record in records} - set(split_names))
    for split in split_names + extras:
        subset = [record for record in records if (record.split or "<unsplit>") == split]
        result[split] = {
            "num_records": len(subset),
            "unique_surfaces": len({r.surface for r in subset if r.surface}),
            "unique_lemmas": len({r.lemma for r in subset if r.lemma}),
            "unique_roots": len({r.root for r in subset if r.root}),
            "unique_abstract_patterns": len({r.abstract_pattern for r in subset if r.abstract_pattern}),
            "unique_concrete_patterns": len({r.concrete_pattern for r in subset if r.concrete_pattern}),
            "pos_distribution": dict(Counter(r.pos or "<missing>" for r in subset)),
        }
    return result
