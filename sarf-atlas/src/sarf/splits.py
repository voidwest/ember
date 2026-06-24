from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def lemma_heldout_split(records: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record.get("lemma", record["id"])].append(dict(record))

    split_names = ["train", "dev", "test"]
    split_rows: list[dict[str, str]] = []
    lemma_to_split = {}
    for index, lemma in enumerate(sorted(grouped)):
        split = split_names[index % len(split_names)]
        lemma_to_split[lemma] = split
        for row in grouped[lemma]:
            row["split"] = split
            split_rows.append(row)

    counts = {name: 0 for name in split_names}
    for row in split_rows:
        counts[row["split"]] += 1

    metadata = {
        "strategy": "lemma_round_robin_heldout",
        "unit": "lemma",
        "toy": True,
        "counts": counts,
        "lemma_to_split": lemma_to_split,
        "not_research_output": True,
    }
    return split_rows, metadata
