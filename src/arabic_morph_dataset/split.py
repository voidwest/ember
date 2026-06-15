from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .models import MorphRecord


SPLIT_STRATEGIES = {
    "random",
    "root_heldout",
    "abstract_pattern_heldout",
    "concrete_pattern_heldout",
    "root_pattern_heldout",
    "lemma_heldout",
}


@dataclass
class UnionFind:
    parent: dict[str, str]

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def split_records(
    records: list[MorphRecord],
    strategy: str,
    seed: int = 13,
    ratios: dict[str, float] | None = None,
) -> tuple[list[MorphRecord], dict[str, object]]:
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unknown split strategy {strategy}; choose one of {sorted(SPLIT_STRATEGIES)}")
    ratios = ratios or {"train": 0.8, "dev": 0.1, "test": 0.1}
    split_names = ["train", "dev", "test"]
    total_ratio = sum(float(ratios.get(name, 0.0)) for name in split_names)
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    normalized = {name: float(ratios.get(name, 0.0)) / total_ratio for name in split_names}

    components = _components(records, strategy)
    rng = random.Random(seed)
    ordered = sorted(
        components.values(),
        key=lambda group: (-len(group), rng.random(), group[0].lemma, group[0].root, group[0].id),
    )

    targets = {name: normalized[name] * len(records) for name in split_names}
    counts = {name: 0 for name in split_names}
    assigned: list[MorphRecord] = []
    component_assignments: dict[str, int] = defaultdict(int)
    for group in ordered:
        split = _choose_split(counts, targets, split_names, len(group))
        counts[split] += len(group)
        component_assignments[split] += 1
        assigned.extend(record.with_split(split) for record in group)

    assigned = sorted(assigned, key=lambda r: r.id)
    report = {
        "strategy": strategy,
        "seed": seed,
        "ratios": normalized,
        "record_counts": counts,
        "component_counts": dict(component_assignments),
        "leakage": leakage_report(assigned, strategy),
    }
    return assigned, report


def _choose_split(counts: dict[str, int], targets: dict[str, float], split_names: list[str], group_size: int = 1) -> str:
    return max(
        split_names,
        key=lambda name: (
            targets[name] - counts[name],
            -(max(0.0, counts[name] + group_size - targets[name])),
            -counts[name],
            -split_names.index(name),
        ),
    )


def _components(records: list[MorphRecord], strategy: str) -> dict[str, list[MorphRecord]]:
    uf = UnionFind(parent={})
    for record in records:
        rid = f"record:{record.id}"
        uf.find(rid)
        for key in _group_keys(record, strategy):
            uf.union(rid, key)

    grouped: dict[str, list[MorphRecord]] = defaultdict(list)
    for record in records:
        grouped[uf.find(f"record:{record.id}")].append(record)
    return grouped


def _group_keys(record: MorphRecord, strategy: str) -> list[str]:
    keys = []
    if record.lemma:
        keys.append(f"lemma:{record.lemma}")
    if strategy == "random" or strategy == "lemma_heldout":
        return keys or [f"record:{record.id}"]
    if strategy == "root_heldout" and record.root:
        keys.append(f"root:{record.root}")
    elif strategy == "abstract_pattern_heldout" and record.abstract_pattern:
        keys.append(f"abstract_pattern:{record.abstract_pattern}")
    elif strategy == "concrete_pattern_heldout" and record.concrete_pattern:
        keys.append(f"concrete_pattern:{record.concrete_pattern}")
    elif strategy == "root_pattern_heldout":
        pattern = record.abstract_pattern or record.concrete_pattern
        if record.root and pattern:
            keys.append(f"root_pattern:{record.root}|{pattern}")
    return keys or [f"record:{record.id}"]


def leakage_report(records: Iterable[MorphRecord], strategy: str) -> dict[str, object]:
    by_split: dict[str, list[MorphRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split or "unsplit"].append(record)

    report: dict[str, object] = {"strategy": strategy, "checks": {}}
    checks: dict[str, object] = {}
    checks["lemma"] = _intersection_check(by_split, lambda r: r.lemma)
    if strategy == "root_heldout":
        checks["root"] = _intersection_check(by_split, lambda r: r.root)
    elif strategy == "abstract_pattern_heldout":
        checks["abstract_pattern"] = _intersection_check(by_split, lambda r: r.abstract_pattern)
    elif strategy == "concrete_pattern_heldout":
        checks["concrete_pattern"] = _intersection_check(by_split, lambda r: r.concrete_pattern)
    elif strategy == "root_pattern_heldout":
        checks["root_pattern"] = _intersection_check(by_split, lambda r: f"{r.root}|{r.abstract_pattern or r.concrete_pattern}" if r.root and (r.abstract_pattern or r.concrete_pattern) else "")
    report["checks"] = checks
    report["passed"] = all(bool(check.get("passed", False)) for check in checks.values() if isinstance(check, dict))
    return report


def _intersection_check(by_split: dict[str, list[MorphRecord]], getter) -> dict[str, object]:
    train = {getter(r) for r in by_split.get("train", []) if getter(r)}
    dev = {getter(r) for r in by_split.get("dev", []) if getter(r)}
    test = {getter(r) for r in by_split.get("test", []) if getter(r)}
    train_dev = sorted(train & dev)
    train_test = sorted(train & test)
    dev_test = sorted(dev & test)
    return {
        "passed": not train_dev and not train_test and not dev_test,
        "train_dev": train_dev,
        "train_test": train_test,
        "dev_test": dev_test,
    }
