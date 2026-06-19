"""Leakage / overlap audit for probe stimuli.

Checks:
- duplicate surface, lemma, root
- duplicate morphological key combinations
- cross-fold leakage (same surface/lemma/root in train+test)
- near-duplicate surface forms
"""

import json
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")
ARABIC_NORMALIZE = {
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0622": "\u0627",  # آ → ا
    "\u0629": "\u0647",  # ة → ه
    "\u0649": "\u064a",  # ى → ي
}


def dediac(s: str) -> str:
    return ARABIC_DIACRITICS.sub("", s)


def normalize_arabic(s: str) -> str:
    """Light normalization: dediac, replace alif variants, ta marbuta → ha, alef maqsura → yeh."""
    s = dediac(s)
    for src, dst in ARABIC_NORMALIZE.items():
        s = s.replace(src, dst)
    return s


def near_duplicates(surfaces: list[str], norm_fn=normalize_arabic) -> dict:
    """Group surfaces that normalize to the same form."""
    groups = defaultdict(list)
    for i, s in enumerate(surfaces):
        key = norm_fn(s or "")
        groups[key].append(i)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    return {
        "n_groups": len(dupes),
        "n_items_in_groups": sum(len(v) for v in dupes.values()),
        "max_group_size": max((len(v) for v in dupes.values()), default=0),
        "examples": [
            {"normalized": k, "count": len(v), "surface_examples": [surfaces[j] for j in v[:3]]}
            for k, v in sorted(dupes.items(), key=lambda x: -len(x[1]))[:10]
        ],
    }


def cross_fold_leakage(rows: list[dict], field: str) -> dict:
    """Check how many values of `field` appear in more than one fold."""
    fold_map = defaultdict(set)
    for r in rows:
        fold = r.get("metadata", {}).get("split", "unknown")
        val = r.get(field) or r.get("expected_surface", "")
        if val:
            fold_map[val].add(fold)

    leakage = []
    for val, folds in fold_map.items():
        if len(folds) > 1:
            leakage.append({"value": val, "folds": sorted(folds)})

    total_unique = len(fold_map)
    leaking_unique = len(leakage)
    return {
        "field": field,
        "total_unique_values": total_unique,
        "values_appearing_in_multiple_folds": leaking_unique,
        "leakage_rate": round(leaking_unique / total_unique * 100, 1) if total_unique else 0,
        "examples": leakage[:10],
    }


def duplicate_report(values: list, name: str) -> dict:
    """Report on duplicate values."""
    cnt = Counter(values)
    dupes = {k: v for k, v in cnt.items() if v > 1}
    if not dupes:
        return {"field": name, "n_unique": len(cnt), "n_duplicate_values": 0, "n_duplicate_items": 0}
    return {
        "field": name,
        "n_unique": len(cnt),
        "n_duplicate_values": len(dupes),
        "n_duplicate_items": sum(dupes.values()),
        "max_dup_count": max(dupes.values()),
        "top_dupes": sorted(dupes.items(), key=lambda x: -x[1])[:10],
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python audit_probe_leakage.py <stimuli.json> [output.json]")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("leakage_audit.json")

    rows = json.loads(src.read_text(encoding="utf-8"))
    print(f"Loaded {len(rows)} rows from {src}")

    surfaces = [r.get("surface", "") or r.get("expected_surface", "") for r in rows]
    lemmas = [r.get("lemma", "") for r in rows]
    roots = [r.get("root", "") for r in rows]
    abstract = [r.get("abstract_pattern", "") for r in rows]
    concrete = [r.get("concrete_pattern", "") for r in rows]
    root_abs = [f"{r.get('root','')}::{r.get('abstract_pattern','')}" for r in rows]
    root_conc = [f"{r.get('root','')}::{r.get('concrete_pattern','')}" for r in rows]
    lemma_abs = [f"{r.get('lemma','')}::{r.get('abstract_pattern','')}" for r in rows]

    report = {
        "n_rows": len(rows),
        "duplicates": {},
        "cross_fold_leakage": {},
        "near_duplicates": {},
    }

    # 1. Duplicate checks
    for name, vals in [
        ("surface", surfaces),
        ("lemma", lemmas),
        ("root", roots),
        ("abstract_pattern", abstract),
        ("concrete_pattern", concrete),
        ("root+abstract_pattern", root_abs),
        ("root+concrete_pattern", root_conc),
        ("lemma+abstract_pattern", lemma_abs),
    ]:
        d = duplicate_report(vals, name)
        report["duplicates"][name] = d
        dup_items = d.get("n_duplicate_items", 0)
        dup_pct = round(dup_items / len(rows) * 100, 1) if dup_items else 0
        print(f"  {name:<30s}: {d['n_unique']:>5d} unique, {d.get('n_duplicate_values',0):>5d} dup values ({dup_items} items, {dup_pct}%)")

    # 2. Cross-fold leakage
    print()
    for field in ["surface", "lemma", "root"]:
        lk = cross_fold_leakage(rows, field)
        report["cross_fold_leakage"][field] = lk
        print(f"  cross-fold {field:<10s}: {lk['values_appearing_in_multiple_folds']} / {lk['total_unique_values']} values leak ({lk['leakage_rate']}%)")

    # 3. Near-duplicate surfaces
    print()
    nd = near_duplicates(surfaces)
    report["near_duplicates"] = nd
    print(f"  near-duplicate surface groups: {nd['n_groups']} ({nd['n_items_in_groups']} items)")
    for ex in nd["examples"]:
        print(f"    '{ex['normalized']}' ×{ex['count']}: {ex['surface_examples']}")

    # 4. Summary
    report["summary"] = {
        "any_surface_duplicates": any(
            d.get("n_duplicate_items", 0) > 0
            for d in [report["duplicates"]["surface"]]
        ),
        "any_root_duplicates": any(
            d.get("n_duplicate_items", 0) > 0
            for d in [report["duplicates"]["root"]]
        ),
        "any_cross_fold_leakage": any(
            lk.get("leakage_rate", 0) > 0
            for lk in report["cross_fold_leakage"].values()
        ),
        "abstract_pattern_leakage_concern": (
            report["duplicates"].get("root+abstract_pattern", {}).get("n_duplicate_items", 0) > 0
            or report["cross_fold_leakage"].get("root", {}).get("leakage_rate", 0) > 0
            or report["cross_fold_leakage"].get("lemma", {}).get("leakage_rate", 0) > 0
        ),
    }

    dst.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {dst}")


if __name__ == "__main__":
    main()
