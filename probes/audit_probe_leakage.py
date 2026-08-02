"""Leakage / overlap audit for probe stimuli.

Checks:
- duplicate surface, lemma, root
- duplicate morphological key combinations
- cross-fold leakage (same surface/lemma/root in train+test)
- near-duplicate surface forms
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .train_linear_probe import (
        atomic_write_text,
        audit_label_revealing_prompt,
        sha256_file,
    )
except ImportError:  # direct script execution
    from train_linear_probe import (
        atomic_write_text,
        audit_label_revealing_prompt,
        sha256_file,
    )

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
        key = norm_fn(s).strip()
        if not key:
            continue
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
    observed_folds = set()
    for r in rows:
        metadata = r.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("stimulus metadata must be an object when present")
        fold = metadata.get("split") or r.get("split")
        if not isinstance(fold, str) or not fold.strip():
            continue
        fold = fold.strip()
        observed_folds.add(fold)
        val = r.get(field)
        if field == "surface" and not val:
            val = r.get("expected_surface")
        if val:
            fold_map[str(val)].add(fold)

    leakage = []
    for val, folds in fold_map.items():
        if len(folds) > 1:
            leakage.append({"value": val, "folds": sorted(folds)})

    total_unique = len(fold_map)
    leaking_unique = len(leakage)
    result = {
        "field": field,
        "observed_folds": sorted(observed_folds),
        "total_unique_values": total_unique,
        "values_appearing_in_multiple_folds": leaking_unique,
        "rows_with_split_metadata": sum(
            1
            for row in rows
            if isinstance(row.get("metadata") or {}, dict)
            and isinstance((row.get("metadata") or {}).get("split") or row.get("split"), str)
        ),
        "leakage_fraction": leaking_unique / total_unique if total_unique else 0.0,
        "leakage_rate_percent": leaking_unique / total_unique * 100.0 if total_unique else 0.0,
        "examples": leakage[:10],
    }
    if len(observed_folds) < 2:
        result["status"] = "insufficient_split_metadata"
    else:
        result["status"] = "evaluated"
    return result


def duplicate_report(values: list, name: str) -> dict:
    """Report on duplicate values."""
    present = [
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    ]
    cnt = Counter(present)
    dupes = {k: v for k, v in cnt.items() if v > 1}
    if not dupes:
        return {
            "field": name,
            "n_present": len(present),
            "n_missing": len(values) - len(present),
            "n_unique": len(cnt),
            "n_duplicate_values": 0,
            "n_duplicate_items": 0,
        }
    return {
        "field": name,
        "n_present": len(present),
        "n_missing": len(values) - len(present),
        "n_unique": len(cnt),
        "n_duplicate_values": len(dupes),
        "n_duplicate_items": sum(dupes.values()),
        "max_dup_count": max(dupes.values()),
        "top_dupes": sorted(dupes.items(), key=lambda x: -x[1])[:10],
    }


def _strict_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _scalar(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"field {field!r} must be scalar when present")
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stimuli", help="stimuli JSON array")
    parser.add_argument("output", nargs="?", help="output JSON path")
    args = parser.parse_args()

    src = Path(args.stimuli)
    if not src.is_file():
        parser.error(f"stimuli file does not exist: {src}")
    dst = Path(args.output) if args.output else src.with_name("leakage_audit.json")

    rows = _strict_json(src)
    if not isinstance(rows, list) or not rows:
        raise ValueError("stimuli must be a non-empty JSON array")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every stimulus must be a JSON object")
    print(f"Loaded {len(rows)} rows from {src}")

    surfaces = [_scalar(r, "surface") or _scalar(r, "expected_surface") for r in rows]
    lemmas = [_scalar(r, "lemma") for r in rows]
    roots = [_scalar(r, "root") for r in rows]
    abstract = [_scalar(r, "abstract_pattern") for r in rows]
    concrete = [_scalar(r, "concrete_pattern") for r in rows]

    def combine(left, right):
        return [f"{a}::{b}" if a and b else "" for a, b in zip(left, right)]

    root_abs = combine(roots, abstract)
    root_conc = combine(roots, concrete)
    lemma_abs = combine(lemmas, abstract)

    report = {
        "schema_version": 2,
        "source": str(src),
        "source_sha256": sha256_file(src),
        "n_rows": len(rows),
        "duplicates": {},
        "cross_fold_leakage": {},
        "near_duplicates": {},
        "prompt_label_exposure": {},
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
        print(f"  cross-fold {field:<10s}: {lk['values_appearing_in_multiple_folds']} / {lk['total_unique_values']} values leak ({lk['leakage_rate_percent']:.3f}%)")

    # 3. Near-duplicate surfaces
    print()
    nd = near_duplicates(surfaces)
    report["near_duplicates"] = nd
    print(f"  near-duplicate surface groups: {nd['n_groups']} ({nd['n_items_in_groups']} items)")
    for ex in nd["examples"]:
        print(f"    '{ex['normalized']}' ×{ex['count']}: {ex['surface_examples']}")

    # 4. Prompt-label exposure. This is separate from train/test overlap: a
    # prompt can directly print the target even when folds are perfectly clean.
    template_counts = Counter(
        template
        for row in rows
        for template in (
            row.get("prompts", {}).keys()
            if isinstance(row.get("prompts"), dict)
            else []
        )
    )
    morphology_tasks = [
        "root",
        "lemma",
        "abstract_pattern",
        "concrete_pattern",
    ]
    for template, count in sorted(template_counts.items()):
        if count != len(rows):
            report["prompt_label_exposure"][template] = {
                "status": "not_checked_partial_template_coverage",
                "rows_with_template": count,
                "total_rows": len(rows),
            }
            continue
        audit = audit_label_revealing_prompt(
            rows,
            morphology_tasks,
            {"probe_template": template, "probe_position": "last"},
        )
        report["prompt_label_exposure"][template] = audit
        print(
            f"  prompt {template:<20s}: {audit['status']} "
            f"({audit.get('revealed_task_row_count', 0)} target-row exposures)"
        )

    # 5. Summary
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
            lk.get("status") == "evaluated"
            and lk.get("leakage_fraction", 0) > 0
            for lk in report["cross_fold_leakage"].values()
        ),
        "abstract_pattern_leakage_concern": (
            report["duplicates"].get("root+abstract_pattern", {}).get("n_duplicate_items", 0) > 0
            or report["cross_fold_leakage"].get("root", {}).get("leakage_fraction", 0) > 0
            or report["cross_fold_leakage"].get("lemma", {}).get("leakage_fraction", 0) > 0
        ),
        "any_prompt_label_exposure": any(
            audit.get("status") == "label_revealed"
            for audit in report["prompt_label_exposure"].values()
        ),
    }

    atomic_write_text(
        dst,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(f"\nSaved to {dst}")


if __name__ == "__main__":
    main()
