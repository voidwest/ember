#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


ROOTS = [
    ("كتب", ("ك", "ت", "ب")),
    ("درس", ("د", "ر", "س")),
    ("علم", ("ع", "ل", "م")),
    ("فتح", ("ف", "ت", "ح")),
    ("قرأ", ("ق", "ر", "أ")),
    ("خرج", ("خ", "ر", "ج")),
    ("دخل", ("د", "خ", "ل")),
    ("عمل", ("ع", "م", "ل")),
    ("سكن", ("س", "ك", "ن")),
    ("جلس", ("ج", "ل", "س")),
    ("حمل", ("ح", "م", "ل")),
    ("حكم", ("ح", "ك", "م")),
    ("طلب", ("ط", "ل", "ب")),
    ("سمع", ("س", "م", "ع")),
    ("نصر", ("ن", "ص", "ر")),
    ("حفظ", ("ح", "ف", "ظ")),
    ("فهم", ("ف", "ه", "م")),
    ("وصف", ("و", "ص", "ف")),
    ("رسم", ("ر", "س", "م")),
    ("زرع", ("ز", "ر", "ع")),
    ("جمع", ("ج", "م", "ع")),
    ("نظر", ("ن", "ظ", "ر")),
    ("شرب", ("ش", "ر", "ب")),
    ("لعب", ("ل", "ع", "ب")),
    ("قرب", ("ق", "ر", "ب")),
    ("بعد", ("ب", "ع", "د")),
    ("كبر", ("ك", "ب", "ر")),
    ("صغر", ("ص", "غ", "ر")),
    ("سفر", ("س", "ف", "ر")),
    ("خدم", ("خ", "د", "م")),
]

ROOT_COUNTS = [42, 34, 29, 24, 21, 18, 16, 14, 13, 12, 11, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "data/arabic_morph_sample/camelmorph_imbalanced_sample.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if any(root != "".join(letters) for root, letters in ROOTS):
        raise AssertionError("fixture root labels and radical tuples disagree")
    records = []
    idx = 1
    for root_index, ((root, letters), count) in enumerate(
        zip(ROOTS, ROOT_COUNTS, strict=True)
    ):
        for i in range(count):
            template = TEMPLATES[(i + root_index) % len(TEMPLATES)]
            record = template(root, letters, i)
            record["analysis_id"] = f"imbalanced-{idx:04d}"
            record["source"] = "synthetic_camelmorph_imbalanced_msa"
            records.append(record)
            idx += 1

    for i in range(20):
        record = rng.choice(records).copy()
        record["analysis_id"] = f"imbalanced-{idx:04d}"
        record["word"] = record["word"] + "ان"
        record["num"] = "d"
        record["is_ambiguous"] = True
        records.append(record)
        idx += 1

    noisy_specs = [
        {"root": "", "pattern": "فَعَلَ", "pattern_concrete": "غفل"},
        {"root": "بنى", "pattern": "", "pattern_concrete": ""},
        {"root": "قال", "pattern": "فَعَلَ", "pattern_concrete": "قال", "lex": ""},
        {"root": "", "pattern": "", "pattern_concrete": ""},
    ]
    for i in range(30):
        spec = noisy_specs[i % len(noisy_specs)]
        record = {
            "word": f"شكل{i}",
            "diac": f"شَكْل{i}",
            "lex": spec.get("lex", f"شَكْل_{i}"),
            "root": spec["root"],
            "pattern": spec["pattern"],
            "pattern_concrete": spec["pattern_concrete"],
            "pos": "noun" if i % 4 else "part",
            "gen": "m",
            "num": "s",
            "stt": "i",
            "cas": "u",
            "analysis_id": f"imbalanced-{idx:04d}",
        }
        records.append(record)
        idx += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not out.is_file():
        raise ValueError(f"output path is not a regular file: {out}")
    ids = [record["analysis_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise AssertionError("fixture generator produced duplicate analysis IDs")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp", dir=out.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    metadata = out.with_suffix(out.suffix + ".metadata.json")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata.name}.", suffix=".tmp", dir=metadata.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "artifact_kind": "synthetic_morphology_fixture",
                    "seed": args.seed,
                    "record_count": len(records),
                    "output_path": str(out.resolve()),
                    "output_sha256": digest,
                },
                handle,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Wrote {len(records)} records to {out}")
    return 0


def verb_perf(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"{c1}{c2}{c3}"
    return base(surface, f"{surface}_1", root, "فَعَلَ", surface, "verb", asp="perf", per="3", gen="m", num="s", vox="act")


def verb_impf(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"ي{c1}{c2}{c3}"
    return base(surface, f"{c1}{c2}{c3}_1", root, "يَفْعَلُ", surface, "verb", asp="impf", per="3", gen="m", num="s", vox="act", mood="ind")


def active_participle(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"{c1}ا{c2}{c3}"
    return base(surface, f"{surface}_1", root, "فَاعِل", surface, "noun", gen="m", num="s", stt="i", cas="u")


def place_noun(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"م{c1}{c2}{c3}"
    return base(surface, f"{surface}_1", root, "مَفْعَل", surface, "noun", gen="m", num="s", stt="i", cas="u")


def feminine_noun(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"م{c1}{c2}{c3}ة"
    return base(surface, f"{surface}_1", root, "مَفْعَلَة", surface, "noun", gen="f", num="s", stt="i", cas="u")


def plural_definite(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"ال{c1}ا{c2}{c3}ون"
    return base(surface, f"{c1}ا{c2}{c3}_1", root, "فَاعِل", f"{c1}ا{c2}{c3}", "noun", gen="m", num="p", stt="d", cas="u")


def verbal_noun(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"ت{c1}{c2}ي{c3}"
    return base(surface, f"{surface}_1", root, "تَفْعِيل", surface, "noun", gen="m", num="s", stt="i", cas="u")


def passive_participle(root: str, letters: tuple[str, str, str], i: int) -> dict[str, str]:
    c1, c2, c3 = letters
    surface = f"م{c1}{c2}و{c3}"
    return base(surface, f"{surface}_1", root, "مَفْعُول", surface, "adj", gen="m", num="s", stt="i", cas="u")


def base(surface: str, lemma: str, root: str, pattern: str, concrete_pattern: str, pos: str, **features: str) -> dict[str, str]:
    record = {
        "word": surface,
        "diac": surface,
        "lex": lemma,
        "root": root,
        "pattern": pattern,
        "pattern_concrete": concrete_pattern,
        "pos": pos,
    }
    record.update(features)
    return record


TEMPLATES = [
    verb_perf,
    verb_impf,
    active_participle,
    place_noun,
    feminine_noun,
    plural_definite,
    verbal_noun,
    passive_participle,
]


if __name__ == "__main__":
    raise SystemExit(main())
