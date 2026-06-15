from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from .models import MorphRecord


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_FORMATTING_MARKS = re.compile(r"[\u0640\u0674]")

FIELD_ALIASES = {
    "surface": ["surface", "word", "form", "token"],
    "surface_dediac": ["surface_dediac", "dediac", "dediacritized", "bw_dediac"],
    "diacritized": ["diacritized", "diac", "diac_surface", "stem"],
    "lemma": ["lemma", "lex", "lemma_diac"],
    "root": ["root"],
    "abstract_pattern": ["abstract_pattern", "pattern", "pattern_abstract", "pat"],
    "concrete_pattern": ["concrete_pattern", "pattern_concrete", "stem_pattern", "pattern_surface"],
    "pos": ["pos", "POS"],
    "analysis_id": ["analysis_id", "analysis", "id"],
}

FEATURE_ALIASES = {
    "gender": ["gender", "gen", "form_gen"],
    "number": ["number", "num", "form_num"],
    "person": ["person", "per"],
    "aspect": ["aspect", "asp"],
    "voice": ["voice", "vox"],
    "mood": ["mood", "mod"],
    "case": ["case", "cas"],
    "definiteness": ["definiteness", "def"],
    "state": ["state", "stt"],
}

VALUE_ALIASES = {
    "f": "fem",
    "m": "masc",
    "s": "sg",
    "d": "def",
    "i": "indef",
    "u": "nom",
    "a": "acc",
    "g": "gen",
    "p": "pl",
    "na": "na",
    "none": "none",
    "-": "",
}

POS_ALIASES = {
    "noun": "NOUN",
    "verb": "VERB",
    "adj": "ADJ",
    "adv": "ADV",
    "pron": "PRON",
    "part": "PART",
    "prep": "ADP",
}


def dediacritize(text: str) -> str:
    return ARABIC_FORMATTING_MARKS.sub("", ARABIC_DIACRITICS.sub("", text or ""))


def clean_lemma(text: str) -> str:
    text = str(text or "").strip()
    return re.sub(r"_\d+$", "", text)


def _first(data: dict[str, Any], aliases: list[str]) -> str:
    for alias in aliases:
        value = data.get(alias)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _normalize_value(value: Any) -> str:
    text = str(value).strip()
    return VALUE_ALIASES.get(text.lower(), text)


def _normalize_pos(pos: str) -> str:
    return POS_ALIASES.get(pos.strip().lower(), pos.strip().upper())


def _is_ambiguous(raw: dict[str, Any]) -> bool:
    explicit = raw.get("is_ambiguous", raw.get("ambiguous"))
    if explicit not in (None, ""):
        if isinstance(explicit, str):
            return explicit.strip().lower() in {"1", "true", "yes", "y"}
        return bool(explicit)
    try:
        num_analyses = int(raw.get("num_analyses", 1))
        return num_analyses > 1
    except (TypeError, ValueError):
        return bool(raw.get("num_analyses"))


def _stable_id(raw: dict[str, Any], source_name: str, analysis_id: str, surface: str, lemma: str, idx: int) -> str:
    fields = [
        source_name,
        analysis_id,
        surface,
        lemma,
        _first(raw, FIELD_ALIASES["root"]),
        _first(raw, FIELD_ALIASES["abstract_pattern"]),
        _first(raw, FIELD_ALIASES["concrete_pattern"]),
        _first(raw, FIELD_ALIASES["pos"]),
        _canonical_raw_features(raw),
        _canonical_raw_record(raw),
        str(idx),
    ]
    seed = "|".join(fields)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def expand_analysis_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for raw in raw_records:
        analyses = raw.get("analyses")
        if isinstance(analyses, list):
            base = {k: v for k, v in raw.items() if k != "analyses"}
            for i, analysis in enumerate(analyses):
                merged = dict(base)
                if isinstance(analysis, dict):
                    merged.update(analysis)
                merged["is_ambiguous"] = len(analyses) > 1
                merged.setdefault("analysis_id", f"{raw.get('id', raw.get('word', 'analysis'))}:{i}")
                expanded.append(merged)
        else:
            expanded.append(raw)
    return expanded


def normalize_raw_record(raw: dict[str, Any], idx: int, source_name: str) -> MorphRecord:
    surface = _first(raw, FIELD_ALIASES["surface"])
    diacritized = _first(raw, FIELD_ALIASES["diacritized"])
    surface_dediac = _first(raw, FIELD_ALIASES["surface_dediac"])
    if not surface and diacritized:
        surface = dediacritize(diacritized)
    if not surface_dediac:
        surface_dediac = dediacritize(surface or diacritized)

    lemma = clean_lemma(_first(raw, FIELD_ALIASES["lemma"]))
    analysis_id = _first(raw, FIELD_ALIASES["analysis_id"])
    record_id = str(raw.get("canonical_id") or raw.get("record_id") or _stable_id(raw, source_name, analysis_id, surface, lemma, idx))

    raw_features = raw.get("features") or {}
    if not isinstance(raw_features, dict):
        raise ValueError("features must be an object if provided")
    features = dict(raw_features)
    for canonical, aliases in FEATURE_ALIASES.items():
        if canonical in features and features[canonical] not in (None, ""):
            features[canonical] = _normalize_value(features[canonical])
            continue
        value = _first(raw, aliases)
        if value:
            normalized = _normalize_value(value)
            if normalized:
                features[canonical] = normalized
    features = {str(k): _normalize_value(v) for k, v in features.items() if _normalize_value(v)}

    metadata = dict(raw.get("metadata") or {})
    metadata.setdefault("raw_index", idx)

    return MorphRecord(
        id=record_id,
        surface=surface,
        surface_dediac=surface_dediac,
        diacritized=diacritized,
        lemma=lemma,
        root=_first(raw, FIELD_ALIASES["root"]),
        abstract_pattern=_first(raw, FIELD_ALIASES["abstract_pattern"]),
        concrete_pattern=_first(raw, FIELD_ALIASES["concrete_pattern"]),
        pos=_normalize_pos(_first(raw, FIELD_ALIASES["pos"])),
        features=features,
        source=source_name,
        analysis_id=analysis_id,
        is_ambiguous=_is_ambiguous(raw),
        metadata=metadata,
    )


def _canonical_raw_features(raw: dict[str, Any]) -> str:
    features = raw.get("features") or {}
    return json.dumps(features, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _canonical_raw_record(raw: dict[str, Any]) -> str:
    return json.dumps(
        {str(k): v for k, v in raw.items() if k not in {"metadata"}},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def normalize_records(raw_records: list[dict[str, Any]], source_name: str) -> tuple[list[MorphRecord], dict[str, Any]]:
    expanded = expand_analysis_records(raw_records)
    records = [normalize_raw_record(raw, idx, source_name) for idx, raw in enumerate(expanded)]
    report = {
        "input_records": len(raw_records),
        "expanded_records": len(expanded),
        "output_records": len(records),
        "missing": {
            "surface": sum(1 for r in records if not r.surface),
            "lemma": sum(1 for r in records if not r.lemma),
            "root": sum(1 for r in records if not r.root),
            "abstract_pattern": sum(1 for r in records if not r.abstract_pattern),
            "concrete_pattern": sum(1 for r in records if not r.concrete_pattern),
        },
        "pos_distribution": dict(Counter(r.pos or "<missing>" for r in records)),
    }
    return records, report
