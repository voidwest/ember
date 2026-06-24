from __future__ import annotations


TOY_RECORDS: list[dict[str, str]] = [
    {
        "id": "toy-001",
        "word": "كتب",
        "lemma": "كتب",
        "root": "كتب",
        "pattern": "فعل",
        "pos": "verb",
        "gender": "",
        "number": "singular",
        "gloss": "he wrote",
    },
    {
        "id": "toy-002",
        "word": "كاتب",
        "lemma": "كاتب",
        "root": "كتب",
        "pattern": "فاعل",
        "pos": "noun",
        "gender": "masculine",
        "number": "singular",
        "gloss": "writer",
    },
    {
        "id": "toy-003",
        "word": "مدرسة",
        "lemma": "مدرسة",
        "root": "درس",
        "pattern": "مفعلة",
        "pos": "noun",
        "gender": "feminine",
        "number": "singular",
        "gloss": "school",
    },
    {
        "id": "toy-004",
        "word": "دارسون",
        "lemma": "دارس",
        "root": "درس",
        "pattern": "فاعل",
        "pos": "noun",
        "gender": "masculine",
        "number": "plural",
        "gloss": "students",
    },
    {
        "id": "toy-005",
        "word": "مفتاح",
        "lemma": "مفتاح",
        "root": "فتح",
        "pattern": "مفعال",
        "pos": "noun",
        "gender": "masculine",
        "number": "singular",
        "gloss": "key",
    },
    {
        "id": "toy-006",
        "word": "فتحوا",
        "lemma": "فتح",
        "root": "فتح",
        "pattern": "فعلوا",
        "pos": "verb",
        "gender": "",
        "number": "plural",
        "gloss": "they opened",
    },
]


def toy_records() -> list[dict[str, str]]:
    return [MorphologyRecord.from_mapping(record).to_dict() for record in TOY_RECORDS]
from .schemas import MorphologyRecord
