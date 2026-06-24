from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class MorphologyRecord:
    id: str
    word: str
    lemma: str
    root: str
    pattern: str
    pos: str
    gender: str = ""
    number: str = ""
    gloss: str = ""

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "MorphologyRecord":
        return cls(
            id=str(row["id"]),
            word=str(row["word"]),
            lemma=str(row.get("lemma", "")),
            root=str(row.get("root", "")),
            pattern=str(row.get("pattern", "")),
            pos=str(row.get("pos", "")),
            gender=str(row.get("gender", "")),
            number=str(row.get("number", "")),
            gloss=str(row.get("gloss", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PromptRecord:
    id: str
    word: str
    prompt: str
    target_pos: str = ""
    target_root: str = ""
    target_pattern: str = ""
    source_lemma: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
