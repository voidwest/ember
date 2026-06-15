from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


REQUIRED_CANONICAL_FIELDS = (
    "id",
    "surface",
    "surface_dediac",
    "diacritized",
    "lemma",
    "root",
    "abstract_pattern",
    "concrete_pattern",
    "pos",
    "features",
    "source",
    "analysis_id",
    "is_ambiguous",
    "metadata",
)


@dataclass(frozen=True)
class MorphRecord:
    id: str
    surface: str
    surface_dediac: str = ""
    diacritized: str = ""
    lemma: str = ""
    root: str = ""
    abstract_pattern: str = ""
    concrete_pattern: str = ""
    pos: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    analysis_id: str = ""
    is_ambiguous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "surface": self.surface,
            "surface_dediac": self.surface_dediac,
            "diacritized": self.diacritized,
            "lemma": self.lemma,
            "root": self.root,
            "abstract_pattern": self.abstract_pattern,
            "concrete_pattern": self.concrete_pattern,
            "pos": self.pos,
            "features": dict(sorted(self.features.items())),
            "source": self.source,
            "analysis_id": self.analysis_id,
            "is_ambiguous": self.is_ambiguous,
            "metadata": dict(sorted(self.metadata.items())),
        }
        if self.split:
            data["split"] = self.split
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MorphRecord":
        return cls(
            id=str(data.get("id", "")),
            surface=str(data.get("surface", "")),
            surface_dediac=str(data.get("surface_dediac", "")),
            diacritized=str(data.get("diacritized", "")),
            lemma=str(data.get("lemma", "")),
            root=str(data.get("root", "")),
            abstract_pattern=str(data.get("abstract_pattern", "")),
            concrete_pattern=str(data.get("concrete_pattern", "")),
            pos=str(data.get("pos", "")),
            features=_dict_field(data.get("features"), "features"),
            source=str(data.get("source", "")),
            analysis_id=str(data.get("analysis_id", "")),
            is_ambiguous=_bool_field(data.get("is_ambiguous", False)),
            metadata=_dict_field(data.get("metadata"), "metadata"),
            split=data.get("split"),
        )

    def with_split(self, split: str) -> "MorphRecord":
        return replace(self, split=split)


def _dict_field(value: Any, field_name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _bool_field(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
