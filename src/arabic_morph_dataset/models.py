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
        if not isinstance(data, dict):
            raise TypeError("canonical morphology record must be an object")
        return cls(
            id=_string_field(data.get("id"), "id"),
            surface=_string_field(data.get("surface"), "surface"),
            surface_dediac=_string_field(data.get("surface_dediac"), "surface_dediac"),
            diacritized=_string_field(data.get("diacritized"), "diacritized"),
            lemma=_string_field(data.get("lemma"), "lemma"),
            root=_string_field(data.get("root"), "root"),
            abstract_pattern=_string_field(data.get("abstract_pattern"), "abstract_pattern"),
            concrete_pattern=_string_field(data.get("concrete_pattern"), "concrete_pattern"),
            pos=_string_field(data.get("pos"), "pos"),
            features=_dict_field(data.get("features"), "features"),
            source=_string_field(data.get("source"), "source"),
            analysis_id=_string_field(data.get("analysis_id"), "analysis_id"),
            is_ambiguous=_bool_field(data.get("is_ambiguous", False)),
            metadata=_dict_field(data.get("metadata"), "metadata"),
            split=_optional_string_field(data.get("split"), "split"),
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
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"is_ambiguous has invalid boolean value {value!r}")
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError("is_ambiguous must be a boolean")


def _string_field(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string_field(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _string_field(value, field_name)
