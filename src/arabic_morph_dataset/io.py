from __future__ import annotations

import csv
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Iterable

from .models import MorphRecord


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            f.write("\n")
    os.replace(tmp, out)


def read_table(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        reader = csv.DictReader(f, dialect=dialect, restkey="__extra_columns__", restval="")
        for line_no, row in enumerate(reader, start=2):
            if row.get("__extra_columns__"):
                raise ValueError(f"{path}:{line_no}: row has extra columns: {row['__extra_columns__']}")
            row.pop("__extra_columns__", None)
            rows.append(dict(row))
        return rows


def read_raw_records(path: str | Path) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix in {".csv", ".tsv"}:
        return read_table(path)
    raise ValueError(f"Unsupported input suffix {suffix}; use .jsonl, .csv, or .tsv")


def read_morph_records(path: str | Path) -> list[MorphRecord]:
    return [MorphRecord.from_dict(row) for row in read_jsonl(path)]


def write_morph_records(path: str | Path, records: Iterable[MorphRecord]) -> None:
    write_jsonl(path, (record.to_dict() for record in records))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, out)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".toml":
        with path.open("rb") as f:
            return _ensure_config_dict(tomllib.load(f), path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML; TOML configs work with the standard library") from exc
        with path.open("r", encoding="utf-8") as f:
            return _ensure_config_dict(yaml.safe_load(f), path)
    raise ValueError("Config must be .toml, .yaml, or .yml")


def _ensure_config_dict(config: Any, path: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping/object at the top level")
    return config
