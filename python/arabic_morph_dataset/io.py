from __future__ import annotations

import csv
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from .models import MorphRecord


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: JSONL record must be an object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    def write_rows(f: TextIO) -> None:
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("JSONL rows must be objects")
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            f.write("\n")

    _atomic_text_write(Path(path), write_rows)


def read_table(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        reader = csv.DictReader(f, dialect=dialect, restkey="__extra_columns__", restval="")
        if not reader.fieldnames:
            raise ValueError(f"{path}: table is missing a header row")
        if any(name is None or not name.strip() for name in reader.fieldnames):
            raise ValueError(f"{path}: table contains an empty column name")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: table contains duplicate column names")
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


def write_json(path: str | Path, data: Any) -> None:
    def write_value(f: TextIO) -> None:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        f.write("\n")

    _atomic_text_write(Path(path), write_value)


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


def _atomic_text_write(path: Path, writer: Callable[[TextIO], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _sync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")
