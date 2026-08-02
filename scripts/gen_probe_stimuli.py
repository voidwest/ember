#!/usr/bin/env python3
"""Convert canonical morphology JSONL to leakage-safe probe stimuli."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _sha256(path: Path) -> str:
    before = _identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if _identity(path) != before:
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest()


def _identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id.strip():
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            if row_id != row_id.strip():
                raise ValueError(f"{path}:{line_number}: id must not have outer whitespace")
            if row_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
            seen_ids.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"canonical JSONL contains no records: {path}")
    return rows


def _string(row: dict[str, Any], field: str, *, required: bool = False) -> str:
    value = row.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"record {row.get('id')!r} field {field!r} must be a string")
    if required and not value.strip():
        raise ValueError(f"record {row.get('id')!r} field {field!r} must not be empty")
    return value


def _finite_json(value: Any, *, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, context=f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains a non-string object key")
            _finite_json(item, context=f"{context}.{key}")
        return
    raise ValueError(f"{context} contains unsupported value type {type(value).__name__}")


def build_stimuli(
    rows: list[dict[str, Any]], *, include_label_revealed_control: bool = False
) -> list[dict[str, Any]]:
    stimuli: list[dict[str, Any]] = []
    for row in rows:
        row_id = _string(row, "id", required=True)
        surface = _string(row, "surface_dediac") or _string(row, "surface", required=True)
        token = _string(row, "surface", required=True)
        lemma = _string(row, "lemma")
        root = _string(row, "root")
        pattern = _string(row, "abstract_pattern")
        concrete_pattern = _string(row, "concrete_pattern")
        pos = _string(row, "pos")
        source = _string(row, "source")
        split = row.get("split", "")
        if split is not None and not isinstance(split, str):
            raise ValueError(f"record {row_id!r} split must be a string or null")
        features = row.get("features", {})
        if not isinstance(features, dict):
            raise ValueError(f"record {row_id!r} features must be an object")
        _finite_json(features, context=f"record {row_id!r} features")

        # The representation under test must not be handed the labels that the
        # downstream probe predicts. The previous generator embedded lemma,
        # root, and pattern verbatim in the default prompt, making those probe
        # scores circular. A label-revealed prompt remains available only as an
        # explicitly named positive-control condition.
        prompt = (
            "Arabic morphology token probe. "
            f"Surface: {surface}\n"
            f"Token: {token}\n"
            "Predict the token morphology."
        )
        prompts = {"morph_context": prompt}
        prompt_contracts = {
            "morph_context": {
                "target_labels_in_prompt": False,
                "revealed_targets": [],
                "intended_use": "label_free_representation_probe",
            }
        }
        if include_label_revealed_control:
            prompts["morph_context_label_revealed_control"] = (
                f"{prompt}\nLemma: {lemma}\nRoot: {root}\nPattern: {pattern}"
            )
            prompt_contracts["morph_context_label_revealed_control"] = {
                "target_labels_in_prompt": True,
                "revealed_targets": ["lemma", "root", "pattern", "abstract_pattern"],
                "intended_use": "label_revealed_positive_control",
            }

        stimuli.append(
            {
                "id": row_id,
                "surface": surface,
                "lemma": lemma,
                "root": root,
                "pattern": pattern,
                "abstract_pattern": pattern,
                "concrete_pattern": concrete_pattern,
                "pos": pos,
                "features": features,
                "expected_surface": _string(row, "surface"),
                "prompts": prompts,
                "prompt_contracts": prompt_contracts,
                "metadata": {"source": source, "split": split or ""},
            }
        )
    return stimuli


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="canonical morphology JSONL")
    parser.add_argument("output", nargs="?", help="output stimuli JSON")
    parser.add_argument(
        "--include-label-revealed-control",
        action="store_true",
        help="add an explicitly named positive-control prompt containing target labels",
    )
    args = parser.parse_args(argv)

    source = Path(args.input)
    if not source.is_file():
        parser.error(f"canonical JSONL does not exist: {source}")
    destination = Path(args.output) if args.output else source.with_name("probe_stimuli.json")
    metadata_destination = destination.with_suffix(
        destination.suffix + ".metadata.json"
    )
    source_resolved = source.resolve()
    destinations = {destination.resolve(), metadata_destination.resolve()}
    if len(destinations) != 2:
        parser.error("stimuli and metadata destinations must be different")
    if source_resolved in destinations:
        parser.error("output paths must not overwrite the canonical JSONL input")
    source_identity = _identity(source)
    rows = _load_rows(source)
    source_sha256 = _sha256(source)
    if _identity(source) != source_identity:
        raise RuntimeError("canonical JSONL changed while stimuli were being built")
    stimuli = build_stimuli(
        rows, include_label_revealed_control=args.include_label_revealed_control
    )
    _atomic_json(destination, stimuli)
    stimuli_sha256 = _sha256(destination)
    _atomic_json(
        metadata_destination,
        {
            "schema_version": SCHEMA_VERSION,
            "source_path": str(source.resolve()),
            "source_sha256": source_sha256,
            "stimuli_sha256": stimuli_sha256,
            "record_count": len(stimuli),
            "default_prompt_contains_target_labels": False,
            "label_revealed_control_included": args.include_label_revealed_control,
        },
    )
    print(f"wrote {len(stimuli)} stimuli to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
