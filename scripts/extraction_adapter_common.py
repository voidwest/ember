"""Shared contract helpers for Ember's llama.cpp extraction adapters.

The Rust runner owns the fresh-run transaction. External adapters write into
the staging paths supplied by the request and must produce a complete artifact
set which the runner validates before publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
OFFSET_UNIT = "unicode_character_index"
_FIELD_RE = re.compile(r"^[\w.\-]+$", re.UNICODE)
REQUEST_SCHEMA_VERSION = 1
ARTIFACT_CONTRACT_VERSION = 2
ARTIFACT_LAYOUT = "ember.layer_sharded_npy.v1"


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def stable_hash_bytes(payload: bytes) -> str:
    value = FNV_OFFSET
    for byte in payload:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def stable_hash(text: str) -> str:
    return stable_hash_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line, parse_constant=reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_index + 1}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record {line_index + 1} in {path} must be an object")
            rows.append((line_index, row))
    if not rows:
        raise ValueError(f"input JSONL contains no samples: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_request(path: Path) -> dict[str, Any]:
    """Load and validate the complete Rust-to-adapter request contract."""
    if not path.is_file():
        raise FileNotFoundError(f"external request does not exist: {path}")
    try:
        request = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_json_constant
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid external request JSON {path}: {error}") from error
    if not isinstance(request, dict):
        raise ValueError("external request must be a JSON object")
    validate_request(request)
    return request


def _required_string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"external request field {field!r} must be a non-empty string")
    return result


def validate_request(request: dict[str, Any]) -> None:
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported external request schema: {request.get('schema_version')!r}"
        )
    if request.get("contract_version") != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            f"unsupported artifact contract: {request.get('contract_version')!r}"
        )
    if request.get("layout") != ARTIFACT_LAYOUT:
        raise ValueError(f"unsupported artifact layout: {request.get('layout')!r}")
    if request.get("backend") != "llama-cpp-external":
        raise ValueError(f"unexpected external backend: {request.get('backend')!r}")

    string_fields = (
        "model_path",
        "input_jsonl_path",
        "output_dir",
        "config_path",
        "manifest_path",
        "samples_path",
        "tokenization_path",
        "positions_path",
        "checksums_path",
        "report_path",
        "prompt_template",
        "sample_id_field",
        "word_field",
        "token_position",
    )
    for field in string_fields:
        _required_string(request, field)
    for field in ("write_logits", "prompt_hashes_only"):
        if not isinstance(request.get(field), bool):
            raise ValueError(f"external request field {field!r} must be a boolean")
    layers = request.get("layers")
    if not isinstance(layers, list) or any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in layers
    ):
        raise ValueError("external request layers must be a list of non-negative integers")
    if layers != sorted(set(layers)):
        raise ValueError("external request layers must be sorted and unique")
    max_seq_len = request.get("max_seq_len")
    if max_seq_len is not None and (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len <= 0
    ):
        raise ValueError("external request max_seq_len must be null or a positive integer")
    if not isinstance(request.get("run_metadata"), (dict, type(None))):
        raise ValueError("external request run_metadata must be an object or null")

    model_path = Path(request["model_path"])
    input_path = Path(request["input_jsonl_path"])
    config_path = Path(request["config_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"canonical config not found: {config_path}")

    manifest_path = Path(request["manifest_path"])
    staging_dir = manifest_path.parent.resolve()
    expected_outputs = {
        "config_path": "config.toml",
        "manifest_path": "manifest.json",
        "samples_path": "samples.jsonl",
        "tokenization_path": "tokenization.jsonl",
        "positions_path": "positions.jsonl",
        "checksums_path": "checksums.json",
        "report_path": "report.json",
    }
    for field, filename in expected_outputs.items():
        candidate = Path(request[field])
        if candidate.name != filename or candidate.parent.resolve() != staging_dir:
            raise ValueError(
                f"external request {field} must be {filename!r} inside the staging directory"
            )
    logits_path = request.get("logits_path")
    if logits_path is not None:
        if not isinstance(logits_path, str) or not logits_path.strip():
            raise ValueError("external request logits_path must be null or a non-empty string")
        candidate = Path(logits_path)
        if candidate.name != "logits.npy" or candidate.parent.resolve() != staging_dir:
            raise ValueError("external request logits_path must be logits.npy in the staging directory")
    if request["write_logits"] != (logits_path is not None):
        raise ValueError("external request write_logits and logits_path disagree")

    config = extraction_config(request)
    matching_fields = (
        "model_path",
        "input_jsonl_path",
        "output_dir",
        "prompt_template",
        "sample_id_field",
        "word_field",
        "token_position",
        "layers",
        "write_logits",
        "prompt_hashes_only",
        "max_seq_len",
        "run_metadata",
    )
    for field in matching_fields:
        if config.get(field) != request.get(field):
            raise ValueError(f"external request {field!r} differs from extraction_config")
    if config.get("backend") != request["backend"]:
        raise ValueError("external request backend differs from extraction_config")


def scalar_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return str(value)
    return None


def template_fields(template: str) -> list[str]:
    fields: list[str] = []
    cursor = 0
    while True:
        start = template.find("{", cursor)
        if start < 0:
            if "}" in template[cursor:]:
                raise ValueError("prompt template contains an unmatched closing brace")
            return fields
        if "}" in template[cursor:start]:
            raise ValueError("prompt template contains an unmatched closing brace")
        double = template.startswith("{{", start)
        content_start = start + (2 if double else 1)
        closing = "}}" if double else "}"
        end = template.find(closing, content_start)
        if end < 0:
            raise ValueError("prompt template contains an unmatched opening brace")
        field = template[content_start:end]
        if not field or not _FIELD_RE.fullmatch(field):
            raise ValueError(f"invalid prompt template field: {field!r}")
        if field not in fields:
            fields.append(field)
        cursor = end + len(closing)


def render_prompt(template: str, row: dict[str, Any]) -> str:
    rendered: list[str] = []
    cursor = 0
    while True:
        start = template.find("{", cursor)
        if start < 0:
            if "}" in template[cursor:]:
                raise ValueError("prompt template contains an unmatched closing brace")
            rendered.append(template[cursor:])
            break
        if "}" in template[cursor:start]:
            raise ValueError("prompt template contains an unmatched closing brace")
        rendered.append(template[cursor:start])
        double = template.startswith("{{", start)
        content_start = start + (2 if double else 1)
        closing = "}}" if double else "}"
        end = template.find(closing, content_start)
        if end < 0:
            raise ValueError("prompt template contains an unmatched opening brace")
        field = template[content_start:end]
        if not field or not _FIELD_RE.fullmatch(field):
            raise ValueError(f"invalid prompt template field: {field!r}")
        text = scalar_text(row.get(field))
        if text is None:
            raise ValueError(f"prompt template field {field!r} is missing or not scalar")
        rendered.append(text)
        cursor = end + len(closing)
    output = "".join(rendered)
    if not output.strip():
        raise ValueError("rendered prompt is empty")
    return output


def extraction_config(request: dict[str, Any]) -> dict[str, Any]:
    config = request.get("extraction_config")
    if not isinstance(config, dict):
        raise ValueError(
            "external request does not include extraction_config; use a current Ember binary"
        )
    return config


def load_samples(
    request: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    position_mode = request["token_position"]
    if position_mode != "prompt_final":
        raise ValueError(
            "llama.cpp adapters currently support token_position=prompt_final only "
            "because their tokenization interfaces do not expose trustworthy offsets"
        )
    sample_id_field = request["sample_id_field"]
    seen_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    for input_index, row in read_jsonl(Path(request["input_jsonl_path"])):
        sample_id = scalar_text(row.get(sample_id_field))
        if sample_id is None:
            raise ValueError(
                f"JSONL record {input_index + 1} is missing scalar "
                f"sample_id_field {sample_id_field!r}"
            )
        if not sample_id.strip():
            raise ValueError(f"JSONL record {input_index + 1} has an empty sample ID")
        if sample_id != sample_id.strip():
            raise ValueError(
                f"JSONL record {input_index + 1} sample ID has outer whitespace"
            )
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample ID {sample_id!r} at record {input_index + 1}")
        seen_ids.add(sample_id)
        prompt = render_prompt(request["prompt_template"], row)
        samples.append(
            {
                "input_index": input_index,
                "sample_id": sample_id,
                "prompt": prompt,
                "prompt_hash": stable_hash(prompt),
            }
        )
    if config.get("prompt_template") != request["prompt_template"]:
        raise ValueError("request prompt_template differs from extraction_config")
    return samples


def sample_order_hash(samples: Iterable[dict[str, Any]]) -> str:
    payload = "".join(
        f"{sample['sample_id']}\t{sample['prompt_hash']}\n" for sample in samples
    )
    return stable_hash(payload)


def common_manifest(
    *,
    request: dict[str, Any],
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    model_max_seq_len: int,
    backend_version: str,
    backend_executable: str,
    backend_details: dict[str, Any],
    logits_shape: list[int] | None,
) -> dict[str, Any]:
    model_path = Path(request["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    config_path = Path(request["config_path"])
    config_bytes = config_path.read_bytes()
    model_sha256 = sha256_file(model_path) if config.get("record_model_sha256") else None
    logits = (
        {"path": "logits.npy", "shape": logits_shape}
        if logits_shape is not None
        else None
    )
    return {
        "schema_version": request["contract_version"],
        "layout": request["layout"],
        "artifact_kind": "ember_hidden_states",
        "created_at_unix": int(time.time()),
        "run_id": config.get("run_id"),
        "run_dir": request["output_dir"],
        "config_path": "config.toml",
        "samples_path": "samples.jsonl",
        "tokenization_path": "tokenization.jsonl",
        "positions_path": "positions.jsonl",
        "checksums_path": "checksums.json",
        "report_path": "report.json",
        "logits_path": "logits.npy" if logits is not None else None,
        "tensor_contract": {
            "storage": "layer-sharded-npy",
            "dtype": "f32",
            "byte_order": "little-endian",
            "sample_axis": 0,
            "hidden_axis": 1,
            "layers": [],
            "logits": logits,
        },
        "sample_count": len(samples),
        "sample_order_hash": sample_order_hash(samples),
        "config_hash": stable_hash_bytes(config_bytes),
        "dtype": "f32",
        "output_format": "npy",
        "model": {
            "path": request["model_path"],
            "architecture": config.get("architecture"),
            "n_layers": 0,
            "embed_dim": 0,
            "max_seq_len": model_max_seq_len,
            "file_size_bytes": model_path.stat().st_size,
            "sha256": model_sha256,
            "gguf_metadata": None,
        },
        "backend": {
            "name": request["backend"],
            "version": backend_version,
            "executable": backend_executable,
            "commit": None,
            "details": backend_details,
        },
        "extraction_config": config,
    }


def write_common_rows(
    *,
    request: dict[str, Any],
    samples: list[dict[str, Any]],
    token_ids: list[list[int]],
) -> list[dict[str, Any]]:
    if len(samples) != len(token_ids):
        raise ValueError("sample/tokenization row count mismatch")
    sample_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for sample_index, (sample, ids) in enumerate(zip(samples, token_ids, strict=True)):
        if not ids:
            raise ValueError(f"sample {sample['sample_id']!r} produced no token IDs")
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in ids):
            raise ValueError(f"sample {sample['sample_id']!r} produced invalid token IDs")
        selected = [len(ids) - 1]
        sample_rows.append(
            {
                "schema_version": request["contract_version"],
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "input_index": sample["input_index"],
                "prompt": None if request["prompt_hashes_only"] else sample["prompt"],
                "prompt_hash": sample["prompt_hash"],
            }
        )
        token_rows.append(
            {
                "schema_version": request["contract_version"],
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "token_ids": ids,
                "token_count": len(ids),
                "prompt_hash": sample["prompt_hash"],
                "offsets": [],
                "offset_unit": OFFSET_UNIT,
            }
        )
        position_rows.append(
            {
                "schema_version": request["contract_version"],
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "position_mode": "prompt_final",
                "pooling": "single",
                "selected_token_positions": selected,
                "source_field": None,
                "source_value": None,
                "source_byte_span": None,
            }
        )
        parity_rows.append(
            {
                "index": sample_index,
                "id": sample["sample_id"],
                "prompt": None if request["prompt_hashes_only"] else sample["prompt"],
                "token_ids": ids,
                "selected_token_positions": selected,
            }
        )
    write_jsonl(Path(request["samples_path"]), sample_rows)
    write_jsonl(Path(request["tokenization_path"]), token_rows)
    write_jsonl(Path(request["positions_path"]), position_rows)
    return parity_rows


def write_report_and_checksums(
    *, request: dict[str, Any], sample_count: int, logits_written: bool
) -> None:
    report = {
        "schema_version": request["contract_version"],
        "layout": request["layout"],
        "status": "complete",
        "sample_count": sample_count,
        "layer_count": 0,
        "layers": [],
        "logits_written": logits_written,
        "resume": {
            "supported_by_contract": False,
            "external_runner_policy": "fresh-run",
            "rule": "the Ember runner publishes a validated staging directory atomically",
        },
    }
    write_json(Path(request["report_path"]), report)
    paths = {
        "config.toml": Path(request["config_path"]),
        "manifest.json": Path(request["manifest_path"]),
        "samples.jsonl": Path(request["samples_path"]),
        "tokenization.jsonl": Path(request["tokenization_path"]),
        "positions.jsonl": Path(request["positions_path"]),
        "report.json": Path(request["report_path"]),
    }
    if logits_written:
        logits_path = request.get("logits_path")
        if not logits_path:
            raise ValueError("logits output was written but request.logits_path is absent")
        paths["logits.npy"] = Path(logits_path)
    checksums = {name: sha256_file(path) for name, path in sorted(paths.items())}
    write_json(Path(request["checksums_path"]), checksums)
