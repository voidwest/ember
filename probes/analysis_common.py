"""Shared validation and row-alignment helpers for representation analyses."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
PROMPT_AUDIT_STATUSES = {
    "passed",
    "not_applicable",
    "label_revealed",
    "not_checked_missing_probe_template_metadata",
    "unverifiable_missing_probe_leakage_audit",
    "unverifiable_missing_prompt_audit",
}
UNVERIFIABLE_PROMPT_AUDIT_STATUSES = {
    "not_checked_missing_probe_template_metadata",
    "unverifiable_missing_probe_leakage_audit",
    "unverifiable_missing_prompt_audit",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activation_metadata_path(path: str | Path) -> Path:
    activation_path = Path(path)
    return activation_path.with_name(f"{activation_path.stem}_metadata.json")


def load_activation_metadata(path: str | Path) -> dict:
    metadata_path = activation_metadata_path(path)
    if not metadata_path.is_file():
        raise ValueError(f"activation metadata sidecar is required: {metadata_path}")
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {metadata_path}")

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"activation metadata must be an object: {metadata_path}")
    return metadata


def _strict_json_text(value: str, context: str):
    def reject_constant(constant):
        raise ValueError(f"non-standard JSON constant {constant!r} in {context}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {context}") from error


def _npz_scalar(data, key: str):
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"artifact field {key!r} must be scalar")
    return value.reshape(-1)[0].item()


def probe_prompt_contract_status(path: str | Path) -> str:
    """Read the prompt leakage status carried by a probe-like NPZ artifact."""
    try:
        with np.load(path, allow_pickle=False) as data:
            audit_text = _npz_scalar(data, "prompt_leakage_audit_json")
            contract_text = _npz_scalar(data, "probe_prompt_contract_json")
    except ValueError as error:
        raise ValueError(f"unsafe or invalid NPZ artifact: {path}") from error
    audit = (
        _strict_json_text(str(audit_text), f"{path}:prompt_leakage_audit_json")
        if audit_text is not None
        else None
    )
    if audit is None and contract_text is not None:
        contract = _strict_json_text(
            str(contract_text), f"{path}:probe_prompt_contract_json"
        )
        if isinstance(contract, dict):
            audit = contract.get("prompt_leakage_audit")
    if not isinstance(audit, dict) or not isinstance(audit.get("status"), str):
        return "unverifiable_missing_prompt_audit"
    status = audit["status"]
    if status not in PROMPT_AUDIT_STATUSES:
        raise ValueError(f"unsupported prompt-contract status {status!r} in {path}")
    return status


def enforce_probe_prompt_contracts(
    paths: list[str | Path],
    *,
    allow_label_revealed: bool = False,
    allow_unverifiable: bool = False,
) -> list[str]:
    """Gate plots/reports that could otherwise disguise prompt leakage."""
    statuses = [probe_prompt_contract_status(path) for path in paths]
    if "label_revealed" in statuses and not allow_label_revealed:
        raise ValueError(
            "artifact was produced from label-revealed prompts; allow it only as an "
            "explicitly marked positive control"
        )
    if (
        any(status in UNVERIFIABLE_PROMPT_AUDIT_STATUSES for status in statuses)
        and not allow_unverifiable
    ):
        raise ValueError(
            "artifact has no verifiable prompt leakage audit; regenerate it or explicitly "
            "allow the legacy input after external verification"
        )
    return statuses


def assert_row_alignment(
    path_a: str | Path,
    path_b: str | Path,
    row_count: int,
    *,
    allow_assumed: bool = False,
) -> str:
    """Require evidence that cross-model activation rows represent the same samples."""
    try:
        metadata_a = load_activation_metadata(path_a)
        metadata_b = load_activation_metadata(path_b)
    except ValueError:
        if allow_assumed:
            return "user_assumed_without_metadata"
        raise
    for label, metadata in (("A", metadata_a), ("B", metadata_b)):
        shape = metadata.get("activation_shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[0] != row_count:
            raise ValueError(
                f"activation metadata {label} has no matching [samples, layers, hidden] shape"
            )
    for label, path, metadata in (
        ("A", path_a, metadata_a),
        ("B", path_b, metadata_b),
    ):
        declared_sha = metadata.get("activations_sha256")
        if declared_sha is not None:
            if not isinstance(declared_sha, str) or not SHA256_PATTERN.fullmatch(
                declared_sha
            ):
                raise ValueError(
                    f"activation metadata {label} has an invalid activations_sha256"
                )
            if declared_sha.lower() != sha256_file(path):
                raise ValueError(
                    f"activation metadata {label} does not match its activation tensor"
                )

    identity_a = _row_identity(metadata_a, row_count)
    identity_b = _row_identity(metadata_b, row_count)
    if identity_a is not None and identity_b is not None:
        if identity_a != identity_b:
            raise ValueError("activation metadata proves that cross-model row identities differ")
        return "metadata_row_identity"

    source_a = _source_identity(metadata_a)
    source_b = _source_identity(metadata_b)
    if source_a is not None and source_b is not None:
        if source_a != source_b:
            raise ValueError("activation metadata source datasets differ")
        rows_a = metadata_a.get("row_indices", list(range(row_count)))
        rows_b = metadata_b.get("row_indices", list(range(row_count)))
        for label, rows in (("A", rows_a), ("B", rows_b)):
            if (
                not isinstance(rows, list)
                or len(rows) != row_count
                or any(isinstance(value, bool) or not isinstance(value, int) for value in rows)
                or len(set(rows)) != row_count
            ):
                raise ValueError(f"activation metadata {label} has invalid row_indices")
        if rows_a != rows_b:
            raise ValueError("activation metadata row index order differs")
        return "source_sha256_and_row_indices"

    if allow_assumed:
        return "user_assumed_without_verifiable_identity"
    raise ValueError(
        "cross-model analysis requires matching token-selection identities or source "
        "SHA-256 metadata; use --assume-row-aligned only after external verification"
    )


def _row_identity(metadata: dict, row_count: int):
    selections = metadata.get("token_selections")
    if not isinstance(selections, list) or len(selections) != row_count:
        return None
    identity = []
    for expected_index, selection in enumerate(selections):
        if not isinstance(selection, dict):
            return None
        index = selection.get("index")
        if index != expected_index:
            return None
        prompt = selection.get("prompt")
        row_id = selection.get("row_id")
        target_span = selection.get("target_span")
        if prompt is not None and not isinstance(prompt, str):
            return None
        if row_id is not None and not isinstance(row_id, (str, int)):
            return None
        if target_span is not None and (
            not isinstance(target_span, list)
            or len(target_span) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in target_span)
        ):
            return None
        if prompt is None and row_id is None:
            return None
        identity.append(
            (
                index,
                prompt,
                row_id,
                tuple(target_span) if isinstance(target_span, list) else None,
            )
        )
    return identity


def _source_identity(metadata: dict):
    for hash_field in ("stimuli_sha256", "benchmark_sha256"):
        value = metadata.get(hash_field)
        if isinstance(value, str) and SHA256_PATTERN.fullmatch(value):
            return ("source_sha256", value.lower())
    return None
