"""A missing or mismatched golden model must never become a passing skip."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_ladder_inputs.py"
spec = importlib.util.spec_from_file_location("ladder_inputs", SCRIPT)
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)


@pytest.fixture
def pinned_inputs(tmp_path):
    records = []
    for family in ladder.FAMILIES:
        for rung in ladder.RUNGS:
            data = f"fixture {family} {rung}".encode()
            (tmp_path / f"{family}-{rung}.gguf").write_bytes(data)
            records.append({"family": family, "rung": rung,
                            "quantizer_commit": ladder.QUANTIZER_COMMIT,
                            "target": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    return tmp_path, manifest


def test_complete_pinned_ladder(pinned_inputs):
    assert len(ladder.validate_inputs(*pinned_inputs)) == 6


@pytest.mark.parametrize("mutation", ["missing", "changed", "truncated"])
def test_missing_or_changed_model_fails(pinned_inputs, mutation):
    root, manifest = pinned_inputs
    model = root / "llama-3.2-1b-q4_k_m.gguf"
    if mutation == "missing":
        model.unlink()
    elif mutation == "changed":
        model.write_bytes(b"x" * model.stat().st_size)
    else:
        model.write_bytes(b"x")
    with pytest.raises(ValueError):
        ladder.validate_inputs(root, manifest)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "quantizer"])
def test_invalid_manifest_fails(pinned_inputs, mutation):
    root, manifest = pinned_inputs
    records = json.loads(manifest.read_text())
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    else:
        records[0]["quantizer_commit"] = "wrong-build"
    manifest.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        ladder.validate_inputs(root, manifest)
