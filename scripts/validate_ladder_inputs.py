#!/usr/bin/env python3
"""Fail closed before a golden run if the complete pinned ladder is unavailable."""

import argparse
import hashlib
import json
from pathlib import Path

FAMILIES = ("llama-3.2-1b", "qwen2.5-1.5b")
RUNGS = ("q8_0", "q6_k", "q4_k_m")
QUANTIZER_COMMIT = "47c786924ad1ab7e91da2cdc72fcdb563780c2bd"


def validate_inputs(model_root: Path, manifest_path: Path) -> list[dict]:
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {(family, rung) for family in FAMILIES for rung in RUNGS}
    indexed = {}
    for record in records:
        key = (record["family"], record["rung"])
        if key not in expected or key in indexed:
            raise ValueError(f"unexpected or duplicate ladder entry: {key}")
        if record["quantizer_commit"] != QUANTIZER_COMMIT:
            raise ValueError(f"unrecognized quantizer pin for {key}")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError(f"incomplete ladder manifest; missing {sorted(expected - set(indexed))}")
    results = []
    for family, rung in sorted(expected):
        path = model_root / f"{family}-{rung}.gguf"
        target = indexed[(family, rung)]["target"]
        if not path.is_file():
            raise ValueError(f"missing pinned model: {path}")
        if path.stat().st_size != target["bytes"]:
            raise ValueError(f"model byte count differs from pin: {path}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != target["sha256"]:
            raise ValueError(f"model SHA-256 differs from pin: {path}: {actual} != {target['sha256']}")
        results.append({"family": family, "rung": rung, "path": str(path), "sha256": actual})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "models/v03-ladder/ladder-manifest.json")
    args = parser.parse_args()
    try:
        results = validate_inputs(args.models_dir, args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"golden ladder preflight failed: {error}\n")
    print(json.dumps({"verified_models": results}, indent=2))


if __name__ == "__main__":
    main()
