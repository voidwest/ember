#!/usr/bin/env python3
"""Real llama.cpp tokenization extractor for Ember external-backend smoke tests.

This helper implements Ember's `llama-cpp-external` request contract by calling
llama.cpp's `llama-tokenize` binary for each prompt. It performs real
GGUF-backed tokenization only. It does not generate text, compute logits, or
extract hidden states.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import time
from pathlib import Path


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3


def stable_hash(text: str) -> str:
    value = FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def render_prompt(template: str, row: dict) -> str:
    rendered = template
    for key, value in row.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def llama_tokenize(binary: str, model_path: str, prompt: str) -> list[int]:
    output = subprocess.run(
        [
            binary,
            "--model",
            model_path,
            "--prompt",
            prompt,
            "--ids",
            "--log-disable",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    parsed = ast.literal_eval(output.stdout.strip())
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"unexpected llama-tokenize output: {output.stdout!r}")
    if not parsed:
        raise ValueError("llama-tokenize returned no token IDs")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    metadata = request.get("run_metadata") or {}
    tokenize_bin = (
        os.environ.get("LLAMA_TOKENIZE_BIN")
        or metadata.get("llama_tokenize_bin")
        or "/tmp/llama.cpp-master/build/bin/llama-tokenize"
    )
    if not Path(tokenize_bin).is_file():
        raise FileNotFoundError(f"llama-tokenize binary not found: {tokenize_bin}")

    run_dir = Path(request["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    tokenization = []
    positions = []
    parity_prompts = []
    order_pairs = []
    for sample_index, row in enumerate(read_jsonl(Path(request["input_jsonl_path"]))):
        sample_id = str(row.get(request["sample_id_field"], sample_index))
        prompt = render_prompt(request["prompt_template"], row)
        prompt_hash = stable_hash(prompt)
        token_ids = llama_tokenize(tokenize_bin, request["model_path"], prompt)
        selected = [len(token_ids) - 1]

        samples.append(
            {
                "schema_version": request["contract_version"],
                "sample_index": sample_index,
                "sample_id": sample_id,
                "input_index": sample_index,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
            }
        )
        tokenization.append(
            {
                "schema_version": request["contract_version"],
                "sample_index": sample_index,
                "sample_id": sample_id,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "prompt_hash": prompt_hash,
                "offsets": [],
            }
        )
        positions.append(
            {
                "schema_version": request["contract_version"],
                "sample_index": sample_index,
                "sample_id": sample_id,
                "position_mode": request["token_position"],
                "pooling": "single",
                "selected_token_positions": selected,
                "source_field": None,
                "source_value": None,
                "source_byte_span": None,
            }
        )
        parity_prompts.append(
            {
                "index": sample_index,
                "id": sample_id,
                "prompt": prompt,
                "token_ids": token_ids,
            }
        )
        order_pairs.append((sample_id, prompt_hash))

    order_payload = "".join(f"{sample_id}\t{prompt_hash}\n" for sample_id, prompt_hash in order_pairs)
    provenance = {
        "real_llama_cpp": True,
        "real_tokenization": True,
        "no_generation": True,
        "no_logits": True,
        "no_hidden_states": True,
        "not_research_output": True,
        "purpose": "real llama.cpp tokenization smoke test for Ember external backend plumbing",
        "llama_tokenize_bin": tokenize_bin,
    }
    extraction_config = {
        "run_id": None,
        "model_path": request["model_path"],
        "architecture": "qwen3",
        "tokenizer_path": None,
        "backend": request["backend"],
        "prompt_template": request["prompt_template"],
        "input_jsonl_path": request["input_jsonl_path"],
        "output_dir": request["output_dir"],
        "layers": request["layers"],
        "token_position": request["token_position"],
        "word_field": request["word_field"],
        "sample_id_field": request["sample_id_field"],
        "batch_size": 1,
        "dtype": "f32",
        "output_format": "npy",
        "prompt_hashes_only": request["prompt_hashes_only"],
        "write_logits": request["write_logits"],
        "resume": False,
        "max_seq_len": request["max_seq_len"],
        "record_model_sha256": False,
        "llama_cpp_binary": None,
        "run_metadata": metadata,
    }
    manifest = {
        "schema_version": request["contract_version"],
        "layout": request["layout"],
        "artifact_kind": "ember_hidden_states",
        "created_at_unix": int(time.time()),
        "run_id": None,
        "run_dir": request["output_dir"],
        "config_path": "config.toml",
        "samples_path": "samples.jsonl",
        "tokenization_path": "tokenization.jsonl",
        "positions_path": "positions.jsonl",
        "checksums_path": "checksums.json",
        "report_path": "report.json",
        "logits_path": None,
        "tensor_contract": {
            "storage": "layer-sharded-npy",
            "dtype": "f32",
            "byte_order": "little-endian",
            "sample_axis": 0,
            "hidden_axis": 1,
            "layers": [],
            "logits": None,
        },
        "sample_count": len(samples),
        "sample_order_hash": stable_hash(order_payload),
        "config_hash": "fnv1a64:0000000000000000",
        "dtype": "f32",
        "output_format": "npy",
        "model": {
            "path": request["model_path"],
            "architecture": "qwen3",
            "n_layers": 0,
            "embed_dim": 0,
            "max_seq_len": 0,
            "file_size_bytes": Path(request["model_path"]).stat().st_size,
            "sha256": None,
            "gguf_metadata": None,
        },
        "backend": {
            "name": request["backend"],
            "version": "llama-tokenize",
            "executable": tokenize_bin,
            "commit": None,
            "details": {
                **provenance,
                "supports_hidden_states": False,
                "supports_logits": False,
            },
        },
        "provenance": provenance,
        "extraction_config": extraction_config,
    }
    report = {
        "schema_version": request["contract_version"],
        "layout": request["layout"],
        "status": "complete",
        **provenance,
    }
    parity_metadata = {
        "engine": "llama.cpp",
        "adapter": "llama-tokenize",
        "model": request["model_path"],
        "arch": "qwen3",
        "prompts": parity_prompts,
        **provenance,
    }

    write_jsonl(Path(request["samples_path"]), samples)
    write_jsonl(Path(request["tokenization_path"]), tokenization)
    write_jsonl(Path(request["positions_path"]), positions)
    Path(request["manifest_path"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(request["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(request["checksums_path"]).write_text("{}\n", encoding="utf-8")
    (run_dir / "metadata.llamacpp.json").write_text(
        json.dumps(parity_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
