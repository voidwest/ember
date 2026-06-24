#!/usr/bin/env python3
"""llama-cpp-python logits adapter for Ember external-backend smoke tests.

This helper implements Ember's `llama-cpp-external` request contract using the
local llama-cpp-python/libllama binding. It evaluates tiny prompts and writes
final-token logits only. It does not generate text or extract hidden states.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if request["layers"]:
        raise ValueError("logits smoke requires layers = []")
    if not request["write_logits"] or not request.get("logits_path"):
        raise ValueError("logits smoke requires write_logits = true")

    metadata = request.get("run_metadata") or {}
    model_arch = metadata.get("model_arch") or metadata.get("arch") or "unknown"
    n_ctx = int(metadata.get("n_ctx") or request.get("max_seq_len") or 256)
    if n_ctx < 16:
        raise ValueError("n_ctx must be at least 16 for logits smoke")

    run_dir = Path(request["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    llama = Llama(
        model_path=request["model_path"],
        logits_all=True,
        n_ctx=n_ctx,
        n_batch=n_ctx,
        verbose=False,
    )
    vocab_size = int(llama.n_vocab())

    samples = []
    tokenization = []
    positions = []
    logits_rows = []
    parity_prompts = []
    order_pairs = []

    for sample_index, row in enumerate(read_jsonl(Path(request["input_jsonl_path"]))):
        sample_id = str(row.get(request["sample_id_field"], sample_index))
        prompt = render_prompt(request["prompt_template"], row)
        prompt_hash = stable_hash(prompt)
        token_ids = [int(token_id) for token_id in llama.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)]
        if not token_ids:
            raise ValueError(f"sample {sample_id} produced no token IDs")
        if len(token_ids) > n_ctx:
            raise ValueError(f"sample {sample_id} has {len(token_ids)} tokens, exceeding n_ctx={n_ctx}")

        llama.reset()
        llama.eval(token_ids)
        last_index = len(token_ids) - 1
        final_logits = np.asarray(llama._scores[last_index, :], dtype=np.float32).copy()
        if final_logits.shape != (vocab_size,):
            raise ValueError(f"unexpected logits shape for sample {sample_id}: {final_logits.shape}")
        logits_rows.append(final_logits)

        selected = [last_index]
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
                "selected_token_positions": selected,
            }
        )
        order_pairs.append((sample_id, prompt_hash))

    logits = np.stack(logits_rows, axis=0).astype(np.float32, copy=False)
    np.save(request["logits_path"], logits)

    order_payload = "".join(f"{sample_id}\t{prompt_hash}\n" for sample_id, prompt_hash in order_pairs)
    provenance = {
        "real_llama_cpp": True,
        "binding": "llama-cpp-python",
        "standalone_llama_cpp_binary": False,
        "real_tokenization": True,
        "real_logits": True,
        "no_generation": True,
        "no_logits": False,
        "no_hidden_states": True,
        "not_research_output": True,
        "purpose": "llama-cpp-python logits-only smoke test for Ember external backend plumbing",
    }
    extraction_config = {
        "run_id": None,
        "model_path": request["model_path"],
        "architecture": model_arch,
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
        "logits_path": "logits.npy",
        "tensor_contract": {
            "storage": "layer-sharded-npy",
            "dtype": "f32",
            "byte_order": "little-endian",
            "sample_axis": 0,
            "hidden_axis": 1,
            "layers": [],
            "logits": {
                "path": "logits.npy",
                "shape": [len(samples), vocab_size],
            },
        },
        "sample_count": len(samples),
        "sample_order_hash": stable_hash(order_payload),
        "config_hash": "fnv1a64:0000000000000000",
        "dtype": "f32",
        "output_format": "npy",
        "model": {
            "path": request["model_path"],
            "architecture": model_arch,
            "n_layers": 0,
            "embed_dim": 0,
            "max_seq_len": n_ctx,
            "file_size_bytes": Path(request["model_path"]).stat().st_size,
            "sha256": None,
            "gguf_metadata": None,
        },
        "backend": {
            "name": request["backend"],
            "version": "llama-cpp-python-logits",
            "executable": "scripts/llama_cpp_python_logits_extract.py",
            "commit": None,
            "details": {
                **provenance,
                "supports_hidden_states": False,
                "supports_logits": True,
                "n_ctx": n_ctx,
            },
        },
        "provenance": provenance,
        "extraction_config": extraction_config,
    }
    report = {
        "schema_version": request["contract_version"],
        "layout": request["layout"],
        "status": "complete",
        "logits_shape": [len(samples), vocab_size],
        **provenance,
    }
    logits_metadata = {
        "engine": "llama.cpp",
        "adapter": "llama-cpp-python",
        "model": request["model_path"],
        "arch": model_arch,
        "logits_path": "logits.npy",
        "logits_shape": [len(samples), vocab_size],
        "prompts": parity_prompts,
        **provenance,
    }

    write_jsonl(Path(request["samples_path"]), samples)
    write_jsonl(Path(request["tokenization_path"]), tokenization)
    write_jsonl(Path(request["positions_path"]), positions)
    Path(request["manifest_path"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(request["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(request["checksums_path"]).write_text("{}\n", encoding="utf-8")
    (run_dir / "metadata.llamacpp-python-logits.json").write_text(
        json.dumps(logits_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
