#!/usr/bin/env python3
"""Real llama.cpp tokenization adapter for Ember external-backend checks.

This adapter intentionally supports prompt-final tokenization artifacts only;
the llama-tokenize CLI does not expose the character offsets required for
word-span extraction.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
from pathlib import Path

try:
    from extraction_adapter_common import (
        common_manifest,
        extraction_config,
        load_request,
        load_samples,
        write_common_rows,
        write_json,
        write_report_and_checksums,
    )
except ModuleNotFoundError:  # imported as scripts.llama_cpp_tokenize_extract
    from scripts.extraction_adapter_common import (
        common_manifest,
        extraction_config,
        load_request,
        load_samples,
        write_common_rows,
        write_json,
        write_report_and_checksums,
    )


def llama_tokenize(
    binary: str, model_path: str, prompt: str, *, timeout_seconds: float = 120.0
) -> list[int]:
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
        timeout=timeout_seconds,
    )
    try:
        parsed = ast.literal_eval(output.stdout.strip())
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"unexpected llama-tokenize output: {output.stdout!r}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"unexpected llama-tokenize output: {output.stdout!r}")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in parsed
    ):
        raise ValueError(f"llama-tokenize returned invalid token IDs: {parsed!r}")
    return parsed


def llama_version(binary: str) -> str:
    try:
        result = subprocess.run(
            [binary, "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[0][:500] if text else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    request = load_request(request_path)
    if request.get("layers"):
        raise ValueError("llama-tokenize does not expose hidden-state layers")
    if request.get("write_logits") or request.get("logits_path"):
        raise ValueError("llama-tokenize does not expose logits")

    config = extraction_config(request)
    metadata = request.get("run_metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata must be an object")
    tokenize_bin = os.environ.get("LLAMA_TOKENIZE_BIN") or metadata.get(
        "llama_tokenize_bin"
    )
    if not isinstance(tokenize_bin, str) or not tokenize_bin.strip():
        raise ValueError(
            "llama-tokenize path required via LLAMA_TOKENIZE_BIN or "
            "run_metadata.llama_tokenize_bin"
        )
    if not Path(tokenize_bin).is_file():
        raise FileNotFoundError(f"llama-tokenize binary not found: {tokenize_bin}")
    if not os.access(tokenize_bin, os.X_OK):
        raise PermissionError(f"llama-tokenize path is not executable: {tokenize_bin}")
    timeout_seconds = metadata.get("timeout_seconds", 120.0)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < float(timeout_seconds) <= 3600
    ):
        raise ValueError("run_metadata.timeout_seconds must be in (0, 3600]")

    staging_dir = Path(request["manifest_path"]).parent
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Ember staging directory is missing: {staging_dir}")
    samples = load_samples(request, config)
    token_rows: list[list[int]] = []
    max_seq_len = request.get("max_seq_len")
    for sample in samples:
        token_ids = llama_tokenize(
            tokenize_bin,
            request["model_path"],
            sample["prompt"],
            timeout_seconds=float(timeout_seconds),
        )
        if max_seq_len is not None and len(token_ids) > max_seq_len:
            raise ValueError(
                f"sample {sample['sample_id']!r} has {len(token_ids)} tokens, "
                f"exceeding max_seq_len={max_seq_len}"
            )
        token_rows.append(token_ids)
    observed_max_tokens = max(len(token_ids) for token_ids in token_rows)

    parity_prompts = write_common_rows(
        request=request,
        samples=samples,
        token_ids=token_rows,
    )
    provenance = {
        "real_llama_cpp": True,
        "real_tokenization": True,
        "no_generation": True,
        "no_logits": True,
        "no_hidden_states": True,
        "not_research_output": True,
        "purpose": "real llama.cpp tokenization check for Ember external backend plumbing",
        "llama_tokenize_bin": str(Path(tokenize_bin).resolve()),
        "supports_hidden_states": False,
        "supports_logits": False,
        "timeout_seconds": float(timeout_seconds),
        "context_limit_source": (
            "request.max_seq_len" if max_seq_len is not None else "observed_token_count_lower_bound"
        ),
    }
    manifest = common_manifest(
        request=request,
        config=config,
        samples=samples,
        model_max_seq_len=int(max_seq_len or observed_max_tokens),
        backend_version=llama_version(tokenize_bin),
        backend_executable=str(Path(tokenize_bin).resolve()),
        backend_details=provenance,
        logits_shape=None,
    )
    write_json(Path(request["manifest_path"]), manifest)
    write_report_and_checksums(
        request=request,
        sample_count=len(samples),
        logits_written=False,
    )
    write_json(
        staging_dir / "metadata.llamacpp.json",
        {
            "engine": "llama.cpp",
            "adapter": "llama-tokenize",
            "model": request["model_path"],
            "arch": config.get("architecture"),
            "prompts": parity_prompts,
            **provenance,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
