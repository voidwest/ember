#!/usr/bin/env python3
"""llama-cpp-python final-token logits adapter for Ember artifact checks."""

from __future__ import annotations

import argparse
import os
import tempfile
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
except ModuleNotFoundError:  # imported as scripts.llama_cpp_python_logits_extract
    from scripts.extraction_adapter_common import (
        common_manifest,
        extraction_config,
        load_request,
        load_samples,
        write_common_rows,
        write_json,
        write_report_and_checksums,
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _final_logits(llama: object, token_count: int, np: object) -> tuple[object, str]:
    """Read final-token logits, preferring the binding's public interface."""
    public_logits = getattr(llama, "eval_logits", None)
    if public_logits is not None:
        retained = list(public_logits)
        if retained:
            return np.asarray(retained[-1], dtype="<f4").copy(), "eval_logits"
    scores = getattr(llama, "_scores", None)
    if scores is not None and len(scores) >= token_count:
        return (
            np.asarray(scores[token_count - 1], dtype="<f4").copy(),
            "_scores_fallback",
        )
    raise RuntimeError("llama-cpp-python did not retain final-token logits")


def _atomic_save_npy(path: Path, array: object, np: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    try:
        import llama_cpp
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "llama_cpp and numpy are required in the adapter environment"
        ) from error

    request = load_request(Path(args.request))
    if request.get("layers"):
        raise ValueError("logits adapter requires layers=[]")
    if not request.get("write_logits") or not request.get("logits_path"):
        raise ValueError("logits adapter requires write_logits=true and a logits path")

    config = extraction_config(request)
    metadata = request.get("run_metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata must be an object")
    requested_limit = request.get("max_seq_len")
    configured_context = metadata.get("n_ctx")
    if configured_context is not None:
        n_ctx = _positive_int(configured_context, "run_metadata.n_ctx")
    elif requested_limit is not None:
        n_ctx = requested_limit
    else:
        n_ctx = 256
    if requested_limit is not None and n_ctx < requested_limit:
        raise ValueError("run_metadata.n_ctx cannot be smaller than max_seq_len")
    effective_limit = min(n_ctx, requested_limit or n_ctx)
    staging_dir = Path(request["manifest_path"]).parent
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Ember staging directory is missing: {staging_dir}")

    llama = llama_cpp.Llama(
        model_path=request["model_path"],
        logits_all=True,
        n_ctx=n_ctx,
        n_batch=n_ctx,
        verbose=False,
    )
    vocab_size = int(llama.n_vocab())
    if vocab_size <= 0:
        raise ValueError(f"llama.cpp reported invalid vocabulary size {vocab_size}")

    samples = load_samples(request, config)
    token_rows: list[list[int]] = []
    logits_rows: list[object] = []
    logits_interfaces: set[str] = set()
    for sample in samples:
        token_ids = [
            int(token_id)
            for token_id in llama.tokenize(
                sample["prompt"].encode("utf-8"), add_bos=True, special=False
            )
        ]
        if not token_ids:
            raise ValueError(f"sample {sample['sample_id']!r} produced no token IDs")
        if len(token_ids) > effective_limit:
            raise ValueError(
                f"sample {sample['sample_id']!r} has {len(token_ids)} tokens, "
                f"exceeding effective context limit {effective_limit}"
            )
        if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
            raise ValueError(f"sample {sample['sample_id']!r} produced out-of-range token IDs")

        llama.reset()
        llama.eval(token_ids)
        final_logits, logits_interface = _final_logits(llama, len(token_ids), np)
        logits_interfaces.add(logits_interface)
        if final_logits.shape != (vocab_size,):
            raise ValueError(
                f"unexpected logits shape for sample {sample['sample_id']!r}: "
                f"{final_logits.shape}, expected {(vocab_size,)}"
            )
        if not np.isfinite(final_logits).all():
            raise ValueError(f"non-finite logits for sample {sample['sample_id']!r}")
        if float(final_logits.std(dtype=np.float64)) == 0.0:
            raise ValueError(
                f"zero-variance logits for sample {sample['sample_id']!r}; "
                "the llama.cpp reference output is not trustworthy"
            )
        token_rows.append(token_ids)
        logits_rows.append(final_logits)

    logits = np.stack(logits_rows, axis=0).astype("<f4", copy=False)
    _atomic_save_npy(Path(request["logits_path"]), logits, np)

    parity_prompts = write_common_rows(
        request=request,
        samples=samples,
        token_ids=token_rows,
    )
    provenance = {
        "real_llama_cpp": True,
        "binding": "llama-cpp-python",
        "real_tokenization": True,
        "real_logits": True,
        "no_generation": True,
        "no_hidden_states": True,
        "not_research_output": True,
        "purpose": "llama-cpp-python logits check for Ember external backend plumbing",
        "supports_hidden_states": False,
        "supports_logits": True,
        "n_ctx": n_ctx,
        "effective_context_limit": effective_limit,
        "logits_interfaces": sorted(logits_interfaces),
        "tokenize_add_bos": True,
        "tokenize_special": False,
    }
    manifest = common_manifest(
        request=request,
        config=config,
        samples=samples,
        model_max_seq_len=n_ctx,
        backend_version=getattr(llama_cpp, "__version__", "unknown"),
        backend_executable=str(Path(__file__).resolve()),
        backend_details=provenance,
        logits_shape=[len(samples), vocab_size],
    )
    write_json(Path(request["manifest_path"]), manifest)
    write_report_and_checksums(
        request=request,
        sample_count=len(samples),
        logits_written=True,
    )
    write_json(
        staging_dir / "metadata.llamacpp-python-logits.json",
        {
            "engine": "llama.cpp",
            "adapter": "llama-cpp-python",
            "model": request["model_path"],
            "arch": config.get("architecture"),
            "logits_path": "logits.npy",
            "logits_shape": [len(samples), vocab_size],
            "prompts": parity_prompts,
            **provenance,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
