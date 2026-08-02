"""extract per-layer encoder activations from Hugging Face models.

This is the encoder-side counterpart to Ember's Rust decoder extractor. It
expects benchmark rows with:

  - `text`: full input string
  - `target_span`: [char_start, char_end] for token/span pooling

The output is a raw `.npy` array shaped `(n_rows, n_layers, hidden_dim)`, where
`n_layers` includes the embedding output plus every encoder layer returned by
`output_hidden_states=True`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .train_linear_probe import (
        atomic_save_npy,
        atomic_write_text,
        sha256_file,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from train_linear_probe import (
        atomic_save_npy,
        atomic_write_text,
        sha256_file,
        validate_activation_tensor,
    )


def require_hf():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "extract_hf_encoder.py requires torch and transformers. "
            "Install the optional encoder stack first."
        ) from exc
    return torch, AutoModel, AutoTokenizer


def span_token_indices(offsets, start: int, end: int) -> list[int]:
    indices = []
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if tok_start < end and tok_end > start:
            indices.append(i)
    return indices


def pool(hidden_states, token_indices: list[int], content_indices: list[int], mode: str):
    stacked = np.stack([h[0].detach().cpu().numpy() for h in hidden_states], axis=0)
    if mode == "cls":
        return stacked[:, 0, :]
    if mode == "last":
        if not content_indices:
            raise ValueError("tokenized input has no non-special content tokens")
        return stacked[:, content_indices[-1], :]
    if mode == "mean":
        if not content_indices:
            raise ValueError("tokenized input has no non-special content tokens")
        return stacked[:, content_indices, :].mean(axis=1)
    if mode == "target_mean":
        if not token_indices:
            raise ValueError("target span did not align to any tokenizer offsets")
        return stacked[:, token_indices, :].mean(axis=1)
    if mode == "target_first":
        if not token_indices:
            raise ValueError("target span did not align to any tokenizer offsets")
        return stacked[:, token_indices[0], :]
    if mode == "target_last":
        if not token_indices:
            raise ValueError("target span did not align to any tokenizer offsets")
        return stacked[:, token_indices[-1], :]
    raise ValueError(f"unknown pooling mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="extract HF encoder hidden states")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--benchmark", required=True, help="JSON rows from build_conllu_benchmark.py")
    parser.add_argument("--output", required=True, help="output .npy path")
    parser.add_argument("--metadata-output", default=None, help="optional metadata JSON path")
    parser.add_argument(
        "--pool",
        choices=["cls", "last", "mean", "target_mean", "target_first", "target_last"],
        default="target_mean",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--revision",
        help="immutable Hugging Face model/tokenizer revision (commit SHA recommended)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if Path(args.output).suffix != ".npy":
        parser.error("--output must end in .npy")

    torch, AutoModel, AutoTokenizer = require_hf()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_file():
        parser.error(f"benchmark file does not exist: {benchmark_path}")

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {benchmark_path}")

    rows = json.loads(
        benchmark_path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("benchmark must be a non-empty JSON array of objects")
    if args.limit is not None:
        rows = rows[: args.limit]
    row_ids = []
    for index, row in enumerate(rows):
        row_id = row.get("id")
        if not isinstance(row_id, (str, int)) or isinstance(row_id, bool) or not str(row_id):
            raise ValueError(f"benchmark row {index} requires a non-empty scalar id")
        row_ids.append(str(row_id))
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("benchmark row ids must be unique")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast Hugging Face tokenizer is required for offset mappings")
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    activations = []
    token_selections = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"benchmark row {i} has no non-empty text")
            encoded = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                truncation=True,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            special_mask = encoded.pop("special_tokens_mask")[0].tolist()
            if len(offsets) != len(special_mask):
                raise ValueError(f"tokenizer returned inconsistent fields for row {i}")
            for token_index, offset in enumerate(offsets):
                if (
                    not isinstance(offset, list)
                    or len(offset) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in offset
                    )
                    or offset[0] < 0
                    or offset[1] < offset[0]
                    or offset[1] > len(text)
                ):
                    raise ValueError(
                        f"tokenizer returned invalid offset for row {i}, token "
                        f"{token_index}: {offset!r}"
                    )
            attention_mask = encoded.get("attention_mask")
            active = attention_mask[0].tolist() if attention_mask is not None else [1] * len(offsets)
            content_indices = [
                index
                for index, (special, attended) in enumerate(zip(special_mask, active))
                if not special and attended
            ]
            covered_end = max((offsets[index][1] for index in content_indices), default=0)
            if text[covered_end:].strip():
                raise ValueError(
                    f"benchmark row {i} was truncated by the tokenizer/model limit"
                )
            span = row.get("target_span")
            if span is not None and (
                not isinstance(span, list)
                or len(span) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
                or not 0 <= span[0] < span[1] <= len(text)
            ):
                raise ValueError(f"benchmark row {i} has invalid target_span {span!r}")
            if args.pool.startswith("target_") and span is None:
                raise ValueError(f"benchmark row {i} requires target_span for pool={args.pool}")
            token_indices = span_token_indices(offsets, span[0], span[1]) if span else []
            if any(index not in content_indices for index in token_indices):
                raise ValueError(f"benchmark row {i} target span mapped to a special/padded token")
            encoded = {k: v.to(args.device) for k, v in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            if not outputs.hidden_states:
                raise RuntimeError("model did not return hidden states")
            activations.append(
                pool(outputs.hidden_states, token_indices, content_indices, args.pool)
            )
            token_selections.append(
                {
                    "index": i,
                    "row_id": row_ids[i],
                    "target_span": span,
                    "token_indices": token_indices,
                    "token_count": len(offsets),
                    "content_token_count": len(content_indices),
                }
            )
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"[{i + 1}/{len(rows)}] extracted")

    arr = np.stack(activations, axis=0).astype(np.float32)
    validate_activation_tensor(arr, args.output, expected_rows=len(rows))
    atomic_save_npy(args.output, arr)
    print(f"wrote {args.output} shape={arr.shape}")

    metadata = {
        "model": args.model,
        "requested_revision": args.revision,
        "benchmark": args.benchmark,
        "benchmark_sha256": sha256_file(args.benchmark),
        "output": args.output,
        "activations_sha256": sha256_file(args.output),
        "pool": args.pool,
        "n_rows": len(rows),
        "activation_shape": list(arr.shape),
        "token_selections": token_selections,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_config": model.config.to_dict(),
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "offset_unit": "unicode_character_index",
    }
    metadata_path = args.metadata_output or args.output.replace(".npy", "_metadata.json")
    atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
