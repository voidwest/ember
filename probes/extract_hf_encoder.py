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


def pool(hidden_states, token_indices: list[int], mode: str):
    stacked = np.stack([h[0].detach().cpu().numpy() for h in hidden_states], axis=0)
    if mode == "cls":
        return stacked[:, 0, :]
    if mode == "last":
        return stacked[:, -1, :]
    if mode == "mean":
        return stacked.mean(axis=1)
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
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    torch, AutoModel, AutoTokenizer = require_hf()

    rows = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    activations = []
    token_selections = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            text = row["text"]
            encoded = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            span = row.get("target_span")
            token_indices = span_token_indices(offsets, span[0], span[1]) if span else []
            encoded = {k: v.to(args.device) for k, v in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            activations.append(pool(outputs.hidden_states, token_indices, args.pool))
            token_selections.append(
                {
                    "index": i,
                    "target_span": span,
                    "token_indices": token_indices,
                    "token_count": len(offsets),
                }
            )
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"[{i + 1}/{len(rows)}] extracted")

    arr = np.stack(activations, axis=0).astype(np.float32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, arr)
    print(f"wrote {args.output} shape={arr.shape}")

    metadata = {
        "model": args.model,
        "benchmark": args.benchmark,
        "output": args.output,
        "pool": args.pool,
        "n_rows": len(rows),
        "activation_shape": list(arr.shape),
        "token_selections": token_selections,
    }
    metadata_path = args.metadata_output or args.output.replace(".npy", "_metadata.json")
    Path(metadata_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
