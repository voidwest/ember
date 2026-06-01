"""compare Ember logits against trusted reference logits.

The reference file should be produced by an external implementation such as
Hugging Face Transformers or llama.cpp for the same model, tokenizer, prompt,
and quantization path. Ember logits can be produced with:

    cargo run --release -- --arch qwen3 --model model.gguf \
      --prompt "The capital of France is" --dump-logits ember_logits.npy
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def top_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits.reshape(-1)))


def top_k(logits: np.ndarray, k: int) -> list[int]:
    flat = logits.reshape(-1)
    k = min(k, flat.size)
    indices = np.argpartition(-flat, np.arange(k))[:k]
    return [int(i) for i in indices[np.argsort(-flat[indices])]]


def sha256_file(path: str | None) -> str | None:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="compare logits against a golden reference")
    parser.add_argument("--ember", required=True, help="Ember .npy logits")
    parser.add_argument("--reference", required=True, help="trusted reference .npy logits")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=None, help="optional model path for SHA-256 provenance")
    parser.add_argument("--model-sha256", default=None, help="precomputed model SHA-256")
    parser.add_argument("--tokenizer", default=None, help="tokenizer path/name used for the prompt")
    parser.add_argument("--gguf-metadata", default=None, help="optional GGUF metadata JSON sidecar")
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()

    ember = np.load(args.ember).astype(np.float32)
    reference = np.load(args.reference).astype(np.float32)
    if ember.shape != reference.shape:
        raise SystemExit(f"shape mismatch: ember={ember.shape} reference={reference.shape}")

    diff = np.abs(ember - reference)
    denom = np.maximum(np.maximum(np.abs(ember), np.abs(reference)), 1.0)
    rel = diff / denom
    max_idx = int(np.argmax(diff))
    ember_top_k = top_k(ember, args.top_k)
    reference_top_k = top_k(reference, args.top_k)
    top_k_overlap = len(set(ember_top_k) & set(reference_top_k)) / max(len(reference_top_k), 1)
    gguf_metadata = None
    if args.gguf_metadata:
        gguf_metadata = json.loads(Path(args.gguf_metadata).read_text(encoding="utf-8"))
    report = {
        "ember": args.ember,
        "reference": args.reference,
        "shape": list(ember.shape),
        "max_abs_diff": float(diff.reshape(-1)[max_idx]),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.reshape(-1)[max_idx]),
        "mean_rel_diff": float(rel.mean()),
        "max_diff_index": max_idx,
        "ember_top_token": top_token(ember),
        "reference_top_token": top_token(reference),
        "top_token_matches": top_token(ember) == top_token(reference),
        "top_k": args.top_k,
        "ember_top_k": ember_top_k,
        "reference_top_k": reference_top_k,
        "top_k_ordered_matches": ember_top_k == reference_top_k,
        "top_k_overlap": float(top_k_overlap),
        "within_tolerance": bool(np.allclose(ember, reference, atol=args.atol, rtol=args.rtol)),
        "atol": args.atol,
        "rtol": args.rtol,
        "model": args.model,
        "model_sha256": args.model_sha256 or sha256_file(args.model),
        "tokenizer": args.tokenizer,
        "gguf_metadata": gguf_metadata,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["within_tolerance"] or not report["top_token_matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
