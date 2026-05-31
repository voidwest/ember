"""compare Ember logits against trusted reference logits.

The reference file should be produced by an external implementation such as
Hugging Face Transformers or llama.cpp for the same model, tokenizer, prompt,
and quantization path. Ember logits can be produced with:

    cargo run --release -- --arch qwen3 --model model.gguf \
      --prompt "The capital of France is" --dump-logits ember_logits.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np


def top_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits.reshape(-1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="compare logits against a golden reference")
    parser.add_argument("--ember", required=True, help="Ember .npy logits")
    parser.add_argument("--reference", required=True, help="trusted reference .npy logits")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
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
    report = {
        "ember": args.ember,
        "reference": args.reference,
        "shape": list(ember.shape),
        "max_abs_diff": float(diff.reshape(-1)[max_idx]),
        "max_rel_diff": float(rel.reshape(-1)[max_idx]),
        "max_diff_index": max_idx,
        "ember_top_token": top_token(ember),
        "reference_top_token": top_token(reference),
        "top_token_matches": top_token(ember) == top_token(reference),
        "within_tolerance": bool(np.allclose(ember, reference, atol=args.atol, rtol=args.rtol)),
        "atol": args.atol,
        "rtol": args.rtol,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["within_tolerance"] or not report["top_token_matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
