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


def load_json(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested_get(obj: dict | None, path: list[str]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_token_ids(obj: dict | None, role: str) -> list[int] | None:
    role_paths = {
        "ember": [
            ["ember_token_ids"],
            ["ember", "token_ids"],
            ["token_audit", "ember_token_ids"],
            ["token_audit", "ember", "token_ids"],
            ["token_audit", "token_ids"],
            ["token_ids"],
        ],
        "reference": [
            ["reference_token_ids"],
            ["reference", "token_ids"],
            ["token_audit", "reference_token_ids"],
            ["token_audit", "reference", "token_ids"],
            ["token_audit", "token_ids"],
            ["token_ids"],
        ],
    }
    for path in role_paths[role]:
        value = nested_get(obj, path)
        if isinstance(value, list) and all(isinstance(v, int) for v in value):
            return value
    return None


def extract_tokenizer_sha256(obj: dict | None) -> str | None:
    for path in [
        ["tokenizer_sha256"],
        ["tokenizer", "sha256"],
        ["token_audit", "tokenizer_sha256"],
        ["run_manifest", "tokenizer", "sha256"],
    ]:
        value = nested_get(obj, path)
        if isinstance(value, str):
            return value
    return None


def token_audit_gate(
    token_audit: dict | None,
    ember_metadata: dict | None,
    reference_metadata: dict | None,
) -> dict:
    ember_ids = extract_token_ids(token_audit, "ember") or extract_token_ids(
        ember_metadata, "ember"
    )
    reference_ids = extract_token_ids(token_audit, "reference") or extract_token_ids(
        reference_metadata, "reference"
    )
    ember_tokenizer_sha256 = extract_tokenizer_sha256(token_audit) or extract_tokenizer_sha256(
        ember_metadata
    )
    reference_tokenizer_sha256 = extract_tokenizer_sha256(reference_metadata)

    failures = []
    if ember_ids is None:
        failures.append("missing Ember token ids")
    if reference_ids is None:
        failures.append("missing reference token ids")
    if ember_ids is not None and reference_ids is not None and ember_ids != reference_ids:
        failures.append("token ids differ")
    if (
        ember_tokenizer_sha256
        and reference_tokenizer_sha256
        and ember_tokenizer_sha256 != reference_tokenizer_sha256
    ):
        failures.append("tokenizer SHA-256 differs")

    return {
        "required": True,
        "passed": not failures,
        "failures": failures,
        "ember_token_count": len(ember_ids) if ember_ids is not None else None,
        "reference_token_count": len(reference_ids) if reference_ids is not None else None,
        "ember_token_ids": ember_ids,
        "reference_token_ids": reference_ids,
        "ember_tokenizer_sha256": ember_tokenizer_sha256,
        "reference_tokenizer_sha256": reference_tokenizer_sha256,
    }


def top_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits.reshape(-1)))


def top_k(logits: np.ndarray, k: int) -> list[int]:
    flat = logits.reshape(-1)
    if k <= 0:
        return []
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


def classify(report: dict, args: argparse.Namespace) -> str:
    notes = report["notes"]
    if not report["shape_check"]["matches"]:
        notes.append("shape mismatch")
        return "golden_fail"

    if (
        args.max_diff_threshold is not None
        and report["max_abs_diff"] is not None
        and report["max_abs_diff"] > args.max_diff_threshold
    ):
        notes.append("max absolute diff exceeds configured threshold")
        return "golden_fail"

    if (
        args.mean_diff_threshold is not None
        and report["mean_abs_diff"] is not None
        and report["mean_abs_diff"] > args.mean_diff_threshold
    ):
        notes.append("mean absolute diff exceeds configured threshold")
        return "golden_fail"

    if report["top_k_overlap_ratio"] < args.topk_overlap_threshold:
        notes.append("top-k overlap below configured threshold")
        return "golden_fail"

    if report["top_1_match"]:
        return "golden_pass"

    notes.append("top-1 differs, but top-k overlap meets configured threshold")
    return "golden_warn"


def main() -> None:
    parser = argparse.ArgumentParser(description="compare logits against a golden reference")
    parser.add_argument("--ember", required=True, help="Ember .npy logits")
    parser.add_argument("--reference", required=True, help="trusted reference .npy logits")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label", default=None, help="optional model/run label")
    parser.add_argument("--model", default=None, help="optional model path for SHA-256 provenance")
    parser.add_argument("--model-sha256", default=None, help="precomputed model SHA-256")
    parser.add_argument("--tokenizer", default=None, help="tokenizer path/name used for the prompt")
    parser.add_argument("--gguf-metadata", default=None, help="optional GGUF metadata JSON sidecar")
    parser.add_argument("--metadata", default=None, help="Ember tokenizer/model metadata JSON sidecar")
    parser.add_argument("--reference-metadata", default=None, help="reference tokenizer/model metadata JSON sidecar")
    parser.add_argument("--token-audit", default=None, help="combined token audit JSON with Ember and reference token ids")
    parser.add_argument("--max-diff-threshold", type=float, default=None)
    parser.add_argument("--mean-diff-threshold", type=float, default=None)
    parser.add_argument("--topk-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--atol", type=float, default=None, help="optional np.allclose absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="optional np.allclose relative tolerance")
    parser.add_argument("--output", required=True, help="JSON report path")
    args = parser.parse_args()

    gguf_metadata = load_json(args.gguf_metadata)
    metadata = load_json(args.metadata)
    reference_metadata = load_json(args.reference_metadata)
    combined_token_audit = load_json(args.token_audit)
    token_audit = token_audit_gate(combined_token_audit, metadata, reference_metadata)
    if not token_audit["passed"]:
        report = {
            "label": args.label,
            "ember": args.ember,
            "reference": args.reference,
            "classification": "token_audit_fail",
            "token_audit": token_audit,
            "metadata": metadata,
            "reference_metadata": reference_metadata,
            "gguf_metadata": gguf_metadata,
            "notes": ["token audit failed before numeric comparison"],
        }
        print(json.dumps(report, indent=2))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1)

    ember = np.load(args.ember).astype(np.float32)
    reference = np.load(args.reference).astype(np.float32)
    shapes_match = ember.shape == reference.shape
    diff = np.abs(ember - reference) if shapes_match else None
    rel = None
    max_idx = None
    if diff is not None:
        denom = np.maximum(np.maximum(np.abs(ember), np.abs(reference)), 1.0)
        rel = diff / denom
        max_idx = int(np.argmax(diff))

    ember_top_1 = top_token(ember)
    reference_top_1 = top_token(reference)
    ember_top_k = top_k(ember, args.top_k)
    reference_top_k = top_k(reference, args.top_k)
    top_k_overlap_count = len(set(ember_top_k) & set(reference_top_k))
    top_k_overlap_ratio = top_k_overlap_count / max(len(reference_top_k), 1)
    within_tolerance = None
    if shapes_match and args.atol is not None and args.rtol is not None:
        within_tolerance = bool(np.allclose(ember, reference, atol=args.atol, rtol=args.rtol))

    report = {
        "label": args.label,
        "ember": args.ember,
        "reference": args.reference,
        "shape_check": {
            "matches": shapes_match,
            "ember_shape": list(ember.shape),
            "reference_shape": list(reference.shape),
        },
        "shape": list(ember.shape),
        "max_abs_diff": float(diff.reshape(-1)[max_idx]) if diff is not None else None,
        "mean_abs_diff": float(diff.mean()) if diff is not None else None,
        "max_rel_diff": float(rel.reshape(-1)[max_idx]) if rel is not None else None,
        "mean_rel_diff": float(rel.mean()) if rel is not None else None,
        "max_diff_index": max_idx,
        "top_1_ember_token_id": ember_top_1,
        "top_1_reference_token_id": reference_top_1,
        "top_1_match": ember_top_1 == reference_top_1,
        "top_k": args.top_k,
        "top_k_ember_ids": ember_top_k,
        "top_k_reference_ids": reference_top_k,
        "top_k_overlap_count": top_k_overlap_count,
        "top_k_overlap_ratio": float(top_k_overlap_ratio),
        "top_k_ordered_matches": ember_top_k == reference_top_k,
        "within_tolerance": within_tolerance,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_diff_threshold": args.max_diff_threshold,
        "mean_diff_threshold": args.mean_diff_threshold,
        "topk_overlap_threshold": args.topk_overlap_threshold,
        "model": args.model,
        "model_sha256": args.model_sha256 or sha256_file(args.model),
        "tokenizer": args.tokenizer,
        "token_audit": token_audit,
        "metadata": metadata,
        "reference_metadata": reference_metadata,
        "gguf_metadata": gguf_metadata,
        "notes": [],
    }
    report["classification"] = classify(report, args)

    # Backwards-compatible field names for older ad hoc consumers.
    report["ember_top_token"] = report["top_1_ember_token_id"]
    report["reference_top_token"] = report["top_1_reference_token_id"]
    report["top_token_matches"] = report["top_1_match"]
    report["ember_top_k"] = report["top_k_ember_ids"]
    report["reference_top_k"] = report["top_k_reference_ids"]
    report["top_k_overlap"] = report["top_k_overlap_ratio"]

    print(json.dumps(report, indent=2))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["classification"] == "golden_fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
