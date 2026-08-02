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

try:
    from .train_linear_probe import atomic_write_text
except ImportError:  # direct script execution
    from train_linear_probe import atomic_write_text


def load_json(path: str | None) -> dict | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {source}")

    value = json.loads(
        source.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


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
        if isinstance(value, list) and value and all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value
        ):
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


def extract_model_sha256(obj: dict | None) -> str | None:
    for path in [
        ["model_sha256"],
        ["model", "sha256"],
        ["run_manifest", "model", "sha256"],
        ["manifest", "model", "sha256"],
    ]:
        value = nested_get(obj, path)
        if isinstance(value, str):
            return value
    return None


def token_audit_gate(
    token_audit: dict | None,
    ember_metadata: dict | None,
    reference_metadata: dict | None,
    explicit_tokenizer_sha256: str | None = None,
) -> dict:
    ember_ids = extract_token_ids(token_audit, "ember") or extract_token_ids(
        ember_metadata, "ember"
    )
    reference_ids = extract_token_ids(token_audit, "reference") or extract_token_ids(
        reference_metadata, "reference"
    )
    ember_tokenizer_sha256 = (
        explicit_tokenizer_sha256
        or extract_tokenizer_sha256(token_audit)
        or extract_tokenizer_sha256(ember_metadata)
    )
    reference_tokenizer_sha256 = extract_tokenizer_sha256(reference_metadata)

    failures = []
    if ember_ids is None:
        failures.append("missing Ember token ids")
    if reference_ids is None:
        failures.append("missing reference token ids")
    if ember_ids is not None and reference_ids is not None and ember_ids != reference_ids:
        failures.append("token ids differ")
    if ember_tokenizer_sha256 is None:
        failures.append("missing Ember tokenizer SHA-256")
    if reference_tokenizer_sha256 is None:
        failures.append("missing reference tokenizer SHA-256")
    if ember_tokenizer_sha256 and not _is_sha256(ember_tokenizer_sha256):
        failures.append("invalid Ember tokenizer SHA-256")
    if reference_tokenizer_sha256 and not _is_sha256(reference_tokenizer_sha256):
        failures.append("invalid reference tokenizer SHA-256")
    if ember_tokenizer_sha256 and reference_tokenizer_sha256 and (
        ember_tokenizer_sha256.lower() != reference_tokenizer_sha256.lower()
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


def provenance_gate(
    ember_metadata: dict | None,
    reference_metadata: dict | None,
    explicit_ember_model_sha256: str | None,
) -> dict:
    ember_sha = explicit_ember_model_sha256 or extract_model_sha256(ember_metadata)
    reference_sha = extract_model_sha256(reference_metadata)
    failures = []
    if ember_sha is None:
        failures.append("missing Ember model SHA-256")
    elif not _is_sha256(ember_sha):
        failures.append("invalid Ember model SHA-256")
    if reference_sha is None:
        failures.append("missing reference model SHA-256")
    elif not _is_sha256(reference_sha):
        failures.append("invalid reference model SHA-256")
    if ember_sha and reference_sha and _is_sha256(ember_sha) and _is_sha256(reference_sha):
        if ember_sha.lower() != reference_sha.lower():
            failures.append("model SHA-256 differs")
    return {
        "required": True,
        "passed": not failures,
        "failures": failures,
        "ember_model_sha256": ember_sha,
        "reference_model_sha256": reference_sha,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def top_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits.reshape(-1)))


def top_k(logits: np.ndarray, k: int) -> list[int]:
    flat = logits.reshape(-1)
    if k <= 0:
        return []
    k = min(k, flat.size)
    return [int(index) for index in np.argsort(-flat, kind="stable")[:k]]


def sha256_file(path: str | None) -> str | None:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    left64 = left.reshape(-1).astype(np.float64)
    right64 = right.reshape(-1).astype(np.float64)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left_norm == right_norm else None
    return float(np.dot(left64, right64) / (left_norm * right_norm))


def classify(report: dict, args: argparse.Namespace) -> str:
    notes = report["notes"]
    if not report["shape_check"]["matches"]:
        notes.append("shape mismatch")
        return "golden_fail"
    if report.get("within_tolerance") is False:
        notes.append("np.allclose tolerance check failed")
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

    if not report["numerical_gate_configured"]:
        notes.append("no numerical error threshold was configured; refusing a golden pass")
        return "golden_warn"

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
    parser.add_argument(
        "--allow-warn",
        action="store_true",
        help="return success for golden_warn (never changes the report classification)",
    )
    parser.add_argument("--output", required=True, help="JSON report path")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not 0.0 <= args.topk_overlap_threshold <= 1.0:
        parser.error("--topk-overlap-threshold must be between 0 and 1")
    for name in ("max_diff_threshold", "mean_diff_threshold", "atol", "rtol"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if (args.atol is None) != (args.rtol is None):
        parser.error("--atol and --rtol must be provided together")
    if args.model_sha256 is not None and not _is_sha256(args.model_sha256):
        parser.error("--model-sha256 must contain exactly 64 hexadecimal digits")

    gguf_metadata = load_json(args.gguf_metadata)
    metadata = load_json(args.metadata)
    reference_metadata = load_json(args.reference_metadata)
    combined_token_audit = load_json(args.token_audit)
    explicit_model_sha = args.model_sha256 or sha256_file(args.model)
    if args.model_sha256 and args.model:
        actual_model_sha = sha256_file(args.model)
        if actual_model_sha.lower() != args.model_sha256.lower():
            raise ValueError("--model-sha256 does not match --model")
    explicit_tokenizer_sha = (
        sha256_file(args.tokenizer) if args.tokenizer and Path(args.tokenizer).is_file() else None
    )
    token_audit = token_audit_gate(
        combined_token_audit,
        metadata,
        reference_metadata,
        explicit_tokenizer_sha,
    )
    provenance = provenance_gate(metadata, reference_metadata, explicit_model_sha)
    if not token_audit["passed"] or not provenance["passed"]:
        report = {
            "schema_version": 2,
            "label": args.label,
            "ember": args.ember,
            "reference": args.reference,
            "model_sha256": provenance["ember_model_sha256"],
            "tokenizer_sha256": token_audit["ember_tokenizer_sha256"],
            "classification": (
                "token_audit_fail" if not token_audit["passed"] else "provenance_fail"
            ),
            "token_audit": token_audit,
            "provenance_gate": provenance,
            "metadata": metadata,
            "reference_metadata": reference_metadata,
            "gguf_metadata": gguf_metadata,
            "notes": ["identity audit failed before numeric comparison"],
        }
        print(json.dumps(report, indent=2, allow_nan=False))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise SystemExit(1)

    ember = np.load(args.ember, allow_pickle=False)
    reference = np.load(args.reference, allow_pickle=False)
    for name, logits in (("Ember", ember), ("reference", reference)):
        if logits.ndim not in {1, 2} or logits.size == 0:
            raise ValueError(f"{name} logits must be a non-empty rank-1 or rank-2 tensor")
        if logits.ndim == 2 and logits.shape[0] != 1:
            raise ValueError(f"{name} golden logits must contain exactly one prompt row")
        if not np.isfinite(logits).all():
            raise ValueError(f"{name} logits contain non-finite values")
        if logits.dtype.kind != "f":
            raise ValueError(f"{name} logits must use a floating-point dtype, got {logits.dtype}")
    shapes_match = ember.shape == reference.shape
    ember64 = ember.astype(np.float64, copy=False)
    reference64 = reference.astype(np.float64, copy=False)
    diff = np.abs(ember64 - reference64) if shapes_match else None
    rel = None
    max_idx = None
    max_rel_idx = None
    if diff is not None:
        denom = np.maximum(np.maximum(np.abs(ember64), np.abs(reference64)), 1.0)
        rel = diff / denom
        max_idx = int(np.argmax(diff))
        max_rel_idx = int(np.argmax(rel))

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
        "schema_version": 2,
        "label": args.label,
        "ember": args.ember,
        "reference": args.reference,
        "shape_check": {
            "matches": shapes_match,
            "ember_shape": list(ember.shape),
            "reference_shape": list(reference.shape),
        },
        "shape": list(ember.shape),
        "ember_dtype": str(ember.dtype),
        "reference_dtype": str(reference.dtype),
        "ember_sha256": sha256_file(args.ember),
        "reference_sha256": sha256_file(args.reference),
        "max_abs_diff": float(diff.reshape(-1)[max_idx]) if diff is not None else None,
        "mean_abs_diff": float(diff.mean()) if diff is not None else None,
        "max_rel_diff": float(rel.reshape(-1)[max_rel_idx]) if rel is not None else None,
        "mean_rel_diff": float(rel.mean()) if rel is not None else None,
        "max_diff_index": max_idx,
        "max_rel_diff_index": max_rel_idx,
        "rmse": float(np.sqrt(np.mean(np.square(ember64 - reference64)))) if shapes_match else None,
        "cosine_similarity": _cosine_similarity(ember, reference) if shapes_match else None,
        "exact_bits_equal": bool(
            ember.dtype == reference.dtype
            and np.array_equal(ember.view(np.uint8), reference.view(np.uint8))
        ) if shapes_match else False,
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
        "model_sha256": provenance["ember_model_sha256"],
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": token_audit["ember_tokenizer_sha256"],
        "token_audit": token_audit,
        "provenance_gate": provenance,
        "metadata": metadata,
        "reference_metadata": reference_metadata,
        "gguf_metadata": gguf_metadata,
        "notes": [],
        "numerical_gate_configured": any(
            value is not None
            for value in [args.max_diff_threshold, args.mean_diff_threshold, args.atol]
        ),
    }
    report["classification"] = classify(report, args)

    # Backwards-compatible field names for older ad hoc consumers.
    report["ember_top_token"] = report["top_1_ember_token_id"]
    report["reference_top_token"] = report["top_1_reference_token_id"]
    report["top_token_matches"] = report["top_1_match"]
    report["ember_top_k"] = report["top_k_ember_ids"]
    report["reference_top_k"] = report["top_k_reference_ids"]
    report["top_k_overlap"] = report["top_k_overlap_ratio"]

    print(json.dumps(report, indent=2, allow_nan=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["classification"] != "golden_pass" and not args.allow_warn:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
