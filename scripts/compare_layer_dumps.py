#!/usr/bin/env python3
"""Compare per-layer hidden-state dumps from Ember and llama.cpp.

Input: two binary files containing per-layer hidden states as f32 flat arrays.

Output: per-layer cosine similarity, L2 norms, mean absolute difference, and
max absolute difference. Optionally writes Markdown and/or JSON reports.

Usage:
    python3 compare_layer_dumps.py \
        --ember ember_35layers.bin \
        --reference llama_35layers.bin \
        --layers 35 \
        --hidden-size 1536 \
        --out-md report.md \
        --out-json report.json
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def load_dump_snapshot(
    path: str, n_layers: int, hidden_size: int
) -> tuple[np.ndarray, str]:
    """Read and hash one immutable f32 dump snapshot.

    Reading the bytes once ensures the compared values and the recorded digest
    describe the same file contents even if another process replaces the source.
    """
    expected_floats = n_layers * hidden_size
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_bytes = expected_floats * np.dtype("<f4").itemsize
    contents = source.read_bytes()
    if len(contents) != expected_bytes:
        raise ValueError(
            f"expected {expected_bytes} bytes in {path}, got {len(contents)}"
        )
    digest = hashlib.sha256(contents).hexdigest()
    data = np.frombuffer(contents, dtype="<f4").copy()
    if len(data) != expected_floats:
        raise ValueError(
            f"expected {expected_floats} floats in {path}, "
            f"got {len(data)}. check --layers ({n_layers}) and "
            f"--hidden-size ({hidden_size})."
        )
    data = data.reshape(n_layers, hidden_size)
    if not np.isfinite(data).all():
        bad = tuple(int(value) for value in np.argwhere(~np.isfinite(data))[0])
        raise ValueError(f"non-finite activation in {path} at {bad}")
    return data, digest


def load_dump(path: str, n_layers: int, hidden_size: int) -> np.ndarray:
    """Load a flat little-endian f32 dump as ``[layers, hidden]``."""
    return load_dump_snapshot(path, n_layers, hidden_size)[0]


def compare(ember: np.ndarray, reference: np.ndarray) -> dict:
    """Return per-layer metrics as a dict."""
    if ember.shape != reference.shape or ember.ndim != 2 or not ember.size:
        raise ValueError(
            f"aligned non-empty rank-2 dumps required, got {ember.shape} and {reference.shape}"
        )
    for label, values in (("Ember", ember), ("reference", reference)):
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{label} dump must use a floating-point dtype")
        if not np.isfinite(values).all():
            bad = tuple(int(value) for value in np.argwhere(~np.isfinite(values))[0])
            raise ValueError(f"non-finite activation in {label} dump at {bad}")
    n_layers = ember.shape[0]
    layers = []
    for i in range(n_layers):
        e = ember[i]
        r = reference[i]
        e64 = e.astype(np.float64)
        r64 = r.astype(np.float64)
        norm_e = float(np.linalg.norm(e64))
        norm_r = float(np.linalg.norm(r64))
        if norm_e == 0.0 or norm_r == 0.0:
            raise ValueError(f"zero-norm hidden state at layer {i}")
        difference = e64 - r64
        cos = float(np.clip(np.dot(e64, r64) / (norm_e * norm_r), -1.0, 1.0))
        layers.append(
            {
                "layer": i,
                "cosine": cos,
                "l2_ember": norm_e,
                "l2_reference": norm_r,
                "mean_abs_diff": float(np.mean(np.abs(difference))),
                "max_abs_diff": float(np.max(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "exact_bits_equal": bool(np.array_equal(e.view(np.uint8), r.view(np.uint8))),
            }
        )
    return {
        "schema_version": 3,
        "dtype": "little-endian float32",
        "shape": list(ember.shape),
        "evaluation": "descriptive numerical comparison; no acceptance threshold",
        "layers": layers,
    }


def print_table(results: dict) -> None:
    """Print a formatted table to stdout."""
    header = (
        f"{'Layer':>5}  {'cosine':>9}  {'L2 ember':>10}  {'L2 ref':>10}  "
        f"{'mean |d|':>10}  {'max |d|':>10}  {'RMSE':>10}  {'exact':>5}"
    )
    print(header)
    print("-" * len(header))
    for l in results["layers"]:
        print(
            f"{l['layer']:5d}  {l['cosine']:9.6f}  {l['l2_ember']:10.2f}  "
            f"{l['l2_reference']:10.2f}  {l['mean_abs_diff']:10.6f}  "
            f"{l['max_abs_diff']:10.6f}  {l['rmse']:10.6f}  "
            f"{str(l['exact_bits_equal']):>5}"
        )


def write_markdown(results: dict, path: str) -> None:
    """Write results as a Markdown table."""
    lines = [
        "# Layer Comparison Report",
        "",
        f"- Ember SHA-256: `{results['ember_sha256']}`",
        f"- Reference SHA-256: `{results['reference_sha256']}`",
        f"- Shape: `{results['shape']}`; dtype: `{results['dtype']}`",
        "- Evaluation: descriptive numerical comparison; no acceptance threshold.",
        "",
        "| Layer | cosine | L2 ember | L2 reference | mean abs diff | max abs diff | RMSE | exact bits |",
        "|-------|--------|----------|-------------|---------------|--------------|------|------------|",
    ]
    for l in results["layers"]:
        lines.append(
            f"| {l['layer']} | {l['cosine']:.6f} | {l['l2_ember']:.2f} | "
            f"{l['l2_reference']:.2f} | {l['mean_abs_diff']:.6f} | "
            f"{l['max_abs_diff']:.6f} | {l['rmse']:.6f} | "
            f"{'yes' if l['exact_bits_equal'] else 'no'} |"
        )
    atomic_write(Path(path), "\n".join(lines) + "\n")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-layer hidden-state dumps")
    parser.add_argument("--ember", required=True, help="Ember layer dump (.bin)")
    parser.add_argument("--reference", required=True, help="Reference layer dump (.bin)")
    parser.add_argument("--layers", type=int, required=True, help="Number of layers")
    parser.add_argument("--hidden-size", type=int, required=True, help="Hidden size per layer")
    parser.add_argument("--out-md", default=None, help="Optional Markdown report path")
    parser.add_argument("--out-json", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if args.layers < 1 or args.hidden_size < 1:
        parser.error("--layers and --hidden-size must be positive")

    input_paths = {Path(args.ember).resolve(), Path(args.reference).resolve()}
    if len(input_paths) != 2:
        parser.error("--ember and --reference must be different files")
    output_values = [value for value in (args.out_md, args.out_json) if value]
    output_paths = [Path(value).resolve() for value in output_values]
    if len(output_paths) != len(set(output_paths)):
        parser.error("--out-md and --out-json must be different paths")
    if any(path in input_paths for path in output_paths):
        parser.error("report paths must not overwrite an input dump")

    ember, ember_sha256 = load_dump_snapshot(
        args.ember, args.layers, args.hidden_size
    )
    reference, reference_sha256 = load_dump_snapshot(
        args.reference, args.layers, args.hidden_size
    )
    results = compare(ember, reference)
    results["ember_path"] = args.ember
    results["reference_path"] = args.reference
    results["ember_sha256"] = ember_sha256
    results["reference_sha256"] = reference_sha256
    print_table(results)

    if args.out_md:
        write_markdown(results, args.out_md)
        print(f"\nMarkdown report written to {args.out_md}")

    if args.out_json:
        atomic_write(
            Path(args.out_json),
            json.dumps(results, indent=2, allow_nan=False) + "\n",
        )
        print(f"JSON report written to {args.out_json}")


if __name__ == "__main__":
    main()
