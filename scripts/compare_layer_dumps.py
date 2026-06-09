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
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_dump(path: str, n_layers: int, hidden_size: int) -> np.ndarray:
    """Load a flat f32 binary file as [n_layers, hidden_size] float32 array."""
    expected_floats = n_layers * hidden_size
    data = np.fromfile(path, dtype=np.float32)
    if len(data) != expected_floats:
        raise ValueError(
            f"expected {expected_floats} floats in {path}, "
            f"got {len(data)}. check --layers ({n_layers}) and "
            f"--hidden-size ({hidden_size})."
        )
    return data.reshape(n_layers, hidden_size)


def compare(ember: np.ndarray, reference: np.ndarray) -> dict:
    """Return per-layer metrics as a dict."""
    n_layers = ember.shape[0]
    layers = []
    for i in range(n_layers):
        e = ember[i]
        r = reference[i]
        cos = float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30))
        layers.append(
            {
                "layer": i,
                "cosine": round(cos, 6),
                "l2_ember": round(float(np.linalg.norm(e)), 2),
                "l2_reference": round(float(np.linalg.norm(r)), 2),
                "mean_abs_diff": round(float(np.mean(np.abs(e - r))), 6),
                "max_abs_diff": round(float(np.max(np.abs(e - r))), 6),
            }
        )
    return {"layers": layers}


def print_table(results: dict) -> None:
    """Print a formatted table to stdout."""
    header = f"{'Layer':>5}  {'cosine':>9}  {'L2 ember':>10}  {'L2 ref':>10}  {'mean |d|':>10}  {'max |d|':>10}"
    print(header)
    print("-" * len(header))
    for l in results["layers"]:
        print(
            f"{l['layer']:5d}  {l['cosine']:9.6f}  {l['l2_ember']:10.2f}  "
            f"{l['l2_reference']:10.2f}  {l['mean_abs_diff']:10.6f}  "
            f"{l['max_abs_diff']:10.6f}"
        )


def write_markdown(results: dict, path: str) -> None:
    """Write results as a Markdown table."""
    lines = [
        "# Layer Comparison Report",
        "",
        "| Layer | cosine | L2 ember | L2 reference | mean abs diff | max abs diff |",
        "|-------|--------|----------|-------------|---------------|--------------|",
    ]
    for l in results["layers"]:
        lines.append(
            f"| {l['layer']} | {l['cosine']:.6f} | {l['l2_ember']:.2f} | "
            f"{l['l2_reference']:.2f} | {l['mean_abs_diff']:.6f} | "
            f"{l['max_abs_diff']:.6f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-layer hidden-state dumps")
    parser.add_argument("--ember", required=True, help="Ember layer dump (.bin)")
    parser.add_argument("--reference", required=True, help="Reference layer dump (.bin)")
    parser.add_argument("--layers", type=int, required=True, help="Number of layers")
    parser.add_argument("--hidden-size", type=int, required=True, help="Hidden size per layer")
    parser.add_argument("--out-md", default=None, help="Optional Markdown report path")
    parser.add_argument("--out-json", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    ember = load_dump(args.ember, args.layers, args.hidden_size)
    reference = load_dump(args.reference, args.layers, args.hidden_size)
    results = compare(ember, reference)
    print_table(results)

    if args.out_md:
        write_markdown(results, args.out_md)
        print(f"\nMarkdown report written to {args.out_md}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"JSON report written to {args.out_json}")


if __name__ == "__main__":
    main()
