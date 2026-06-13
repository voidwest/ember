#!/usr/bin/env python3
"""Run pairwise CCA/RSA geometry comparisons across saved activation tensors."""

import argparse
import json
from pathlib import Path

import numpy as np

from cca_analysis import cca_cross_model
from rsa_analysis import rsa_cross_model


def load_activations(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        if "activations" in data:
            return data["activations"]
        return data[list(data.keys())[0]]
    return np.load(path)


def normalized_layer_alignment(matrix: np.ndarray) -> list[dict]:
    rows = []
    if matrix.size == 0:
        return rows
    denom_a = max(matrix.shape[0] - 1, 1)
    denom_b = max(matrix.shape[1] - 1, 1)
    for layer_a in range(matrix.shape[0]):
        layer_b = int(np.nanargmax(matrix[layer_a]))
        rows.append(
            {
                "layer_a": layer_a,
                "layer_b": layer_b,
                "layer_a_norm": layer_a / denom_a,
                "layer_b_norm": layer_b / denom_b,
                "score": float(matrix[layer_a, layer_b]),
            }
        )
    return rows


def parse_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL:ACTIVATIONS")
    label, path = value.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must be LABEL:ACTIVATIONS")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description="pairwise cross-model CCA/RSA")
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        metavar="LABEL:ACTIVATIONS",
        help="model label and activation .npy/.npz path; may be repeated",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="correlation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activations = {
        label: load_activations(path)
        for label, path in args.model
    }
    manifest = {
        "models": [
            {
                "label": label,
                "path": path,
                "shape": list(activations[label].shape),
            }
            for label, path in args.model
        ],
        "pairs": [],
    }

    labels = [label for label, _ in args.model]
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            acts_a = activations[label_a]
            acts_b = activations[label_b]
            cca = cca_cross_model(acts_a, acts_b)
            rsa = rsa_cross_model(acts_a, acts_b, args.metric)
            pair_name = f"{label_a}__{label_b}"
            npz_path = out_dir / f"{pair_name}_geometry.npz"
            np.savez(
                npz_path,
                cca_cross_model=cca,
                rsa_cross_model=rsa,
            )
            manifest["pairs"].append(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "path": str(npz_path),
                    "cca_shape": list(cca.shape),
                    "rsa_shape": list(rsa.shape),
                    "cca_alignment": normalized_layer_alignment(cca),
                    "rsa_alignment": normalized_layer_alignment(rsa),
                }
            )

    manifest_path = out_dir / "cross_model_geometry_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
