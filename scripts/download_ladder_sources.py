#!/usr/bin/env python3
"""Download the exact FP16 sources of the frozen v0.3 quantization ladder."""

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = (
    ("llama-3.2-1b", "second-state/Llama-3.2-1B-Instruct-GGUF", "ffae973dc8b47a497fae462fc1f882437e258bbf", "Llama-3.2-1B-Instruct-f16.gguf"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "91cad51170dc346986eccefdc2dd33a9da36ead9", "qwen2.5-1.5b-instruct-fp16.gguf"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    manifest = json.loads((Path(__file__).resolve().parents[1] / "models/v03-ladder/ladder-manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = []
    for family, repo, revision, filename in SOURCES:
        expected = next(row["source"] for row in manifest if row["family"] == family)
        path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=args.out))
        if path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"source size differs from frozen pin: {path}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise SystemExit(f"source SHA-256 differs from frozen pin: {path}")
        provenance.append({"family": family, "repo": repo, "revision": revision, "filename": filename, "path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size})
        print(f"verified {family}: {actual}", flush=True)
    (args.out / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
