#!/usr/bin/env python3
"""Record or verify the built llama.cpp reference, including binary identities."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def snapshot(root: Path, commit: str, build_dir: str = "build", quantizer_only: bool = False) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("rev-parse", "HEAD") != commit:
        raise ValueError("reference checkout differs from requested commit")
    if git("diff", "HEAD", "--"):
        raise ValueError("reference checkout has tracked changes")
    build = root / build_dir
    binaries = ["llama-quantize"] if quantizer_only else ["llama", "llama-bench", "llama-quantize"]
    paths = [build / "CMakeCache.txt", *(build / "bin" / name for name in binaries)]
    libraries = sorted(set((build / "bin").glob("*.dylib")) | set((build / "bin").glob("*.so*")))
    if not libraries:
        raise ValueError("reference shared libraries are missing")
    files = {}
    for path in paths + libraries:
        with path.open("rb") as handle:
            files[str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"commit": commit, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("commit")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--quantizer-only", action="store_true")
    args = parser.parse_args()
    try:
        actual = snapshot(args.root, args.commit, args.build_dir, args.quantizer_only)
        stamp = args.root / args.build_dir / "ember-quantizer-build.json" if args.quantizer_only else args.root / "ember-reference-build.json"
        if args.mode == "record":
            temporary = stamp.with_suffix(".tmp")
            temporary.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
            temporary.replace(stamp)
        elif json.loads(stamp.read_text(encoding="utf-8")) != actual:
            raise ValueError("reference build changed since setup; rebuild with setup_llama_cpp.sh")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"reference build provenance failed: {error}\n")


if __name__ == "__main__":
    main()
