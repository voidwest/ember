#!/usr/bin/env python3
"""Tooling for moving large tracked research data out of git (non-destructive).

Nothing here deletes, untracks or rewrites anything. It supports the plan in
docs/research-data.md:

  manifest  Scan tracked files under data/ and research/ at or above a size
            threshold and write scripts/research_data_manifest.json: path,
            bytes and SHA-256 per file, with byte-identical copies grouped.
  report    Print what could move and what it would save (from the manifest).
  pack      Build the release asset: one deterministic tarball holding each
            distinct blob once, named blobs/<sha256> (dedupe by construction).
  fetch     Restore every manifest path from that asset (URL or local file),
            verifying each file against the committed manifest's SHA-256.
  verify    Check the files on disk against the manifest.

The committed manifest is the trust anchor: every restored file must match
its recorded SHA-256, whatever the asset's origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "research_data_manifest.json"
SCHEMA = "ember.research-data-manifest.v1"
ROOTS = ("data", "research")
DEFAULT_THRESHOLD = 256 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files(roots: tuple[str, ...]) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", *roots],
                         check=True, capture_output=True).stdout
    return sorted(name for name in out.decode().split("\0") if name)


# Code that would break if a file were no longer checked out.
CODE_PATHS = ("src", "tests", "scripts", ".github", "probes", "python", "bindings",
              "benches", "examples", "tools", "pyproject.toml")


def references(path: str) -> list[str]:
    """Tracked code files that name `path` literally (a heuristic: paths
    assembled at runtime are not found)."""
    result = subprocess.run(["git", "-C", str(ROOT), "grep", "-l", "-F", path, "--", *CODE_PATHS],
                            capture_output=True, text=True)
    return sorted(name for name in result.stdout.split()
                  if name != "scripts/research_data_manifest.json")


def build_manifest(threshold: int) -> dict:
    files = []
    for name in tracked_files(ROOTS):
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        if size >= threshold:
            files.append({"path": name, "bytes": size, "sha256": sha256_file(path),
                          "referenced_by": references(name)})
    groups = defaultdict(list)
    for entry in files:
        groups[entry["sha256"]].append(entry["path"])
    duplicates = [
        {"sha256": sha, "bytes": next(f["bytes"] for f in files if f["sha256"] == sha),
         "paths": paths}
        for sha, paths in sorted(groups.items()) if len(paths) > 1
    ]
    total = sum(f["bytes"] for f in files)
    unique = sum(next(f["bytes"] for f in files if f["sha256"] == sha) for sha in groups)
    return {
        "schema": SCHEMA,
        "roots": list(ROOTS),
        "threshold_bytes": threshold,
        "summary": {
            "files": len(files),
            "bytes": total,
            "distinct_blobs": len(groups),
            "distinct_bytes": unique,
            "duplicate_bytes": total - unique,
        },
        "asset": {
            "name": "ember-research-data-<version>.tar",
            "layout": "blobs/<sha256>, one entry per distinct blob",
            "url": None,
            "sha256": None,
            "note": "url/sha256 are filled in by the maintainer when the asset is published",
        },
        "files": files,
        "duplicates": duplicates,
    }


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise SystemExit(f"unsupported manifest schema: {manifest.get('schema')!r}")
    return manifest


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return str(n)


def cmd_manifest(args) -> None:
    manifest = build_manifest(args.threshold)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    s = manifest["summary"]
    print(f"wrote {args.manifest}: {s['files']} files, {human(s['bytes'])}, "
          f"{s['distinct_blobs']} distinct ({human(s['distinct_bytes'])})")


def cmd_report(args) -> None:
    manifest = load_manifest(args.manifest)
    s = manifest["summary"]
    all_tracked = 0
    for name in tracked_files(ROOTS):
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            all_tracked += path.stat().st_size
    print(f"tracked under {', '.join(ROOTS)}: {human(all_tracked)}")
    print(f"files >= {human(manifest['threshold_bytes'])}: {s['files']} files, {human(s['bytes'])}")
    print(f"  distinct blobs: {s['distinct_blobs']} ({human(s['distinct_bytes'])}); "
          f"byte-identical copies: {human(s['duplicate_bytes'])}")
    print(f"checkout shrinks by {human(s['bytes'])} if moved; the asset is "
          f"{human(s['distinct_bytes'])} (duplicates stored once)")
    referenced = [e for e in manifest["files"] if e.get("referenced_by")]
    print(f"named by code/tests (need the fetch step wired in before they move): "
          f"{len(referenced)} files, {human(sum(e['bytes'] for e in referenced))}")
    for entry in referenced:
        print(f"  {entry['path']} <- {', '.join(entry['referenced_by'])}")
    by_dir = defaultdict(int)
    for entry in manifest["files"]:
        by_dir["/".join(entry["path"].split("/")[:3])] += entry["bytes"]
    print("largest directories:")
    for directory, size in sorted(by_dir.items(), key=lambda item: -item[1])[:args.top]:
        print(f"  {human(size):>10}  {directory}")
    print(f"duplicate groups: {len(manifest['duplicates'])}")
    for group in sorted(manifest["duplicates"], key=lambda g: -g["bytes"] * (len(g["paths"]) - 1))[:args.top]:
        saved = group["bytes"] * (len(group["paths"]) - 1)
        print(f"  {len(group['paths'])} copies x {human(group['bytes'])} (saves {human(saved)}) "
              f"sha256 {group['sha256'][:12]}")
        for path in group["paths"]:
            print(f"      {path}")


def cmd_pack(args) -> None:
    manifest = load_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    blobs = {}
    for entry in manifest["files"]:
        path = args.root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"{entry['path']} no longer matches the manifest; regenerate it")
        blobs.setdefault(entry["sha256"], path)
    with tarfile.open(args.out, "w", format=tarfile.PAX_FORMAT) as tar:
        for sha, path in sorted(blobs.items()):
            info = tarfile.TarInfo(f"blobs/{sha}")
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    print(f"packed {len(blobs)} blobs into {args.out}")
    print(f"sha256 {sha256_file(args.out)}  {args.out.name}")
    print("not published; see docs/research-data.md for the manual publish step")


def open_asset(source: str, workdir: Path) -> Path:
    if "://" not in source:
        return Path(source)
    target = workdir / "asset.tar"
    with urllib.request.urlopen(source) as response, target.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    return target


def cmd_fetch(args) -> None:
    manifest = load_manifest(args.manifest)
    source = args.asset or (manifest["asset"] or {}).get("url")
    expected_asset = args.asset_sha256 or (manifest["asset"] or {}).get("sha256")
    if not source:
        raise SystemExit("no asset configured: pass --asset URL|FILE (the manifest has no url yet)")
    missing = [e for e in manifest["files"]
               if not (args.root / e["path"]).is_file() or sha256_file(args.root / e["path"]) != e["sha256"]]
    if not missing:
        print(f"all {len(manifest['files'])} files already present and verified")
        return
    with tempfile.TemporaryDirectory() as tmp:
        asset = open_asset(source, Path(tmp))
        if expected_asset and sha256_file(asset) != expected_asset.lower():
            raise SystemExit("asset SHA-256 does not match the pinned value")
        wanted = {e["sha256"] for e in missing}
        with tarfile.open(asset, "r") as tar:
            for member in tar.getmembers():
                sha = member.name.removeprefix("blobs/")
                if not member.isfile() or member.name != f"blobs/{sha}" or sha not in wanted:
                    continue
                blob = Path(tmp) / sha
                with tar.extractfile(member) as src, blob.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha256_file(blob) != sha:
                    raise SystemExit(f"blob {sha} in the asset is corrupt")
        restored = 0
        for entry in missing:
            blob = Path(tmp) / entry["sha256"]
            if not blob.is_file():
                raise SystemExit(f"asset lacks blob {entry['sha256']} for {entry['path']}")
            dest = args.root / entry["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blob, dest)
            restored += 1
    print(f"restored {restored} file(s); all verified against the manifest")


def cmd_verify(args) -> None:
    manifest = load_manifest(args.manifest)
    bad = []
    for entry in manifest["files"]:
        path = args.root / entry["path"]
        if not path.is_file():
            bad.append(f"missing {entry['path']}")
        elif sha256_file(path) != entry["sha256"]:
            bad.append(f"sha256 mismatch {entry['path']}")
    for line in bad:
        print(line, file=sys.stderr)
    if bad:
        raise SystemExit(1)
    print(f"all {len(manifest['files'])} manifest files present and verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="checkout that pack reads from and fetch/verify restore into or check (default: this repo)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    p.set_defaults(func=cmd_manifest)
    p = sub.add_parser("report")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("pack")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_pack)
    p = sub.add_parser("fetch")
    p.add_argument("--asset", help="asset URL or local path (default: the manifest's asset.url)")
    p.add_argument("--asset-sha256", help="pinned asset SHA-256 (default: the manifest's asset.sha256)")
    p.set_defaults(func=cmd_fetch)
    p = sub.add_parser("verify")
    p.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.manifest = args.manifest.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
