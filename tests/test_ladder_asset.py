"""Round-trip the golden-ladder release asset scripts on a synthetic ladder.

scripts/package_ladder_asset.sh packs rungs into per-rung tarballs pinned by
an index; scripts/fetch_ladder_asset.sh downloads (here via file://), checks
every SHA-256 and extracts. The real rungs are ~7.6 GB, so a six-rung ladder
of tiny files with its own manifest stands in for them.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_ladder_inputs import FAMILIES, QUANTIZER_COMMIT, RUNGS  # noqa: E402

pytestmark = pytest.mark.skipif(
    not shutil.which("bash") or not shutil.which("curl") or sys.version_info < (3, 11),
    reason="requires bash, curl and Python 3.11+",
)


def synthetic_ladder(root: Path) -> Path:
    ladder = root / "ladder"
    ladder.mkdir()
    records = []
    for family in FAMILIES:
        for rung in RUNGS:
            data = f"GGUF synthetic {family} {rung}\n".encode() * 50
            (ladder / f"{family}-{rung}.gguf").write_bytes(data)
            records.append({
                "family": family,
                "rung": rung,
                "quantizer_commit": QUANTIZER_COMMIT,
                "target": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
            })
    (ladder / "ladder-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return ladder


def run(*args, check=True):
    return subprocess.run(["bash", *map(str, args)], cwd=ROOT, capture_output=True,
                          text=True, check=check, timeout=120)


def test_package_then_fetch_round_trips_and_fails_closed(tmp_path):
    ladder = synthetic_ladder(tmp_path)
    manifest = ladder / "ladder-manifest.json"
    out = tmp_path / "dist"
    packed = run(ROOT / "scripts/package_ladder_asset.sh", "--version", "t1",
                 "--ladder", ladder, "--manifest", manifest, "--out", out)
    assert "not published" in packed.stdout
    index = out / "ember-ladder-t1.index.json"
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    assert index_sha in packed.stdout
    parts = json.loads(index.read_text())["parts"]
    assert len(parts) == len(FAMILIES) * len(RUNGS)
    sums = (out / "SHA256SUMS").read_text()
    assert f"{index_sha}  {index.name}" in sums

    # Deterministic: repacking yields identical bytes.
    again = tmp_path / "dist2"
    run(ROOT / "scripts/package_ladder_asset.sh", "--version", "t1",
        "--ladder", ladder, "--manifest", manifest, "--out", again)
    assert (again / "SHA256SUMS").read_text() == sums

    # Refuses to overwrite an existing asset.
    assert run(ROOT / "scripts/package_ladder_asset.sh", "--version", "t1", "--ladder", ladder,
               "--manifest", manifest, "--out", out, check=False).returncode != 0

    dest = tmp_path / "fetched"
    fetched = run(ROOT / "scripts/fetch_ladder_asset.sh", "--index-url", index.as_uri(),
                  "--index-sha256", index_sha, "--dest", dest, "--manifest", manifest)
    assert "ladder verified" in fetched.stdout
    for family in FAMILIES:
        for rung in RUNGS:
            name = f"{family}-{rung}.gguf"
            assert (dest / name).read_bytes() == (ladder / name).read_bytes()
    assert not (dest / ".fetch").exists()

    # A wrong index pin fails before any part is fetched.
    wrong = run(ROOT / "scripts/fetch_ladder_asset.sh", "--index-url", index.as_uri(),
                "--index-sha256", "0" * 64, "--dest", tmp_path / "wrong",
                "--manifest", manifest, check=False)
    assert wrong.returncode != 0 and "index SHA-256 mismatch" in wrong.stderr

    # A tampered part fails its pinned hash.
    part = out / parts[0]["name"]
    part.write_bytes(part.read_bytes()[:-1] + b"\x01")
    tampered = run(ROOT / "scripts/fetch_ladder_asset.sh", "--index-url", index.as_uri(),
                   "--index-sha256", index_sha, "--dest", tmp_path / "tampered",
                   "--manifest", manifest, check=False)
    assert tampered.returncode != 0 and "SHA-256 mismatch" in tampered.stderr


def test_package_fails_closed_on_an_incomplete_ladder(tmp_path):
    ladder = synthetic_ladder(tmp_path)
    (ladder / f"{FAMILIES[0]}-{RUNGS[0]}.gguf").unlink()
    result = run(ROOT / "scripts/package_ladder_asset.sh", "--version", "t1", "--ladder", ladder,
                 "--manifest", ladder / "ladder-manifest.json", "--out", tmp_path / "dist",
                 check=False)
    assert result.returncode != 0
    assert "missing pinned model" in result.stderr
    assert not (tmp_path / "dist" / "ember-ladder-t1.index.json").exists()
