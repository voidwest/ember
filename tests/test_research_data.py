"""Research-data tooling: the committed manifest matches the checkout, and
pack -> fetch restores byte-identical, deduplicated files (synthetic root)."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts/research_data.py"
FETCH = ROOT / "scripts/fetch_research_data.sh"


def run(*args, check=True):
    return subprocess.run([*map(str, args)], cwd=ROOT, capture_output=True, text=True,
                          check=check, timeout=120)


def test_committed_manifest_matches_the_tracked_files():
    manifest = json.loads((ROOT / "scripts/research_data_manifest.json").read_text())
    assert manifest["schema"] == "ember.research-data-manifest.v1"
    assert manifest["files"], "manifest lists no files"
    for entry in manifest["files"]:
        assert entry["path"].split("/")[0] in manifest["roots"]
    run(sys.executable, TOOL, "verify")


@pytest.mark.skipif(not shutil.which("bash"), reason="requires bash")
def test_pack_then_fetch_restores_deduplicated_files(tmp_path):
    source = tmp_path / "source"
    payloads = {"data/a/x.jsonl": b"alpha\n" * 100, "data/b/x.jsonl": b"alpha\n" * 100,
                "research/r.jsonl": b"beta\n" * 100}
    files = []
    for rel, data in payloads.items():
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_bytes(data)
        files.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "ember.research-data-manifest.v1", "roots": ["data", "research"],
        "threshold_bytes": 0, "summary": {}, "asset": {"url": None, "sha256": None},
        "files": files, "duplicates": [],
    }))
    asset = tmp_path / "asset.tar"
    packed = run(sys.executable, TOOL, "--manifest", manifest, "--root", source, "pack", "--out", asset)
    assert "packed 2 blobs" in packed.stdout  # the duplicate is stored once
    asset_sha = hashlib.sha256(asset.read_bytes()).hexdigest()

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    missing = run("bash", FETCH, "--manifest", manifest, "--root", checkout, "--verify", check=False)
    assert missing.returncode != 0
    run("bash", FETCH, "--manifest", manifest, "--root", checkout,
        "--asset", asset, "--asset-sha256", asset_sha)
    for rel, data in payloads.items():
        assert (checkout / rel).read_bytes() == data
    run("bash", FETCH, "--manifest", manifest, "--root", checkout, "--verify")

    # A wrong asset pin is refused before anything is restored.
    wrong = run("bash", FETCH, "--manifest", manifest, "--root", tmp_path / "other",
                "--asset", asset, "--asset-sha256", "0" * 64, check=False)
    assert wrong.returncode != 0 and "does not match" in wrong.stderr
    assert not (tmp_path / "other").exists()
