"""A commit text file alone must not authenticate changed reference binaries."""
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reference_build.py"


@pytest.fixture
def reference(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "source.c").write_text("original\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.c"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    (tmp_path / "build/bin").mkdir(parents=True)
    for name in ("llama", "llama-bench", "llama-quantize", "libllama.dylib"):
        (tmp_path / "build/bin" / name).write_bytes(b"original fixture bytes")
    (tmp_path / "build/CMakeCache.txt").write_text("fixture settings\n")
    result = run("record", tmp_path, commit)
    assert result.returncode == 0, result.stderr
    return tmp_path, commit


def run(mode, root, commit):
    return subprocess.run([sys.executable, str(SCRIPT), mode, str(root), commit], capture_output=True, text=True)


def test_unchanged_build_passes(reference):
    assert run("verify", *reference).returncode == 0


@pytest.mark.parametrize("path", ["build/bin/llama", "build/bin/libllama.dylib", "build/CMakeCache.txt", "source.c"])
def test_mutation_fails(reference, path):
    root, commit = reference
    (root / path).write_text("changed\n")
    assert run("verify", root, commit).returncode != 0


def test_wrong_commit_fails(reference):
    root, _ = reference
    assert run("verify", root, "0" * 40).returncode != 0


def test_missing_library_fails(reference):
    root, commit = reference
    (root / "build/bin/libllama.dylib").unlink()
    assert run("verify", root, commit).returncode != 0


def test_added_library_fails(reference):
    root, commit = reference
    (root / "build/bin/libinjected.dylib").write_bytes(b"unexpected")
    assert run("verify", root, commit).returncode != 0


def test_quantize_records_commands_and_preserves_existing_rungs(reference, tmp_path):
    import json
    import os

    root, commit = reference
    binary = root / "build/bin/llama-quantize"
    binary.write_text('#!/bin/sh\ncp "$1" "$2"\n')
    binary.chmod(0o755)
    subprocess.run([sys.executable, str(SCRIPT), "record", str(root), commit, "--quantizer-only"], check=True)
    (root / "COMMIT").write_text(commit + "\n")
    source = root / "source with spaces.gguf"
    source.write_bytes(b"fake source for orchestration test")
    out = root / "ladder"
    command = ["bash", str(SCRIPT.with_name("quantize_ladder.sh")), "--model-llama", str(source), "--out", str(out), "--jobs", "2"]
    env = dict(os.environ, PYTHON=sys.executable, LLAMA_CPP_DIR=str(root), QUANTIZER_BUILD_DIR="build")
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    manifest_path = out / "ladder-manifest.json"
    before = manifest_path.read_bytes()
    records = json.loads(before)
    assert len(records) == 3
    assert all(r["argv"][-1] == "2" and r["argv"][1] == str(source) for r in records)
    assert all(r["target"]["bytes"] == source.stat().st_size for r in records)
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert "existing rung" in second.stderr
    assert manifest_path.read_bytes() == before
    assert all(Path(r["target"]["path"]).read_bytes() == source.read_bytes() for r in records)
