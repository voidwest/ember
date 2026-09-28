"""Exercise download publication/resume without network or real model weights."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_models.sh"
PAYLOAD = b"GGUF fixture complete bytes"


@pytest.fixture
def download(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    curl = tools / "curl"
    curl.write_text(f"#!{sys.executable}\n" + '''import os, pathlib, sys
args = sys.argv[1:]
assert args[args.index('-C') + 1] == '-'
out = pathlib.Path(args[args.index('-o') + 1])
assert out.name.endswith('.part')
pathlib.Path(os.environ['CALLS']).write_text('called')
data = os.environ['PAYLOAD'].encode()
if os.environ.get('FAIL') == '1':
    out.write_bytes(data[:5])
    raise SystemExit(22)
existing = out.read_bytes() if out.exists() else b''
assert data.startswith(existing)
with out.open('ab') as handle:
    handle.write(data[len(existing):])
if os.environ.get('RACE') == '1':
    pathlib.Path(str(out)[:-5]).write_bytes(b'other writer')
''')
    curl.chmod(0o755)
    root = tmp_path / "models with spaces"
    root.mkdir()
    calls = tmp_path / "calls"

    def run(**overrides):
        env = dict(os.environ, MODEL_DIR=str(root), PATH=str(tools) + os.pathsep + os.environ["PATH"], CALLS=str(calls), PAYLOAD=PAYLOAD.decode())
        env.update(overrides)
        return subprocess.run(["/bin/bash", "-c", 'source "$1"; download "https://example.invalid/model" "fixture.gguf" "$2" "$3"', "test", str(SCRIPT), hashlib.sha256(PAYLOAD).hexdigest(), str(len(PAYLOAD))], env=env, capture_output=True, text=True)

    return root, calls, run


def test_verified_download_is_published(download):
    root, _, run = download
    result = run()
    assert result.returncode == 0, result.stderr
    assert (root / "fixture.gguf").read_bytes() == PAYLOAD
    assert not (root / "fixture.gguf.part").exists()
    assert not (root / "fixture.gguf.download.lock").exists()


def test_interruption_resumes_partial_not_final_file(download):
    root, _, run = download
    assert run(FAIL="1").returncode != 0
    assert not (root / "fixture.gguf").exists()
    assert (root / "fixture.gguf.part").read_bytes() == PAYLOAD[:5]
    assert run().returncode == 0
    assert (root / "fixture.gguf").read_bytes() == PAYLOAD


def test_bad_hash_never_publishes(download):
    root, _, run = download
    result = run(PAYLOAD="x" * len(PAYLOAD))
    assert result.returncode != 0
    assert not (root / "fixture.gguf").exists()
    assert (root / "fixture.gguf.part").exists()


@pytest.mark.parametrize("correct", [True, False])
def test_existing_file_is_verified_and_preserved(download, correct):
    root, calls, run = download
    data = PAYLOAD if correct else b"existing user data"
    (root / "fixture.gguf").write_bytes(data)
    assert (run().returncode == 0) == correct
    assert not calls.exists()
    assert (root / "fixture.gguf").read_bytes() == data


def test_completed_partial_publishes_without_range_request(download):
    root, calls, run = download
    (root / "fixture.gguf.part").write_bytes(PAYLOAD)
    assert run().returncode == 0
    assert not calls.exists()
    assert (root / "fixture.gguf").read_bytes() == PAYLOAD


def test_publication_does_not_replace_racing_writer(download):
    root, _, run = download
    assert run(RACE="1").returncode != 0
    assert (root / "fixture.gguf").read_bytes() == b"other writer"
    assert (root / "fixture.gguf.part").read_bytes() == PAYLOAD


def test_partial_symlink_is_not_written(download, tmp_path):
    root, calls, run = download
    other = tmp_path / "other"
    other.write_bytes(b"keep")
    (root / "fixture.gguf.part").symlink_to(other)
    assert run().returncode != 0
    assert not calls.exists()
    assert other.read_bytes() == b"keep"


def test_existing_lock_prevents_concurrent_writes(download):
    root, calls, run = download
    (root / "fixture.gguf.download.lock").mkdir()
    assert run().returncode != 0
    assert not calls.exists()
