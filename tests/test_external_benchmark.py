"""Smoke tests for the model-free external benchmark harness."""

from __future__ import annotations

import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "external_benchmark.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_benchmark_captures_pairwise_outputs(tmp_path: Path) -> None:
    code = (
        "import sys; "
        "sys.stdout.buffer.write(b'out\\x00\\n'); "
        "sys.stderr.buffer.write(b'err\\n')"
    )
    spec = {
        "schema": "ember.external-benchmark.v1",
        "id": "pytest-smoke",
        "description": "no model is needed for this harness test",
        "inputs": {"model": {"sha256": "0" * 64}},
        "runtimes": [
            {"id": "ember", "command": [sys.executable, "-c", code], "inherit_env": False},
            {"id": "external", "command": [sys.executable, "-c", code], "inherit_env": False},
        ],
        "cases": [
            {"id": "case", "warmups": 1, "repetitions": 2, "timeout_s": 10}
        ],
    }
    spec_path = tmp_path / "spec.json"
    output = tmp_path / "run"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec_path), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest_digest = _sha256(output / "manifest.json")
    assert (output / "manifest.sha256").read_text(encoding="ascii").startswith(manifest_digest)
    assert manifest["schema"] == "ember.external-benchmark.v1"
    assert manifest["execution"]["command_shell"] is False

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    pair = summary["pairwise_comparisons"][0]
    assert pair["runtime_a"] == "ember"
    assert pair["runtime_b"] == "external"
    assert pair["stdout_hash_equal"] is True
    assert "correctness oracle" in pair["interpretation"]

    stdout_files = sorted(output.glob("trials/*/case/run-000/stdout.bin"))
    assert len(stdout_files) == 2
    assert all(path.read_bytes() == b"out\x00\n" for path in stdout_files)


def test_external_benchmark_rejects_unpaired_json_surrogate_preflight(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(
        b'{"schema":"ember.external-benchmark.v1","id":"surrogate",'
        b'"runtimes":[{"id":"r","command":["echo"],'
        b'"metadata":{"bad":"\\ud800"}}],"cases":[{"id":"c"}]}'
    )
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec_path), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not output.exists()


def test_external_benchmark_rejects_surrogate_cwd_preflight(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(
        b'{"schema":"ember.external-benchmark.v1","id":"cwd-surrogate",'
        b'"runtimes":[{"id":"r","command":["echo"],"cwd":"\\ud800"}],'
        b'"cases":[{"id":"c"}]}'
    )
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec_path), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not output.exists()


def test_external_benchmark_revalidates_executable_each_trial(tmp_path: Path) -> None:
    executable = tmp_path / "runtime.py"
    replacement = f"#!{sys.executable}\nprint('replacement')\n"
    executable.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import atexit\n"
        "path = Path(__file__)\n"
        "def replace():\n"
        f"    path.write_text({replacement!r}, encoding='utf-8')\n"
        "atexit.register(replace)\n"
        "print('first')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    spec = {
        "schema": "ember.external-benchmark.v1",
        "id": "executable-revalidation",
        "runtimes": [
            {
                "id": "runtime",
                "command": [str(executable)],
                "cwd": str(tmp_path),
                "inherit_env": False,
            }
        ],
        "cases": [{"id": "case", "warmups": 0, "repetitions": 2, "timeout_s": 10}],
    }
    spec_path = tmp_path / "spec.json"
    output = tmp_path / "run"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec_path), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    records = json.loads((output / "results.json").read_text(encoding="utf-8"))["records"]
    assert [record["status"] for record in records] == ["ok", "identity-mismatch"]
    assert records[0]["identity_revalidated"] is True
    assert records[1]["returncode"] is None
    assert "executable" in records[1]["identity_error"]
    assert records[1]["stdout"]["bytes"] == 0


def test_external_benchmark_revalidates_working_directory_each_trial(tmp_path: Path) -> None:
    if os.name != "posix":
        return
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    code = (
        "from pathlib import Path\n"
        "cwd = Path.cwd()\n"
        "old = cwd.with_name(cwd.name + '.old')\n"
        "if not old.exists():\n"
        "    cwd.rename(old)\n"
        "    cwd.mkdir()\n"
        "print('first')\n"
    )
    spec = {
        "schema": "ember.external-benchmark.v1",
        "id": "cwd-revalidation",
        "runtimes": [
            {
                "id": "runtime",
                "command": [sys.executable, "-c", code],
                "cwd": str(cwd),
                "inherit_env": False,
            }
        ],
        "cases": [{"id": "case", "warmups": 0, "repetitions": 2, "timeout_s": 10}],
    }
    spec_path = tmp_path / "spec.json"
    output = tmp_path / "run"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec_path), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    records = json.loads((output / "results.json").read_text(encoding="utf-8"))["records"]
    assert [record["status"] for record in records] == ["ok", "identity-mismatch"]
    assert "cwd identity" in records[1]["identity_error"]
    assert records[1]["stdout"]["bytes"] == 0
