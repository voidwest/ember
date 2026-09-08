"""Validate CI campaign orchestration with fake builds and fake fuzz binaries."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["gguf_loader", "gguf_to_llama", "npy_bytes", "kv_snapshot_manifest",
           "tokenizer_json", "wav_bytes", "image_bytes", "image_preprocess"]


@pytest.fixture
def fuzz_runner(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "fuzz").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copyfile(ROOT / "scripts/run_parser_fuzz.sh", tmp_path / "scripts/run_parser_fuzz.sh")
    for lock in (tmp_path / "Cargo.lock", tmp_path / "fuzz/Cargo.lock"):
        lock.write_text("locked fixture\n")
    log = tmp_path / "calls.jsonl"
    fake_cargo = tmp_path / "bin/cargo"
    fake_cargo.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
root = pathlib.Path.cwd()
with open(os.environ["FUZZ_TEST_LOG"], "a") as output:
    output.write(json.dumps({"cargo": sys.argv[1:]}) + "\\n")
step = sys.argv[2] if sys.argv[2] != "fuzz" else sys.argv[3]
if os.environ.get("FUZZ_TEST_FAIL_STEP") == step:
    sys.exit(19)
if os.environ.get("FUZZ_TEST_TAMPER_STEP") == step:
    (root / "fuzz/Cargo.lock").write_text("unexpected resolution")
if step == "build":
    for target in json.loads(os.environ["FUZZ_TEST_TARGETS"]):
        if target == os.environ.get("FUZZ_TEST_MISSING_TARGET"):
            continue
        path = root / "fuzz/target" / os.environ["FUZZ_TARGET_TRIPLE"] / "release" / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!" + sys.executable + "\\n" + os.environ["FUZZ_TEST_BINARY"])
        path.chmod(0o755)
''')
    fake_cargo.chmod(0o755)
    binary = '''
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[0]).name
with open(os.environ["FUZZ_TEST_LOG"], "a") as output:
    output.write(json.dumps({"target": target, "args": sys.argv[1:], "asan": os.environ["ASAN_OPTIONS"]}) + "\\n")
if target == os.environ.get("FUZZ_TEST_FAIL_TARGET"):
    prefix = next(arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("-artifact_prefix="))
    pathlib.Path(prefix, "crash-fixture").write_bytes(b"hostile input")
    sys.exit(42)
'''
    env = dict(os.environ, PATH=str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
               FUZZ_TOOLCHAIN="nightly-2026-05-26", FUZZ_TARGET_TRIPLE="fixture-target",
               FUZZ_TEST_LOG=str(log), FUZZ_TEST_TARGETS=json.dumps(TARGETS),
               FUZZ_TEST_BINARY=binary, ASAN_OPTIONS="allocator_may_return_null=1")

    def run(**overrides):
        result = subprocess.run(["bash", "scripts/run_parser_fuzz.sh"], cwd=tmp_path,
                                env=dict(env, **overrides), text=True, capture_output=True, timeout=20)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls

    return tmp_path, run


def test_builds_once_then_runs_all_eight_campaigns_with_original_limits(fuzz_runner):
    root, run = fuzz_runner
    result, calls = run()
    assert result.returncode == 0, result.stderr
    cargo = [row["cargo"] for row in calls if "cargo" in row]
    assert cargo == [
        ["+nightly-2026-05-26", "fetch", "--locked", "--manifest-path", "fuzz/Cargo.toml"],
        ["+nightly-2026-05-26", "fuzz", "check", "--target", "fixture-target",
         "--target-dir", "fuzz/target", "--sanitizer", "address"],
        ["+nightly-2026-05-26", "fuzz", "build", "--target", "fixture-target",
         "--target-dir", "fuzz/target", "--sanitizer", "address"],
    ]
    campaigns = [row for row in calls if "target" in row]
    assert [row["target"] for row in campaigns] == TARGETS
    for campaign in campaigns:
        target = campaign["target"]
        assert campaign["args"] == [f"-artifact_prefix={root}/fuzz/artifacts/{target}/",
                                     "-max_total_time=30", "-max_len=262144", "-timeout=10",
                                     "-rss_limit_mb=2048", f"fuzz/corpus_work/{target}"]
        assert campaign["asan"] == "allocator_may_return_null=1:detect_odr_violation=0"


@pytest.mark.parametrize("step", ("fetch", "check", "build"))
def test_failed_build_or_changed_lock_prevents_campaigns(fuzz_runner, step):
    _, run = fuzz_runner
    result, calls = run(FUZZ_TEST_TAMPER_STEP=step)
    assert result.returncode != 0
    assert "changed a locked dependency graph" in result.stderr
    assert not any("target" in row for row in calls)


def test_crash_remains_failure_and_preserves_artifact(fuzz_runner):
    root, run = fuzz_runner
    result, calls = run(FUZZ_TEST_FAIL_TARGET="npy_bytes")
    assert result.returncode == 42
    assert [row["target"] for row in calls if "target" in row] == TARGETS[:3]
    assert (root / "fuzz/artifacts/npy_bytes/crash-fixture").read_bytes() == b"hostile input"


def test_missing_built_target_cannot_silently_shorten_campaigns(fuzz_runner):
    _, run = fuzz_runner
    result, _ = run(FUZZ_TEST_MISSING_TARGET="gguf_loader")
    assert result.returncode != 0
    assert "missing built fuzz target" in result.stderr


def test_workflow_retains_pins_readonly_permissions_and_failure_upload():
    workflow = yaml.load((ROOT / ".github/workflows/fuzz.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"]["FUZZ_TOOLCHAIN"] == "nightly-2026-05-26"
    steps = workflow["jobs"]["fuzz"]["steps"]
    upload = next(step for step in steps if step.get("name") == "upload crash artifacts")
    assert upload["if"] == "failure()"
    assert upload["with"]["path"] == "fuzz/artifacts/"
    assert all(len(step["uses"].rsplit("@", 1)[1]) == 40 for step in steps if "uses" in step)
    assert any("cargo-fuzz@0.13.2 --locked" in step.get("run", "") for step in steps)
