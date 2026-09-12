"""Tests for the optional PyO3 binding (`bindings/python`).

The binding is not built by default cargo commands: these tests skip when the
extension is not installed. CI installs it (`pip install ./bindings/python`)
before running pytest; locally run `maturin develop -m bindings/python/Cargo.toml`.
"""

import json
import subprocess
from pathlib import Path

import pytest

ember = pytest.importorskip("ember")

ROOT = Path(__file__).resolve().parents[1]

MINIMAL_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "[UNK]",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": {"[UNK]": 0, "a": 1},
        "merges": [],
    },
}


def write_tokenizer(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(MINIMAL_TOKENIZER))
    return path


def test_version_is_exposed():
    assert isinstance(ember.__version__, str)
    assert ember.__version__


def test_inspect_tokenizer_report_and_sha256(tmp_path):
    path = write_tokenizer(tmp_path)
    report = ember.inspect(str(path), sha256=True)
    assert report["kind"] == "tokenizer"
    assert report["tokenizer"]["vocab_size"] == 2
    assert report["gguf"] is None and report["kv_snapshot"] is None
    assert len(report["sha256"]) == 64
    assert report["file"] == str(path)


def test_inspect_accepts_pathlike_and_defaults_without_sha256(tmp_path):
    path = write_tokenizer(tmp_path)
    report = ember.inspect(path)  # pathlib.Path exercises os.PathLike
    assert report["kind"] == "tokenizer"
    assert report["sha256"] is None


def test_inspect_unknown_file_is_a_report_not_an_error(tmp_path):
    path = tmp_path / "mystery.bin"
    path.write_bytes(b"not a model")
    report = ember.inspect(str(path), sha256=True)
    assert report["kind"] == "unknown"
    assert report["sha256"] is None
    assert report["notes"], "unknown kind must carry a remediation note"


def test_inspect_invalid_gguf_raises_runtime_error(tmp_path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"GGUF")  # magic claims GGUF; body missing
    with pytest.raises(RuntimeError, match="structural validation"):
        ember.inspect(str(path))


def test_diff_report_without_externals(tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"definitely not gguf")
    report = ember.diff(str(path), against=[])
    assert report["schema"] == "ember.diff.v1"
    assert report["ember"]["runtime"] == "ember"
    assert report["ember"]["outcome"] in {
        "ACCEPT",
        "STRUCTURED_REJECT",
        "PANIC",
        "PROCESS_CRASH",
        "TIMEOUT",
        "RESOURCE_LIMIT_OR_EXTERNAL_KILL",
        "NOT_COMPARABLE",
        "HARNESS_ERROR",
    }
    assert report["externals"] == []
    assert report["agreement"]["all_agree"] is True
    assert report["agreement"]["distinct_outcomes"] == [report["ember"]["outcome"]]


def test_diff_rejects_unknown_runtime(tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"junk")
    with pytest.raises(ValueError, match="unknown runtime"):
        ember.diff(str(path), against=["vllm"])


def test_diff_rejects_nonpositive_timeout(tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"junk")
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout_secs"):
            ember.diff(str(path), against=[], timeout_secs=bad)


def test_diff_duplicate_runtimes_are_ignored(tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"junk")
    report = ember.diff(str(path), against=["candle", "candle", "CANDLE"], timeout_secs=1)
    # A missing external binary is HARNESS_ERROR, never an exception; the
    # duplicate names must collapse into one side.
    assert len(report["externals"]) == 1
    assert report["externals"][0]["runtime"] == "candle"


def test_binding_matches_cli_json(tmp_path):
    binary = ROOT / "target" / "release" / "ember"
    if not binary.exists():
        pytest.skip("release binary not built; CLI parity needs target/release/ember")
    path = write_tokenizer(tmp_path)
    binding = ember.inspect(str(path), sha256=True)
    completed = subprocess.run(
        [str(binary), "inspect", str(path), "--json", "--sha256"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == binding


def test_plan_rejects_bad_execution_and_missing_model(tmp_path):
    missing = tmp_path / "missing.gguf"
    with pytest.raises(RuntimeError, match="unknown --execution value"):
        ember.plan(str(missing), execution="turbo")
    with pytest.raises(RuntimeError, match="structural validation"):
        ember.plan(str(missing))


def test_plan_matches_cli_json_when_a_model_exists(tmp_path):
    model = ROOT / "models" / "v03-ladder" / "llama-3.2-1b-q8_0.gguf"
    binary = ROOT / "target" / "release" / "ember"
    if not model.exists() or not binary.exists():
        pytest.skip("plan parity needs a local llama GGUF and target/release/ember")
    binding = ember.plan(str(model))
    assert binding["architecture"] == "llama"
    assert binding["execution"] == "planned"
    plan_file = tmp_path / "plan.json"
    subprocess.run(
        [str(binary), "inspect", str(model), "plan", "--output", str(plan_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_plan = json.loads(plan_file.read_text())
    # Build-identity fields legitimately differ between locally built binaries.
    for document in (binding["plan"], cli_plan):
        document["provenance"]["plan_build_time"] = "<time>"
        document["provenance"]["git_commit"] = "<commit>"
        document.pop("plan_hash", None)
    assert binding["plan"] == cli_plan


def test_diff_corpus_campaign(tmp_path):
    seed_a = tmp_path / "seed_a.bin"
    seed_b = tmp_path / "seed_b.bin"
    seed_a.write_bytes(
        b"GGUF\x03\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    seed_b.write_bytes(b"not a gguf but long enough to mutate")
    report = ember.diff_corpus(
        n=2,
        out_dir=str(tmp_path / "out"),
        seed=7,
        mode="raw",
        against=[],
        jobs=1,
        timeout_secs=5,
        seeds=[str(seed_a), str(seed_b)],
    )
    assert report["run_tag"] == "raw-2-7"
    assert report["summary"]["n_mutations"] == 2
    assert report["summary"]["seed_cases"] == 2
    assert report["stats"]["failed_cases"] == 0
    assert set(report["stats"]["per_target"]) == {"ember"}
    log_path = Path(report["log_path"])
    assert log_path.is_file()
    assert Path(report["summary_path"]).is_file()
    # One JSONL record per case (ember side only).
    assert len(log_path.read_text().splitlines()) == 2


def test_diff_corpus_argument_validation(tmp_path):
    out_dir = str(tmp_path / "out")
    with pytest.raises(ValueError, match="n must be positive"):
        ember.diff_corpus(n=0, out_dir=out_dir, against=[])
    with pytest.raises(ValueError, match="jobs must be positive"):
        ember.diff_corpus(n=1, out_dir=out_dir, jobs=0, against=[])
    with pytest.raises(ValueError, match="unknown mode"):
        ember.diff_corpus(n=1, out_dir=out_dir, mode="bytes", against=[])
    with pytest.raises(ValueError, match="unknown runtime"):
        ember.diff_corpus(n=1, out_dir=out_dir, against=["vllm"])


def test_diff_corpus_refuses_frozen_tree():
    frozen = ROOT / "research" / "embersec" / "comparative" / "scratch-out"
    with pytest.raises(RuntimeError, match="frozen"):
        ember.diff_corpus(n=1, out_dir=str(frozen), against=[], seeds=["README.md"])
