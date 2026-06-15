import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_morph_dataset.exporters import make_probe_records, make_sft_examples
from arabic_morph_dataset.filters import apply_filters
from arabic_morph_dataset.io import read_raw_records
from arabic_morph_dataset.normalize import dediacritize, normalize_records
from arabic_morph_dataset.report import make_summary_report
from arabic_morph_dataset.split import leakage_report, split_records
from arabic_morph_dataset.stats import dataset_stats
from arabic_morph_dataset.validate import validate_canonical, validate_canonical_rows, validate_probe_records, validate_sft_examples
from arabic_morph_dataset.cli import entrypoint
from arabic_morph_dataset.models import MorphRecord


SAMPLE = ROOT / "data/arabic_morph_sample/camelmorph_sample.jsonl"
IMBALANCED_SAMPLE = ROOT / "data/arabic_morph_sample/camelmorph_imbalanced_sample.jsonl"


def sample_records():
    records, _ = normalize_records(read_raw_records(SAMPLE), "test_sample")
    filtered, _ = apply_filters(
        records,
        {
            "drop_missing_root": True,
            "drop_missing_pattern": True,
            "drop_missing_lemma": True,
            "drop_ambiguous": True,
        },
    )
    return filtered


def test_normalization_maps_camel_style_fields():
    records, report = normalize_records([{"word": "المكتبات", "diac": "ٱلْمَكْتَبَاتُ", "lex": "مَكْتَبَة_1", "root": "كتب", "pattern": "مَفْعَلَة", "pattern_concrete": "مكتبة", "pos": "noun", "gen": "f", "num": "p", "stt": "d"}], "unit")
    record = records[0]
    assert record.surface == "المكتبات"
    assert record.surface_dediac == "المكتبات"
    assert record.lemma == "مَكْتَبَة"
    assert record.pos == "NOUN"
    assert record.features["gender"] == "fem"
    assert record.features["number"] == "pl"
    assert record.features["state"] == "def"
    assert report["output_records"] == 1
    assert dediacritize("كَتَبَ") == "كتب"
    assert dediacritize("كـَتَبَ") == "كتب"


def test_normalization_preserves_not_applicable_features():
    records, _ = normalize_records([{"word": "باب", "lex": "بَاب_1", "root": "بوب", "pattern": "فَعَل", "pattern_concrete": "باب", "pos": "noun", "gen": "NA"}], "unit")
    assert records[0].features["gender"] == "na"


def test_normalization_rejects_malformed_features():
    try:
        normalize_records([{"word": "باب", "lex": "بَاب_1", "features": "gender"}], "unit")
    except ValueError as exc:
        assert "features must be an object" in str(exc)
    else:
        raise AssertionError("malformed features should fail")


def test_num_analyses_zero_is_not_ambiguous():
    records, _ = normalize_records([{"word": "x", "lex": "x", "num_analyses": 0}], "unit")
    assert records[0].is_ambiguous is False


def test_schema_validation_catches_duplicate_ids():
    records = sample_records()
    duplicated = [records[0], records[0]]
    report = validate_canonical(duplicated)
    assert not report["passed"]
    assert duplicated[0].id in report["duplicate_ids"]


def test_schema_validation_catches_missing_raw_fields():
    report = validate_canonical_rows([{"id": "bad", "surface": "كتب"}])
    assert not report["passed"]
    missing_fields = {item["field"] for item in report["missing_required"]}
    assert "metadata" in missing_fields
    assert "features" in missing_fields


def test_from_dict_parses_false_string_boolean():
    assert MorphRecord.from_dict({"id": "x", "surface": "x", "is_ambiguous": "false"}).is_ambiguous is False


def test_each_split_strategy_has_no_required_leakage():
    records = sample_records()
    for strategy in [
        "random",
        "lemma_random",
        "root_heldout",
        "abstract_pattern_heldout",
        "concrete_pattern_heldout",
        "root_pattern_heldout",
        "lemma_heldout",
    ]:
        split, report = split_records(records, strategy=strategy, seed=3, ratios={"train": 0.6, "dev": 0.2, "test": 0.2})
        assert len(split) == len(records)
        assert report["leakage"]["passed"], strategy
        if strategy == "random":
            assert report["leakage"]["checks"] == {}
        if strategy == "lemma_heldout":
            assert set(report["leakage"]["checks"]) == {"lemma"}


def test_filter_rejects_string_pos_allowlist():
    try:
        apply_filters(sample_records(), {"pos_allowlist": "NOUN"})
    except ValueError as exc:
        assert "pos_allowlist must be a list" in str(exc)
    else:
        raise AssertionError("string pos_allowlist should fail")


def test_leakage_detection_reports_root_overlap():
    records = sample_records()
    a = records[0].with_split("train")
    b = next(record for record in records if record.root == a.root and record.id != a.id).with_split("test")
    report = leakage_report([a, b], "root_heldout")
    assert not report["passed"]
    assert a.root in report["checks"]["root"]["train_test"]


def test_sft_formatting_has_json_assistant_content():
    split, _ = split_records(sample_records(), "lemma_heldout", seed=5)
    examples = make_sft_examples(split, ["analyze_form", "root_pattern", "feature_bundle"])
    assert examples
    first = examples[0]
    assert first["messages"][0]["role"] == "user"
    assert first["messages"][1]["role"] == "assistant"
    assert json.loads(first["messages"][1]["content"])
    assert validate_sft_examples(examples)["passed"]


def test_sft_validation_checks_task_schema():
    report = validate_sft_examples(
        [
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "{\"foo\":\"bar\"}"},
                ],
                "metadata": {"task": "analyze_form"},
            }
        ]
    )
    assert not report["passed"]
    assert "missing keys" in report["errors"][0]["error"]


def test_probe_formatting_and_validation():
    split, _ = split_records(sample_records(), "root_pattern_heldout", seed=5)
    probes = make_probe_records(split, "root_pattern_heldout")
    assert probes[0]["split_type"] == "root_pattern_heldout"
    assert "messages" not in probes[0]
    assert validate_probe_records(probes)["passed"]
    assert not validate_probe_records([{**probes[0], "root": None}])["passed"]


def test_empty_outputs_are_valid_but_marked_empty():
    assert validate_canonical([])["passed"]
    assert validate_canonical([])["empty"]
    assert validate_sft_examples([])["passed"]
    assert validate_sft_examples([])["empty"]
    assert validate_probe_records([])["passed"]
    assert validate_probe_records([])["empty"]


def test_stats_generation_on_sample_data():
    split, _ = split_records(sample_records(), "root_heldout", seed=7)
    stats = dataset_stats(split)
    assert stats["num_records"] == 18
    assert stats["unique_roots"] == 5
    assert stats["pos_distribution"]["VERB"] > 0
    assert sum(stats["split_counts"].values()) == 18


def test_summary_report_on_imbalanced_fixture():
    records, _ = normalize_records(read_raw_records(IMBALANCED_SAMPLE), "imbalanced_test")
    filtered, filter_report = apply_filters(
        records,
        {
            "drop_missing_root": True,
            "drop_missing_pattern": True,
            "drop_missing_lemma": True,
            "drop_ambiguous": True,
            "pos_allowlist": ["NOUN", "VERB", "ADJ"],
        },
    )
    report = make_summary_report(filtered, filter_report, seed=17, ratios={"train": 0.7, "dev": 0.15, "test": 0.15})
    assert report["records"]["input"] == 393
    assert report["records"]["kept"] == 343
    assert report["records"]["dropped"] == 50
    assert report["unique_roots"] == 30
    assert report["unique_abstract_patterns"] == 7
    assert report["split_leakage"]["root_heldout"]
    assert report["split_leakage"]["abstract_pattern_heldout"]
    assert report["split_leakage"]["concrete_pattern_heldout"]
    assert report["top_20_roots"][0] == {"root": "كتب", "count": 42}


def test_cli_rejects_empty_task_list(tmp_path):
    output = tmp_path / "sft.jsonl"
    assert entrypoint(["make-sft", "--input", str(ROOT / "data/arabic_morph_sample/out/canonical.jsonl"), "--output", str(output), "--tasks", ""]) == 1


def test_package_module_entrypoint_runs_stats():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arabic_morph_dataset",
            "stats",
            "--input",
            str(ROOT / "data/arabic_morph_sample/out/canonical.jsonl"),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["num_records"] == 18
