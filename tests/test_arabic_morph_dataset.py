import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_morph_dataset.exporters import make_probe_records, make_sft_examples
from arabic_morph_dataset.filters import apply_filters
from arabic_morph_dataset.io import read_raw_records
from arabic_morph_dataset.normalize import dediacritize, normalize_records
from arabic_morph_dataset.split import leakage_report, split_records
from arabic_morph_dataset.stats import dataset_stats
from arabic_morph_dataset.validate import validate_canonical, validate_probe_records, validate_sft_examples


SAMPLE = ROOT / "data/arabic_morph_sample/camelmorph_sample.jsonl"


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


def test_schema_validation_catches_duplicate_ids():
    records = sample_records()
    duplicated = [records[0], records[0]]
    report = validate_canonical(duplicated)
    assert not report["passed"]
    assert duplicated[0].id in report["duplicate_ids"]


def test_each_split_strategy_has_no_required_leakage():
    records = sample_records()
    for strategy in [
        "random",
        "root_heldout",
        "abstract_pattern_heldout",
        "concrete_pattern_heldout",
        "root_pattern_heldout",
        "lemma_heldout",
    ]:
        split, report = split_records(records, strategy=strategy, seed=3, ratios={"train": 0.6, "dev": 0.2, "test": 0.2})
        assert len(split) == len(records)
        assert report["leakage"]["passed"], strategy


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


def test_probe_formatting_and_validation():
    split, _ = split_records(sample_records(), "root_pattern_heldout", seed=5)
    probes = make_probe_records(split, "root_pattern_heldout")
    assert probes[0]["split_type"] == "root_pattern_heldout"
    assert "messages" not in probes[0]
    assert validate_probe_records(probes)["passed"]


def test_stats_generation_on_sample_data():
    split, _ = split_records(sample_records(), "root_heldout", seed=7)
    stats = dataset_stats(split)
    assert stats["num_records"] == 18
    assert stats["unique_roots"] == 5
    assert stats["pos_distribution"]["VERB"] > 0
    assert sum(stats["split_counts"].values()) == 18
