import json
from pathlib import Path

import numpy as np
import pytest

from probes.build_conllu_benchmark import build_rows, reconstruct_text
from scripts.benchmark_threads import parse_benchmark
from scripts.compare_layer_dumps import compare, load_dump_snapshot
from scripts.extraction_adapter_common import render_prompt
from scripts.run_smoke import parse_time_output
from stimuli.generate_stimuli import apply_pattern, render_prompts


def test_prompt_rendering_is_single_pass_and_rejects_bad_templates():
    assert render_prompt("{first}|{second}", {"first": "{second}", "second": "secret"}) == (
        "{second}|secret"
    )
    with pytest.raises(ValueError, match="unmatched opening brace"):
        render_prompt("prefix {field", {"field": "value"})


def test_benchmark_parser_distinguishes_tokens_from_decode_evaluations():
    parsed = parse_benchmark(
        "prefill: 10 tokens in 20.000ms -> 500.000 tok/s\n"
        "decode: 4 evals in 100.000ms -> 40.000 eval/s\n"
    )
    assert parsed["prefill"]["unit"] == "tokens"
    assert parsed["decode"]["unit"] == "decode_evaluations"
    with pytest.raises(ValueError, match="both prefill and decode"):
        parse_benchmark("prefill: 10 tokens in 20.000ms -> 500.000 tok/s\n")
    with pytest.raises(ValueError, match="inconsistent"):
        parse_benchmark(
            "prefill: 10 tokens in 20.000ms -> 500.000 tok/s\n"
            "decode: 4 evals in 100.000ms -> 400.000 eval/s\n"
        )


def test_smoke_parser_converts_gnu_elapsed_and_keeps_benchmark_units():
    stderr = (
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.50\n"
        "Maximum resident set size (kbytes): 12345\n"
        "prefill: 10 tokens in 20.000ms -> 500.000 tok/s\n"
        "decode: 4 evals in 100.000ms -> 40.000 eval/s\n"
    )
    rss, elapsed, seconds, prompt_tokens, decode_evaluations, prefill_rate, decode_rate = (
        parse_time_output(stderr)
    )
    assert (rss, elapsed, seconds) == (12345, "1:02:03.50", 3723.5)
    assert (prompt_tokens, decode_evaluations) == (10, 4)
    assert (prefill_rate, decode_rate) == (500.0, 40.0)


def test_layer_dump_loader_and_metrics_use_little_endian_f32(tmp_path: Path):
    path = tmp_path / "layers.bin"
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    path.write_bytes(values.tobytes())
    loaded, digest = load_dump_snapshot(str(path), 2, 2)
    result = compare(loaded, values.copy())
    assert len(digest) == 64
    assert result["dtype"] == "little-endian float32"
    assert all(layer["exact_bits_equal"] for layer in result["layers"])
    with pytest.raises(ValueError, match="non-finite"):
        compare(np.array([[np.nan]], dtype=np.float32), np.ones((1, 1), dtype=np.float32))


def test_conllu_reconstruction_honors_space_after_and_alignment_is_strict(
    tmp_path: Path,
):
    sentence = [
        ["1", "لا", "_", "PART", "_", "_", "0", "root", "_", "SpaceAfter=No"],
        ["2", "شيء", "_", "NOUN", "_", "_", "1", "dep", "_", "_"],
    ]
    assert reconstruct_text(sentence) == "لاشيء"

    path = tmp_path / "unaligned.conllu"
    path.write_text(
        "# sent_id = s1\n"
        "# text = مختلف\n"
        "1\tكلمة\t_\tNOUN\t_\t_\t0\troot\t_\t_\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="could not align token"):
        build_rows(str(path), min_label_count=1)
    audit = {}
    assert build_rows(
        str(path), min_label_count=1, allow_unaligned=True, audit=audit
    ) == []
    assert audit["unaligned_count"] == 1


def test_nonce_pattern_substitution_does_not_rewrite_inserted_radicals():
    assert apply_pattern("f-l-m", "fa3ala") == "falama"
    row = render_prompts(
        [
            {
                "root": "f-l-m",
                "pattern": "fa3ala",
                "expected_surface": "falama",
            }
        ]
    )[0]
    assert row["prompt_contracts"]["en_zero"]["target_labels_in_prompt"] is True
    assert (
        row["prompt_contracts"]["en_surface_probe"]["target_labels_in_prompt"]
        is False
    )
    assert json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False)) == row
