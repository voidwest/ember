"""Hermetic checks for extraction reuse, RSA storage, and shared artifact I/O."""

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from probes import artifact_io
from probes.extract_hf_encoder import extract_rows, pool
from probes.rsa_analysis import rsa_cross_model, rsa_layer_matrix, rsa_matrix


class ArrayTensor:
    """Only the CPU tensor interface used by extraction; no model dependencies."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, index):
        return ArrayTensor(self.values[index])

    def tolist(self):
        return self.values.tolist()

    def to(self, device):
        assert device == "cpu"
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        assert kwargs == {
            "return_tensors": "pt",
            "return_offsets_mapping": True,
            "return_special_tokens_mask": True,
            "truncation": True,
        }
        words = list(re.finditer(r"\S+", text))
        ids = [101, *(sum(map(ord, word.group())) for word in words), 102, 0]
        return {
            "input_ids": ArrayTensor([ids]),
            "offset_mapping": ArrayTensor([[[0, 0], *(list(w.span()) for w in words), [0, 0], [0, 0]]]),
            "special_tokens_mask": ArrayTensor([[1, *([0] * len(words)), 1, 1]]),
            "attention_mask": ArrayTensor([[*([1] * (len(words) + 2)), 0]]),
        }


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, input_ids, attention_mask, output_hidden_states):
        assert output_hidden_states is True
        self.calls += 1
        rng = np.random.default_rng(int(input_ids.values.sum()))
        states = tuple(
            ArrayTensor(rng.normal(size=(*input_ids.values.shape, 7)).astype(np.float32))
            for _ in range(3)
        )
        return SimpleNamespace(hidden_states=states)


def legacy_pool(stacked, targets, content, mode):
    if mode == "cls":
        return stacked[:, 0, :]
    if mode == "last":
        return stacked[:, content[-1], :]
    if mode == "mean":
        return stacked[:, content, :].mean(axis=1)
    if mode == "target_mean":
        return stacked[:, targets, :].mean(axis=1)
    index = targets[0] if mode == "target_first" else targets[-1]
    return stacked[:, index, :]


MODES = ("cls", "last", "mean", "target_mean", "target_first", "target_last")


@pytest.mark.parametrize("mode", MODES)
def test_grouped_extraction_matches_rowwise_outputs_and_metadata(mode):
    rows = [
        {"text": "كتب الولد", "target_span": [0, 3]},
        {"text": "قرأت البنت", "target_span": [5, 10]},
        {"text": "كتب الولد", "target_span": [4, 9]},
        {"text": "كتب الولد", "target_span": [0, 9]},
    ]
    row_ids = ["a", "b", "c", "d"]
    tokenizer, model = FakeTokenizer(), FakeModel()
    actual, metadata = extract_rows(rows, row_ids, tokenizer, model, mode, "cpu")
    assert tokenizer.calls == ["كتب الولد", "قرأت البنت"]
    assert model.calls == 2

    expected = []
    expected_metadata = []
    for index, row in enumerate(rows):
        encoded = FakeTokenizer()(row["text"], return_tensors="pt", return_offsets_mapping=True,
                                  return_special_tokens_mask=True, truncation=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        special = encoded.pop("special_tokens_mask")[0].tolist()
        active = encoded["attention_mask"][0].tolist()
        content = [i for i, (s, a) in enumerate(zip(special, active)) if not s and a]
        start, end = row["target_span"]
        targets = [i for i, (a, b) in enumerate(offsets) if a < b and a < end and b > start]
        outputs = FakeModel()(**encoded, output_hidden_states=True)
        stacked = np.stack([h.values[0] for h in outputs.hidden_states])
        expected.append(legacy_pool(stacked, targets, content, mode))
        expected_metadata.append({
            "index": index, "row_id": row_ids[index], "target_span": row["target_span"],
            "token_indices": targets, "token_count": len(offsets),
            "content_token_count": len(content),
        })
    np.testing.assert_array_equal(actual, np.stack(expected).astype(np.float32))
    assert metadata == expected_metadata
    assert actual.flags.owndata


@pytest.mark.parametrize("mode", MODES)
def test_pooled_output_owns_only_selected_values(mode):
    rng = np.random.default_rng(710)
    states = [ArrayTensor(rng.normal(size=(1, 128, 17)).astype(np.float32)) for _ in range(3)]
    stacked = np.stack([h.values[0] for h in states])
    result = pool(states, [4, 5], list(range(1, 127)), mode)
    np.testing.assert_array_equal(result, legacy_pool(stacked, [4, 5], list(range(1, 127)), mode))
    assert result.flags.owndata
    assert result.base is None
    assert result.nbytes == 3 * 17 * np.dtype(np.float32).itemsize


@pytest.mark.parametrize("span, message", [([0, 99], "invalid target_span"), (None, "requires target_span")])
def test_repeated_text_still_validates_every_target_span(span, message):
    rows = [{"text": "a b", "target_span": [0, 1]}, {"text": "a b", "target_span": span}]
    model = FakeModel()
    with pytest.raises(ValueError, match=f"row 1.*{message}"):
        extract_rows(rows, ["a", "b"], FakeTokenizer(), model, "target_mean", "cpu")
    assert model.calls == 0


@pytest.mark.parametrize("problem, message", [
    ("truncation", "truncated"),
    ("offset", "invalid offset"),
    ("special", "special/padded"),
    ("attention", "attention mask"),
])
def test_grouped_extraction_preserves_tokenizer_validation(problem, message):
    def tokenizer(text, **kwargs):
        encoded = FakeTokenizer()(text, **kwargs)
        if problem == "truncation":
            encoded["offset_mapping"].values[0, 2] = [0, 0]
        elif problem == "offset":
            encoded["offset_mapping"].values[0, 1, 0] = -1
        elif problem == "special":
            encoded["special_tokens_mask"].values[0, 1] = 1
        else:
            encoded["attention_mask"] = ArrayTensor([[1]])
        return encoded

    with pytest.raises(ValueError, match=message):
        extract_rows([{"text": "a b", "target_span": [0, 1]}], ["a"], tokenizer,
                     FakeModel(), "target_mean", "cpu")


def legacy_rsa_vectors(activations, metric):
    upper = np.triu_indices(len(activations), k=1)
    return np.stack([
        (1 - squareform(pdist(activations[:, layer], metric=metric)))[upper]
        for layer in range(activations.shape[1])
    ])


@pytest.mark.parametrize("metric", ("correlation", "cosine", "euclidean"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_condensed_rsa_matches_full_square_reference(metric, dtype):
    rng = np.random.default_rng(988)
    a = rng.normal(size=(31, 4, 11)).astype(dtype)
    b = rng.normal(size=(31, 3, 7)).astype(dtype)
    vectors_a, vectors_b = legacy_rsa_vectors(a, metric), legacy_rsa_vectors(b, metric)
    expected_cross = np.array([
        [np.corrcoef(x, y)[0, 1] for y in vectors_b] for x in vectors_a
    ])
    # Layer analyses must never reconstruct a square RDM or its triangle indices.
    with patch("probes.rsa_analysis.squareform", side_effect=AssertionError("square RDM")), \
         patch("probes.rsa_analysis.np.triu_indices", side_effect=AssertionError("triangle indices")):
        np.testing.assert_allclose(rsa_layer_matrix(a, metric), np.corrcoef(vectors_a), atol=2e-14, rtol=0)
        np.testing.assert_allclose(rsa_cross_model(a, b, metric), expected_cross, atol=2e-14, rtol=0)
        assert rsa_layer_matrix(a[:, :1], metric).shape == (1, 1)
    np.testing.assert_array_equal(
        rsa_matrix(a[:, 0], metric), 1 - squareform(pdist(a[:, 0], metric=metric))
    )


def test_rsa_rejects_invalid_or_undefined_correlations():
    for tensor, metric in [
        (np.ones((3, 2, 4)), "correlation"),
        (np.ones((3, 2, 4)), "euclidean"),
        (np.full((3, 2, 4), np.nan), "cosine"),
        (np.ones((2, 2, 4)), "euclidean"),
        (np.ones((3, 0, 4)), "euclidean"),
    ]:
        with pytest.raises(ValueError):
            rsa_layer_matrix(tensor, metric)
    with pytest.raises(ValueError, match="equal aligned sample counts"):
        rsa_cross_model(np.ones((3, 1, 2)), np.ones((4, 1, 2)))


def test_shared_artifact_writers_preserve_formats_and_replace_atomically(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    artifact_io.atomic_save_npy(tmp_path / "array.npy", values)
    np.testing.assert_array_equal(np.load(tmp_path / "array.npy", allow_pickle=False), values)
    artifact_io.atomic_savez(tmp_path / "array.npz", values=values, label=np.array("كتب"))
    with np.load(tmp_path / "array.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["values"], values)
        assert archive["label"].item() == "كتب"
    text_path = tmp_path / "nested" / "text.json"
    artifact_io.atomic_write_text(text_path, "old")
    artifact_io.atomic_write_text(text_path, "كتب\n" * 300000)
    expected = ("كتب\n" * 300000).encode("utf-8")
    assert text_path.read_bytes() == expected
    assert artifact_io.sha256_file(text_path) == hashlib.sha256(expected).hexdigest()
    assert sorted(p.name for p in text_path.parent.iterdir()) == ["text.json"]


@pytest.mark.parametrize("failure", ("write", "fsync", "replace"))
def test_atomic_write_failure_preserves_destination_and_cleans_temp(tmp_path, failure):
    output = tmp_path / "saved.npy"
    output.write_bytes(b"original")
    target = {"write": "numpy.save", "fsync": "probes.artifact_io.os.fsync",
              "replace": "probes.artifact_io.os.replace"}[failure]
    with patch(target, side_effect=OSError("injected write failure")):
        with pytest.raises(OSError, match="injected"):
            artifact_io.atomic_save_npy(output, np.ones((2, 3)))
    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]


def test_figure_writer_preserves_suffix_and_cleans_failed_save(tmp_path):
    output = tmp_path / "figure.svg"

    def savefig(path, **kwargs):
        assert path.suffix == ".svg"
        assert kwargs == {"dpi": 100}
        path.write_text("<svg/>")

    artifact_io.atomic_save_figure(SimpleNamespace(savefig=savefig), output, dpi=100)
    assert output.read_text() == "<svg/>"

    def fail(path):
        path.write_text("partial")
        raise RuntimeError("interrupted figure")

    with pytest.raises(RuntimeError, match="interrupted"):
        artifact_io.atomic_save_figure(SimpleNamespace(savefig=fail), output)
    assert output.read_text() == "<svg/>"
    assert list(tmp_path.iterdir()) == [output]


def test_artifact_helpers_remain_importable_without_training_dependencies():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, "-B", "-c",
        "import sys; from probes import artifact_io; "
        "assert not {'numpy', 'sklearn', 'torch'} & sys.modules.keys()",
    ], cwd=root, check=True)


@pytest.mark.parametrize("script", ("extract_hf_encoder", "rsa_analysis", "benchmark_summary"))
def test_changed_clis_support_direct_and_package_imports(script):
    root = Path(__file__).resolve().parents[1]
    for invocation in ([f"probes/{script}.py"], ["-m", f"probes.{script}"]):
        result = subprocess.run(
            [sys.executable, "-B", *invocation, "--help"],
            cwd=root, capture_output=True, text=True, check=True,
        )
        assert "usage:" in result.stdout
