"""scripts/convert_linear_sae.py: a linear-SAE state dict becomes a SAELens
folder, and a pickle that names anything else is refused without running."""

import collections
import json
import pickle
import struct
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import convert_linear_sae as convert  # noqa: E402


def _write_state_dict(path, tensors):
    """Write `tensors` as PyTorch writes a state dict, without PyTorch: fake
    `torch` modules exist only while the pickle is written, so the pickle names
    the same globals as a real one."""
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args):
        raise AssertionError("never called while writing")

    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    torch.FloatStorage, torch._utils, utils._rebuild_tensor_v2 = FloatStorage, utils, _rebuild_tensor_v2

    class Storage:
        def __init__(self, key, values):
            self.key, self.values = key, values

    class Tensor:
        def __init__(self, storage, shape):
            self.storage, self.shape = storage, shape

        def __reduce__(self):
            stride, step = [], 1
            for size in reversed(self.shape):
                stride.insert(0, step)
                step *= size
            return (_rebuild_tensor_v2, (self.storage, 0, self.shape, tuple(stride), False, collections.OrderedDict()))

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", FloatStorage, obj.key, "cpu", obj.values.size)
            return None

    storages, state = {}, collections.OrderedDict()
    for index, (name, array) in enumerate(tensors.items()):
        storage = Storage(str(index), np.ascontiguousarray(array, dtype="<f4").reshape(-1))
        storages[storage.key] = storage
        state[name] = Tensor(storage, tuple(array.shape))
    saved = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = torch, utils
    try:
        buffer = __import__("io").BytesIO()
        Pickler(buffer, protocol=2).dump(state)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        for key, storage in storages.items():
            archive.writestr(f"archive/data/{key}", storage.values.tobytes())


def _read_safetensors(path):
    data = Path(path).read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + n])
    base = 8 + n
    return {
        name: np.frombuffer(data[base + meta["data_offsets"][0] : base + meta["data_offsets"][1]], dtype="<f4").reshape(meta["shape"])
        for name, meta in header.items()
    }


def _linear_sae(d_in=3, d_sae=5):
    rng = np.random.default_rng(0)
    return {
        "encoder.weight": rng.standard_normal((d_sae, d_in)),
        "encoder.bias": rng.standard_normal(d_sae),
        "decoder.weight": rng.standard_normal((d_in, d_sae)),
        "decoder.bias": rng.standard_normal(d_in),
    }


def test_a_linear_state_dict_becomes_a_saelens_folder(tmp_path):
    state = _linear_sae()
    source = tmp_path / "sae.pt"
    _write_state_dict(source, state)
    out = tmp_path / "saelens"
    assert convert.main([str(source), str(out), "--model-name", "m", "--hook-name", "blocks.0.hook_resid_post", "--license", "apache-2.0"]) == 0

    tensors = _read_safetensors(out / "sae_weights.safetensors")
    f32 = lambda array: np.asarray(array, dtype="<f4")
    assert np.array_equal(tensors["W_enc"], f32(state["encoder.weight"]).T)
    assert np.array_equal(tensors["b_enc"], f32(state["encoder.bias"]))
    assert np.array_equal(tensors["W_dec"], f32(state["decoder.weight"]).T)
    assert np.array_equal(tensors["b_dec"], f32(state["decoder.bias"]))
    cfg = json.loads((out / "cfg.json").read_text())
    assert (cfg["architecture"], cfg["d_in"], cfg["d_sae"]) == ("standard", 3, 5)
    assert cfg["apply_b_dec_to_input"] is False and cfg["normalize_activations"] == "none"
    assert cfg["converted_from"]["license"] == "apache-2.0"

    # The encode that Ember applies to the converted folder is the model's.
    x = np.array([0.5, -1.0, 2.0])
    expected = np.maximum(state["encoder.weight"] @ x + state["encoder.bias"], 0)
    got = np.maximum(x @ tensors["W_enc"].astype(np.float64) + tensors["b_enc"], 0)
    assert np.allclose(got, expected, atol=1e-6)


def test_a_pickle_that_names_anything_else_is_refused_without_running(tmp_path):
    marker = tmp_path / "ran"
    # GLOBAL os.system, then call it: what a hostile .pt can hold.
    payload = b"cos\nsystem\n(S'touch " + str(marker).encode() + b"'\ntR."
    source = tmp_path / "hostile.pt"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
    with pytest.raises(pickle.UnpicklingError, match="os.system"):
        convert.read_state_dict(source)
    assert not marker.exists(), "the pickle's call must never run"
    assert convert.main([str(source), str(tmp_path / "out"), "--model-name", "m", "--hook-name", "h"]) == 1


def test_wrong_tensors_or_shapes_are_refused(tmp_path):
    state = _linear_sae()
    state["decoder.weight"] = state["decoder.weight"][:, :4]
    source = tmp_path / "bad.pt"
    _write_state_dict(source, state)
    with pytest.raises(ValueError, match="decoder.weight has shape"):
        convert.to_saelens(convert.read_state_dict(source))
    extra = _linear_sae()
    extra["threshold"] = np.ones(5)
    _write_state_dict(source, extra)
    with pytest.raises(ValueError, match="extra"):
        convert.to_saelens(convert.read_state_dict(source))
