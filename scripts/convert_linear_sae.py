"""Convert the PyTorch state dict of a linear sparse autoencoder to a SAELens
folder that Ember's `[[dictionaries]]` can read.

The input is a `.pt` file that holds only an `OrderedDict` in the
`torch.nn.Linear` layout, for a ReLU SAE:

    encoder.weight [d_sae, d_in]    encoder.bias [d_sae]
    decoder.weight [d_in, d_sae]    decoder.bias [d_in]

so that f = relu(encoder(x)) and x_hat = decoder(f). The output folder gets
`sae_weights.safetensors` (W_enc = encoder.weight.T, b_enc, W_dec =
decoder.weight.T, b_dec, all F32) and a `cfg.json` with the SAELens fields
that Ember reads, plus a `converted_from` record of the source.

The pickle is never run as code. The unpickler accepts only an OrderedDict,
float storages and the tensor rebuild functions, and refuses every other
global, so a file that pickles a model class or a callable fails before
anything runs. PyTorch is not needed.

usage:
    python scripts/convert_linear_sae.py IN.pt OUT_DIR \\
        --model-name meta-llama/Llama-3.2-1B-Instruct \\
        --hook-name blocks.9.hook_resid_post \\
        [--source "repo@revision file"] [--license apache-2.0]

The architecture and the hook are not in the file. Check them: capture the
site with `features` and look at `l0` and `relative_error` in the feature
artifact. A wrong hook or a missing input offset shows as a poor fit.
"""

import argparse
import collections
import hashlib
import io
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

STORAGE_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16}
TENSORS = ("encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias")


class _Storage:
    def __init__(self, name):
        self.dtype = STORAGE_DTYPES[name]


def _rebuild_tensor(storage, offset, size, stride, *_unused):
    itemsize = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    ).copy()


def _rebuild_parameter(data, *_unused):
    return data


class _StateDictUnpickler(pickle.Unpickler):
    def __init__(self, data, archive, prefix):
        super().__init__(io.BytesIO(data))
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if (module, name) == ("torch._utils", "_rebuild_parameter"):
            return _rebuild_parameter
        if module == "torch" and name in STORAGE_DTYPES:
            return _Storage(name)
        raise pickle.UnpicklingError(
            f"the file names {module}.{name}; only a plain state dict of tensors is accepted"
        )

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _kind, storage, key, _location, numel = pid
        if not isinstance(storage, _Storage):
            raise pickle.UnpicklingError(f"unsupported storage {storage!r}")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        values = np.frombuffer(raw, dtype=storage.dtype)
        if values.size != numel:
            raise pickle.UnpicklingError(f"storage {key} holds {values.size} values, not {numel}")
        return values


def read_state_dict(path):
    """The tensors of a `.pt` state dict, as NumPy arrays, without running it."""
    with zipfile.ZipFile(path) as archive:
        pickles = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise ValueError(f"{path}: expected one data.pkl, found {len(pickles)}")
        prefix = pickles[0][: -len("/data.pkl")]
        state = _StateDictUnpickler(archive.read(pickles[0]), archive, prefix).load()
    if not isinstance(state, dict):
        raise ValueError(f"{path}: the pickle holds a {type(state).__name__}, not a state dict")
    return dict(state)


def to_saelens(state):
    """SAELens tensors from a linear-SAE state dict, with the shapes checked."""
    missing = [name for name in TENSORS if name not in state]
    extra = sorted(set(state) - set(TENSORS))
    if missing or extra:
        raise ValueError(f"expected exactly {list(TENSORS)}; missing {missing}, extra {extra}")
    d_sae, d_in = state["encoder.weight"].shape
    expected = {
        "encoder.weight": (d_sae, d_in),
        "encoder.bias": (d_sae,),
        "decoder.weight": (d_in, d_sae),
        "decoder.bias": (d_in,),
    }
    for name, shape in expected.items():
        if state[name].shape != shape:
            raise ValueError(f"{name} has shape {state[name].shape}; expected {shape}")
        if not np.all(np.isfinite(state[name])):
            raise ValueError(f"{name} holds a non-finite value")
    as_f32 = lambda array: np.ascontiguousarray(array, dtype="<f4")
    return {
        "W_enc": as_f32(state["encoder.weight"].T),
        "b_enc": as_f32(state["encoder.bias"]),
        "W_dec": as_f32(state["decoder.weight"].T),
        "b_dec": as_f32(state["decoder.bias"]),
    }


def write_safetensors(path, tensors):
    """Tensors in name order, after an 8-byte-aligned JSON header."""
    header, offset = {}, 0
    for name in sorted(tensors):
        size = tensors[name].nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name in sorted(tensors):
            out.write(tensors[name].tobytes())


def convert(source, out_dir, model_name, hook_name, source_note=None, license_id=None):
    """Write `cfg.json` and `sae_weights.safetensors` into `out_dir`."""
    tensors = to_saelens(read_state_dict(source))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_safetensors(out_dir / "sae_weights.safetensors", tensors)
    d_in, d_sae = tensors["W_enc"].shape
    cfg = {
        "architecture": "standard",
        "d_in": int(d_in),
        "d_sae": int(d_sae),
        "activation_fn": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "converted_from": {
            "file": Path(source).name,
            "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "source": source_note,
            "license": license_id,
            "mapping": "W_enc = encoder.weight.T, b_enc = encoder.bias, "
            "W_dec = decoder.weight.T, b_dec = decoder.bias",
        },
    }
    with open(out_dir / "cfg.json", "w") as out:
        json.dump(cfg, out, indent=2)
        out.write("\n")
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="the .pt state dict")
    parser.add_argument("out_dir", help="the SAELens folder to write")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hook-name", required=True)
    parser.add_argument("--source", dest="source_note", help="where the file came from")
    parser.add_argument("--license", dest="license_id", help="the SAE's license")
    args = parser.parse_args(argv)
    try:
        cfg = convert(
            args.source, args.out_dir, args.model_name, args.hook_name,
            args.source_note, args.license_id,
        )
    except (pickle.UnpicklingError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    for name in ("sae_weights.safetensors", "cfg.json"):
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        print(f"{name}: sha256 {digest}")
    print(f"d_in {cfg['d_in']}, d_sae {cfg['d_sae']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
