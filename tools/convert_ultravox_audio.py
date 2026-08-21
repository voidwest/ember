#!/usr/bin/env python3
"""Convert Ultravox audio tower + projector from HuggingFace safetensors
into ember's audio mmproj GGUF layout.

Usage:
    python tools/convert_ultravox_audio.py <model.safetensors> <out.audio.gguf>

Follows the vision mmproj convention exactly (see
tools/convert_smolvlm_mmproj.py): tensors are written unchanged from the HF
state dict (payload = HF row-major), converted to f32. Ember's
`take_f32` + `gguf_to_row_major_f32` then produce the expected layouts:
linears arrive as [in, out] row-major, convs as [out, in, k].

metadata:
  general.architecture                = "ultravox-audio"
  ultravox.audio.num_mel_bins         = 128
  ultravox.audio.d_model              = 1280
  ultravox.audio.encoder_layers       = 32
  ultravox.audio.encoder_ffn_dim      = 5120
  ultravox.audio.max_source_positions = 1500
  ultravox.audio.layer_norm_eps       = 1e-5
  ultravox.stack_factor               = 8
  ultravox.projector.act              = "swiglu"

tensors (prefix `a.`):
  a.audio_tower.conv1.weight/.bias            Conv1d [1280, 128, 3]
  a.audio_tower.conv2.weight/.bias            Conv1d [1280, 1280, 3]
  a.audio_tower.position_embedding.weight     [1500, 1280]
  a.audio_tower.layers.{i}.self_attn.{q,k,v,out}_proj.weight/.bias
  a.audio_tower.layers.{i}.self_attn_layer_norm.weight/.bias
  a.audio_tower.layers.{i}.fc1.weight/.bias   [5120, 1280]
  a.audio_tower.layers.{i}.fc2.weight/.bias   [1280, 5120]
  a.audio_tower.layers.{i}.final_layer_norm.weight/.bias
  a.audio_tower.layer_norm.weight/.bias
  a.projector.ln_pre.weight                   [10240]
  a.projector.linear_1.weight                 [4096, 10240]
  a.projector.ln_mid.weight                   [2048]
  a.projector.linear_2.weight                 [4096, 2048]
"""
import sys

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

sys.path.insert(0, "gguf-py")
import gguf  # noqa: E402


def main() -> None:
    src, out_path = sys.argv[1], sys.argv[2]

    tensors = load_file(src)
    keys = set(tensors.keys())

    def has(name: str) -> bool:
        return name in keys

    def get(name: str) -> torch.Tensor:
        return tensors[name]

    def f32(name: str) -> np.ndarray:
        return get(name).contiguous().to(torch.float32).numpy()

    # derive hyperparameters from the tensors themselves (the v0_5 repo
    # ships no standalone config for the tower)
    d_model = f32("audio_tower.conv1.bias").shape[0]
    n_mels = f32("audio_tower.conv1.weight").shape[1]
    n_layers = max(
        int(k.split(".")[2]) for k in keys if k.startswith("audio_tower.layers.")
    ) + 1
    ffn_dim = f32("audio_tower.layers.0.fc1.bias").shape[0]
    pos_len = f32("audio_tower.embed_positions.weight").shape[0]
    llm_hidden = f32("multi_modal_projector.linear_2.weight").shape[0]
    proj_hidden = f32("multi_modal_projector.linear_1.weight").shape[0]
    mid = f32("multi_modal_projector.linear_2.weight").shape[1]
    assert mid * 2 == proj_hidden, "swiglu convention assumes mid == hidden/2"

    out = gguf.GGUFWriter(out_path, "ultravox-audio")
    out.add_uint32("ultravox.audio.num_mel_bins", n_mels)
    out.add_uint32("ultravox.audio.d_model", d_model)
    out.add_uint32("ultravox.audio.encoder_layers", n_layers)
    out.add_uint32("ultravox.audio.encoder_ffn_dim", ffn_dim)
    out.add_uint32("ultravox.audio.max_source_positions", pos_len)
    out.add_float32("ultravox.audio.layer_norm_eps", 1e-5)
    out.add_uint32("ultravox.stack_factor", 8)
    out.add_string("ultravox.projector.act", "swiglu")
    out.add_uint32("ultravox.text.hidden_size", llm_hidden)

    def add(name_gguf: str, arr: np.ndarray) -> None:
        out.add_tensor(name_gguf, arr)

    add("a.audio_tower.conv1.weight", f32("audio_tower.conv1.weight"))
    add("a.audio_tower.conv1.bias", f32("audio_tower.conv1.bias"))
    add("a.audio_tower.conv2.weight", f32("audio_tower.conv2.weight"))
    add("a.audio_tower.conv2.bias", f32("audio_tower.conv2.bias"))
    add(
        "a.audio_tower.position_embedding.weight",
        f32("audio_tower.embed_positions.weight"),
    )

    for i in range(n_layers):
        p_hf = f"audio_tower.layers.{i}"
        p_gg = f"a.audio_tower.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            add(f"{p_gg}.self_attn.{proj}.weight",
                f32(f"{p_hf}.self_attn.{proj}.weight"))
            if has(f"{p_hf}.self_attn.{proj}.bias"):
                add(f"{p_gg}.self_attn.{proj}.bias",
                    f32(f"{p_hf}.self_attn.{proj}.bias"))
        add(f"{p_gg}.self_attn_layer_norm.weight",
            f32(f"{p_hf}.self_attn_layer_norm.weight"))
        add(f"{p_gg}.self_attn_layer_norm.bias",
            f32(f"{p_hf}.self_attn_layer_norm.bias"))
        add(f"{p_gg}.fc1.weight", f32(f"{p_hf}.fc1.weight"))
        add(f"{p_gg}.fc1.bias", f32(f"{p_hf}.fc1.bias"))
        add(f"{p_gg}.fc2.weight", f32(f"{p_hf}.fc2.weight"))
        add(f"{p_gg}.fc2.bias", f32(f"{p_hf}.fc2.bias"))
        add(f"{p_gg}.final_layer_norm.weight",
            f32(f"{p_hf}.final_layer_norm.weight"))
        add(f"{p_gg}.final_layer_norm.bias",
            f32(f"{p_hf}.final_layer_norm.bias"))

    add("a.audio_tower.layer_norm.weight", f32("audio_tower.layer_norm.weight"))
    add("a.audio_tower.layer_norm.bias", f32("audio_tower.layer_norm.bias"))

    add("a.projector.ln_pre.weight", f32("multi_modal_projector.ln_pre.weight"))
    add("a.projector.linear_1.weight", f32("multi_modal_projector.linear_1.weight"))
    add("a.projector.ln_mid.weight", f32("multi_modal_projector.ln_mid.weight"))
    add("a.projector.linear_2.weight", f32("multi_modal_projector.linear_2.weight"))

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()

    print(f"wrote {out_path}")
    print(
    f"dims: mel={n_mels} d_model={d_model} layers={n_layers} ffn={ffn_dim} "
    f"pos={pos_len} mid={mid} llm_hidden={llm_hidden}"
    )


if __name__ == "__main__":
    main()
