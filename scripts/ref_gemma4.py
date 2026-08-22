#!/usr/bin/env python3
"""FP32 reference forward pass for Gemma 4 E2B (google/gemma-4-E2B-it).

Loads ONLY the language model from the original safetensors into
transformers' Gemma4TextModel at float32 (the two giant embedding tables
stay bf16 in storage; gathered rows are upcast exactly since bf16 -> f32 is
lossless), runs one prompt, and dumps:

  hidden_states.npy  [n_layers+1, seq, 1536]  (entry 0 = scaled embeddings,
                      then block outputs; transformers 5.x overwrites the
                      last entry with the FINAL-NORMED state via
                      capture_outputs(tie_last_hidden_states=True))
  final_normed.npy   [seq, 1536]
  logits.npy         [seq, vocab]             (tied head + softcap 30)
  input_ids.npy

Parity reference for gemma4: ember matches this fp32 path stage-by-stage at
cos >= 0.999 through all 35 blocks (see scripts/gemma4_forensics.py).
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import (
    ROPE_INIT_FUNCTIONS,
    Gemma4TextConfig,
    Gemma4TextModel,
)

SRC = "/home/west/ember-work/gemma4-src"



def assign(root, path, tensor):
    """Replace a module attribute with a new Parameter/buffer carrying
    `tensor` (meta-device params reject in-place set_data)."""
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parts[-1]
    old = getattr(parent, leaf)
    if isinstance(old, torch.nn.Parameter):
        setattr(parent, leaf, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(parent, leaf, tensor)


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/west/ember-work/ref_gemma4"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(SRC)
    ids = tok(prompt)["input_ids"]
    if not ids or ids[0] != tok.bos_token_id:
        ids = [tok.bos_token_id] + ids
    print("token ids:", ids)

    cfg_full = json.load(open(os.path.join(SRC, "config.json")))
    cfg = Gemma4TextConfig(**cfg_full["text_config"])
    # meta-device init: no parameter is materialized until real weights are
    # assigned (a naive f32 init alone needs >9 GB for the PLE table)
    with torch.device("meta"):
        model = Gemma4TextModel(cfg)
    # materialize the small rotary module on cpu and rebuild its inv_freq
    # buffers exactly as __init__ would (meta tensors cannot run)
    model.rotary_emb = model.rotary_emb.to_empty(device="cpu")
    for layer_type in set(cfg.layer_types):
        rope_params = cfg.rope_parameters[layer_type]
        rope_type = rope_params["rope_type"]
        rope_init_fn = (
            model.rotary_emb.compute_default_rope_parameters
            if rope_type == "default"
            else ROPE_INIT_FUNCTIONS[rope_type]
        )
        kwargs = {"device": torch.device("cpu"), "layer_type": layer_type}
        if layer_type == "full_attention" and rope_type == "proportional":
            kwargs["head_dim_key"] = "global_head_dim"
        inv_freq, scaling = rope_init_fn(cfg, **kwargs)
        model.rotary_emb.register_buffer(
            f"{layer_type}_inv_freq", inv_freq, persistent=False
        )
        model.rotary_emb.register_buffer(
            f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False
        )
        setattr(model.rotary_emb, f"{layer_type}_attention_scaling", scaling)
    model.eval()

    # ---- load state dict -------------------------------------------------
    # Direct assignment into parameters; the giant embedding tables are
    # sliced to just the prompt's token rows so the whole model fits RAM.
    import gc

    f32 = torch.float32

    class _SafeTensors:
        """Lazy per-tensor access over the safetensors file."""

        def __init__(self, path):
            from safetensors import safe_open

            self.f = safe_open(path, framework="pt")

        def get(self, name: str) -> torch.Tensor:
            return self.f.get_tensor(f"model.language_model.{name}")

    st = _SafeTensors(os.path.join(SRC, "model.safetensors"))

    input_ids_t = torch.tensor([ids])
    uniq = sorted(set(ids))
    remap = {t: i for i, t in enumerate(uniq)}
    local_ids = torch.tensor([[remap[t] for t in ids]])

    with torch.no_grad():
        # main embedding + PLE table are NOT assigned here (shape-changing
        # set_data is rejected); they are replaced by patched lookups over
        # prompt-row slices below
        full_rows = st.get("embed_tokens.weight")[uniq].to(f32)

        # PLE table slice, bf16 storage
        ple_rows = st.get("embed_tokens_per_layer.weight")[uniq]
        ple_scale = cfg.hidden_size_per_layer_input ** 0.5

        assign(model.norm, "weight", st.get("norm.weight").to(f32))
        assign(model.per_layer_model_projection, "weight", st.get("per_layer_model_projection.weight").to(f32))
        assign(model.per_layer_projection_norm, "weight", st.get("per_layer_projection_norm.weight").to(f32))

        n_layers = cfg.num_hidden_layers
        names = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.k_norm.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "post_feedforward_layernorm.weight",
            "per_layer_input_gate.weight",
            "per_layer_projection.weight",
            "post_per_layer_input_norm.weight",
            "layer_scalar",
        ]
        for i in range(n_layers):
            layer = model.layers[i]
            for name in names:
                try:
                    t = st.get(f"layers.{i}.{name}")
                except Exception:
                    continue  # shared-KV layers lack k/v tensors in HF
                # shared-KV modules may not exist on the layer even though
                # the checkpoint ships the tensors
                probe = layer
                try:
                    for part in name.split(".")[:-1]:
                        probe = getattr(probe, part)
                except AttributeError:
                    continue
                assign(layer, name, t.to(f32))

    del st
    gc.collect()

    # ---- patched PLE lookup over the sliced table ------------------------
    model.embed_tokens_per_layer.forward = (
        lambda input_ids_: ple_rows[[remap[int(t)] for t in input_ids_[0]]]
        .unsqueeze(0)
        .float()
        * ple_scale
    )
    # patched main embedding over the sliced table (exact: bf16 rows upcast);
    # Gemma scales embeddings by sqrt(hidden)
    embed_scale = cfg.hidden_size ** 0.5
    model.embed_tokens.forward = (
        lambda input_ids_: full_rows[[remap[int(t)] for t in input_ids_[0]]]
        .unsqueeze(0)
        * embed_scale
    )

    with torch.no_grad():
        out = model(input_ids_t, output_hidden_states=True)
        hs = out.hidden_states  # tuple(len = n_layers+1)
        hidden = torch.stack([h[0] for h in hs]).numpy().astype(np.float32)

        # transformers 5.x already applies the final norm to
        # last_hidden_state; applying model.norm() again double-norms
        normed = out.last_hidden_state

        # full-vocab logits against the ORIGINAL table, computed in row
        # chunks so the 262144x1536 matrix never sits in RAM
        from safetensors import safe_open

        with safe_open(
            os.path.join(SRC, "model.safetensors"), framework="pt"
        ) as sf:
            emb_full = sf.get_tensor(
                "model.language_model.embed_tokens.weight"
            )
            rows = []
            B = 8192
            for start in range(0, emb_full.shape[0], B):
                block = emb_full[start : start + B].float()
                rows.append(normed[0] @ block.T)  # [seq, B]
            raw_logits = torch.cat(rows, dim=-1)  # [seq, vocab]
        logits = 30.0 * torch.tanh(raw_logits / 30.0)

    np.save(os.path.join(out_dir, "hidden_states.npy"), hidden)
    np.save(os.path.join(out_dir, "final_normed.npy"), normed[0].numpy())
    np.save(os.path.join(out_dir, "logits.npy"), logits.numpy().T.astype(np.float32))
    np.save(os.path.join(out_dir, "input_ids.npy"), np.array(ids, dtype=np.int64))
    top = torch.topk(logits[-1, :], 10)
    print("top-10 next tokens:", tok.convert_ids_to_tokens(top.indices.tolist()))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
