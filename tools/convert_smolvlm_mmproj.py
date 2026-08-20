#!/usr/bin/env python3
"""Convert SmolVLM-256M-Instruct (or any Idefics3-family) vision tower +
connector from HuggingFace safetensors into ember's mmproj GGUF layout.

Usage:
    python tools/convert_smolvlm_mmproj.py <model_dir> <out.mmproj.gguf>

Layout (documented in docs/multimodal-foundation-plan.md):

metadata:
  general.architecture            = "smolvlm-vision"
  smolvlm.vision.patch_size       = 16
  smolvlm.vision.image_size       = 512
  smolvlm.vision.hidden_size      = 768
  smolvlm.vision.num_hidden_layers= 12
  smolvlm.vision.num_attention_heads = 12
  smolvlm.vision.intermediate_size= 3072
  smolvlm.vision.layer_norm_eps   = 1e-6 (float32)
  smolvlm.scale_factor            = 4
  smolvlm.text.hidden_size        = 576

tensors (GGUF dims = reversed HF shape; payload unchanged HF row-major;
linears use the same [in, out]-dims convention the llama loader uses):
  v.vision.embeddings.patch_embedding.weight   [768, 3, 16, 16] (dims 16,16,3,768)
  v.vision.embeddings.patch_embedding.bias     [768]
  v.vision.embeddings.position_embedding.weight [1024, 768] (dims 768, 1024)
  v.vision.encoder.layers.{i}.layer_norm1.weight/.bias   [768]
  v.vision.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight/.bias
  v.vision.encoder.layers.{i}.layer_norm2.weight/.bias
  v.vision.encoder.layers.{i}.mlp.fc1.weight/.bias   [768, 3072]
  v.vision.encoder.layers.{i}.mlp.fc2.weight/.bias   [3072, 768]
  v.vision.post_layernorm.weight/.bias
  v.connector.modality_projection.proj.weight   [576, 12288] (dims 12288, 576)
"""
import json
import sys

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, "gguf-py")
import gguf  # noqa: E402


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    vcfg = cfg["vision_config"]
    tcfg = cfg["text_config"]

    tensors = safetensors.torch.load_file(f"{model_dir}/model.safetensors")
    out = gguf.GGUFWriter(out_path, "smolvlm-vision")

    out.add_uint32("smolvlm.vision.patch_size", vcfg["patch_size"])
    out.add_uint32("smolvlm.vision.image_size", vcfg["image_size"])
    out.add_uint32("smolvlm.vision.hidden_size", vcfg["hidden_size"])
    out.add_uint32("smolvlm.vision.num_hidden_layers", vcfg["num_hidden_layers"])
    out.add_uint32("smolvlm.vision.num_attention_heads", vcfg["num_attention_heads"])
    out.add_uint32("smolvlm.vision.intermediate_size", vcfg["intermediate_size"])
    out.add_float32("smolvlm.vision.layer_norm_eps", float(vcfg["layer_norm_eps"]))
    out.add_uint32("smolvlm.scale_factor", cfg["scale_factor"])
    out.add_uint32("smolvlm.text.hidden_size", tcfg["hidden_size"])

    def add(name: str, key: str) -> None:
        t = tensors[key]
        t = t.contiguous().to(torch.float16)
        out.add_tensor(name, t.numpy())

    prefix = "model."
    add("v.vision.embeddings.patch_embedding.weight", prefix + "vision_model.embeddings.patch_embedding.weight")
    add("v.vision.embeddings.patch_embedding.bias", prefix + "vision_model.embeddings.patch_embedding.bias")
    add("v.vision.embeddings.position_embedding.weight", prefix + "vision_model.embeddings.position_embedding.weight")
    for i in range(vcfg["num_hidden_layers"]):
        p = f"v.vision.encoder.layers.{i}."
        hf = f"{prefix}vision_model.encoder.layers.{i}."
        add(p + "layer_norm1.weight", hf + "layer_norm1.weight")
        add(p + "layer_norm1.bias", hf + "layer_norm1.bias")
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            add(p + f"self_attn.{proj}.weight", hf + f"self_attn.{proj}.weight")
            add(p + f"self_attn.{proj}.bias", hf + f"self_attn.{proj}.bias")
        add(p + "layer_norm2.weight", hf + "layer_norm2.weight")
        add(p + "layer_norm2.bias", hf + "layer_norm2.bias")
        add(p + "mlp.fc1.weight", hf + "mlp.fc1.weight")
        add(p + "mlp.fc1.bias", hf + "mlp.fc1.bias")
        add(p + "mlp.fc2.weight", hf + "mlp.fc2.weight")
        add(p + "mlp.fc2.bias", hf + "mlp.fc2.bias")
    add("v.vision.post_layernorm.weight", prefix + "vision_model.post_layernorm.weight")
    add("v.vision.post_layernorm.bias", prefix + "vision_model.post_layernorm.bias")
    add("v.connector.modality_projection.proj.weight", prefix + "connector.modality_projection.proj.weight")

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
