#!/usr/bin/env python3
"""Convert an MMS-TTS VITS model (transformers layout) into ember's TTS GGUF.

Usage:
    python tools/convert_mms_vits.py <model_dir> <out.vits.gguf>

<source> is a local directory holding `model.safetensors`, `config.json`,
`vocab.txt` as published for e.g. facebook/mms-tts-ara (CC-BY-NC-4.0).

Inference path converted (training-only modules — posterior encoder and the
SDP's posterior branch — are skipped):

```text
ids [T] -> embed(vocab,192) * sqrt(192)
        -> 6 x { rel-pos attention(2 heads) + FFN(conv k3 relu) }
        -> project conv1d(192->384, k1)  -> prior_means / prior_log_vars
        -> SDP(reverse): conv_pre(k1) -> DDSConv x3 -> conv_proj
           -> [ConvFlow3, ConvFlow2, ElementwiseAffine] reverse
        -> durations -> monotonic expansion of hidden + priors
        -> flow reverse x4: conv_pre(k1) -> WaveNet(4 layers) -> conv_post
        -> HiFi-GAN: conv_pre(k7) -> up x[8,8,2,2] + resblocks{3,7,11}
           -> conv_post(k7) -> tanh -> PCM @16 kHz
```

Weight-normalized convs (`weight_g`/`weight_v` pairs under flow.flows.*.
wavenet.*) are deparametrized here: weight = g * v / ||v||_out.

metadata:
  general.architecture   = "mms-vits"
  vits.vocab_size / hidden_size / num_layers / num_heads / window_size
  vits.ffn_dim / ffn_kernel_size / flow_size / wavenet_layers
  vits.prior_flows / dp_dds_layers / dp_flows / dp_bins / dp_tail_bound
  vits.upsample_rates/kernels / resblock_kernels/dilations
  vits.sample_rate=16000 hop=256 leaky=0.1 ln_eps=1e-5
  vits.vocab             (raw vocab.txt bytes, utf-8)

tensors (prefix `v.`):
  v.embed                       [vocab, H]
  v.layer.{i}.attn.{q,k,v,out}  Linear weights [H,H] + bias [H]
  v.layer.{i}.rel_{k,v}         [1, 2*W+1, hd]
  v.layer.{i}.ffn1/ffn2         Conv1d weights [dim,in,k] + bias
  v.layer.{i}.ln1/ln2           LayerNorm scale/bias [H]
  v.project                     Conv1d(192->2*F, k1) weight+bias
  v.sdp.conv_pre/proj, v.sdp.dds.{d,p}.{0..2}, v.sdp.dds.ln1/ln2.{0..2}
  v.sdp.flow{j}.affine.translate/log_scale      (j=0)
  v.sdp.flow{j}.conv_pre/proj, dds...,          (j=1..3 ConvFlows)
  v.flow{j}.conv_pre/post, v.flow{j}.wavenet.in/res_skip.{0..3}
  v.hifigan.conv_pre/post
  v.up{i}                        ConvTranspose1d weight [in,out,k]
  v.rb{i}{j}.c1{k}.w/.b, .c2{k}.w/.b   resblocks i*3+j, convs k=0..2
"""
import json
import sys

import numpy as np
import torch  # noqa: F401
from safetensors.torch import load_file

sys.path.insert(0, "gguf-py")
import gguf  # noqa: E402


def main() -> None:
    src_dir, out_path = sys.argv[1], sys.argv[2]

    sd = load_file(f"{src_dir}/model.safetensors")
    # transformers-format config lives at the repo root (hf-config.json in a
    # converted dir); the per-model config.json uses raw VITS names instead.
    import os

    cfg_path = f"{src_dir}/config.json"
    if os.path.exists(f"{src_dir}/hf-config.json"):
        cfg_path = f"{src_dir}/hf-config.json"
    cfg = json.load(open(cfg_path))
    vocab = open(f"{src_dir}/vocab.txt", encoding="utf-8").read().splitlines()

    def f32(name: str) -> np.ndarray:
        return sd[name].contiguous().to(torch.float32).numpy()

    def wn(name: str) -> np.ndarray:
        """Deparametrize torch weight_norm: w = g * v / ||v||_out.
        For Conv1d weights [out, in, k] the norm spans ALL dims except out."""
        g = sd[f"{name}.weight_g"].to(torch.float32)
        v = sd[f"{name}.weight_v"].to(torch.float32)
        norm = v.norm(dim=(1, 2), keepdim=True)
        return (g * v / norm).contiguous().numpy()

    has_g = any(k.endswith(".weight_g") for k in sd)

    def conv(name: str) -> tuple[np.ndarray, np.ndarray]:
        if has_g and f"{name}.weight_g" in sd:
            return wn(name), f32(f"{name}.bias")
        return f32(f"{name}.weight"), f32(f"{name}.bias")

    out = gguf.GGUFWriter(out_path, "mms-vits")
    out.add_uint32("vits.vocab_size", len(vocab))
    out.add_uint32("vits.hidden_size", cfg["hidden_size"])
    out.add_uint32("vits.num_layers", cfg["num_hidden_layers"])
    out.add_uint32("vits.num_heads", cfg["num_attention_heads"])
    out.add_uint32("vits.window_size", cfg.get("window_size", 4))
    out.add_uint32("vits.ffn_dim", cfg["ffn_dim"])
    out.add_uint32("vits.ffn_kernel_size", cfg["ffn_kernel_size"])
    out.add_uint32("vits.flow_size", cfg["flow_size"])
    out.add_uint32("vits.wavenet_layers", cfg["prior_encoder_num_wavenet_layers"])
    out.add_uint32("vits.prior_flows", cfg["prior_encoder_num_flows"])
    out.add_uint32("vits.dp_dds_layers", cfg["depth_separable_num_layers"])
    out.add_uint32("vits.dp_flows", cfg["duration_predictor_num_flows"])
    out.add_uint32("vits.dp_bins", cfg["duration_predictor_flow_bins"])
    out.add_float32("vits.dp_tail_bound", cfg["duration_predictor_tail_bound"])
    out.add_uint32("vits.sample_rate", cfg.get("sampling_rate", 16000))
    out.add_uint32("vits.hop_length", cfg.get("hop_length", 256))
    out.add_float32("vits.leaky_relu_slope", cfg.get("leaky_relu_slope", 0.1))
    out.add_float32("vits.ln_eps", cfg.get("layer_norm_eps", 1e-5))
    # The tokenizer's declared pad_token is an ADDED token for VitsTokenizer:
    # text pieces are split on every occurrence and each occurrence emits its
    # bare id WITHOUT surrounding blank frames. For MMS checkpoints this is a
    # real vocab character (e.g. 'ا' -> 0 for ara), which changes the id
    # stream wherever that letter appears. Recorded here so inference can
    # reproduce the reference pipeline exactly.
    pad_tok = None
    tok_cfg_path = f"{src_dir}/tokenizer_config.json"
    if os.path.exists(tok_cfg_path):
        try:
            pad_tok = json.load(open(tok_cfg_path)).get("pad_token")
        except Exception:
            pad_tok = None
    if not pad_tok:
        pad_tok = vocab[0] if vocab else ""
    out.add_string("vits.pad_token", pad_tok)
    out.add_array("vits.vocab", vocab)

    def add(name: str, arr: np.ndarray) -> None:
        out.add_tensor(name, np.ascontiguousarray(arr))

    # -- text encoder ----------------------------------------------------
    add("v.embed", f32("text_encoder.embed_tokens.weight"))
    L = cfg["num_hidden_layers"]
    for i in range(L):
        p = f"text_encoder.encoder.layers.{i}"
        g = f"v.layer.{i}"
        for n, short in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"),
                         ("out_proj", "o")):
            add(f"{g}.attn.{short}.w", f32(f"{p}.attention.{n}.weight"))
            add(f"{g}.attn.{short}.b", f32(f"{p}.attention.{n}.bias"))
        if cfg.get("window_size"):
            add(f"{g}.rel_k", f32(f"{p}.attention.emb_rel_k"))
            add(f"{g}.rel_v", f32(f"{p}.attention.emb_rel_v"))
        add(f"{g}.ffn1.w", f32(f"{p}.feed_forward.conv_1.weight"))
        add(f"{g}.ffn1.b", f32(f"{p}.feed_forward.conv_1.bias"))
        add(f"{g}.ffn2.w", f32(f"{p}.feed_forward.conv_2.weight"))
        add(f"{g}.ffn2.b", f32(f"{p}.feed_forward.conv_2.bias"))
        add(f"{g}.ln1.w", f32(f"{p}.layer_norm.weight"))
        add(f"{g}.ln1.b", f32(f"{p}.layer_norm.bias"))
        add(f"{g}.ln2.w", f32(f"{p}.final_layer_norm.weight"))
        add(f"{g}.ln2.b", f32(f"{p}.final_layer_norm.bias"))
    add("v.project.w", f32("text_encoder.project.weight"))
    add("v.project.b", f32("text_encoder.project.bias"))

    # -- stochastic duration predictor (reverse path only) ---------------
    add("v.sdp.conv_pre.w", f32("duration_predictor.conv_pre.weight"))
    add("v.sdp.conv_pre.b", f32("duration_predictor.conv_pre.bias"))
    add("v.sdp.conv_proj.w", f32("duration_predictor.conv_proj.weight"))
    add("v.sdp.conv_proj.b", f32("duration_predictor.conv_proj.bias"))

    def dds(prefix_src: str, prefix_gguf: str) -> None:
        n = cfg["depth_separable_num_layers"]
        for j in range(n):
            add(
                f"{prefix_gguf}.d{j}.w",
                f32(f"{prefix_src}.convs_dilated.{j}.weight"),
            )
            add(f"{prefix_gguf}.d{j}.b", f32(f"{prefix_src}.convs_dilated.{j}.bias"))
            add(
                f"{prefix_gguf}.p{j}.w",
                f32(f"{prefix_src}.convs_pointwise.{j}.weight"),
            )
            add(f"{prefix_gguf}.p{j}.b", f32(f"{prefix_src}.convs_pointwise.{j}.bias"))
            add(f"{prefix_gguf}.ln1_{j}.w", f32(f"{prefix_src}.norms_1.{j}.weight"))
            add(f"{prefix_gguf}.ln1_{j}.b", f32(f"{prefix_src}.norms_1.{j}.bias"))
            add(f"{prefix_gguf}.ln2_{j}.w", f32(f"{prefix_src}.norms_2.{j}.weight"))
            add(f"{prefix_gguf}.ln2_{j}.b", f32(f"{prefix_src}.norms_2.{j}.bias"))

    dds("duration_predictor.conv_dds", "v.sdp.dds")

    # flows: index 0 = ElementwiseAffine, 1.. = ConvFlow
    nf = cfg["duration_predictor_num_flows"]
    for j in range(nf + 1):
        src_j = f"duration_predictor.flows.{j}"
        g_j = f"v.sdp.flow{j}"
        if j == 0:
            add(f"{g_j}.translate", f32(f"{src_j}.translate"))
            add(f"{g_j}.log_scale", f32(f"{src_j}.log_scale"))
        else:
            add(f"{g_j}.conv_pre.w", f32(f"{src_j}.conv_pre.weight"))
            add(f"{g_j}.conv_pre.b", f32(f"{src_j}.conv_pre.bias"))
            dds(f"{src_j}.conv_dds", f"{g_j}.dds")
            add(f"{g_j}.conv_proj.w", f32(f"{src_j}.conv_proj.weight"))
            add(f"{g_j}.conv_proj.b", f32(f"{src_j}.conv_proj.bias"))

    # -- prior flow block (reverse) ---------------------------------------
    pf = cfg["prior_encoder_num_flows"]
    wl = cfg["prior_encoder_num_wavenet_layers"]
    for j in range(pf):
        src_j = f"flow.flows.{j}"
        g_j = f"v.flow{j}"
        w, b = conv(f"{src_j}.conv_pre")
        add(f"{g_j}.conv_pre.w", w)
        add(f"{g_j}.conv_pre.b", b)
        w, b = conv(f"{src_j}.conv_post")
        add(f"{g_j}.conv_post.w", w)
        add(f"{g_j}.conv_post.b", b)
        for k in range(wl):
            w, b = conv(f"{src_j}.wavenet.in_layers.{k}")
            add(f"{g_j}.wn.in{k}.w", w)
            add(f"{g_j}.wn.in{k}.b", b)
            w, b = conv(f"{src_j}.wavenet.res_skip_layers.{k}")
            add(f"{g_j}.wn.rs{k}.w", w)
            add(f"{g_j}.wn.rs{k}.b", b)

    # -- HiFi-GAN decoder --------------------------------------------------
    add("v.hifigan.conv_pre.w", f32("decoder.conv_pre.weight"))
    add("v.hifigan.conv_pre.b", f32("decoder.conv_pre.bias"))
    nu = len(cfg["upsample_rates"])
    for i in range(nu):
        add(f"v.up{i}.w", f32(f"decoder.upsampler.{i}.weight"))
        add(f"v.up{i}.b", f32(f"decoder.upsampler.{i}.bias"))
    nk = len(cfg["resblock_kernel_sizes"])
    total_rb = nu * nk
    for r in range(total_rb):
        stage_i, blk_j = divmod(r, nk)
        src_r = f"decoder.resblocks.{r}"
        g_r = f"v.rb{stage_i}{blk_j}"
        ndil = len(cfg["resblock_dilation_sizes"][0])
        for k in range(ndil):
            add(f"{g_r}.c1{k}.w", f32(f"{src_r}.convs1.{k}.weight"))
            add(f"{g_r}.c1{k}.b", f32(f"{src_r}.convs1.{k}.bias"))
            add(f"{g_r}.c2{k}.w", f32(f"{src_r}.convs2.{k}.weight"))
            add(f"{g_r}.c2{k}.b", f32(f"{src_r}.convs2.{k}.bias"))
    add("v.hifigan.conv_post.w", f32("decoder.conv_post.weight"))

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
