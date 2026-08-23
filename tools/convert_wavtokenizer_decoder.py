#!/usr/bin/env python3
"""Convert the WavTokenizer (Vocos/ISTFT) speech DECODER from the OuteAI
interface checkpoint into ember's codec GGUF layout.

Usage:
    python tools/convert_wavtokenizer_decoder.py <decoder_model.pt> <out.codec.gguf>

Source: OuteAI/wavtokenizer-large-75token-interface (`decoder_model.pt`,
torch.save dict with `model_state_dict` + `codebook_weights`). The encoder
is NOT converted: speech synthesis only ever decodes codec tokens.

Only the decoder side of WavTokenizer is used by ember's TTS path
(OuteTTS emits codec tokens; this GGUF turns tokens into 24 kHz PCM):

```text
codes [T] -> codebook lookup [512, T]
          -> embed Conv1d(512->768, k7 p3)
          -> pos_net: ResnetBlock x2 -> time attention -> ResnetBlock x2
             -> GroupNorm(32)
          -> AdaLayerNorm(bandwidth_id=0)
          -> 12 x ConvNeXt block (dwconv k7, AdaLayerNorm, 768<->2304)
          -> LayerNorm -> Linear(768->1282) -> mag*exp(i*phase)
          -> iSTFT (n_fft 1280, hop 320, hann, "same" trim) -> PCM
```

metadata:
  general.architecture   = "wavtokenizer-decoder"
  wt.sample_rate         = 24000
  wt.n_fft               = 1280
  wt.hop_length          = 320
  wt.codebook_bins       = 4096
  wt.latent_dim          = 512
  wt.dim                 = 768
  wt.intermediate_dim    = 2304
  wt.convnext_layers     = 12
  wt.group_norm_groups   = 32
  wt.group_norm_eps      = 1e-6
  wt.layer_norm_eps      = 1e-6

tensors (prefix `w.`, payloads written unchanged in torch layout, f32):
  w.codebook                        [4096, 512]
  w.embed.weight/.bias              Conv1d [768, 512, 7]
  w.pos_net.{0,1,3,4}.norm1/conv1/norm2/conv2...
  w.pos_net.2.norm/q/k/v/proj_out...  (1x1 convs = linears over channels)
  w.pos_net.5.weight/.bias            final GroupNorm affine
  w.adanorm.scale/.shift             [4, 768] per-bandwidth conditioning
  w.convnext.{i}.dwconv.weight/.bias [768, 1, 7] depthwise
  w.convnext.{i}.norm.scale/.shift   AdaLayerNorm params for the block
  w.convnext.{i}.pwconv1/pwconv2...  [2304, 768] / [768, 2304]
  w.convnext.{i}.gamma               [768] layer scale
  w.final_layer_norm.weight/.bias
  w.head.out.weight/.bias            [1282, 768]
  w.window                           [1280] hann (copied verbatim)
"""
import sys

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "gguf-py")
import gguf  # noqa: E402


def main() -> None:
    src, out_path = sys.argv[1], sys.argv[2]

    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    msd = ckpt["model_state_dict"]
    codebook = ckpt["codebook_weights"].contiguous().to(torch.float32).numpy()

    def f32(name: str) -> np.ndarray:
        return msd[name].contiguous().to(torch.float32).numpy()

    dim = f32("backbone.embed.bias").shape[0]
    latent = f32("backbone.embed.weight").shape[1]
    n_fft = f32("head.out.weight").shape[0] - 2
    intermediate = f32("backbone.convnext.0.pwconv1.weight").shape[0]
    convnext_layers = 1 + max(
        int(k.split(".")[2]) for k in msd if k.startswith("backbone.convnext.")
    )
    adanorm_bands = f32("backbone.norm.scale.weight").shape[0]

    out = gguf.GGUFWriter(out_path, "wavtokenizer-decoder")
    out.add_uint32("wt.sample_rate", 24000)
    out.add_uint32("wt.n_fft", n_fft)
    out.add_uint32("wt.hop_length", 320)
    out.add_uint32("wt.codebook_bins", codebook.shape[0])
    out.add_uint32("wt.latent_dim", latent)
    out.add_uint32("wt.dim", dim)
    out.add_uint32("wt.intermediate_dim", intermediate)
    out.add_uint32("wt.convnext_layers", convnext_layers)
    out.add_uint32("wt.group_norm_groups", 32)
    out.add_float32("wt.group_norm_eps", 1e-6)
    out.add_float32("wt.layer_norm_eps", 1e-6)
    out.add_uint32("wt.adanorm_bands", adanorm_bands)

    def add(name_gguf: str, arr: np.ndarray) -> None:
        out.add_tensor(name_gguf, arr)

    add("w.codebook", codebook)

    add("w.embed.weight", f32("backbone.embed.weight"))
    add("w.embed.bias", f32("backbone.embed.bias"))

    # pos_net blocks: 0/1/3/4 resnet, 2 attention, 5 group norm
    for i in (0, 1, 3, 4):
        p = f"backbone.pos_net.{i}"
        g = f"w.pos_net.{i}"
        add(f"{g}.norm1.weight", f32(f"{p}.norm1.weight"))
        add(f"{g}.norm1.bias", f32(f"{p}.norm1.bias"))
        add(f"{g}.conv1.weight", f32(f"{p}.conv1.weight"))
        add(f"{g}.conv1.bias", f32(f"{p}.conv1.bias"))
        add(f"{g}.norm2.weight", f32(f"{p}.norm2.weight"))
        add(f"{g}.norm2.bias", f32(f"{p}.norm2.bias"))
        add(f"{g}.conv2.weight", f32(f"{p}.conv2.weight"))
        add(f"{g}.conv2.bias", f32(f"{p}.conv2.bias"))

    p_attn = "backbone.pos_net.2"
    g_attn = "w.pos_net.2"
    for name in ("norm", "q", "k", "v", "proj_out"):
        add(f"{g_attn}.{name}.weight", f32(f"{p_attn}.{name}.weight"))
        add(f"{g_attn}.{name}.bias", f32(f"{p_attn}.{name}.bias"))

    add("w.pos_net.5.weight", f32("backbone.pos_net.5.weight"))
    add("w.pos_net.5.bias", f32("backbone.pos_net.5.bias"))

    # backbone-level AdaLayerNorm (pre-ConvNeXt conditioning)
    add("w.adanorm.scale", f32("backbone.norm.scale.weight"))
    add("w.adanorm.shift", f32("backbone.norm.shift.weight"))

    for i in range(convnext_layers):
        p = f"backbone.convnext.{i}"
        g = f"w.convnext.{i}"
        add(f"{g}.dwconv.weight", f32(f"{p}.dwconv.weight"))
        add(f"{g}.dwconv.bias", f32(f"{p}.dwconv.bias"))
        add(f"{g}.norm.scale", f32(f"{p}.norm.scale.weight"))
        add(f"{g}.norm.shift", f32(f"{p}.norm.shift.weight"))
        add(f"{g}.pwconv1.weight", f32(f"{p}.pwconv1.weight"))
        add(f"{g}.pwconv1.bias", f32(f"{p}.pwconv1.bias"))
        add(f"{g}.pwconv2.weight", f32(f"{p}.pwconv2.weight"))
        add(f"{g}.pwconv2.bias", f32(f"{p}.pwconv2.bias"))
        add(f"{g}.gamma", f32(f"{p}.gamma"))

    add("w.final_layer_norm.weight", f32("backbone.final_layer_norm.weight"))
    add("w.final_layer_norm.bias", f32("backbone.final_layer_norm.bias"))

    add("w.head.out.weight", f32("head.out.weight"))
    add("w.head.out.bias", f32("head.out.bias"))
    add("w.window", f32("head.istft.window"))

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
