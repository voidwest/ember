#!/usr/bin/env python3
"""Reference capture for the WavTokenizer DECODER ladder (Track E5).

Loads the OuteAI wavtokenizer-large-75token-interface decoder (the exact
checkpoint OuteTTS uses), decodes deterministic code sequences, and dumps
every meaningful boundary to .npy for comparison against ember's
`tts::wavtokenizer::WavTokenizerDecoder`.

The reference modules are used UNMODIFIED from the outetts wheel
(decoder/{modules,spectral_ops,heads,models}.py, imported as a top-level
`decoder` package exactly as their dynamic loading expects); only the
package `__init__` (which drags in loguru/llama-cpp/etc.) is bypassed.

Boundaries captured (per code length):
  codes_{n}         [T]        input codec tokens (documented LCG)
  0_features_{n}    [512, T]   codebook lookup output
  1_embed_{n}       [768, T]   after embed Conv1d
  2_posnet_{n}      [768, T]   pos_net output incl. final GroupNorm
  3_adanorm_{n}     [T, 768]   after backbone AdaLayerNorm
  4_convnext_{i}_{n}[768, T]   ConvNeXt blocks i = 0, mid, last
  5_backbone_final_{n}[T, 768] final LayerNorm output
  6_mag/6_phase_{n} [641, T]
  7_waveform_{n}    [S]        float32 PCM at 24 kHz

Codes come from an explicit documented LCG so Rust reproduces them:
    s = s * 6364136223846793005 + 1442695040888963407 (mod 2**64)
    code = ((s >> 33) % 4096)

Usage:
    python scripts/ref_wavtokenizer.py <decoder_dir> <out_dir> [--tokens N ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
WT_ROOT = os.environ.get(
    "OUTETTS_WHEEL_ROOT", "/tmp/opencode/.ref_outetts/outetts/wav_tokenizer"
)
sys.path.insert(0, WT_ROOT)

# heads.py imports two private torchaudio helpers only used by the IMDCT
# head variants we do not instantiate; stub them so no torchaudio needed.
if "torchaudio" not in sys.modules:
    try:
        import torchaudio  # noqa: F401
    except ModuleNotFoundError:
        import types

        ta = types.ModuleType("torchaudio")
        fn = types.ModuleType("torchaudio.functional")
        ff = types.ModuleType("torchaudio.functional.functional")

        def _hz_to_mel(*_a, **_k):
            raise NotImplementedError("stub: unused by ISTFTHead")

        _mel_to_hz = _hz_to_mel
        ff._hz_to_mel = _hz_to_mel
        ff._mel_to_hz = _mel_to_hz
        ta.functional = fn
        fn.functional = ff
        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.functional"] = fn
        sys.modules["torchaudio.functional.functional"] = ff

from decoder.models import VocosBackbone  # noqa: E402
from decoder.heads import ISTFTHead  # noqa: E402


class WavDecoder(nn.Module):
    """Decoder-only shim identical to outetts wav_tokenizer/model.py."""

    def __init__(self, backbone, head, codebook_weights):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.register_buffer("codebook_weights", codebook_weights)

    def codes_to_features(self, codes):
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)
        n_bins = self.codebook_weights.size(0) // len(codes)
        offsets = torch.arange(0, n_bins * len(codes), n_bins)
        idx = codes + offsets.view(-1, 1, 1)
        feats = torch.nn.functional.embedding(idx, self.codebook_weights).sum(dim=0)
        return feats.transpose(1, 2)

    def forward(self, features_input, bandwidth_id=None):
        x = self.backbone(features_input, bandwidth_id=bandwidth_id)
        return self.head(x)


def lcg_codes(n_tokens: int, seed: int) -> list:
    s = seed & 0xFFFFFFFFFFFFFFFF
    codes = []
    for _ in range(n_tokens):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        codes.append((s >> 33) % 4096)
    return codes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("decoder_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--tokens", type=int, nargs="+", default=[37, 150, 750])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.decoder_dir, "config.json")) as f:
        cfg = json.load(f)
    backbone = VocosBackbone(**cfg["backbone_config"])
    head = ISTFTHead(**cfg["head_config"])
    ckpt = torch.load(
        os.path.join(args.decoder_dir, "decoder_model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model = WavDecoder(backbone, head, ckpt["codebook_weights"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("reference decoder loaded:", sum(p.numel() for p in model.parameters()) / 1e6, "M params")

    manifest = {"lengths": {}, "bandwidth_id": 0}
    for n_tokens in args.tokens:
        codes = torch.tensor([lcg_codes(n_tokens, seed=n_tokens)], dtype=torch.int64)
        bandwidth_id = torch.tensor([0])
        tag = str(n_tokens)
        with torch.inference_mode():
            feats = model.codes_to_features(codes)
            emb = model.backbone.embed(feats)
            # step through pos_net capturing every sub-stage
            bn = model.backbone
            h = emb
            for ri, blk in enumerate(bn.pos_net):
                try:
                    h = blk(h, cond_embedding_id=bandwidth_id)
                except TypeError:
                    h = blk(h)
                np.save(f"{args.out_dir}/pn_{ri}_{tag}.npy", h.squeeze(0).numpy())
            posnet = h.clone()
            h = model.backbone.norm(h.transpose(1, 2), cond_embedding_id=bandwidth_id)
            adanorm = h.clone()
            h = h.transpose(1, 2)
            traced_blocks = {}
            for bi, blk in enumerate(model.backbone.convnext):
                h = blk(h, cond_embedding_id=bandwidth_id)
                if bi in (0, len(model.backbone.convnext) // 2, len(model.backbone.convnext) - 1):
                    traced_blocks[bi] = h.clone()
            xf = model.backbone.final_layer_norm(h.transpose(1, 2))
            head_out = model.head.out(xf)
            mag_raw, p = head_out.chunk(2, dim=2)
            mag = torch.clip(torch.exp(mag_raw), max=1e2)
            S = mag * (torch.cos(p) + 1j * torch.sin(p))
            audio = model.head.istft(S.transpose(1, 2)).squeeze().numpy()

        np.save(f"{args.out_dir}/codes_{tag}.npy", codes.numpy()[0])
        np.save(f"{args.out_dir}/0_features_{tag}.npy", feats.squeeze(0).numpy())
        np.save(f"{args.out_dir}/1_embed_{tag}.npy", emb.squeeze(0).numpy())
        np.save(f"{args.out_dir}/2_posnet_{tag}.npy", posnet.squeeze(0).numpy())
        np.save(f"{args.out_dir}/3_adanorm_{tag}.npy", adanorm.squeeze(0).numpy())
        for bi, t in traced_blocks.items():
            np.save(f"{args.out_dir}/4_convnext_{bi}_{tag}.npy", t.squeeze(0).numpy())
        np.save(f"{args.out_dir}/5_backbone_final_{tag}.npy", xf.squeeze(0).numpy())
        np.save(f"{args.out_dir}/6_mag_{tag}.npy", mag.squeeze(0).permute(1, 0).numpy())
        np.save(f"{args.out_dir}/6_phase_{tag}.npy", p.squeeze(0).permute(1, 0).numpy())
        np.save(f"{args.out_dir}/7_waveform_{tag}.npy", audio.astype(np.float32))
        manifest["lengths"][tag] = {
            "samples": int(audio.shape[0]),
            "frames": int(codes.shape[1]),
        }
        print(f"[{tag}] {n_tokens} tokens -> {audio.shape[0]} samples @24kHz")

    manifest["lcg"] = "s = s*6364136223846793005 + 1442695040888963407 mod 2^64; code=(s>>33)%4096"
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote reference dumps to {args.out_dir}")


if __name__ == "__main__":
    main()
