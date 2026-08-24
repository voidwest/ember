#!/usr/bin/env python3
"""Capture reference MMS-TTS (VITS) inference boundaries from transformers.

Usage:
  python scripts/ref_vits.py <model_dir> "<arabic text>" <out_dir>

<model_dir> must contain model.safetensors + config.json(transformers names)
+ vocab.txt (facebook/mms-tts-ara layout; hf-config.json is accepted as the
config when config.json uses raw VITS names).

The run is made DETERMINISTIC by setting noise_scale = 0.0 and
noise_scale_duration = 0.0 on the loaded model: every stochastic draw then
contributes zero and both engines compute identical math. This mirrors
ember's inference contract (greedy/deterministic synthesis).

Dumped boundaries ([C, T] channel-major float32 .npy unless stated):
  00_input_ids            int64 [T]
  01_embed_scaled         embed * sqrt(H),        [T, H] row-major
  02_encoder_out          after 6 layers,         [T, H]
  03_prior_means          project split,          [T, F]
  04_prior_logvars                                [T, F]
  05_log_duration         SDP reverse output      [1, T]
  06_durations            ceil(exp*mask)          [T] int64
  07_expanded_hidden      monotonic expansion     [S, H]
  08_prior_latents        expanded means (noise=0)[F, S]
  09_flow_z               flow reverse output     [F, S]
  10_waveform             tanh(...)               [1, N]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from transformers import VitsModel, VitsTokenizer


def load(model_dir: str):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(model_dir, "hf-config.json")
    # stage a clean loadable dir
    tmp = os.path.join(model_dir, "_hf_loadable")
    os.makedirs(tmp, exist_ok=True)
    import shutil

    cfg = json.load(open(cfg_path))
    # the published root config lags the weights: trust the checkpoint's
    # embedding rows for vocab_size
    ckpt_vocab = load_file(os.path.join(model_dir, "model.safetensors"))[
        "text_encoder.embed_tokens.weight"
    ].shape[0]
    if cfg.get("vocab_size") != ckpt_vocab:
        print(f"note: config vocab_size {cfg.get('vocab_size')} -> {ckpt_vocab} (from ckpt)")
        cfg["vocab_size"] = int(ckpt_vocab)
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
    # transformers' VitsTokenizer expects the vocab under `vocab.json`
    vocab_src = os.path.join(tmp, "vocab.json")
    if not os.path.exists(vocab_src):
        src_vocab = os.path.join(model_dir, "vocab.txt")
        lines = open(src_vocab, encoding="utf-8").read().splitlines()
        json.dump({tok: i for i, tok in enumerate(lines)}, open(vocab_src, "w"), ensure_ascii=False)
    for extra in ("tokenizer_config.json", "preprocessor_config.json",
                  "model.safetensors", "special_tokens_map.json"):
        p = os.path.join(model_dir, extra)
        if os.path.exists(p) and not os.path.exists(os.path.join(tmp, extra)):
            os.symlink(p, os.path.join(tmp, extra))
    tok = VitsTokenizer.from_pretrained(tmp)
    model = VitsModel.from_pretrained(tmp)
    model.eval()
    # deterministic synthesis contract
    model.noise_scale = 0.0
    model.noise_scale_duration = 0.0
    return tok, model


def main():
    model_dir, text, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    tok, model = load(model_dir)

    inputs = tok(text=text, return_tensors="pt")
    ids = inputs["input_ids"]
    pad = torch.ones_like(ids).unsqueeze(-1).to(torch.float32)

    def dump(name, arr):
        arr = arr.detach().to(torch.float32).numpy()
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
        print(f"  {name}: {arr.shape}")

    with torch.no_grad():
        te = model.text_encoder(input_ids=ids, padding_mask=pad)
        hidden = te.last_hidden_state            # [1, T, H]
        means = te.prior_means                   # [1, T, F]
        logvars = te.prior_log_variances
        dump("00_input_ids", ids.squeeze(0))
        dump(
            "01_embed_scaled",
            model.text_encoder.embed_tokens(ids) * np.sqrt(model.config.hidden_size),
        )
        dump("02_encoder_out", hidden.squeeze(0))
        dump("03_prior_means", means.squeeze(0))
        dump("04_prior_logvars", logvars.squeeze(0))

        hidden_c = hidden.transpose(1, 2)        # [1, H, T]
        pad_c = pad.transpose(1, 2)
        log_duration = model.duration_predictor(
            hidden_c, pad_c, None, reverse=True, noise_scale=model.noise_scale_duration
        )
        print("shapes:", tuple(hidden.shape), tuple(means.shape), tuple(log_duration.shape))
        dump("05_log_duration", log_duration.squeeze(0))

        # ---- exact replication of VitsModel.forward (deterministic) -----
        hidden_c2 = hidden.transpose(1, 2)                    # [1,H,T]
        input_padding_mask = pad_c
        log_d = model.duration_predictor(
            hidden_c2, input_padding_mask, None,
            reverse=True, noise_scale=model.noise_scale_duration,
        )
        length_scale = 1.0 / model.speaking_rate
        duration = torch.ceil(torch.exp(log_d) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()
        dump("06_durations", duration.squeeze())

        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - torch.nn.functional.pad(valid_indices, [0, 0, 1, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        prior_means_x = torch.matmul(attn.squeeze(1), means).transpose(1, 2)
        prior_logvars_x = torch.matmul(attn.squeeze(1), logvars).transpose(1, 2)
        dump("07_expanded_hidden", torch.matmul(attn.squeeze(1), hidden).squeeze(0))
        prior_latents = prior_means_x + torch.randn_like(prior_means_x)             * torch.exp(prior_logvars_x) * model.noise_scale
        dump("08_prior_latents", prior_latents.squeeze(0))

        latents = model.flow(prior_latents, output_padding_mask, None, reverse=True)
        dump("09_flow_z", latents.squeeze(0))

        spectrogram = latents * output_padding_mask
        waveform = model.decoder(spectrogram, None)
        dump("10_waveform", waveform.squeeze(0))

    manifest = {
        "text": text,
        "noise_scale": model.noise_scale,
        "noise_scale_duration": model.noise_scale_duration,
        "speaking_rate": model.speaking_rate,
        "n_mel_frames": int(predicted_lengths.item()),
        "waveform_samples": int(waveform.shape[-1]),
        "sample_rate": model.config.sampling_rate,
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
