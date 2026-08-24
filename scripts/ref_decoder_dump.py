#!/usr/bin/env python3
"""Per-substage reference dumps of the MMS-VITS HiFi-GAN decoder.

Replicates ref_vits.py deterministically (noise_scale=0) and walks
model.decoder manually, dumping every substage to <out_dir>/dec/.

Dumps ([C, T] float32 .npy):
  d_convpre      conv_pre(spectrogram)
  d_up{i}        after upsampler[i]
  d_rb{i}{j}     resblocks[i*3+j] output (pre-average)
  d_stage{i}     after block average i
  d_prepost      final leaky_relu
  d_pretanh      conv_post output (no bias)
  10_waveform    tanh output [1, N]
"""
import json
import os
import sys

import numpy as np
import torch

from transformers import VitsModel, VitsTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_vits import load  # noqa: E402


def dump(root, name, arr):
    arr = arr.detach().to(torch.float32).cpu().numpy()
    np.save(os.path.join(root, name + ".npy"), arr)


def main():
    model_dir, text, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    dec_dir = os.path.join(out_dir, "dec")
    os.makedirs(dec_dir, exist_ok=True)
    tok, model = load(model_dir)

    inputs = tok(text=text, return_tensors="pt")
    ids = inputs["input_ids"]
    pad = torch.ones_like(ids).unsqueeze(-1).to(torch.float32)

    with torch.no_grad():
        te = model.text_encoder(input_ids=ids, padding_mask=pad)
        hidden = te.last_hidden_state
        means = te.prior_means
        logvars = te.prior_log_variances

        log_d = model.duration_predictor(
            hidden.transpose(1, 2), pad.transpose(1, 2), None, reverse=True,
            noise_scale=model.noise_scale_duration,
        )
        duration = torch.ceil(torch.exp(log_d) * pad.transpose(1, 2))
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()
        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype)
        output_padding_mask = (indices.unsqueeze(0) < predicted_lengths.unsqueeze(1))
        output_padding_mask = output_padding_mask.unsqueeze(1).to(pad.dtype)

        attn_mask = torch.unsqueeze(pad.transpose(1, 2), 2) * torch.unsqueeze(output_padding_mask, -1)
        b, _, ol, il = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(b * il, 1)
        indices = torch.arange(ol, dtype=duration.dtype)
        valid = (indices.unsqueeze(0) < cum_duration).to(attn_mask.dtype).view(b, il, ol)
        padded = valid - torch.nn.functional.pad(valid, [0, 0, 1, 0])[:, :-1]
        attn = padded.unsqueeze(1).transpose(2, 3) * attn_mask

        prior_means_x = torch.matmul(attn.squeeze(1), means).transpose(1, 2)
        prior_latents = prior_means_x  # noise_scale = 0

        latents = model.flow(prior_latents, output_padding_mask, None, reverse=True)
        spectrogram = latents * output_padding_mask
        dump(dec_dir, "d_spec", spectrogram.squeeze(0))

        dec = model.decoder
        h = dec.conv_pre(spectrogram)
        dump(dec_dir, "d_convpre", h.squeeze(0))
        slope = model.config.leaky_relu_slope
        for i in range(dec.num_upsamples):
            h = torch.nn.functional.leaky_relu(h, slope)
            h = dec.upsampler[i](h)
            dump(dec_dir, f"d_up{i}", h.squeeze(0))
            res_state = dec.resblocks[i * dec.num_kernels](h)
            dump(dec_dir, f"d_rb{i}0", res_state.squeeze(0))
            for j in range(1, dec.num_kernels):
                res_state = res_state + dec.resblocks[i * dec.num_kernels + j](h)
                dump(dec_dir, f"d_rb{i}{j}", res_state.squeeze(0))
            h = res_state / dec.num_kernels
            dump(dec_dir, f"d_stage{i}", h.squeeze(0))
        h = torch.nn.functional.leaky_relu(h)
        dump(dec_dir, "d_prepost", h.squeeze(0))
        h = dec.conv_post(h)
        dump(dec_dir, "d_pretanh", h.squeeze(0))
        wave = torch.tanh(h)
        dump(dec_dir, "d_tanh", wave.squeeze(0))

    manifest = {"frames": int(predicted_lengths.item()), "samples": int(wave.shape[-1])}
    json.dump(manifest, open(os.path.join(dec_dir, "manifest.json"), "w"))
    print("wrote", dec_dir)


if __name__ == "__main__":
    main()
