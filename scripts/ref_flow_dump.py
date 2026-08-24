#!/usr/bin/env python3
"""Per-substage reference dumps of the MMS-VITS prior flow stack (reverse).

Replicates scripts/ref_vits.py up to 08_prior_latents, then steps through
model.flow (VitsResidualCouplingBlock) manually, dumping every meaningful
substage of every ResidualCouplingLayer to <out_dir>/flow/.

All tensor dumps are channel-major [C, T] float32 .npy, named:
  f{j}_in          input to coupling layer j (post-flip)
  f{j}_x0 / _x1    channel split halves [96, T]
  f{j}_convpre     conv_pre(x0)                       [192, T]
  f{j}_wn{i}_h     wavenet.in_layers[i](cur)          [384, T]
  f{j}_wn{i}_acts  tanh*sigmoid gate                  [192, T]
  f{j}_wn{i}_rs    res_skip_layers[i](acts)           [384 or 192, T]
  f{j}_wn{i}_cur   cur after residual add             [192, T]
  f{j}_wnout       wavenet outputs accumulator        [192, T]
  f{j}_mean        conv_post(wnout)                   [96, T]
  f{j}_x1new       x1 - mean                          [96, T]
  f{j}_out         concat(x0, x1new)                  [192, T]
  f{j}_flip        channel-flipped output             [192, T]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from transformers import VitsModel, VitsTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_vits import load  # noqa: E402


def dump(root, name, arr):
    arr = arr.detach().to(torch.float32).cpu().numpy()
    np.save(os.path.join(root, name + ".npy"), arr)


def main():
    model_dir, text, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    flow_dir = os.path.join(out_dir, "flow")
    os.makedirs(flow_dir, exist_ok=True)
    tok, model = load(model_dir)

    inputs = tok(text=text, return_tensors="pt")
    ids = inputs["input_ids"]
    pad = torch.ones_like(ids).unsqueeze(-1).to(torch.float32)

    with torch.no_grad():
        te = model.text_encoder(input_ids=ids, padding_mask=pad)
        hidden = te.last_hidden_state
        means = te.prior_means
        logvars = te.prior_log_variances

        hidden_c = hidden.transpose(1, 2)
        log_d = model.duration_predictor(
            hidden_c, pad.transpose(1, 2), None, reverse=True,
            noise_scale=model.noise_scale_duration,
        )
        length_scale = 1.0 / model.speaking_rate
        duration = torch.ceil(torch.exp(log_d) * pad.transpose(1, 2) * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(pad.dtype)

        attn_mask = torch.unsqueeze(pad.transpose(1, 2), 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - torch.nn.functional.pad(valid_indices, [0, 0, 1, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        prior_means_x = torch.matmul(attn.squeeze(1), means).transpose(1, 2)
        prior_latents = prior_means_x  # noise_scale = 0

        # ---- manual walk through the flow stack --------------------------
        x = prior_latents
        mask = output_padding_mask
        wn_layers = len(model.flow.flows[0].wavenet.in_layers)
        for flow in reversed(model.flow.flows):
            x = torch.flip(x, [1])
            first_half, second_half = torch.split(x, [flow.half_channels] * 2, dim=1)
            h_states = flow.conv_pre(first_half) * mask
            j = model.flow.flows.index(flow) if False else None  # name below
            break
        # simpler explicit loop with names by module position
        flows = list(model.flow.flows)
        x = prior_latents
        for pos in range(len(flows) - 1, -1, -1):
            flow = flows[pos]
            x = torch.flip(x, [1])
            dump(flow_dir, f"f{pos}_in", x.squeeze(0))
            first_half, second_half = torch.split(x, [flow.half_channels] * 2, dim=1)
            dump(flow_dir, f"f{pos}_x0", first_half.squeeze(0))
            dump(flow_dir, f"f{pos}_x1", second_half.squeeze(0))

            h_states = flow.conv_pre(first_half) * mask
            dump(flow_dir, f"f{pos}_convpre", h_states.squeeze(0))

            wn = flow.wavenet
            cur = h_states
            outputs = torch.zeros_like(cur)
            for i in range(wn.num_layers):
                h = wn.in_layers[i](cur)
                dump(flow_dir, f"f{pos}_wn{i}_h", h.squeeze(0))
                zeros = torch.zeros_like(h)
                acts = fused(wn.hidden_size if hasattr(wn, "hidden_size") else 192, h, zeros)
                dump(flow_dir, f"f{pos}_wn{i}_acts", acts.squeeze(0))
                rs = wn.res_skip_layers[i](acts)
                dump(flow_dir, f"f{pos}_wn{i}_rs", rs.squeeze(0))
                if i < wn.num_layers - 1:
                    cur = (cur + rs[:, : wn.hidden_size, :]) * mask
                    outputs = outputs + rs[:, wn.hidden_size :, :]
                else:
                    outputs = outputs + rs
                dump(flow_dir, f"f{pos}_wn{i}_cur", cur.squeeze(0))
                dump(flow_dir, f"f{pos}_wn{i}_acc", outputs.squeeze(0))
            mean = flow.conv_post(outputs) * mask
            dump(flow_dir, f"f{pos}_mean", mean.squeeze(0))
            x1new = (second_half - mean) * mask
            dump(flow_dir, f"f{pos}_x1new", x1new.squeeze(0))
            x = torch.cat([first_half, x1new], dim=1)
            dump(flow_dir, f"f{pos}_out", x.squeeze(0))
            dump(flow_dir, f"f{pos}_flip", torch.flip(x, [1]).squeeze(0))

        dump(out_dir, "09_flow_z_manual", x.squeeze(0))

    manifest = {"text": text, "frames": int(predicted_lengths.item())}
    json.dump(manifest, open(os.path.join(flow_dir, "manifest.json"), "w"))
    print("wrote", flow_dir)


def fused(num_channels, a, b):
    from transformers.models.vits.modeling_vits import fused_add_tanh_sigmoid_multiply
    n = torch.IntTensor([num_channels])
    return fused_add_tanh_sigmoid_multiply(a, b, n[0])


if __name__ == "__main__":
    main()
