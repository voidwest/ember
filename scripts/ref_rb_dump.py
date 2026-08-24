#!/usr/bin/env python3
"""Substep reference dumps inside HiFi-GAN residual blocks (rb{stage}{blk}).

Walks decoder.resblocks[i*3+j] manually from the d_up{i} inputs dumped by
ref_decoder_dump.py, dumping per-iteration tensors:
  rb{s}{b}_it{d}_c1       leaky -> convs1[d] output
  rb{s}{b}_it{d}_residual input snapshot
  rb{s}{b}_it{d}_c2presid leaky -> convs2[d] output (pre-residual-add)
  rb{s}{b}_it{d}_out      after residual add

Usage: ref_rb_dump.py <model_dir> <dec_dir_with_d_up_i> <out_dir>
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
    model_dir, dec_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    tok, model = load(model_dir)
    os.makedirs(out_dir, exist_ok=True)

    slope = model.config.leaky_relu_slope
    with torch.no_grad():
        for stage in range(len(model.decoder.upsampler)):
            h = torch.from_numpy(
                np.load(os.path.join(dec_dir, f"d_up{stage}.npy"))
            ).unsqueeze(0).to(torch.float32)
            channels = h.shape[1]
            num_kernels = model.decoder.num_kernels
            for j in range(num_kernels):
                rb = model.decoder.resblocks[stage * num_kernels + j]
                hs = h.clone()
                for d, (conv1, conv2) in enumerate(zip(rb.convs1, rb.convs2)):
                    residual = hs.clone()
                    hs = torch.nn.functional.leaky_relu(hs, rb.leaky_relu_slope)
                    hs = conv1(hs)
                    dump(out_dir, f"rb{stage}{j}_it{d}_c1", hs.squeeze(0))
                    hs = torch.nn.functional.leaky_relu(hs, rb.leaky_relu_slope)
                    hs = conv2(hs)
                    dump(out_dir, f"rb{stage}{j}_it{d}_c2presid", hs.squeeze(0))
                    dump(out_dir, f"rb{stage}{j}_it{d}_residual", residual.squeeze(0))
                    hs = hs + residual
                    dump(out_dir, f"rb{stage}{j}_it{d}_out", hs.squeeze(0))

    print("wrote", out_dir)


if __name__ == "__main__":
    main()
