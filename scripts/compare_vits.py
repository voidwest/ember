#!/usr/bin/env python3
"""Compare ember MMS-VITS dumps against scripts/ref_vits.py reference."""
import sys
import numpy as np
import os


def load(d, name):
    p = os.path.join(d, f"{name}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


def main():
    ref_dir, emb_dir = sys.argv[1], sys.argv[2]
    names = [
        "01_embed_scaled",
        "02_encoder_out",
        "03_prior_means",
        "05_log_duration",
        "06_durations",
        "07_expanded_hidden",
        "08_prior_latents",
        "09_flow_z",
        "10_waveform",
    ]
    all_ok = True
    for n in names:
        r = load(ref_dir, n)
        e = load(emb_dir, n)
        if r is None or e is None:
            print(f"[skip] {n}: missing ({r is None}/{e is None})")
            continue
        if r.shape != e.shape:
            # ember ladder dumps are flat; reshape to the reference layout
            if e.size == r.size:
                e = e.reshape(r.shape)
            else:
                print(f"[FAIL] {n}: shape {e.shape} != ref {r.shape}")
                all_ok = False
                continue
        diff = np.abs(r.astype(np.float64) - e.astype(np.float64))
        rms = float(np.sqrt((r.astype(np.float64) ** 2).mean())) or 1.0
        max_abs = float(diff.max())
        rel = float(np.sqrt((diff**2).mean()) / rms)
        ok = rel <= 2e-2 or max_abs <= 1e-3
        print(
            f"[{'ok ' if ok else 'FAIL'}] {n:22s} max_abs={max_abs:.3e} "
            f"rms_rel={rel:.3e}"
        )
        all_ok &= ok
    print("ALL WITHIN GATES" if all_ok else "DIVERGENCE PRESENT")


if __name__ == "__main__":
    main()
