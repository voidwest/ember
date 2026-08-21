#!/usr/bin/env python3
"""Compare ember audio validation dumps against the transformers reference.

Usage: python scripts/compare_ultravox.py <ref_dir> <ember_dir>

Reports per-boundary max abs / RMS-relative error and flags the first
divergent boundary (same methodology as compare_smolvlm.py). The
generation-level gate is exact token equality.
"""
import json
import os
import sys

import numpy as np


def load_bin(path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    assert arr.size == int(np.prod(shape)), f"{path}: {arr.size} != {shape}"
    return arr.reshape(shape)


def report(name, ref, got, tol_rel=2e-2):
    """RMS-relative tolerance: |ref-got| / rms(ref) <= tol_rel."""
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    if ref.shape != got.shape:
        print(f"[FAIL] {name}: shape {got.shape} != ref {ref.shape}")
        return False
    diff = np.abs(ref - got)
    rms = float(np.sqrt((ref * ref).mean())) or 1.0
    max_abs = float(diff.max())
    max_rel = max_abs / rms
    ok = max_rel <= tol_rel
    print(
        f"[{'ok ' if ok else 'FAIL'}] {name:24s} max_abs={max_abs:.3e} "
        f"max_abs/rms={max_rel:.3e} (rms={rms:.3f})"
    )
    return ok


def main():
    ref_dir, emb_dir = sys.argv[1], sys.argv[2]
    emb_manifest = json.load(open(os.path.join(emb_dir, "manifest.json")))
    ref_manifest = json.load(open(os.path.join(ref_dir, "manifest.json")))

    all_ok = True

    print("== waveform ==")
    ref = np.load(os.path.join(ref_dir, "0_waveform.npy"))
    got = load_bin(os.path.join(emb_dir, "0_waveform.bin"),
                   tuple(emb_manifest["shapes"]["0_waveform"]))
    d = float(np.abs(ref - got).max()) if ref.shape == got.shape else 9.0
    print(f"[{'ok ' if d <= 2e-5 else 'FAIL'}] {'waveform':24s} max_abs={d:.3e}")
    all_ok &= d <= 2e-5

    def cmp(name, tol_rel=2e-2):
        nonlocal all_ok
        rpath = os.path.join(ref_dir, f"{name}.npy")
        gpath = os.path.join(emb_dir, f"{name}.bin")
        if not os.path.exists(rpath) or not os.path.exists(gpath):
            print(f"[skip] {name}: missing artifact")
            return
        ref = np.load(rpath)
        got = load_bin(gpath, tuple(emb_manifest["shapes"][name]))
        all_ok &= report(name, ref, got, tol_rel)

    cmp("2_mel_features", 2e-4)
    cmp("3_conv1_output")
    for name in sorted(os.listdir(emb_dir)):
        if name.startswith("4_layer_") and name.endswith(".bin"):
            cmp(name.replace(".bin", ""))

    cmp("5_encoder_output")
    cmp("6_projector_output")
    cmp("7_assembled_embeddings")

    print("== first LLM logits ==")
    ref = np.load(os.path.join(ref_dir, "8_first_logits.npy"))
    got = load_bin(os.path.join(emb_dir, "8_first_logits.bin"),
                   tuple(emb_manifest["shapes"]["8_first_logits"]))
    all_ok &= report("8_first_logits", ref, got)

    print("== per-step logits (top-1 agreement) ==")
    ref_steps = np.load(os.path.join(ref_dir, "step_logits.npy"))
    n_steps_ref = ref_steps.shape[0]
    vocab = ref_steps.shape[1]
    got_steps = load_bin(os.path.join(emb_dir, "step_logits.bin"),
                         tuple(emb_manifest["shapes"]["step_logits"]))
    agree = sum(
        int(np.argmax(ref_steps[i]) == np.argmax(got_steps[i]))
        for i in range(min(n_steps_ref, got_steps.shape[0]))
    )
    total = min(n_steps_ref, got_steps.shape[0])
    ok = agree == total
    print(f"[{'ok ' if ok else 'FAIL'}] top-1 agreement: {agree}/{total} steps")
    all_ok &= ok

    print("== generation ==")
    ref_ids = ref_manifest["generation_ids"]
    got_ids = emb_manifest["generation_ids"]
    match = ref_ids[:total] == got_ids[:total]
    print(f"[{'ok ' if match else 'FAIL'}] generation ids match ({total} compared): {match}")
    print(f"ref  generation: {ref_manifest['generated_text']!r}")
    print(f"ember generation: {emb_manifest['generated_text']!r}")
    all_ok &= match

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
