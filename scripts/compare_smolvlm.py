#!/usr/bin/env python3
"""Compare ember multimodal validation dumps against the transformers reference.

Usage: python scripts/compare_smolvlm.py <ref_dir> <ember_dir>

Reports per-boundary max abs / max rel error and flags the first divergent
boundary. Artifacts:
  1_pixels, 2_patch_embeddings, 3_layer_{0,1,5,11}, 4_encoder_output,
  5_projector_output, 6_assembled_embeddings, 7_first_logits,
  8_generation_ids (json in manifest).
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
    """RMS-relative tolerance: |ref-got| / rms(ref) <= tol_rel.

    The reference (torch fp32) and ember use different accumulation orders
    (per-head matmuls, BLAS kernels), so activations drift slowly through
    the stack; absolute tolerances are meaningless across scales. The
    generation-level gate is exact token equality.
    """
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
    print(f"[{'ok ' if ok else 'FAIL'}] {name:24s} max_abs={max_abs:.3e} max_abs/rms={max_rel:.3e} (rms={rms:.3f})")
    return ok


def main():
    ref_dir, emb_dir = sys.argv[1], sys.argv[2]
    emb_manifest = json.load(open(os.path.join(emb_dir, "manifest.json")))
    ref_manifest = json.load(open(os.path.join(ref_dir, "manifest.json")))
    shapes = emb_manifest["shapes"]

    print("== text/tokenization ==")
    print("ref  text:", repr(ref_manifest["text"]))
    print("ember ids len:", emb_manifest["input_ids_len"], "ref:", ref_manifest["input_ids_len"])
    ref_ids = np.load(os.path.join(ref_dir, "input_ids.npy"))
    got_ids = np.array(emb_manifest["input_ids"])
    match = ref_ids.shape[0] == len(got_ids) and bool(np.all(ref_ids == got_ids))
    print(f"[{'ok ' if match else 'FAIL'}] input_ids identical: {match}")
    if not match:
        print("  first divergence at", np.argmax(ref_ids != got_ids) if ref_ids.shape[0] == len(got_ids) else "len")

    print("== pixel tensor ==")
    ref_px = np.load(os.path.join(ref_dir, "1_pixels.npy"))
    got_px = load_bin(os.path.join(emb_dir, "1_pixels.bin"), shapes["1_pixels"])
    report("1_pixels", ref_px, got_px, tol_rel=5e-4)

    print("== vision encoder ==")
    ref_patch = np.load(os.path.join(ref_dir, "2_patch_embeddings.npy"))
    got_patch = load_bin(os.path.join(emb_dir, "2_patch_embeddings.bin"), shapes["2_patch_embeddings"]).reshape(ref_patch.shape)
    report("2_patch_embeddings", ref_patch, got_patch, tol_rel=2e-2)
    for i in [0, 1, 5, 11]:
        ref_l = np.load(os.path.join(ref_dir, f"3_layer_{i}.npy"))
        got_l = load_bin(os.path.join(emb_dir, f"3_layer_{i}.bin"), shapes[f"3_layer_{i}"]).reshape(ref_l.shape)
        report(f"3_layer_{i}", ref_l, got_l)
    ref_enc = np.load(os.path.join(ref_dir, "4_encoder_output.npy"))
    got_enc = load_bin(os.path.join(emb_dir, "4_encoder_output.bin"), shapes["4_encoder_output"]).reshape(ref_enc.shape)
    report("4_encoder_output", ref_enc, got_enc, tol_rel=2e-2)

    print("== connector ==")
    ref_proj = np.load(os.path.join(ref_dir, "5_projector_output.npy"))
    got_proj = load_bin(os.path.join(emb_dir, "5_projector_output.bin"), shapes["5_projector_output"]).reshape(ref_proj.shape)
    report("5_projector_output", ref_proj, got_proj, tol_rel=2e-2)

    print("== assembled LLM input embeddings ==")
    ref_asm = np.load(os.path.join(ref_dir, "6_assembled_embeddings.npy"))
    got_asm = load_bin(os.path.join(emb_dir, "6_assembled_embeddings.bin"), shapes["6_assembled_embeddings"])
    report("6_assembled_embeddings", ref_asm, got_asm, tol_rel=2e-2)

    print("== first LLM logits ==")
    ref_logits = np.load(os.path.join(ref_dir, "7_first_logits.npy"))
    got_logits = load_bin(os.path.join(emb_dir, "7_first_logits.bin"), shapes["7_first_logits"]).reshape(ref_logits.shape)
    report("7_first_logits", ref_logits, got_logits, tol_rel=1e-2)

    print("== per-step logits (top-1 agreement) ==")
    ref_steps = np.load(os.path.join(ref_dir, "step_logits.npy"))
    got_steps = load_bin(os.path.join(emb_dir, "step_logits.bin"), shapes["step_logits"])
    if ref_steps.shape == got_steps.shape:
        agree = [int(np.argmax(ref_steps[s])) == int(np.argmax(got_steps[s])) for s in range(ref_steps.shape[0])]
        print(f"[{'ok ' if all(agree) else 'FAIL'}] top-1 agreement: {sum(agree)}/{len(agree)} steps")
        for s, a in enumerate(agree):
            if not a:
                print(f"  divergent at step {s}")
    else:
        print(f"[FAIL] step_logits shape {got_steps.shape} != ref {ref_steps.shape}")

    print("== generation ==")
    ref_gen = np.load(os.path.join(ref_dir, "8_generation_ids.npy")).tolist()
    got_gen = emb_manifest["generation_ids"]
    n = min(len(ref_gen), len(got_gen))
    same = ref_gen[:n] == got_gen[:n]
    print(f"[{'ok ' if same else 'FAIL'}] generation ids match ({n} compared): {same}")
    if not same:
        for i in range(n):
            if ref_gen[i] != got_gen[i]:
                print(f"  first divergent token at {i}: ref {ref_gen[i]} got {got_gen[i]}")
                break
    print("ref  generation:", repr(ref_manifest["generated_text"]))
    print("ember generation:", repr(emb_manifest["generated_text"]))


if __name__ == "__main__":
    main()
