#!/usr/bin/env python3
"""Compare ember's wavtokenizer decoder dumps against the reference.

Usage:
    python scripts/compare_wavtokenizer.py <ref_dir> <ember_dir>

Both directories hold {codes,0_features,1_embed,2_posnet,3_adanorm,
4_convnext_{i},5_backbone_final,6_mag,6_phase,7_waveform}_{n}.bin/.npy
(ember side is .bin f32 little-endian; shapes come from the reference .npy).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = len(data) // 4
    return np.array(struct.unpack(f"<{n}f", data), dtype=np.float32)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    diff = a.astype(np.float64) - b.astype(np.float64)
    max_abs = float(np.abs(diff).max()) if diff.size else 0.0
    rms_a = float(np.sqrt((a.astype(np.float64) ** 2).mean())) or 1e-30
    rms_diff = float(np.sqrt((diff**2).mean()))
    rms_rel = rms_diff / rms_a
    cos = 1.0
    if diff.size:
        na = float(np.linalg.norm(a.astype(np.float64))) or 1e-30
        nb = float(np.linalg.norm(b.astype(np.float64))) or 1e-30
        cos = float((a.astype(np.float64) * b.astype(np.float64)).sum() / (na * nb))
    return {"max_abs": max_abs, "rms_rel": rms_rel, "cos": cos}


def main() -> None:
    ref_dir, ember_dir = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    lengths = sorted(manifest["lengths"], key=int)
    overall_ok = True

    for tag in lengths:
        print(f"=== token length {tag} ===")
        codes_ref = np.load(ref_dir / f"codes_{tag}.npy")
        codes_emb = load_bin(ember_dir / f"codes_{tag}.bin")
        assert (
            codes_ref == codes_emb[: len(codes_ref)]
        ).all(), "code sequences differ — LCG mismatch"

        pairs = [
            ("0_features", "0_features"),
            ("1_embed", "1_embed"),
            ("2_posnet", "2_posnet"),
            ("3_adanorm", "3_adanorm"),
            ("5_backbone_final", "5_backbone_final"),
            ("6_mag", "6_mag"),
            ("6_phase", "6_phase"),
            ("7_waveform", "7_waveform"),
        ]
        # convnext traced blocks present in BOTH dirs (stems carry _tag)
        import re
        for p in sorted(ref_dir.glob(f"4_convnext_*_{tag}.npy")):
            m = re.fullmatch(rf"4_convnext_(\d+)_{tag}", p.stem)
            if m and (ember_dir / f"{p.stem}.bin").exists():
                pairs.append((f"4_convnext_{m.group(1)}", f"4_convnext_{m.group(1)}"))
                # mark as full-stem pair via a third element convention
                pairs[-1] = (p.stem, p.stem)

        for ref_name, emb_name in pairs:
            is_full_stem = ref_name.startswith("4_convnext_")
            ref_base = ref_name if is_full_stem else f"{ref_name}_{tag}"
            emb_base = emb_name if is_full_stem else f"{emb_name}_{tag}"
            ref = np.load(ref_dir / f"{ref_base}.npy")
            emb = load_bin(ember_dir / f"{emb_base}.bin")
            assert emb.size == ref.size, (
                f"{ref_name}: ember has {emb.size} elements, ref {ref.size}"
            )
            emb = emb[: ref.size].reshape(ref.shape)
            m = metrics(ref, emb)
            ok = m["rms_rel"] <= 2e-2 or m["max_abs"] <= 2e-3 * max(
                1.0, float(np.abs(ref).max())
            )
            # waveform gets its own stricter sanity: correlation
            extra = ""
            if ref_name.startswith("7_"):
                extra = f" peak_ref={np.abs(ref).max():.3f} peak_emb={np.abs(emb).max():.3f}"
                ok = ok and m["cos"] > 0.999
            flag = "ok " if ok else "FAIL"
            overall_ok &= ok
            print(
                f"  [{flag}] {ref_name:22s} max_abs={m['max_abs']:.3e} "
                f"rms_rel={m['rms_rel']:.3e} cos={m['cos']:.6f}{extra}"
            )

    print("ALL BOUNDARIES WITHIN GATES" if overall_ok else "GATE FAILURES PRESENT")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
