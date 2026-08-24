#!/usr/bin/env python3
"""Build the Phase 5 Session 2 Arabic speech validation bank (Track B1).

Sources — chosen for legal reproducibility and small size:

* google/fleurs test split, configs ar_sa / ar_eg / ar_ae (CC-BY-4.0).
  Read MSA sentences with regional speaker accents; transcripts included.
* Derived conditions generated deterministically from those clips:
  - quiet   : gain 0.25 (-12 dB)
  - noisy   : + white noise at SNR 10 dB
  - rate48k : linear resample to 48 kHz (device-rate path exercise)
  - long    : same-speaker concatenation to >30 s
  - codeswitch: EN segment (jfk.wav) + AR segment concatenated

Everything lands in research/banks/arabic_speech_001/ with a manifest.json
recording provenance, license, transcript and sha256 per entry. The bank is
gitignored like all research data.
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf  # noqa: F401  (via librosa dependency of datasets)

ROOT = Path("/home/west/ember")
BANK = ROOT / "research" / "banks" / "arabic_speech_001"
JFK = Path("/home/west/luminal/examples/whisper/assets/jfk.wav")

TARGETS = [
    # FLEURS publishes a single Arabic config (Egyptian-accented read MSA);
    # dialect breadth beyond this is recorded honestly in the manifest.
    ("ar_eg", 12),
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_entry(entries, name, pcm, sr, meta):
    import wave

    path = BANK / f"{name}.wav"
    assert pcm.dtype == np.float32 or pcm.dtype == np.float64
    data = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)
    entry = {
        "id": name,
        "file": path.name,
        "sample_rate": sr,
        "samples": int(len(pcm)),
        "seconds": round(len(pcm) / sr, 3),
        "sha256_wav": sha256(path.read_bytes()),
        **meta,
    }
    entries.append(entry)
    print(f"  {name}: {entry['seconds']}s @ {sr}")


def main():
    from datasets import load_dataset

    BANK.mkdir(parents=True, exist_ok=True)
    entries = []

    for config, n in TARGETS:
        ds = load_dataset("google/fleurs", config, split="test", streaming=True)
        got = 0
        for i, row in enumerate(ds):
            if got >= n:
                break
            audio = row["audio"]
            pcm = audio["array"].astype(np.float32)
            if np.abs(pcm).max() < 1e-4 or len(pcm) < 8000:
                continue
            pcm = pcm / max(1.0, np.abs(pcm).max()) * 0.9
            name = f"{config}_test_{i:04d}"
            write_entry(
                entries,
                name,
                pcm,
                audio["sampling_rate"],
                {
                    "source": f"google/fleurs {config} test[{i}]",
                    "license": "CC-BY-4.0",
                    "transcript": row.get("transcription", ""),
                    "transcript_raw": row.get("raw_transcription", ""),
                    "condition": "clean",
                    "language": "ar",
                    "dialect_tag": config,
                },
            )
            got += 1
        assert got == n, f"{config}: only {got}/{n} clips"

    # ---- derived conditions over a deterministic pick -------------------
    base = [e for e in entries if e["condition"] == "clean"]
    rng = np.random.default_rng(20260823)

    def read(entry):
        import wave

        with wave.open(str(BANK / entry["file"]), "rb") as w:
            assert w.getframerate() == entry["sample_rate"]
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def linear_resample(x, sr_from, sr_to):
        t_in = np.arange(len(x)) / sr_from
        n_out = int(round(len(x) * sr_to / sr_from))
        t_out = np.arange(n_out) / sr_to
        return np.interp(t_out, t_in, x).astype(np.float32)

    # quiet + noisy versions of two clips each
    for tag, gain in (("quiet", 0.25),):
        for e in [base[0], base[4]]:
            x = read(e) * gain
            write_entry(
                entries,
                f"{e['id']}_{tag}",
                x,
                e["sample_rate"],
                {**{k: e[k] for k in ("source", "license", "transcript", "dialect_tag")},
                 "condition": tag,
                 "derived_from": e["id"],
                 "language": "ar"},
            )
    for e in [base[1], base[5]]:
        x = read(e)
        snr_db = 10.0
        sig_pw = float(np.mean(x**2))
        noise = rng.standard_normal(len(x)).astype(np.float32)
        noise *= np.sqrt(sig_pw / 10 ** (snr_db / 10) / float(np.mean(noise**2)))
        write_entry(
            entries,
            f"{e['id']}_noisy",
            x + noise,
            e["sample_rate"],
            {**{k: e[k] for k in ("source", "license", "transcript", "dialect_tag")},
             "condition": "noisy",
             "derived_from": e["id"],
             "snr_db": snr_db,
             "language": "ar"},
        )

    # device-rate exercise: resample two clips to 48 kHz
    for e in [base[2], base[6]]:
        x = read(e)
        write_entry(
            entries,
            f"{e['id']}_rate48k",
            linear_resample(x, e["sample_rate"], 48_000),
            48_000,
            {**{k: e[k] for k in ("source", "license", "transcript", "dialect_tag")},
             "condition": "rate48k",
             "derived_from": e["id"],
             "language": "ar"},
        )

    # one >30 s long-form item (concatenate clips from the same config)
    pool = [e for e in base if e["condition"] == "clean"][:5]
    xs = [read(e) for e in pool]
    long_x = np.concatenate([np.concatenate([x, np.zeros(16000, np.float32)]) for x in xs])
    write_entry(
        entries,
        "ar_sa_long_30s",
        long_x,
        16_000,
        {"source": "+".join(e["source"] for e in pool),
         "license": "CC-BY-4.0",
         "transcript": " ".join(e["transcript"] for e in pool),
         "condition": "long",
         "derived_from": ",".join(e["id"] for e in pool),
         "language": "ar",
         "dialect_tag": "ar_eg"},
    )

    # code-switch boundary: EN segment + AR segment concatenated
    import wave as _w

    with _w.open(str(JFK), "rb") as w:
        assert w.getframerate() == 16_000, "jfk fixture expected at 16k"
        en = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    ar = read(base[3])
    cs = np.concatenate([en[: int(6 * 16_000)], np.zeros(4000, np.float32), ar])
    write_entry(
        entries,
        "codeswitch_en_ar_001",
        cs,
        16_000,
        {"source": f"jfk.wav (public domain) + {base[3]['source']}",
         "license": "public domain + CC-BY-4.0",
         "transcript": "[EN jfk opening] + " + base[3]["transcript"],
         "condition": "codeswitch",
         "derived_from": base[3]["id"],
         "language": "ar-en",
         "dialect_tag": base[3]["dialect_tag"]},
    )

    manifest = {
        "bank": "arabic_speech_001",
        "created": "2026-08-23",
        "purpose": "Phase 5 Session 2 Track B runtime/functional validation bank",
        "note": (
            "Levantine-specific speech was not obtainable under the "
            "reproducibility constraints this session; Levantine coverage "
            "remains on the TEXT side (Phase 5 J-track batteries). Recorded "
            "honestly in the report."
        ),
        "entries": entries,
    }
    (BANK / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    total_s = sum(e["seconds"] for e in entries)
    print(f"\nbank complete: {len(entries)} entries, {total_s:.1f} s total -> {BANK}")


if __name__ == "__main__":
    sys.exit(main())
