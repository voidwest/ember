# EmberSEC Phase 2 — Hostile Multimodal Inputs (read-only threat-surface audit)

**Date:** 2026-08-31 · **Branch/commit:** main @ `6ed3a133` (dirty worktree; audit read-only, no files changed)
**Scope:** image, audio, video, and fusion paths from attacker-controlled bytes/files into tensors and the LLM.
**Method:** direct source reading; every finding is code-verified (file:line). Nothing was executed; no exploit,
memory-corruption, or dependency-CVE claims are made. All findings are crash / panic / resource-amplification /
semantic-integrity observations unless stated otherwise.

---

## 1. Path maps

**Image** (PNG/JPEG bytes or file):
`ImageInput::File|Bytes|Pixels` (`src/multimodal/request.rs:34`) →
`image::ImageReader` 0.25.10 (features png+jpeg, pure Rust) (`src/multimodal/image.rs:102-121`) →
`to_rgb8` bitmap → `rgb8_to_tensor` f32 CHW `[3,h,w]` (image.rs:123-136) →
`preprocess` (resize longest edge 2048 → tile 512 + global tile → rescale/normalize) (image.rs:146-280) →
`batch_encode_images` (`src/multimodal/batch.rs:52`) →
`VisionTransformer` (conv patch embed, learned pos embed, 12 layers bidirectional attention, fast-exp softmax)
(`src/multimodal/vision.rs:106-350`) → `PixelShuffleConnector` (vision.rs:604) →
`SmolVlmAssembler` placeholder expansion + scatter (`src/multimodal/assembler.rs:215-318`) →
`EmbeddingSequence` → normal Llama prefill.

**Audio** (WAV bytes or file, or raw samples):
`AudioInput::File|Bytes|Samples` (`src/multimodal/audio.rs:71`) →
hand-rolled RIFF/WAVE parser `decode_wav_bytes` (audio.rs:233-350) →
`DecodedAudio` → `to_mono_16k` + windowed-sinc `resample` (audio.rs:361-429) →
`log_mel_spectrogram_full` (audio.rs:570-646, deliberately ungated for long-form) →
`AudioEncoder` conv1/conv2 + 32 Whisper layers (`src/multimodal/audio_encoder.rs:167-350`) →
`UltravoxProjector` stack-8 → assembler → Llama. Long-form chunking: `ultravox.rs:328-414`.

**Video:** directory of PNG frames → `decode_rgb` per frame (`src/cli_video.rs:74-84`) →
`VideoInput::Frames` → frame sampler → same vision path.

---

## 2. Prioritized findings

> **Status marker (freeze 2026-08-31):** the findings below describe the
> audited state (main @ `6ed3a133`); **all P1/P2 items were fixed** in
> `50b3af30` + `75183e67` — see §4 for the fix map. Do not read any finding
> below as an open vulnerability.

### P1 — crash or resource amplification on attacker-controlled input (fix first)

- **AUD-1 · Panic on unsupported WAV format.** `decode_wav_bytes`'s `read_one` has a
  `_ => panic!("unsupported wav format tag ...")` arm (`src/multimodal/audio.rs:334`). A WAV with
  `format_tag` ∉ {1, 3, 0xFFFE} (e.g. tag 6 = A-law, 8-bit; tag 2 = MS ADPCM with 8-bit) and a non-empty
  data chunk reaches it. Trivial malformed file → process panic (abort in CLI/server). Fix: return `Err`.
  Code-verified; not executed.

- **AUD-2 · Resample amplification + division-by-zero.** `resample` computes
  `ratio = to_rate / from_rate` and `out_len = (samples.len() * ratio).round() as usize`
  (`src/multimodal/audio.rs:365-367`); `to_mono_16k` calls it for any rate ≠ 16000 (audio.rs:418-428).
  - `sample_rate = 0` (valid u32 in the WAV header, or `AudioInput::Samples`) → `ratio = inf` →
    `out_len = usize::MAX` → `vec![0.0f32; usize::MAX]` → capacity-overflow panic / OOM abort.
  - `sample_rate = 1` → 16 000× output amplification; a 256 MiB WAV (the existing cap) → ~2 TB f32
    allocation. The 256 MiB file cap does not bound this.
  Fix: validate `sample_rate` to a sane range (e.g. 8k–48k) before resample; use `checked_mul`/checked
  length for `out_len`; cap total output samples.

- **AUD-3 · Long-form CPU/RAM amplification (bounded but severe).** A 256 MiB WAV ≈ 30+ minutes of audio.
  `log_mel_spectrogram_full` is deliberately ungated (audio.rs:565-569) and allocates ~5–7 GB of f64
  working set for a max-size file (padded + frames + power + mel buffers), then `encode_mel_chunked`
  (ultravox.rs:328-414) runs ~267 chunked 3000-frame × 1280-dim × 32-layer encodes → minutes–hours CPU.
  Multiple `--audio` flags multiply this with no count or total-duration admission limit
  (`src/cli_audio.rs:27-30`). DoS on a 16 GB host with one or two files.

- **IMG-1 · Image decode has no explicit limits.** `decode_rgb_bytes`/`decode_rgb` never call
  `ImageReader::limits(...)` (`src/multimodal/image.rs:102-121`), so only the image crate's defaults apply:
  `max_alloc = 512 MiB` (non-strict for some decoders — zune-jpeg may not honor it), **no width/height
  caps**. After decode, `rgb8_to_tensor` allocates f32 at 12 B/px — 4× the bitmap (`image.rs:126`),
  so a ~512 MiB decoded bitmap becomes a ~2 GiB f32 tensor before preprocessing. `cli_multimodal`
  accepts unbounded repeated `--image` (`src/cli_multimodal.rs:26-29`) → N × ~2 GiB peak → OOM abort on
  this host. Fix: set explicit `Limits` (max width/height + tighter alloc) and a per-request
  total-pixel/media budget at admission.

### P2 — panic on public-API shape misuse (not reachable via the current CLI, but pub API)

- **PAN-1 · Vision rank/shape assumptions.** `encode_impl` indexes `pixels.shape()[0..3]` before any
  ndim check (`src/multimodal/vision.rs:118-126`); rank < 4 → index panic. Release mode has only
  `debug_assert`s for channel/size (vision.rs:125-126); a height not divisible by `patch_size` silently
  truncates the patch grid (correctness), and `vec![0.0f32; n_images * num_patches * patch_dim]`
  (vision.rs:133) is unchecked.
- **PAN-2 · Audio encoder asserts.** `encode_inner` does `assert_eq!(mel.shape()[0], cfg.num_mel_bins)`
  and indexes `shape()[0]/[1]` (`src/multimodal/audio_encoder.rs:175-176`); a 1-D tensor → index panic,
  wrong bin count → assert panic. `encode`/`encode_traced` are `pub`.
- **PAN-3 · batch_encode_images division by zero.** `scale2 = scale_factor * scale_factor` unchecked
  (`src/multimodal/batch.rs:59`) and `rows = ... / scale2` (batch.rs:121); `scale_factor = 0` from any
  caller → divide-by-zero panic. `patch_size = 0` similarly.
- **PAN-4 · conv1d kernel assert** `assert_eq!(kernel, 3)` (`audio_encoder.rs:297`) — internal invariant
  after load-time dim checks; listed for completeness.

### P3 — robustness / validated-state seams

- **VAL-1 · No validated boundary types.** `ImageInput`/`AudioInput` are raw enums; geometry (dims,
  rate, duration, byte budget) is re-derived at each stage with inconsistent checks. The natural Phase-2
  seam is exactly `ParsedImage → ValidatedImageInput` and `ParsedAudio → ValidatedAudioInput` immediately
  after decode (capture: dims, format, byte_len, channel layout; audio: sample_rate, duration_s, sample
  count), with a request-level media budget (max parts, max total pixels/samples) at the CLI/API
  admission boundary. The model wrappers then consume only validated types.
- **VAL-2 · Cache keys are 64-bit non-cryptographic.** `MediaId` uses `DefaultHasher` (fixed keys,
  request.rs:126-160); `PreprocessFingerprint` is a custom 64-bit mix (cache.rs:29-57). A collision could
  reuse features across distinct media → semantic corruption, not memory unsafety; bounded by the cache
  byte budget. Fine for a local CLI; document or move to 128-bit before any multi-tenant/server exposure.
- **VAL-3 · Video frame count unbounded.** `cli_video` loads every PNG in `--frames-dir` with no
  count/total-pixel cap (`src/cli_video.rs:74-84`).
- **VAL-4 · Zero-size edge is contained but duplicated.** `preprocess` guards h/w ≥ 1 only in the
  resize branch (image.rs:171-172); a `[3,0,0]` input under a no-resize config flows to a 0-row tensor
  and fails at the assembler (error, not panic). `tile_grid_for` (image.rs:286-327) duplicates the
  rounding math — divergence risk, not a security issue.
- **VAL-5 · Fuzz gap.** `fuzz/` covers gguf_loader, gguf_to_llama, kv_snapshot_manifest, npy_bytes,
  tokenizer_json — **no wav, png/jpeg, or multimodal request fuzz targets**; `tests/multimodal.rs`
  exercises only valid shapes. Suggested matrix: `decode_wav_bytes` (headers/format tags/rates/chunk
  arithmetic), `decode_rgb_bytes` (png/jpeg bombs, truncated files), `ImageInput::Pixels`/`AudioInput::Samples`
  shape fuzz against `preprocess`/`to_mono_16k`/encoder entry points.
- **VAL-6 · Unsafe reachability after media decode.** Enabled decoders (png 0.18, zune-jpeg 0.5) are pure
  Rust — no C/FFI in the decode path. The only `unsafe` reachable downstream are `matrixmultiply::sgemm`
  (`src/tensor.rs:197-214, 262-279`, guarded by shape asserts + contiguous-layout invariants) and the
  AVX2 `fast_exp_raw` (`src/simd.rs:1544-1608`, length-checked wrappers). No unsafe/FFI finding.

---

## 3. Verdict

**Enough for a real Phase 2 — yes.** Two code-verified panic paths (AUD-1, AUD-2) and one unbounded
resource amplification (AUD-2, AUD-3) sit on attacker-controlled WAV bytes/headers; the image side lacks
explicit decode limits and any media-count admission bound; public encoder entry points panic on
malformed shapes; and there is zero fuzz coverage for any multimodal input.

**Recommended Phase-2 work order (no implementation performed in this audit):**
1. AUD-1/AUD-2: `Err` instead of panic; validate `sample_rate`; checked output lengths. + regression tests.
2. IMG-1/VAL-3: explicit `image::Limits` + per-request media/pixel/sample budget at CLI/API admission.
3. VAL-1: introduce `ValidatedImageInput`/`ValidatedAudioInput` at the decode seam; wrappers consume only validated types.
4. VAL-5: fuzz targets for wav bytes, png/jpeg bytes, and the `Pixels`/`Samples` shape surface.
5. PAN-1/2/3: convert public encoder shape panics to `Result`.

Deferred (unchanged roadmap): execution identity (Phase III), attestation (IV), quantized-inference
security (V).

*This document is a local, untracked audit artifact (repo convention for security docs); it changes no
tracked state.*

---

## 4. Remediation status (2026-08-31, implementation pass)

Implemented (all code-verified by the full test suite: 402 lib + 52 bin +
integration suites; `cargo fmt --check` and `cargo clippy --all-targets
--all-features -- -D warnings` clean):

| Finding | Fix | Where |
|---|---|---|
| AUD-1 panic on unsupported WAV format | `read_one` returns `Err` instead of `panic!` | `src/multimodal/audio.rs` |
| AUD-2 zero/absurd sample rates + resample amplification | `sample_rate > 0` validated at WAV parse; `resample` returns `Result`, rejects non-finite output length and caps output at `MAX_RESAMPLE_OUTPUT_SAMPLES` (2 h @ 48 kHz) | `src/multimodal/audio.rs` |
| AUD-3 long-form CPU/RAM DoS | `ValidatedAudioInput::from_audio_input` enforces `MAX_AUDIO_SECONDS` (1 h) and `MAX_AUDIO_SEGMENTS` (16) at the model boundary (`Ultravox::build_mel`/`build_inputs`) | `src/multimodal/audio.rs`, `src/ultravox.rs` |
| IMG-1 no image decode limits | explicit `image::Limits`: 8192 px/edge + 256 MiB alloc budget on both decode paths | `src/multimodal/image.rs` |
| PAN-1 vision shape panics | `encode_impl`/`attention_impl` validate rank/channels/size/patch/heads and return `CpuError`; checked patch-buffer product | `src/multimodal/vision.rs` |
| PAN-2 audio encoder asserts | `encode_inner` validates mel rank/bins; `conv1d` kernel-3 assert → `CpuError` | `src/multimodal/audio_encoder.rs` |
| PAN-3 batch geometry div-by-zero | `patch_size`/`scale_factor` validated; `scale²` via `checked_mul` | `src/multimodal/batch.rs` |
| VAL-1 validated-state seam | `ValidatedImageInput` (request.rs) + `ValidatedAudioInput` (audio.rs); wrappers decode through them; `MAX_IMAGES_PER_REQUEST`/`MAX_VIDEO_FRAMES` admission caps | `src/multimodal/request.rs`, `src/smolvlm.rs`, `src/cli_video.rs` |
| VAL-5 fuzz gap | new targets: `wav_bytes`, `image_bytes`, `image_preprocess` (registered in `fuzz/Cargo.toml`) | `fuzz/fuzz_targets/` |

Regression tests added in `tests/multimodal.rs` (9 tests) and lib unit suites.

**Commit/state note (final):** all hardening is committed on `main` and
pushed to `origin/main`:
- `50b3af30` — image/video boundary hardening (image limits, validated image
  seam, batch guards, video frame cap, fuzz targets),
- `00e6ced7` — the previously pending staged batch (multimodal phase 4/5,
  gemma4 parity, loader/npy/tokenizer/kv hardening, fuzz harness, docs),
- `75183e67` — audio/vision hardening delta (WAV panic/amplification fixes,
  encoder shape checks, validated audio seam + duration/segment caps, tests).
The pre-existing unstaged leftovers (`src/quant.rs`, the deleted
`data/test_*.npy`, and the two untracked files) were left untouched.

Deferred: VAL-2 (cache-key width), VAL-4 (tile-grid mirror dedup), and the
remaining encoder pub-API hardening beyond the shape checks above.
