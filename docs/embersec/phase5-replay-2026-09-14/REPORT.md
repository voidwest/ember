# Phase V replay audit — 14 September 2026

The corrected Phase V scale sweep and distribution scan were recreated from archived source and the seven current model files. Both fresh kernel runs produced exactly the original 576-row JSONL, byte for byte. The fresh scale scan matched every field of the saved distribution JSON for all seven files.

This is a new, recorded replay. It does not recover the original executable, original dependency lock, process environment, or scan-time model hashes.

## Checks and outcomes

| Check | Outcome |
|---|---|
| Archived source identity | 146 Git blobs checked; only the standalone crate's absolute path dependency changed |
| Current model identity | SHA-256 computed before and after each scan; hashes and stat identities unchanged for all seven files |
| Original scanner replay | All saved distribution fields equal, including quantiles, histograms, signed counts, and Q4_K minimum summaries |
| Independent scanner | Separate GGUF header parser and mmap/stride extraction agree on all d-word counts, tensor counts, and exponent histograms |
| Corpus total | 136,307,712 d words; maximum occupied encoded exponent 11; zero words in the single-bit-to-nonfinite eligible exponent buckets |
| Kernel replay | 576 rows identical to the original JSONL; second run identical to first |
| Scale conversion and mutations | Independent binary16 arithmetic and Python's half conversion agree with all recorded trials |
| Direct validator test | All 512 scan-magnitude trials accepted; all four non-finite controls rejected; zero disagreement between acceptance and output finiteness within these 576 fixtures |
| Instrumentation preservation | Instrumented validator sweep outputs identical to uninstrumented sweep outputs |
| Ember loader cross-check | 45 sampled scale words matched across seven files; zero mismatches |

“Identical distribution” means equality of every parsed JSON field. The new JSON formatting differs. “Identical kernel output” means equality of the entire saved JSONL byte stream, including metrics rounded by the original serializer. No claim is made about unsaved higher-precision intermediate arrays.

## Build and execution

Source commit: `ae550f0fbcf9dea1a5507f5695b077d2cc77d02b`.

Rust: 1.92.0 (`ded5c06cf`, LLVM 21.1.3). Cargo: 1.92.0. Host and CPU flags are recorded in [environment.json](environment.json).

The standalone correction crate originally supplied no Cargo.lock. The offline build resolved the locally cached compatible dependencies, with default features enabled (including GUI dependencies), and captured the resulting [Cargo.lock](Cargo.lock). Subsequent build invocations used `--offline --locked`. The dev profile retains the archived `opt-level=2`. Build parallelism and Rayon thread count were both four. No engine, kernel, or sweep algorithm was patched for the uninstrumented replay.

The first build completed successfully in approximately 629 seconds. The retained command logs cover the subsequent locked build and all replay/validation commands; the initial compilation transcript is not included. The archive has no .git directory, so its source identity comes from the independent Git-blob comparison, not an embedded build-time Git label.

Each sweep process ran with `EMBER_VERIFY_QUANT=0`; other `EMBER_*` variables were removed. Seeds and requested scales are unchanged. The detector check uses [validate_faults.rs](validate_faults.rs), a separate copy of the original sweep with calls to `validate_integrity()` and logged verdicts immediately after each mutation. The original sweep source remains unchanged.

## Current file identities

These hashes identify the files read on 14 September 2026. Exact distribution agreement cannot prove that every payload byte matches the files used for the original scan. The historical scan did not save content hashes.

| Filename | d words | Replay SHA-256 |
|---|---:|---|
| `Llama-3.2-1B-Instruct.Q4_K_M.gguf` | 4,827,136 | `f3cdd84d4a33483d749ddbe9cf13433b763ce41352f58b86cc67718325a38885` |
| `Llama-3.2-1B-Instruct.Q6_K.gguf` | 4,827,136 | `3e22c35a5214a758faf2ca6bdd175aab574a4f8d2914e81f90375393bc0bf3df` |
| `Llama-3.2-1B-Instruct-Q8_0.gguf` | 38,617,088 | `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3` |
| `Qwen3-0.6B-Q8_0.gguf` | 18,624,512 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| `qwen2.5-1.5b-instruct-q4_k_m.ember.gguf` | 6,941,184 | `b66e0350b994a95e26e9c41f05410c39f1ec84838b96144f494f87a5aaee8bf5` |
| `qwen2.5-1.5b-instruct-q6_k.ember.gguf` | 6,941,184 | `c6bc806dd29f9dd3f32e320d90cd6f3facf94f2bdff0b13fc8311113a7f354d1` |
| `qwen2.5-1.5b-instruct-q8_0.gguf` | 55,529,472 | `d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8` |

## Kernel output identity

Both repeated sweeps and the instrumented detector sweep share the original JSONL SHA-256:

```text
fbf34d0d1f02f819d1141012c01ae7f90e2266ab9e4b336218960a5faf19abc6
```

The newly built sweep executable SHA-256 is:

```text
5ac7b57bac9a449d731ba49556d2f50c9d502a87eaa501df97360646aeb8716a
```

An executable hash identifies this build; it is not a promise that another compiler or build path will yield identical executable bytes.

## Verify the saved evidence

From this directory, run:

```bash
python3 verify.py
```

The verifier checks the package hashes and recounts the saved results. It does not build a model or run inference. [SHA256SUMS](SHA256SUMS) is an integrity inventory, not a digital signature or trusted timestamp.

The exact execution scripts are included with their original local paths. [reproduce.py](reproduce.py) is a portable wrapper for a new checkout/build, repeated kernel sweeps, detector logging, and a fresh scan:

```bash
python3 reproduce.py --repo /path/to/ember --models /path/to/gguf-files \
  --out /path/to/new-output-directory --offline
```

The output directory must not already exist. Omit `--offline` if dependencies need downloading. It requires Python with NumPy, Git, Rust/Cargo, the captured lockfile, and all seven named model files. The portable wrapper was syntax-checked; the recorded execution used replay_kernels.py and replay_scan.py. The independent scanner and loader spot checks are separate recorded checks, not operations performed by the portable wrapper.

## What remains unestablished

The recreation supports the corrected kernel and scale-population results. The earlier payload-bit campaigns were not rerun. It provides no end-to-end accuracy, layer-level drift distribution, hardware fault rate, or adversarial bit-selection result. The validator experiment now directly supports the detector limitation for these fixtures. It does not certify detection behavior for arbitrary corruption.

The helper's round-to-nearest-even comment still disagrees with its general implementation. All scale settings used here nevertheless match independent round-to-nearest-even conversion. No source correction was needed to reproduce the reported rows.
