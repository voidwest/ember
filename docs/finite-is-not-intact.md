# Finite Is Not Intact: Scale-Bit Faults in GGUF Quantized Kernels

**EmberSEC Phase V · Technical lab note · Draft, 14 September 2026**

## Replay record

- **Archived source commit:** `ae550f0fbcf9dea1a5507f5695b077d2cc77d02b` (correction package; the original executable/build identity was not recorded in that package).
- **Sweep binary:** `sweep`, from `docs/embersec/phase5-correction-2026-09-03/sweep/Cargo.toml`.
- **Replay setting:** `EMBER_VERIFY_QUANT=0`. The original process environment was not saved. The sweep directly invokes the kernels after mutation; this is not a validator-detection run.
- **Fixed seeds:** K payload constructor `0x5EED` (Xor64 initialized with `0x243F6A8885A308D3 ^ seed`), K activations `7`; Q8 payload construction and activations `11`.
- **Seven input-file SHA-256 values:** not recorded in the correction package; all seven are explicitly marked unavailable in the [file inventory](#file-inventory). Do not substitute a hash from another experiment or a same-named current file.

[Replay commands](#replay-commands) · [Saved artifact hashes](#artifact-identity-at-draft-preparation)

## Abstract

EmberSEC's first scale-fault fixture produced infinity after a single bit flip. Its scale was 1.0. A later scan of 136,307,712 scale words across seven GGUF files found no exponent that could reach NaN/Inf through one bit flip. In the corrected kernel sweep, all 512 trials at magnitudes drawn from that scan stayed finite; four of 64 synthetic-control trials produced non-finite outputs.

The finite outputs could still move substantially. Relative L2 drift reached thousands to tens of thousands in the synthetic kernels. Ember's header-finiteness check accepts the corrupted scales responsible for those changes.

These measurements describe kernel-fixture drift. End-to-end accuracy and physical fault feasibility were not tested.

## 1. Question and scope

Packed quantization stores numerical payloads together with parameters that determine their interpretation. Changing a payload bit changes part of a block's quantized representation; changing a shared scale can alter the contribution of many values at once. We ask two bounded questions:

1. Does the single-bit mechanism that produces a non-finite scale in a synthetic fixture occur in the scale population of the scanned model files?
2. If corrupted scales remain finite, does header-finiteness validation establish that their computation remains close to the original?

The fault model is one software-injected bit flip in the binary16 `d` field of an otherwise unchanged packed weight. It models a content mutation, without demonstrating Rowhammer, a DRAM error rate, an attacker-selected physical address, or an exploit. File corruption before loading and memory corruption after loading are distinct detection scenarios.

The experiment uses these layouts:

| Format | Values per block | Bytes per block | Location of binary16 `d` |
|---|---:|---:|---|
| Q8_0 | 32 | 34 | bytes 0–1 |
| Q4_K | 256 | 144 | bytes 0–1 |
| Q6_K | 256 | 210 | bytes 208–209 |

Q4_K also contains a binary16 minimum parameter and packed local scale/minimum fields. Q6_K contains integer local scales. Only `d` is mutated in the corrected sweep.

## 2. Why the original result needed correction

The initial deterministic fixture used `d = 1.0`. In binary16, this is `0x3c00`. Flipping bit 14 gives `0x7c00`, positive infinity. The analogous change to `−1.0` gives negative infinity. Whether this mechanism occurs in model weights depends on their scale exponents.

Binary16 has a five-bit exponent field in word bits 10–14. A finite word becomes a non-finite word through a single exponent-bit flip only when its original encoded exponent differs from 31 by exactly one bit. The eligible finite exponents are therefore:

```text
15, 23, 27, 29, 30  →  31
```

A sign or fraction-bit flip cannot change a finite exponent to 31. Once exponent 31 is reached, the fraction distinguishes infinity from NaN.

The saved distribution scan contains no occupied exponent bucket above 11. Consequently, **none of the scanned `d` words can become NaN or infinity through a single-bit change to that word**. This is a corpus-specific encoding result, stronger than merely observing no such events in a small trial set.

This is not an inference-safety theorem: a finite but greatly enlarged scale can still contribute to overflow downstream.

## 3. Evidence and experimental design

### 3.1 Scale population

The saved scan covers seven files: Llama-3.2-1B and Qwen2.5-1.5B in Q8_0, Q4_K_M, and Q6_K variants, plus Qwen3-0.6B Q8_0. Mixed-format files contribute to their actual constituent tensor formats. Across Q8_0, Q4_K, and Q6_K tensors, the scan reports:

- **136,307,712 `d` words**;
- no initially non-finite `d` words;
- maximum positive `d` of **0.09326171875**;
- maximum occupied encoded exponent of **11**.

These are counts within the saved corpus, not a random sample of all GGUF models. The scanner reads packed scale words and records per-file/per-format statistics. The analysis here independently totals the saved counts and checks the occupied exponent buckets; it does not repeat the full model-file scan.

### File inventory

These are the exact filenames keyed in the saved scan. Counts include only the scanned Q8_0, Q4_K, and Q6_K `d` fields, not all tensor values or every header field.

| Scanned GGUF filename | `d` words | SHA-256 at scan time |
|---|---:|---|
| `Llama-3.2-1B-Instruct.Q4_K_M.gguf` | 4,827,136 | Not recorded |
| `Llama-3.2-1B-Instruct.Q6_K.gguf` | 4,827,136 | Not recorded |
| `Llama-3.2-1B-Instruct-Q8_0.gguf` | 38,617,088 | Not recorded |
| `Qwen3-0.6B-Q8_0.gguf` | 18,624,512 | Not recorded |
| `qwen2.5-1.5b-instruct-q4_k_m.ember.gguf` | 6,941,184 | Not recorded |
| `qwen2.5-1.5b-instruct-q6_k.ember.gguf` | 6,941,184 | Not recorded |
| `qwen2.5-1.5b-instruct-q8_0.gguf` | 55,529,472 | Not recorded |

The correction package does not bind these filenames to file-content hashes. A separate Phase I record contains a Llama Q8_0 hash, but it does not establish the identity of the file scanned here. The counts and exponent result are therefore attributed to the saved scan record; the original seven file identities cannot be independently established from this package alone.

### 3.2 Kernel sweep

The correction harness constructs deterministic synthetic weights with eight output rows and 256 input features. This is **not a layer of Llama-3.2-1B**. The 136-million-word distribution scan is the only part of this experiment that touched real model files. K-quant payload bytes come from a fixed pseudorandom seed. Q8_0 payloads are generated by quantizing deterministic synthetic values. Each fixture uses a uniform requested `d`, converted to binary16, across its blocks. Q4_K's minimum parameter remains fixed at approximately `−0.02`.

For each requested scale, the harness flips each of the 16 bits of `d` in the first block, separately, and calls the real Ember kernel through `k_decode` or `q8_decode`. Activations are fixed synthetic vectors. The experiment includes 11 scale settings with magnitudes taken from the scanned `d` population for Q4_K, 10 for Q6_K, and 11 for Q8_0, plus `+1.0` controls for all formats and a `−1.0` control for Q6_K.

Thus the 576 rows comprise **512 trials at those scale magnitudes and 64 control trials**. The settings are fixed representative magnitudes. They were not sampled independently from the 136 million words. The payloads and activations remain synthetic throughout.

For pristine output vector `y` and faulted output vector `y′`, the recorded measures are:

- whether every component of `y′` is finite;
- maximum absolute component difference;
- relative L2 difference, `||y′ − y||₂ / ||y||₂`;
- whether the index of the largest output component changes.

The implementation floors the squared denominator at the smallest positive normal float32 value. Metrics are computed in float32 and written with limited decimal precision. Historical field names refer to “logits” and “top1”; here these mean **kernel output components and their argmax**, not vocabulary logits or generated answers.

## 4. Results

The following values were recounted directly from the saved JSONL rows:

| Format | Production-scale trials | Non-finite outputs | Maximum relative L2 drift | Maximum absolute difference | Bit-14 output-argmax changes |
|---|---:|---:|---:|---:|---:|
| Q4_K | 176 | 0 | 4,976.151 | 62,578.38 | 11/11 |
| Q6_K | 160 | 0 | 30,831.56 | 57,895.96 | 5/10 |
| Q8_0 | 176 | 0 | 6,297.217 | 939,410.2 | 0/11 |

All four non-finite trials occurred in the synthetic controls, at bit 14. The real-scale trials also have finite mutated `d` words, as checked directly from their saved bit patterns.

Relative L2, maximum absolute difference (L∞), and output-argmax change are different estimands. Q6_K has the largest relative L2 maximum in this fixture; Q8_0 has the largest absolute difference, about 939,410, while retaining argmax in all 11 bit-14 trials. None is an accuracy-drop measurement or a format-level security ranking.

The Q8_0 result illustrates why output-argmax stability alone is insufficient: none of its 11 real-scale bit-14 trials changed argmax, despite very large vector changes. This observation is tied to the fixed output geometry. It does not establish that Q8_0 protects a model's predictions better than either K format.

The formats also differ in block width, payload construction, and activation setup. These measurements do not support a controlled ranking of format resilience. Large relative errors should be read alongside absolute differences because the pristine vector norm affects the ratio.

The earlier payload sweep remains supporting exploratory evidence in the Phase V report. Its finite outputs do not prove that payload corruption is harmless, and its fault sites and fixtures are not a matched comparison establishing that scale faults always dominate payload faults.

## 5. What the integrity check detects

Ember's `QuantizedWeight::validate_integrity()` checks expected byte length and finite Q8_0 scales. `KQuantWeight::validate_integrity()` checks expected byte length and finite Q4_K `d`/minimum or Q6_K `d` fields. The optional `EMBER_VERIFY_QUANT` load hook invokes this validation on constructed quantized weights.

Under those predicates, a bit flip that changes a finite `d` to another finite `d` while preserving layout is accepted. The finite scale faults above satisfy that condition. This is a source-level deduction from the validator, rather than a newly executed detector benchmark over all 576 trials.

| Event | Covered by the described check? |
|---|---|
| Inconsistent expected packed byte length | Yes |
| NaN/Inf in a checked scale header at validation time | Yes |
| Finite but incorrect scale at validation time | No |
| Payload-only corruption preserving layout | No |
| Mutation after a one-time load check | No continuing detection |

The check establishes a limited form of numerical validity. It does not compare packed bytes with a trusted original. Authenticity or content-integrity mechanisms would need to establish that reference and check the relevant bytes at the appropriate time. This note does not evaluate the cost or effectiveness of such a replacement.

## 6. Interpretation and related work

Quantization scaling factors are an established concern in fault resilience. Fasfous et al.'s *Mind the Scaling Factors: Resilience Analysis of Quantized Adversarially Robust CNNs* (DATE 2022) relates quantization scaling factors to hardware-fault susceptibility in CNNs. This note examines the corresponding issue in GGUF block formats at the kernel boundary.

Choosing `d = 1.0` made the initial test exercise a mechanism absent from the scanned population. Replacing it with magnitudes from that population changed the result: the outputs stayed finite, sometimes with large errors. A check designed to reject NaN/Inf headers leaves those errors undetected.

## 7. Limits

Each format was tested in one fixed geometry, so the results cannot estimate typical drift across layers or models. The sweep changes one bit of `d` at a time. Multi-bit faults and other header fields remain untested.

The experiment ends at kernel outputs. It provides no measurements of model accuracy, perplexity, generated answers, physical fault rates, or adversarial fault selection. The seven-file scan supports a claim about that recorded corpus only.

### Artifact status

The vendored Cargo dependency points to `/home/west/ember`; the legacy analysis script reads `/tmp/opencode/phase5/sweep.jsonl`. The commands below relocate the dependency and give the new sweep output an explicit path.

There is also a conversion discrepancy to preserve: the sweep's float-to-half comment says round-to-nearest-even, while the implementation uses `round()`. This note's bit-level reasoning uses the saved `d_bits` and `faulted_bits`. Original build/environment identity and scan-time model hashes are missing, which limits exact historical replay.

Preparing this note involved saved-array checks and source inspection. No new model execution, fault campaign, or validator benchmark was run.

### Replay commands

To rerun the archived synthetic kernel sweep, create a separate checkout of the correction commit and point its vendored crate at that checkout. This produces new measurements; it does not recreate missing historical provenance. It requires Git, Python 3, a compatible Rust toolchain, and access to the crate dependencies. No model files are used.

```bash
git worktree add --detach /tmp/embersec-phase5 ae550f0fbcf9dea1a5507f5695b077d2cc77d02b
python3 - <<'PYTHON'
from pathlib import Path
p = Path('/tmp/embersec-phase5/docs/embersec/phase5-correction-2026-09-03/sweep/Cargo.toml')
p.write_text(p.read_text().replace('/home/west/ember', '/tmp/embersec-phase5'))
PYTHON
EMBER_VERIFY_QUANT=0 cargo run \
  --manifest-path /tmp/embersec-phase5/docs/embersec/phase5-correction-2026-09-03/sweep/Cargo.toml \
  --bin sweep -- /tmp/embersec-phase5-replay.jsonl
```

For a new real-file scan, supply seven explicit file paths and save their hashes alongside the output. This requires NumPy. New hashes identify that new scan only.

```bash
# Set these seven positional arguments to the files in the inventory.
set -- /path/to/file1.gguf /path/to/file2.gguf /path/to/file3.gguf \
  /path/to/file4.gguf /path/to/file5.gguf /path/to/file6.gguf /path/to/file7.gguf
sha256sum "$@" > /tmp/embersec-phase5-models.sha256
python3 /tmp/embersec-phase5/docs/embersec/phase5-correction-2026-09-03/gguf_d_scan.py \
  "$@" --out /tmp/embersec-phase5-distribution.json \
  --samples /tmp/embersec-phase5-samples.json
```

## Evidence and references

1. [EmberSEC Phase V report and historical correction](embersec/quantized-inference-security.md).
2. [Per-trial sweep data](embersec/phase5-correction-2026-09-03/sweep.jsonl) and [saved scale distributions](embersec/phase5-correction-2026-09-03/d_distribution.json).
3. [Correction sweep source](embersec/phase5-correction-2026-09-03/sweep/src/main.rs), [GGUF scanner](embersec/phase5-correction-2026-09-03/gguf_d_scan.py), and [legacy analysis script](embersec/phase5-correction-2026-09-03/analyze_sweep.py).
4. [Kernel fault harness](../src/quant_fault.rs), [Q8_0 validation](../src/quant.rs), and [K-quant validation](../src/quant_k.rs).
5. Fasfous et al. (2022), [*Mind the Scaling Factors: Resilience Analysis of Quantized Adversarially Robust CNNs*](https://iris.polito.it/handle/11583/2964392), DATE 2022. Related-work positioning is preliminary, not a systematic novelty review.

### Artifact identity at draft preparation

SHA-256 values identify the saved evidence inspected for this draft; they do not supply missing upstream model provenance.

```text
sweep.jsonl
fbf34d0d1f02f819d1141012c01ae7f90e2266ab9e4b336218960a5faf19abc6
d_distribution.json
c04c7eb992c617b6a2409f8491e17e2d1326144fd58514a69dcd47d3df1cc465
sweep/src/main.rs
33d65cd423665b8c59950490ff9c0b61b9d132755369197af428893856f197c7
gguf_d_scan.py
63db42db2b117bdd34b0b085fd9a8b9bd7e5c2bd9efa437353a25ae578067fad
```
