# Finite Is Not Intact: Scale-Bit Faults in GGUF Quantized Kernels

**EmberSEC Phase V · Technical lab note · Draft, 14 September 2026**

## Abstract

A fault-injection result can depend critically on the values chosen for the test. EmberSEC's initial synthetic experiment showed that a single bit flip in a binary16 quantization scale could produce infinity. A subsequent scan of 136,307,712 scale words across seven GGUF model files found that none of their encoded exponents could become the NaN/Inf exponent through one bit flip. A distribution-informed experiment using Ember's actual quantized matrix-vector kernels likewise produced no non-finite outputs in 512 trials at production-scale magnitudes, while reproducing non-finite outputs in four of 64 synthetic-control trials. Yet finite output drift reached relative L2 errors of approximately 5,000–31,000 in the tested geometries. The result exposes a distinction between numerical validity and content integrity: a finite-header check accepts the finite scale corruptions responsible for these large changes. This note reports the corrected experiment, its analytical boundary, and the limitations of the associated load-time check. It does not measure end-to-end model accuracy or physical fault feasibility.

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

Q4_K also contains a binary16 minimum parameter and packed local scale/minimum fields. Q6_K contains integer local scales. The corrected sweep targets `d`; it is not an exhaustive experiment over every header field.

## 2. Why the original result needed correction

The initial deterministic fixture used `d = 1.0`. In binary16, this is `0x3c00`. Flipping bit 14 gives `0x7c00`, positive infinity. The analogous change to `−1.0` gives negative infinity. The mechanism is real, but the fixture does not establish its prevalence in model weights.

Binary16 has a five-bit exponent field in word bits 10–14. A finite word becomes a non-finite word through a single exponent-bit flip only when its original encoded exponent differs from 31 by exactly one bit. The eligible finite exponents are therefore:

```text
15, 23, 27, 29, 30  →  31
```

A sign or fraction-bit flip cannot change a finite exponent to 31. Once exponent 31 is reached, the fraction distinguishes infinity from NaN.

The saved distribution scan contains no occupied exponent bucket above 11. Consequently, **none of the scanned `d` words can become NaN or infinity through a single-bit change to that word**. This is a corpus-specific encoding result, stronger than merely observing no such events in a small trial set.

It is also narrower than an inference-safety theorem. A finite but greatly enlarged scale can still contribute to overflow in a different downstream computation. The exponent argument establishes the finiteness of the mutated header, not of every possible activation, accumulation, or model output.

## 3. Evidence and experimental design

### 3.1 Scale population

The saved scan covers seven files: Llama-3.2-1B and Qwen2.5-1.5B in Q8_0, Q4_K_M, and Q6_K variants, plus Qwen3-0.6B Q8_0. Mixed-format files contribute to their actual constituent tensor formats. Across Q8_0, Q4_K, and Q6_K tensors, the scan reports:

- **136,307,712 `d` words**;
- no initially non-finite `d` words;
- maximum positive `d` of **0.09326171875**;
- maximum occupied encoded exponent of **11**.

These are counts within the saved corpus, not a random sample of all GGUF models. The scanner reads packed scale words and records per-file/per-format statistics. The analysis here independently totals the saved counts and checks the occupied exponent buckets; it does not repeat the full model-file scan.

### 3.2 Kernel sweep

The correction harness constructs deterministic synthetic weights with eight output rows and 256 input features. K-quant payload bytes come from a fixed pseudorandom seed. Q8_0 payloads are generated by quantizing deterministic synthetic values. Each fixture uses a uniform requested `d`, converted to binary16, across its blocks. Q4_K's minimum parameter remains fixed at approximately `−0.02`.

For each requested scale, the harness flips each of the 16 bits of `d` in the first block, separately, and calls the real Ember kernel through `k_decode` or `q8_decode`. Activations are fixed synthetic vectors. The experiment includes 11 distribution-informed scale settings for Q4_K, 10 for Q6_K, and 11 for Q8_0, plus `+1.0` controls for all formats and a `−1.0` control for Q6_K.

Thus the 576 rows comprise **512 distribution-informed trials and 64 control trials**. The settings are fixed representative magnitudes, not independent random draws from the 136 million words. “Production-scale” describes the scale magnitudes; it does not mean the harness used intact production weight blocks or captured model activations.

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

Quantization scaling factors are an established concern in fault resilience. Fasfous et al.'s *Mind the Scaling Factors: Resilience Analysis of Quantized Adversarially Robust CNNs* (DATE 2022) relates quantization scaling factors to hardware-fault susceptibility in CNNs. Our experiment concerns different formats and a different workload boundary; it should not be presented as the discovery that scales matter.

The contribution here is a reproducible case study with three connected observations: a synthetic scale selected a non-finite mechanism absent from the scanned scale population; distribution-informed scale faults still caused large finite kernel-output changes; and a finite-header validator does not detect those changes. The correction strengthens the measurement lesson by separating a valid bit-level demonstration from an unsupported population interpretation.

## 7. Limits and publication readiness

This is a kernel-level lab result. It supplies no end-to-end model accuracy, perplexity, safety-behavior, or generated-answer measurements. It supplies no physical fault rate or adversarial fault-selection study. One fixed geometry per format cannot estimate typical drift, and seven files cannot establish a universal property of quantized models. Multi-bit faults and other header fields remain outside the corrected sweep.

The current artifact package supports inspection and independent saved-row recounting. Before an archival release, it needs portable paths, exact provenance and content hashes for all seven scanned model files, and a pinned build environment for a fresh kernel replay. The vendored Cargo dependency points to `/home/west/ember`, and the legacy analysis script reads `/tmp/opencode/phase5/sweep.jsonl`. The sweep's custom float-to-half conversion also deserves an independent check: its comment says round-to-nearest-even, while its implementation uses `round()`. The saved `d_bits` and `faulted_bits` are the authoritative inputs for this note's bit-level reasoning.

No new model execution, fault campaign, or validator benchmark was performed while preparing this draft. Numerical checks were confined to existing saved artifacts; implementation claims were checked by source inspection.

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
