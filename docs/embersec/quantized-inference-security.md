# EmberSEC Phase V — Quantized Inference Security: Faults, Determinism, Integrity

**Status:** landed (main, freeze tag `embersec-freeze-2026-08-31`); Threat model and measured findings below; all measurements
are from the hermetic, deterministic test harness (synthetic seeded blocks —
no model files required).

---

## 1. Surface map

Ember executes three quantized weight formats (block layouts from GGUF):

| Format | Block bytes | Block contents |
|---|---|---|
| Q8_0 | 34 | f16 `d` scale (bytes 0-1) + 32 int8 quants (2-33) |
| Q4_K | 144 | f16 `d` (0-1), f16 `min` (2-3), 12-byte packed scale/min pairs (4-15), 128 nibble quants (16-143) |
| Q6_K | 210 | 128 ql (0-127), 64 qh (128-191), 16 int8 scales (192-207), f16 `d` (208-209) |

K-quant tensors run compressed-resident through integer dot kernels
(`k_quant_matmul.rs`); Q8_0 runs packed through SIMD kernels; the eager-f32
path dequantizes at load. Determinism is already contract-tested
serial ≡ parallel, scalar ≡ SIMD dispatch, and bit-identical presplit.

## 2. Fault model

A fault is a single corrupted byte inside a loaded quantized tensor (DRAM
bit flip, rowhammer-style disturbance, or a mutated/corrupted file that
passes length checks). File-level integrity is covered elsewhere (model
SHA-256 in run manifests, bundle payload hashes); this phase covers the
*in-memory* content after a valid load. Where the corrupted byte lands
determines the damage:

- **Payload byte** (quants): the affected dequantized values change by one
  or a few quantization steps — bounded logit drift, no crash.
- **Scale/header byte** (f16 `d`/`min`, or int8 scales): a single bit can
  turn the scale into NaN/Inf, or flip its sign — unbounded drift and
  non-finite logits.
- **Layout byte** (length/shape): impossible to corrupt silently; the
  constructors enforce exact block-count math and would reject the tensor.

## 3. Findings from the deterministic harness (single-bit faults)

| Fault site | Count | Median rel-L2 | p95 / max rel-L2 | Max abs logit Δ | Non-finite? |
|---|---|---|---|---|---|
| Q4_K payload (bit 3 of every nibble byte) | 1024 | 0.0011 | 0.0098 / 0.32 | 5748 | never |
| Q6_K payload (bit 3 of every quant byte) | 1536 | 0.0104 | 0.085 / 0.16 | 797 | never |
| Q8_0 payload (bit 5 of one quant byte/block) | 8 | 0.0405 | 0.052 | 0.67 | never |
| Q4_K f16 `d`, exponent bits 10–14 | 40 attempts (8 blocks; early-stop per block) | — | — | — | 8 block hits (bit 14) |
| Q6_K f16 `d`, exponent bits 10–14 | 40 attempts (8 blocks; early-stop per block) | — | — | — | 8 block hits (bit 14) |

**Interpretation.** Payload faults are graceful degradation: median logit
drift is ~0.1-4% relative, top-1 flips only when the pristine margin is
small (the same near-threshold-flip mechanism documented in the Arabic
quantization pilots).

The scale check is an analytical FP16 layout finding, not a percentage
estimate. IEEE-754 binary16 stores its exponent in bits 10–14. The current
test uses `d = 1.0`, probes those five positions, and stops after the first
non-finite result in each block. With eight synthetic blocks per dtype, that
is 5 × 8 = 40 attempted flips per dtype; bit 14 sets the exponent to the
all-ones value and produces `Inf`, giving eight block hits. The test does not
sweep all 16 bits, vary the starting `d`, or estimate a population hit rate.
There is no confidence interval or per-trial result artifact; this is one
repeatable deterministic unit-test invocation. Q8_0's `d` header is covered
by integrity testing, not this exponent sweep.

Downstream behavior on non-finite logits:
- the CLI's logit validation **bails with a structured error** (safe);
- the sampler's `argmax_token` **asserts on NaN** (crash, contained);
- the K-matmul `validate` rejects non-finite activations before touching
  `dst` (activations are the other side of the same boundary).

## 4. Integrity validation

- `QuantizedWeight::validate_integrity()` / `KQuantWeight::validate_integrity()`:
  block-layout math plus per-block finite-header checks (Q8_0 `d`; Q4_K `d`
  and `min`; Q6_K `d`). Any NaN/Inf header is an error.
- Load-time hook: `EMBER_VERIFY_QUANT=1|true|yes` runs the check on every
  constructed quantized weight at load (cost: one pass over the packed
  bytes; off by default because compressed-resident Q8 loads deliberately
  avoid touching file pages).
- `data_mut()` seams (owned weights only; `None` for mmap-backed) make the
  harness and future tooling possible without weakening the read-only
  mapping invariant.

## 5. Harness (`ember::quant_fault`)

`inject_bit_flip` (bounds-checked), `measure_impact` (max-abs, rel-L2,
top-1 flip, finiteness), `k_decode` / `q8_decode` (the real inference
kernels over synthetic seeded weights). Tests: payload faults bounded and
finite for all three formats, scale faults demonstrably produce non-finite
logits, corrupted headers detected by `validate_integrity`, bounds-checked
injection.

## 6. Recommendations

1. Keep the CLI's non-finite logit validation (structured bail) — it is the
   crash-safe boundary for the scale-fault class.
2. For long-running server contexts, run with `EMBER_VERIFY_QUANT=1` once at
   load; the one-pass cost is negligible relative to model load.
3. If top-1 stability under payload faults matters (e.g., benchmarking),
   margin-aware reporting (already the pilot methodology) is the right tool;
   do not add redundancy to the quant formats for this threat.
4. Determinism: no change needed — the existing serial ≡ parallel and
   scalar ≡ SIMD parity tests are the contract; the harness reuses the same
   kernels so fault measurements and determinism guarantees share one code
   path.
