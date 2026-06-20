# Bug Report: 2026-06-20 Paper Revision

## Summary

During the 1.4k → 5k scale-up of Arabic morphology probing results, a systematic
re-examination uncovered five distinct categories of errors in the paper,
extraction pipeline, and interpretation. All have been fixed. This document
records what was wrong, how it was found, and what was done.

---

## 1. Qwen2.5-1.5B: Silently Broken Model (All Results Invalid)

**Severity**: Critical — affected the original 1,416-token paper's Table 3 and all
5k re-extraction attempts.

**What was wrong**: Qwen2.5-1.5B GGUF uses `qwen2` architecture keys internally
(`general.architecture: qwen2`). Ember's `--arch qwen3` dispatch loads `qwen3.*`
GGUF tensor keys. The tensor mapping is incorrect — weights are loaded into wrong
parameter slots. The forward pass produces structured-but-wrong hidden states.

**Evidence**:
- Generation test: `echo "The capital of France is" | ember --arch qwen3 --model Qwen2.5-1.5B-Instruct-Q8_0.gguf` → `"followingVILLEVILLEVILLEVILLE"` (garbage)
- Activation statistics at 1.4k: range [-1567, +1257] (extreme, should be ~[-5, +5])
- Probe accuracy at 5k: POS 60% (below 62.9% majority), gender 65% (-29pp Δchar)
- Both the original 1.4k GGUF (`qwen2.5-1.5b-instruct-q8_0.gguf`, 1.89 GB, SHA d7efb07) and the new 5k GGUF (`Qwen2.5-1.5B-Instruct-Q8_0.gguf`, 1.65 GB, SHA 7185d30) produce garbage — different files, same architecture mismatch

**Fix**: Retracted Qwen2.5 from all paper claims. Abstract, introduction, methods,
results, discussion, conclusion, table captions, and appendix all updated to
reflect N=2 models. Added explicit retraction explanation noting it's an Ember
implementation issue, not a model property.

---

## 2. "Final-Token" = the Period, Not the Arabic Word

**Severity**: High — the paper's methods were ambiguous and the interpretation of
results depended on knowing which token was probed.

**What was wrong**: The paper stated "we extract final-token hidden states" but
never specified that the "final token" is the period (".") at the end of the
fixed prompt template ("Predict the token morphology."), not the final subword
token of the Arabic stimulus word. This was discovered by tracing the extraction
code in `src/main.rs`:

```rust
// select_probe_indices -> ProbePosition::Last ->
//   non_special_token_indices(offsets, token_ids.len()).last()
// non_special_token_indices filters tokens where start == end (special tokens)
// The last token with start != end is "." for all stimuli and both tokenizers
```

**Evidence**:
- Llama tokenizer (GPT-2 BPE): token [47] = ".", token [46] = " morphology"
- Qwen3 tokenizer: token [52] = ".", token [51] = " morphology"
- Both consistently produce "." as the final non-special token
- Token-level char n-gram baseline on the period: 62.9% (majority — no signal)
- Token-level char baseline on "morphology": also majority (constant token)

**Fix**: Added explicit clarification in six locations:
- Abstract: "All experiments extract hidden states from the final token of a
  fixed prompt template — which, for our template, is the period ('.')"
- Methods §4.1: Detailed explanation with code-level justification
- Appendix §A: Explicit "Probed token position" paragraph
- Limitations: Updated from "Multi-token words may distribute..." to "We extract
  from the period, not from Arabic stimulus tokens"
- Dataset §3: Tokenization discussion updated
- Layer 0 definition: "output of first transformer block (post-attention,
  post-FFN), not raw token embedding"

---

## 3. Llama "L0 = Embedding Layer" Was Wrong

**Severity**: High — affected the architectural interpretation of results.

**What was wrong**: The paper (and our analysis) characterized Llama's best
layer as "the embedding layer" and claimed "Llama embeddings encode morphology."
Reading the actual extraction code at `src/llama.rs:1024`:

```rust
for (li, block) in self.blocks.iter().enumerate() {
    x = block.forward(backend, &x)?;   // line 1024: forward pass
    let data = backend.data(&x);        // line 1025: collect AFTER forward
```

Layer 0 activations are collected AFTER `block.forward()` — meaning they are the
output of the first transformer block (self-attention + FFN), not the raw token
embedding lookup. The period token at position 47 receives context from all
preceding tokens through block-0 self-attention, which is how it encodes the
Arabic stimulus information.

**Evidence**:
- Direct code read: `x = block.forward(backend, &x)?;` then `let data = backend.data(&x);`
- Token-level char baseline: period token = 62.9% (majority), confirming the
  period carries no surface POS signal by itself
- Block-0 attention propagates Arabic word context to the period position

**Fix**: Updated all references from "embedding layer" to "first transformer
block output." Added explicit note in methods §4.1 and appendix §A.

---

## 4. "Monotonic Decline" Was False

**Severity**: Medium — incorrect characterization of Llama's layerwise pattern.

**What was wrong**: The paper draft claimed Llama showed "monotonic decline"
in POS accuracy across layers. Direct re-computation of the per-layer values:

```
Llama POS lemma-heldout per-layer:
L0: 84.6%  L4: 81.8%  L7: 83.0%  L10: 80.8%
L1: 84.4%  L5: 82.2%  L8: 80.5%  L11: 79.4%
L2: 82.0%  L6: 81.9%  L9: 80.5%  L15: 76.6%
```

Four upward steps (L4, L5, L7, L10) disprove monotonicity. L1 is near-identical
to L0 (84.4% vs 84.6%). The gap from best to worst is 8.0pp — a weak downward
trend, not a clean monotonic decline.

**Fix**: Replaced "monotonic decline" with "declining weakly over subsequent
layers (84.6% → 76.6% over 16 layers, with minor oscillations; not a clean
monotonic decline)" in §5.3.

---

## 5. Fabricated 1.4k Numbers Introduced During Edits

**Severity**: Critical — the agent introduced numbers that don't exist in any
data file and contradicted the original paper's verified values.

**What was wrong**: During the paper revision, three fabricated claims were
introduced:

| Claim | Fabricated | Actual (from heldout_probe_results.json) |
|-------|-----------|------------------------------------------|
| Qwen3 1.4k POS accuracy | 79.9% | 86.0% |
| Llama 1.4k POS accuracy | 82.1% | 84.0% |
| Llama 1.4k Δchar | +2.0pp | +14.5pp |
| "Llama gap flipped from marginal" | fabricated narrative | Original always showed +14.5pp |

The original paper correctly reported Δchar lifts (+16.5%, +14.5%, +8.7%).
During editing, fabricated raw accuracy numbers and a false narrative about
Llama's gap "flipping" were introduced from memory rather than from data files.

**How found**: The original paper's reviewer noticed the numbers didn't match the
original uploaded manuscript. Cross-referencing against the actual
`heldout_probe_results.json` files revealed the discrepancies. The fabricated
values (79.9%, 82.1%, +2.0pp) don't appear in any probe output file.

**Fix**: Restored correct values from actual data files. Verified each number
against source JSON. Removed the entire "Llama gap flipped" narrative.

---

## 6. Char N-Gram Baseline Discrepancy

**Severity**: Low — did not affect conclusions but indicated inconsistent
methodology between verification scripts.

**What was wrong**: Two different char n-gram baseline computations produced
different values for the same Llama POS lemma-heldout data:

| Script | ngram_range | Surface source | Result |
|--------|------------|----------------|--------|
| verification.py | (2,5) | surface_dediac field | 67.2% |
| run_heldout_probes.py | (1,4) | surface + manual diacritic strip | 69.3% |

The 2.1pp difference comes from two independent configuration choices. Neither
is wrong, but the discrepancy went unnoticed until both were run side-by-side.

**Fix**: Paper consistently uses the heldout script's canonical values
(69.3% char baseline for Llama POS lemma-heldout, producing Δ=+15.2pp).
Documented the difference in the verification notes.

---

## 7. CI Failures After Code Changes

**Severity**: Low — blocked CI but did not affect correctness.

**What was wrong**: The batched extraction code changes introduced:
- `AttentionSpec` new field `block_boundaries` — integration test at
  `tests/integration.rs:645` wasn't updated
- Clippy `needless_range_loop` warnings in the modified attention loops
- Clippy `dead_code` on `Gpt2::forward_pooled_with_blocks`
- Clippy `manual_is_multiple_of` in the extraction progress print
- `cargo fmt` formatting differences

**Fix**: All resolved in commit `9dbf343`. Tests pass (72/72), clippy clean,
fmt clean.

---

## Remaining Issues (Not Bugs, Documented Limitations)

1. **Gender Δchar asymmetry**: Qwen3 +12.3% vs Llama +4.4% for gender
   lemma-heldout. Flagged in discussion §6.2 as unexamined; may reflect
   architectural divergence or tokenization artifact.

2. **Abstract pattern heldout incomplete**: 443-class Ridge OVR failed to
   complete under heldout splits within reasonable time. Only random CV
   results available. Paper correctly states these tasks are "not meaningfully
   evaluable under closed-set heldout classification."

3. **N=2 models**: All architectural claims scoped to "in these two models,
   not claimed as a general architectural law."

---

## Lessons

1. **Numbers must be traceable to source files.** Every claim in the paper
   should have a corresponding entry in a probe output JSON or a direct
   computation script. Numbers from memory are wrong.

2. **"What does the code actually do?" beats "what do we think it does?"**
   The period-token and L0-as-post-attention discoveries came from reading
   `src/main.rs` and `src/llama.rs`, not from reasoning about the architecture.

3. **Silent failures are worse than crashes.** Qwen2.5 loaded without errors
   but produced garbage. A generation-quality smoke test would have caught this
   immediately. Should be added to the extraction pipeline.

4. **Char baseline methodology must be pinned.** Different n-gram ranges and
   surface normalization produce different baselines. The canonical values
   come from `run_heldout_probes.py` with its specific configuration.
