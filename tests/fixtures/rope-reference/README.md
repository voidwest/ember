# Pinned reference RoPE fixture

`llama-q6-position15.json` contains the first 64-element Q and K head before
and after rotation at absolute position 15, Arabic frozen prompt evaluation 10.
Expected outputs come from the unchanged llama.cpp reference callback capture,
not Ember. The JSON records the reference commit, model SHA-256 and capture
SHA-256. Frequency factors are the pinned model's f32 `rope_freqs.weight` tensor;
head dimension and base frequency come from its metadata.

Local source evidence: `q6-isolated-rope/tensors.json`, with capture-on/off equality
for all 128,256 logits at evaluation 10. Values are exact f32 values serialized
with enough decimal digits to round-trip. The model is not needed to run the
regression. Inputs/outputs are also permuted into split-half layout to test the
same mathematical pairs through that implementation; this is not independent
Qwen model validation or evidence for other positions/platforms.

Exact reference-bit assertions are scoped to macOS AArch64, where this fixture
was collected and libm rounding was validated. No cross-platform bit guarantee
is inferred from this fixture.

Against retained pre-correction output, this fixture detects 22/64 Q and 21/64 K
bit differences. Recurrence-only output still differs in 8/64 Q and 4/64 K values,
so the fixture detects omission of the fused-order correction as well. Evidence:
`rope-candidate-broad/fixture-discrimination.json`.
