# ARM CPU quantized kernels

AArch64 systems with NEON and signed byte dot products use native Q4_K/Q6_K kernels. Q8_0 also requires FP16 conversion support. CPU features are detected at runtime; portable scalar implementations remain available on other systems. No Metal backend or nightly Rust is required.

Q8_0 decode processes four output columns together, reusing activation loads. Prompt tiles process two or four activation rows together, reusing weight loads. Q4_K and Q6_K unpack compressed weights into NEON registers and apply signed integer dot products; their four-row prompt tiles reuse the unpacked weights. All paths retain the scalar block accumulation and scaling order. SDOT uses a small inline assembly wrapper because Rust 1.92 does not expose its intrinsic on stable.

`--k-strategy auto` selects the native ARM tier when supported. `--k-strategy arm` requires it and records `compressed_arm` execution with `q4-k-q8-k-neon-dotprod` or `q6-k-q8-k-neon-dotprod` kernel identity. `--k-strategy scalar` remains an explicit scalar reference. The global strategy option precedes subcommands:

```sh
RAYON_NUM_THREADS=8 cargo run --locked --release -- --k-strategy arm bench-decode \
  --model models/arm-validation/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --execution planned --tokens 128 --warmups 2 --repetitions 5
```

Feature-unavailable explicit tiers fail unless `--k-allow-fallback` is set. Embedding lookup remains row dequantization; recording an ARM projection tier does not mislabel the embedding operation as a dot product. Execution plans record and validate their NEON+dotprod requirement.

The unit tests cover integer extremes, non-aligned Q8 buffers, prompt/output tails, scalar parity, and nonzero destination accumulation. Model validation can explicitly require ARM:

```sh
EMBER_PARITY_MODEL=models/arm-validation/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
EMBER_PARITY_TOKENIZER=tokenizer.json EMBER_PARITY_TIER=arm \
EMBER_PARITY_REQUIRE_ARM=1 RAYON_NUM_THREADS=8 \
cargo test --locked --release --test k_parity \
  arm_q8_k_is_bit_exact_with_scalar_on_frozen_prompts
```

That test compares all prefill activations and logits, each decode logit vector, and greedy tokens across six English/Arabic prompts. It requires exact equality between scalar and ARM K-quant arithmetic; it is separate from the approximate Q8_K-versus-f32 oracle gate. Model-gated tests report skips without their model environment variables, so an ordinary unit-test run is not evidence of real-model parity.

ARM planned normalization uses the same NEON reduction tree as reference
normalization, including fused residual normalization. This avoids scalar/NEON
rounding differences being amplified by subsequent Q8_K activation packing.
The fused path reuses its destination and does not allocate scratch storage.
Q8's existing fast decode path deliberately retains its established scalar
reduction order (with a vectorized apply step), preserving old generation and
next-token results. It is separate from planned K-quant normalization.

The eager-f32 validation oracle borrows mmap-backed K-quant payloads while
dequantizing. It does not allocate a duplicate compressed tensor; allocation
estimates reflect that saving. Reader-only loading still budgets the owned
compressed buffer, and all existing allocation limits remain unchanged.

These ARM changes introduced execution-plan kernel revision 3. The working
RoPE correction uses candidate revision 4; its broader validation is pending.
Existing revision-1/2/3
plans remain readable and verifiable offline; live execution requires rebuilding
the plan. Bundle reproduction rebuilds a current plan from the source inputs.

## M1 Pro tuning

Q4_K accumulates consecutive dot products directly into SDOT integer lanes,
removing separate vector additions while preserving exact scalar results. The
four-output Q8 decode tile remains intentional: eight- and sixteen-output
variants were substantially slower on the tested M1 Pro.

With more than eight ARM workers, Q8 output work is split into chunks of at
most 256 rows (aligned to the kernel tile). This lets Rayon redistribute work
across the M1 Pro's unequal core types. Smaller pools retain the previous
partitioning; using the smaller chunks at four workers regressed performance.

The large Q8 output projection can use four-row interleaved weight storage on
ARM with dot-product and FP16 support. The model's existing packed cache stores
and reuses this architecture-neutral layout. On the supplied Llama 3.2 1B
model, the uncached experiment added approximately 266 MiB of peak RSS and
50 ms of packing work; the dedicated decode comparison improved by about 2%
at four workers. Cache validation and numerical parity apply to the packed
bytes as well as the original row layout. `EMBER_PACKED_CACHE=0` disables disk
caching, not the in-memory layout.

For this machine and these models, start with `RAYON_NUM_THREADS=4` for Q8_0
and `RAYON_NUM_THREADS=10` for Q4_K_M. The fastest tested Q8 route uses
`--execution reference`. These are model-specific settings, not universal
thread recommendations. Background macOS work can materially reduce throughput,
especially with ten workers.

Keep the pinned Rust toolchain and existing thin-LTO release profile. Native
CPU targeting, one codegen unit, fat LTO, and profile-guided optimization did
not provide a convincing overall gain in the tested workloads. An experimental
macOS QoS override was also rejected; no scheduling-priority override is applied.

FP16 KV-cache attention also has a native ARM path. Four half values convert
with FCVTL and multiply in NEON registers. Dot products then add the four
products in their original order, with Rust's negative-zero sum identity;
weighted additions use separate multiplication and addition. This preserves
the scalar result rather than changing the reduction tree or enabling FMA.
Runtime FP16 detection retains the scalar fallback. Exact tests cover every
finite half value, signed zeros, slice offsets and SIMD tails; model checks
cover the frozen reproduction and 1,282-token prompt continuations.
