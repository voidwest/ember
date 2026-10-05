# Ember support matrix

What "supported" means in Ember, and which surfaces are covered by it.

A surface is **supported** only when all five hold:

1. a documented contract exists (this file plus the linked doc/module docs);
2. at least one automated test covers that contract;
3. known limitations are documented;
4. failures surface cleanly (typed error or explicit warning — no silent fallback);
5. a reproducible validation path exists for numerical claims (`docs/validation.md`).

Everything else is one of: **experimental** (runnable, unstable, weaker or no
automated coverage), **internal** (implementation detail, no promise), or
**unsupported** (fails closed with a clear error).

The runtime mirrors this file: loading a GGUF whose architecture is outside the
matrix logs a warning, and `--strict-support` turns that warning into a failure.

## Model families

| Declared `general.architecture` | Level | Evidence / limitations |
|---|---|---|
| `llama` | Supported | Golden logits vs pinned llama.cpp (v0.3 ladder); decode, KV-cached continuation, tracing, extraction |
| `qwen2` (Qwen2/Qwen2.5) | Supported | Golden checked on Qwen2.5-1.5B (top-1 agreement 100%, cosine ≥ 0.9963); q/k/v biases loaded |
| `qwen3` | Experimental | Execution path works and is exercised locally; golden-logit target still pending — treat outputs as research-grade |
| `gemma3` / `gemma4` | Experimental | Dense text-only path; the final-logit parity record is incomplete and the user-facing validation note marks outputs untrusted. No MoE, no multimodal, no K-quant loader support for this family |
| `gpt2` | Internal | Loader/negative-control baseline; generation is not a supported user path |

The declared metadata string decides the level (`qwen2` and `qwen3` are
distinct even though both dispatch through the shared llama-family engine).

## Quantization

| GGUF tensor type | Level | Execution |
|---|---|---|
| F32 / F16 / BF16 | Supported | Converted to f32 at load; generic kernels |
| Q8_0 | Supported | Block-compressed, mmap-backed; scalar/AVX2/AVX-512/ARM dot-product decode; packed VNNI layout cached to disk on supported x86 CPUs |
| Q4_K / Q6_K | Supported | Compressed-resident scalar/AVX2/ARM dot-product kernels (AVX-512 tier opt-in via `EMBER_K_AVX512`); ARM arithmetic is checked exactly against scalar; oracle and llama.cpp ladders remain separate. The production numerical gate is the pinned llama.cpp golden ladder; the cross-tier eager-f32 comparison is a cosine envelope on shared-prefix steps, not exact greedy-token equality ([validation](validation.md)) |
| Q2_K / Q3_K / Q5_K | Experimental | Eager-f32 fallback, recorded per tensor and warned at load; expect 2.6–4.5× more RAM and much slower decode |
| Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_1 / Q8_K | Unsupported | The loader rejects them with a typed error |

## Execution paths

| Path | Level | Notes |
|---|---|---|
| `reference` | Supported | The v0.3 generic hooked path; the oracle for parity tests |
| `planned` | Supported | Default for llama/qwen3 decode; tokens bit-identical to reference, logits within the documented 1e-3 envelope |
| `planned-fused` | Evolving | Frozen fusion set F1–F5 with hook-driven de-fusion; some kv subcommands reject it |

## Other surfaces

| Surface | Level | Notes |
|---|---|---|
| CLI documented in `docs/usage.md` | Supported | Behavior changes follow `docs/api-stability.md` |
| Versioned artifacts (`ember.experiment.v1`, `ember.bundle.v1`, `ember.kv-snapshot.v1`, `ember.agent.trace.v1`) | Supported | Unknown schema majors fail closed |
| Rust library API | **No stable subset yet** | Everything public is experimental until the 1.0 API freeze; `#[doc(hidden)]` modules are internal (see `docs/api-stability.md`) |
| Python binding (`inspect`, `plan`, `diff`, `diff_corpus`) | Experimental (pinned to the CLI/JSON contracts) | Built and tested on x86_64 Linux CI |
| Agent runtime + built-in tools | Evolving | No shell/network/delete tools by design; limits documented in `docs/agent-runtime.md`; `--sandbox-root` is a path guard, not an OS sandbox |
| Tracing (`ember.agent.trace.v1`) | Evolving | Prompt/generated text is opt-in (`--trace-content`); see `docs/trace-schema.md` |
| Hidden-state capture/intervention (v0.5 experiments, probes) | Experimental by policy | Weaker stability promise is deliberate; the schemas are versioned |
| Multimodal image (SmolVLM), video (SmolVLM2), audio (Ultravox) | Experimental | Correctness rests on reference ladders and hermetic tests; E2E weight tests are env-gated and not in CI |
| Speech output (OuteTTS + WavTokenizer, MMS-VITS) | Experimental | Reference weights are CC-BY-NC-4.0; not product-usable as shipped |
| Live duplex voice | Experimental | Non-default `audio` feature; requires local audio hardware; never exercised in CI |
| Linux x86_64 | Supported | Primary CI target (AVX2 baseline, scalar fallback always available) |
| Linux aarch64 | Supported (headless) | CI target; NEON elementwise/dequant kernels; native Q8_0/Q4_K/Q6_K dot-product kernels when CPU features permit, scalar fallback otherwise |
| macOS (arm64, x86_64) | Unclaimed | CI builds, tests and lints the headless crate and compile-checks the GUI on both architectures; the only model-level gate there is the arm64 golden-path run of the pinned research example, and there is no general support promise; macOS ARM has local kernel/model validation described in [ARM kernels](arm-kernels.md) |
| Windows | Unclaimed | No CI tier or general support promise |

## Changing this file

Adding a row to the supported table requires a linked test or golden record;
moving a row down to experimental only requires a note. Removing support is a
breaking change and follows `docs/api-stability.md`.
