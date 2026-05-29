# ember

[![rust](https://img.shields.io/badge/rust-1.92-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

a lightweight cpu-first llm inference engine in rust. runs quantized models
without heavy framework dependencies. also serves as a **hidden-state probing
platform** for linguistics research - extracting and analyzing per-layer
representations from decoder-only language models.

research write-up: https://voidwest.dev/ember

## features

- **gguf v3 loader**: reads gguf model files, supports f32, f16, and q8_0 dtypes.
- **on-the-fly dequantization**: q8_0 weights stay in block-compressed form in
  memory (~4x smaller than f32) and are dequantized during matmul.  the 3B
  model uses ~3.4 GB of ram instead of ~12.8 GB.
- **block-wise sgemm**: quantized matmuls dequantize in blocks of 256 columns
  and multiply with `matrixmultiply::sgemm` - 5x faster prefill than the
  scalar path.
- **backend trait**: model code is generic over a `Backend` trait for linear ops,
  embeddings, and element-wise math - swap cpu for gpu later without rewriting
  those paths. (attention is cpu-scalar for now; see design notes.)
- **explicit memory**: no hidden allocations in the inference hot path.
- **alloc-first design**: core tensor types and model code avoid `std` where practical, using `alloc` for vec-backed storage.
- **hidden-state probing**: extract per-layer activations at any token position.
  probe mode (`--probe`) feeds stimuli through the model and saves full
  hidden-state tensors as `.npy` for downstream analysis.
- **probing pipeline**: python scripts for linear probes (with task-specific
  splits, control tasks, and selectivity), CCA, RSA, divergence analysis,
  cross-model comparison plots, and
  tokenizer fertility analysis.

## what this demonstrates

- **systems programming in rust**: manual memory layout for the kv cache
  (`[layer][head][pos][head_dim]`), explicit stride math for tensor indexing,
  no hidden allocations in the hot path.
- **generic backend architecture**: the transformer is written against a
  `Backend` trait - the same model code works on cpu today and could run
  on gpu tomorrow without modification.
- **ml fundamentals**: causal multi-head attention with kv caching,
  numerically stable softmax (handles all-masked rows), layer norm,
  gelu activation, top-k/top-p sampling.
- **file format parsing**: gguf v3 loader with f32, f16, and q8_0
  quantization support.
- **memory-conscious inference**: q8_0 weights stay quantized in memory
  and are dequantized in blocks during matmul - the 3.2B model runs in
  ~3.4 GB of ram on consumer hardware.
- **edge case handling**: uniform fallback when every logit is -inf,
  categorical sampling with inverse cdf, nucleus cutoff logic.

## usage

```bash
cargo run --release -- --model gpt2.Q8_0.gguf --prompt "hello"
```

### flags

| flag | default | description |
|------|---------|-------------|
| `-m`, `--model` | `gpt2.Q8_0.gguf` | path to gguf model file |
| `--arch` | `gpt2` | model architecture: `gpt2` or `llama` |
| `--tokenizer` | arch-dependent | path to tokenizer.json (`tokenizer-gpt2.json` for gpt-2, `tokenizer.json` for llama) |
| `-p`, `--prompt` | `The` | text prompt to complete |
| `-n`, `--max-tokens` | `20` | tokens to generate |
| `-t`, `--temperature` | `0.8` | sampling temp (0 = greedy) |
| `--top-k` | (none) | top-k sampling |
| `--top-p` | (none) | nucleus sampling |
| `-i`, `--interactive` | (none) | repl mode after first prompt |
| `--demo` | (none) | fixed prompts with timing and deterministic output |
| `--delay-ms` | `0` | delay between tokens in demo mode (0 = instant) |
| `--benchmark` | (none) | print prefill/decode timing to stderr |
| `--probe` | (none) | run probe mode: extract hidden states from each block |
| `--probe-stimuli` | `stimuli/nonce_root_pattern.json` | path to stimuli json for probe mode |
| `--probe-output` | `data/activations.npy` | output path for probe activations (.npy) |
| `--probe-template` | `en_zero` | stimulus prompt key to probe (`en_zero`, `en_one`, `ar_zero`, `ar_one`, or generated controls) |
| `--probe-position` | `last` | hidden-state position to pool: `last`, `root`, `pattern`, or `prompt_mean` |
| `--probe-generate-tokens` | `16` | continuation length for probe behavioral scoring |

### demo mode

```bash
cargo run --release -- --demo
```

runs through a fixed set of prompts using greedy sampling (temperature 0)
for deterministic, repeatable output. useful for screen recordings
(`asciinema`, `script`, terminal capture) and benchmarking.

each prompt reports its completion, token counts, and per-phase timing.
a summary table at the end shows aggregate throughput across all prompts.

### interactive mode

```bash
cargo run --release -i
```

commands inside the repl: `/quit`, `/help`, `/stats`.

### probe mode

```bash
cargo run --release -- --probe --model Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama
```

feeds each stimulus from the stimuli json file through the model and collects
pooled per-layer hidden states at the selected prompt position. saves a 3d
`.npy` array `(n_stimuli, n_layers, embed_dim)` plus `_correctness.json` and
`_metadata.json` sidecars with next-token predictions, generated continuations,
match results, and the exact prompt template, position, model, shape, and token
selections used.
works with both gpt-2 and llama architectures through the `ForwardModel`
trait.

the `probes/` directory contains python scripts for downstream analysis:

| script | purpose |
|--------|---------|
| `train_linear_probe.py` | linear and small-MLP probes with task-specific CV splits, control tasks, and selectivity |
| `cca_analysis.py` | canonical correlation analysis, layer similarity matrices |
| `rsa_analysis.py` | representational similarity analysis, distance metrics |
| `divergence_analysis.py` | correct-vs-incorrect hidden state divergence |
| `tokenizer_fertility.py` | subword tokenization comparison across tokenizers |
| `plot_results.py` | visualization: probe accuracy, CCA/RSA heatmaps, cross-model comparison, fertility |
| `plot_root_scale_comparison.py` | compact root-accuracy comparison across Llama model scales |
| `run_probe_matrix.py` | repeatable model/template/position probe matrix runner |

stimuli are defined in `stimuli/` and generated by `stimuli/generate_stimuli.py`.
the current stimulus set targets **Arabic nonce root-pattern morphology** (200
stimuli: 20 roots x 10 patterns, from Alakeel et al. 2026).
pass `--include-ablations` to add masked-root, masked-pattern, both-masked, and
fake-pattern control prompts without changing the default stimulus output.

generated probe outputs (`*_activations.npy`, `*_activations_correctness.json`,
`*_activations_metadata.json`, `.npz` bundles, ad hoc plots, logs, and Python
bytecode caches) are ignored.
checked-in fixtures and published figures are kept small and explicit.

for hardening runs, `train_linear_probe.py --probe-kind mlp` tests whether
features that drop under linear probing remain recoverable non-linearly, and
`run_probe_matrix.py --dry-run` prints the full extraction/analysis command
matrix for model, prompt-template, and probe-position ablations.

## testing

```bash
cargo fmt -- --check
cargo test
cargo clippy -- -D warnings
```

the integration suite covers tensor operations, sampling, tokenizer loading,
and an in-memory GGUF parser fixture. the model smoke test also runs a GPT-2
forward pass when `gpt2.Q8_0.gguf` is present locally; otherwise it skips so CI
does not need to download large model weights.

### llama models

ember supports llama-compatible architectures via `--arch llama`. the following
models have been tested:

- **Llama 3.2 1B Instruct** (`Llama-3.2-1B-Instruct-Q8_0.gguf`) - 1.2B params, Q8_0 (~1.3 GB)
- **Llama 3.2 3B Instruct** (`Llama-3.2-3B-Instruct-Q8_0.gguf`) - 3.2B params, Q8_0 (~3.4 GB)
- **Llama 3.1 8B Instruct** (`meta-llama-3.1-8b-instruct.Q8_0.gguf`) - 8B params, Q8_0 (~8.5 GB)
- **Qwen2.5 1.5B Instruct** (`qwen2.5-1.5b-instruct-q8_0.gguf`) - 1.5B params, Q8_0 (~1.8 GB)

qwen2.5 models use the same `--arch llama` flag - ember auto-detects the
architecture from gguf metadata and supports qwen2-family models through
the same inference path.

download a quantized gguf from huggingface (e.g.
[unsloth/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF)),
then run:

```bash
cargo run --release -- \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --arch llama \
  --prompt "The capital of France is" \
  -n 30 \
  --temperature 0
```

> **note**: if `--tokenizer` is omitted, ember picks `tokenizer-gpt2.json`
> for `--arch gpt2` and `tokenizer.json` for `--arch llama`.

> **note**: interactive (`-i`) and demo (`--demo`) modes are not yet wired
> for llama. the single-prompt generation path and probe (`--probe`) mode
> work with both architectures.

## research: arabic morphology probing

ember has been used to probe how Llama 3.2 models (1B, 3B, 8B) represent
Arabic nonce root-pattern morphology. key findings:

- **root identity becomes less linearly decodable at scale**: root probe
  accuracy drops from 100% (1B, all layers) -> 78% (3B mid-layers) -> 70%
  (8B mid-layers), forming a deepening U-shaped curve.
- **pattern identity becomes more surface-accessible at scale**: pattern probe
  accuracy at layer 0 rises from 20% (1B) -> 100% (3B) -> 68.5% (8B), and
  all models reach 100% by early layers (L3-L11 depending on scale).
- **no model produces Arabic output**: all three scales generate "The" for
  every Arabic prompt - morphological information is encoded in hidden states
  but not used for generation in these English-centric models.
- **Llama 3 tokenizer is balanced for Arabic**: ar/en token ratio is 1.2x
  vs 2.4x for GPT-2's tokenizer on the same prompts.

full research write-up: https://voidwest.dev/ember

## architecture

the entry point is `main.rs` -> `generate()` (or `generate_llama()` for llama models), which runs a two-phase loop:

1. **prefill** - forward pass on the full prompt, populating the kv cache.
2. **decode** - one token at a time, reading from the cache.

model components live in `src/model.rs` as generic `Block<B>`, `Attention<B>`,
`Mlp<B>`, `LayerNorm<B>`, and `Gpt2<B>`. tensors are `CpuTensor` in
`src/tensor.rs`. the gguf parser is `src/loader.rs`.

```text
main.rs              entry point, cli args, dispatch, probe mode
|- loader.rs         gguf v3 parser, tensor loading
|- model.rs          gpt-2 + llama transformer, ForwardModel trait
|  |- backend.rs     backend trait + cpu backend impl
|  |- tensor.rs      row-major f32 tensor, rope, silu, elemul
|  `- kv_cache.rs    flat k/v cache, gqa-aware (n_kv_heads)
|- sampler.rs        temperature, top-k, top-p sampling
|- tokenizer.rs      huggingface tokenizer wrapper
|- quant.rs          q8_0 block dequantization + QuantizedWeight
`- probes/           python probe scripts (linear, cca, rsa, divergence)
```

## design notes

- **backend trait**: the transformer is generic - `CpuBackend` is the default,
  but any type implementing `Backend` works. the trait abstracts linear ops,
  element-wise math, layer norm, attention, and tensor lifecycle. the current
  attention backend is still scalar cpu code, but the model no longer owns
  those kernels directly.
- **q8_0 quantization**: 8-bit block quantization (fp16 scale + 32 int8
  values per block). weights stay in this quantized form in memory and
  are dequantized on the fly during matmul - ~4x smaller than f32 at
  rest with minimal perplexity loss.
- **kv cache**: flat `[layer][head][seq_position][head_dim]` layout. prefill
  stores k/v for all prompt tokens; decode reads from cache and appends one
  token at a time. uses `n_kv_heads` (not `n_heads`) for the head dimension,
  supporting grouped-query attention with zero overhead for mha models
  (`n_heads == n_kv_heads`).

## design justifications

these are the non-obvious trade-offs made in this codebase.

**transposed embeddings on load.** gguf stores token/position embeddings as
`[vocab, embed]`. the loader transposes them so `index_select` picks a row
directly - one contiguous slice per token - instead of gathering strided
elements at inference time. the cost is one transpose at load; the benefit
is simpler and faster lookups in the hot loop.

**`load_from_cpu` on the backend trait.** the method loads host-side f32
data into a backend tensor. for `CpuBackend` this is a thin wrapper around
`CpuTensor::from_data`; a future gpu backend would copy the data to device
memory here. the name was chosen over `from_cpu` to avoid tripping
`clippy::wrong_self_convention` (which expects `from_*` to be a constructor
without `&self`).

**`n_layers` is stored but never read.** the kv cache allocates per-layer
storage using `n_layers` in `new()`, then never reads the field again. it
exists only to size the flat buffer. removing it would require threading
the layer count through every cache method or hardcoding it. storing it is
the more explicit path.

**`matrixmultiply` for cpu matmul.** both f32 and q8_0 matmuls go through
`matrixmultiply::sgemm` - pure rust, no blas linking, decent simd.  the
`Backend` trait means faster kernels can be swapped in under a new backend
type without touching model code.  this is a pragmatic default, not a
final answer.

**softmax returns uniform for all-masked input.** when every logit is -inf
(fully masked row), softmax normally produces NaN. this code detects that
case and returns `1/n` per position. it costs one extra branch per row and
prevents the generation loop from producing NaNs on degenerate input.

## prerequisites

- rust stable toolchain
- a gguf model file (e.g. gpt2 in q8_0)
- a tokenizer file for the model (`tokenizer.json` for llama, `tokenizer-gpt2.json` for gpt-2; both are included in the repo)

## current limitations

- attention math is abstracted behind the `Backend` trait, but the only
  implementation today is scalar cpu code.
- the lm head (128K output features) is the throughput bottleneck during
  decode - each token does 501 sgemm calls for the lm head alone. batching
  or a fused/deferred lm-head path is the next obvious optimization target.
- model loader supports gpt-2 and llama architectures (gguf tensor names are
  hardcoded per arch). demo and interactive modes are not yet wired for
  llama; single-prompt generation and probe mode work with both.
- not fully no_std - file i/o and mmap require std.

## license

mit
