# ember

[![rust](https://img.shields.io/badge/rust-1.92-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

a lightweight cpu-first llm inference engine in rust. runs quantized models
without heavy framework dependencies.

write-up: https://voidwest.github.io/ember/

## features

- **gguf v3 loader**: reads gguf model files, supports f32 and q8_0 dtypes.
- **backend trait**: model code is generic over a `Backend` trait for linear ops,
  embeddings, and element-wise math — swap cpu for gpu later without rewriting
  those paths. (attention is cpu-scalar for now; see design notes.)
- **explicit memory**: no hidden allocations in the inference hot path.
- **alloc-first design**: core tensor types and model code avoid `std` where practical, using `alloc` for vec-backed storage.

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
- **file format parsing**: gguf v3 loader with f32 and q8_0 quantization
  support, including fp16 scale dequantization.
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
| `--tokenizer` | `tokenizer.json` | path to tokenizer.json |
| `-p`, `--prompt` | `The` | text prompt to complete |
| `-n`, `--max-tokens` | `20` | tokens to generate |
| `-t`, `--temperature` | `0.8` | sampling temp (0 = greedy) |
| `--top-k` | (none) | top-k sampling |
| `--top-p` | (none) | nucleus sampling |
| `-i`, `--interactive` | (none) | repl mode after first prompt |
| `--demo` | (none) | fixed prompts with timing and deterministic output |
| `--benchmark` | (none) | print prefill/decode timing to stderr |

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

## architecture

the entry point is `main.rs` → `generate()`, which runs a two-phase loop:

1. **prefill** - forward pass on the full prompt, populating the kv cache.
2. **decode** - one token at a time, reading from the cache.

model components live in `src/model.rs` as generic `Block<B>`, `Attention<B>`,
`Mlp<B>`, `LayerNorm<B>`, and `Gpt2<B>`. tensors are `CpuTensor` in
`src/tensor.rs`. the gguf parser is `src/loader.rs`.

```text
main.rs              entry point, cli args, generation loop
├─ loader.rs         gguf v3 parser, tensor loading + dequant
├─ model.rs          gpt-2 transformer blocks
│  ├─ backend.rs     backend trait + cpu backend impl
│  ├─ tensor.rs      row-major f32 tensor with basic ops
│  └─ kv_cache.rs    flat key/value cache for incremental decode
├─ sampler.rs        temperature, top-k, top-p sampling
├─ tokenizer.rs      huggingface tokenizer wrapper
└─ quant.rs          q8_0 block dequantization
```

## design notes

- **backend trait**: the transformer is generic — `CpuBackend` is the default,
  but any type implementing `Backend` works. the trait abstracts linear ops,
  element-wise math, layer norm, and tensor lifecycle. **attention is a known
  exception**: the forward methods extract raw f32 slices via `data()` and run
  the attention math in scalar cpu loops, bypassing the backend abstraction.
  adding `fn attention(...)` to the trait is planned; for now the trait is
  honest about what it covers.
- **q8_0 quantization**: 8-bit block quantization (fp16 scale + 32 int8 values
  per block). reduces model size ~4× with minimal perplexity loss.
- **kv cache**: flat `[layer][head][seq_position][head_dim]` layout. prefill
  stores k/v for all prompt tokens; decode reads from cache and appends one
  token at a time.

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

**`matrixmultiply` as a placeholder.** the matmul delegates to
`matrixmultiply::sgemm` - pure rust, correct, no blas linking. it's not
optimal (no simd), but the `Backend` trait means simd kernels can be
swapped in under a new backend type without touching model code. the
crates.io crate is a deliberate stepping stone, not a final choice.

**softmax returns uniform for all-masked input.** when every logit is -inf
(fully masked row), softmax normally produces NaN. this code detects that
case and returns `1/n` per position. it costs one extra branch per row and
prevents the generation loop from producing NaNs on degenerate input.

## prerequisites

- rust stable toolchain
- a gguf model file (e.g. gpt2 in q8_0)
- tokenizer.json for the model (gpt-2's is included in the repo; for other models, point `--tokenizer` at their tokenizer file or use `download_gpt2_tokenizer_blocking()` from the library)

## current limitations

- matmul is scalar - no simd optimization yet.
- model loader is gpt-2 specific (gguf tensor names are hardcoded).
- not fully no_std - file i/o and mmap require std.
- attention math runs in cpu scalar loops even when a future gpu backend is
  plugged in — the `Backend` trait doesn't yet include `fn attention(...)`.

## license

mit
