# ember

[![rust](https://img.shields.io/badge/rust-1.85-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

a lightweight, cpu-first llm inference engine in rust. designed for running
quantized models without heavy framework dependencies.

## features

- **gguf v3 loader**: reads gguf model files, supports f32 and q8_0 dtypes.
- **backend trait**: model code is generic over a `Backend` trait — swap cpu
  for gpu later without rewriting the transformer.
- **explicit memory**: no hidden allocations in the inference hot path.
- **no_std ready**: core types avoid `std` where possible. uses `alloc` only.

## what this demonstrates

- **systems programming in rust**: manual memory layout for the kv cache
  (`[layer][head][pos][head_dim]`), explicit stride math for tensor indexing,
  no hidden allocations in the hot path.
- **generic backend architecture**: the transformer is written against a
  `Backend` trait — the same model code works on cpu today and could run
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

### interactive mode

```bash
cargo run --release -i
```

commands inside the repl: `/quit`, `/help`, `/stats`.

## architecture

the entry point is `main.rs` → `generate()`, which runs a two-phase loop:

1. **prefill** — forward pass on the full prompt, populating the kv cache.
2. **decode** — one token at a time, reading from the cache.

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
  but any type implementing `Backend` works. this means the model code is
  written once and reused across hardware targets.
- **q8_0 quantization**: 8-bit block quantization (fp16 scale + 32 int8 values
  per block). reduces model size ~4× with minimal perplexity loss.
- **kv cache**: flat `[layer][head][seq_position][head_dim]` layout. prefill
  stores k/v for all prompt tokens; decode reads from cache and appends one
  token at a time.

## prerequisites

- rust stable toolchain
- a gguf model file (e.g. gpt2 in q8_0)
- tokenizer.json for the model

## current limitations

- matmul is scalar — no simd optimization yet.
- model loader is gpt-2 specific (gguf tensor names are hardcoded).
- not fully no_std — file i/o and mmap require std.

## license

mit
