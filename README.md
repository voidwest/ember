# ember

A lightweight, CPU-first LLM inference engine written in Rust. Designed for efficiency and portability, ember provides a modular architecture for running quantized models without the overhead of heavy deep-learning frameworks.

## features

- **gguf support**: native loader for GGUF model files (v3).
- **quantization**: built-in support for Q8_0 dequantization for reduced memory footprint.
- **backend agnostic**: core logic is generic over a `Backend` trait, allowing easy swaps between CPU and future GPU implementations.
- **explicit memory**: no hidden allocations during inference to ensure predictable performance.
- **no_std friendly**: core types avoid `std` where possible to facilitate future embedded ports.

## tech stack

- **rust 2021**
- **tokenizers** (huggingface)
- **half** (f16 support)
- **memmap2**
- **anyhow/thiserror** (robust error handling)

## getting started

### prerequisites

- rust toolchain (latest stable)
- a GGUF model file (e.g., `gpt2.Q8_0.gguf`)
- a corresponding `tokenizer.json`

### setup

1. clone the repository into your workspace.
2. place your `.gguf` and `tokenizer.json` in the project root.

### usage

the engine currently supports GPT-2 style architectures. to run a basic inference:

```bash
cargo run --release
````

the current main.rs demonstrates loading a model, encoding a prompt, and predicting the next token using the CpuBackend.

## architecture

```text
┌────────────────┐
│   GGUF Model   │
└────────┬───────┘
         │
┌────────▼───────┐      ┌──────────────┐
│  GgufLoader    ├──────┤ Tokenizer    │
└────────┬───────┘      └──────┬───────┘
         │                     │
┌────────▼───────┐      ┌──────▼───────┐
│  CpuBackend    │◄─────┤ Tensor Ops   │
└────────┬───────┘      └──────┬───────┘
         │                     │
┌────────▼───────┐      ┌──────▼───────┐
│  Transformer   │      │ KV Cache     │
│    Blocks      │◄─────┤ (Context)    │
└────────┬───────┘      └──────────────┘
         │
┌────────▼──────────┐
│  Logits / Token   │
└───────────────────┘
```

## design decisions

### why a custom backend trait?

to decouple the model logic from the hardware. by using the Backend trait, the transformer implementation remains identical whether running on a single-threaded CPU or a high-performance SIMD/GPU backend.

### why explicit memory?

inference engines often suffer from "allocation churn." ember prioritizes explicit memory management and avoids hidden Vec allocations in the hot path to keep execution deterministic.

### why Q8_0 quantization?

Q8_0 offers a significant reduction in model size with minimal perplexity loss. it is the "gold standard" for balanced local CPU inference.

## known limitations

* naive matmul: the current matrix multiplication is a basic implementation and needs SIMD optimization.
* GPT-2 specific: the model loader is currently tuned for GPT-2 architecture patterns.
* incomplete no_std: while designed for it, the project still relies on alloc and some std file I/O for loading.

## roadmap

* [ ] SIMD-accelerated CPU kernels (AVX/NEON)
* [ ] Q4_0 and Q4_K quantization support
* [ ] llama/mistral architecture support
* [ ] full no_std certification for embedded targets
* [ ] interactive CLI chat mode
* [ ] full developer blog writeup

## license
MIT

