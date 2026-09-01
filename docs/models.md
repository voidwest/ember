# Ember models and quantization

Supported families, quantization encodings, and
model-specific notes. Moved from the top-level README.

## supported models and quantization

| CLI architecture | Supported model family | Status and boundaries |
|------------------|------------------------|-----------------------|
| `gpt2` | GPT-2 GGUF | baseline generation and tensor tests; experiment hooks are not integrated |
| `llama` | dense LLaMA-family decoders | generation, KV-cached decode, tracing, extraction, and both built-in experiments |
| `qwen3` | Qwen3 dense decoders | explicit split-half RoPE and pre-RoPE QK normalization; generation, tracing, extraction, and both experiments |
| `qwen3` | Qwen2/Qwen2.5 metadata through the shared loader | experimental; tokenizer and real-model validation remain model-specific |
| `gemma4` | dense text-only Gemma 4 | generation, tracing, extraction, packed gate/up dispatch, and both experiments; no MoE or multimodal path |

The GGUF loader accepts these tensor encodings:

| GGUF tensor type | In-memory treatment | CPU execution |
|------------------|---------------------|---------------|
| F32 | materialized as f32 | generic f32 kernels |
| F16 | converted once to f32 while loading | generic f32 kernels |
| BF16 | converted once to f32 while loading | generic f32 kernels |
| Q8_0 | retained block-compressed, mmap-backed for file loads | scalar/AVX2/AVX-512 decode and tiled/packed prefill dispatch |
| Q2_K / Q3_K / Q4_K / Q5_K / Q6_K | dequantized once to f32 while loading | generic f32 kernels |

K-quant tensors are dequantized to f32 at load (transcribed from llama.cpp;
validated against llama.cpp logits and fp16 source tensors: see
`src/quant_k.rs`). A GGUF may mix the listed types, as real models commonly
do for norms, embeddings, and linear weights. “Supported” here means Ember
has an execution path; external golden-logit or activation-reference status
is tracked separately in the validation tables below.

Note: the K-quant path is a research loader, not a deployment path :
dequant-to-f32 makes Q4/Q6 models use 2.6–4.5× more RAM and run 16–52×
slower than Q8 (the Q8 path keeps weights compressed and uses the fast
decode path).


### llama models

ember supports llama-compatible architectures via `--arch llama`. qwen-family
ggufs run through the same llama-family model path; use `--arch qwen3` for
qwen3-specific metadata handling. the following models have been tested:

- **llama 3.2 1b instruct** (`Llama-3.2-1B-Instruct-Q8_0.gguf`) - 1.2b params, q8_0 (~1.3 gb)
- **llama 3.2 3b instruct** (`Llama-3.2-3B-Instruct-Q8_0.gguf`) - 3.2b params, q8_0 (~3.4 gb)
- **llama 3.1 8b instruct** (`meta-llama-3.1-8b-instruct.Q8_0.gguf`) - 8b params, q8_0 (~8.5 gb)
- **qwen2.5 1.5b instruct** (`qwen2.5-1.5b-instruct-q8_0.gguf`) - 1.5b params, q8_0 (~1.8 gb)

Both `--arch llama` and `--arch qwen3` dispatch to the shared Llama-family
implementation; the GGUF `general.architecture` metadata selects the `llama`,
`qwen2`, or `qwen3` configuration keys. The smoke wrapper labels Qwen2.5 as
`qwen3` and passes `tokenizer-qwen2.5.json` explicitly. Qwen2/Qwen2.5
attention projections carry q/k/v biases, which the loader now picks up;
a golden-logit check on Qwen2.5-1.5B matches llama.cpp (top-1 agreement,
max abs diff 0.29).


### support status

| architecture | loads | generates | probe smoke | full 200-stimulus probe | golden checked |
|--------------|-------|-----------|-------------|--------------------------|----------------|
| gpt-2 | yes | yes | yes | not standard | no |
| llama | yes | yes | yes | yes, local/cloud depending on size | no |
| qwen2.5 | yes, via `--arch qwen3` (attention projection biases loaded) | yes, coherent after bias fix | selected smoke runs | pending | yes, 1.5B vs llama.cpp (top-1 match, max diff 0.29) |
| qwen3 | yes, via `--arch qwen3` | yes | yes, 5-stimulus local smoke | yes, Qwen3 0.6B local run | no |
| gemma4 | yes | yes, coherent English | one-stimulus local smoke | pending | no (cosine ~0.87; L0 bit-identical; remaining gap unresolved) |

hidden-state probe results should be treated as research-grade only after a
trusted-reference logits or activation check exists for the exact architecture,
model file, tokenizer, and quantization path. gemma4 golden-logit checks now cover block layout, PLE, global projection,
embedding scaling, layer scales, GELU tanh, RoPE freq_factors, and BF16
loading. RMSNorm amplification of small upstream differences is the current
working explanation for the remaining cosine gap, not a completed root-cause
proof. See `docs/gemma4-parity-investigation.md` and
`docs/layer-dump-tooling.md` for details.

Ember can emit last-prompt logits for external golden checks:

```bash
cargo run --release -- \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" \
  --dump-logits data/qwen3_france_logits.npy
```

Compare against trusted reference logits with token metadata from both sides:

```bash
python probes/check_golden_logits.py \
  --ember data/qwen3_france_logits.npy \
  --reference reference/qwen3_france_logits.npy \
  --metadata data/qwen3_france_logits_metadata.json \
  --reference-metadata reference/qwen3_france_logits_metadata.json \
  --output data/qwen3_france_golden_report.json
```

Probe classifiers scale activations by default and use a higher logistic
regression iteration limit to avoid premature convergence failures:

```bash
python3 probes/train_linear_probe.py \
  --activations data/activations.npy \
  --stimuli stimuli/nonce_root_pattern_surface.json \
  --max-iter 2000 \
  --scale
```

Use `--no-scale` only when intentionally comparing against an unscaled probe
baseline.


### gemma 4 text models

ember supports dense text-only gemma 4 models via `--arch gemma4`. the path
targets e2b/e4b/31b-style ggufs with f32, f16, or q8_0 weights. it rejects
moe gemma 4 models, multimodal inputs, speculative drafter models, and
k-quantized ggufs in this first pass.

the gemma 4 loader handles long-context rope without cloning per-layer tables,
uses packed q8 per-layer embeddings without full dequantization, projects
per-layer embedding chunks through `blk.N.proj.weight`, and supports probe mode
for hidden-state extraction. a one-stimulus smoke probe on
`gemma-4-E2B-it.Q8_0.gguf` produced activations with shape `(1, 35, 1536)`.

```bash
cargo run --release -- \
  --arch gemma4 \
  --model models/gemma-4-E2B-it.Q8_0.gguf \
  --tokenizer tokenizer-gemma4.json \
  --prompt "The capital of France is" \
  -n 8 --temperature 0 --benchmark
```

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
> for `--arch gpt2`, `tokenizer.json` for llama/qwen, and
> `tokenizer-gemma4.json` for `--arch gemma4`.

> **note**: demo (`--demo`), single-prompt generation, and probe (`--probe`)
> mode work across the supported model families. Interactive mode (`-i`)
> remains GPT-2-only.
