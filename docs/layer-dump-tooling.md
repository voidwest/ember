# Layer Dump Tooling

Tooling for comparing Ember per-layer hidden states against llama.cpp per-layer
hidden states. Used to identify where a model begins to diverge during GGUF
parity debugging. Introduced during Gemma 4 parity work.

## What this tooling does

1. Runs a single prompt through both Ember and llama.cpp.
2. Captures per-layer hidden states (last prompt token) from both.
3. Compares them layer-by-layer: cosine similarity, L2 norms, mean/max absolute
   difference.
4. Produces a table, optional Markdown report, and optional JSON metrics.

## Prerequisites

- Rust toolchain (for Ember)
- C++17 compiler (for llama.cpp helper)
- Python 3 with `numpy` (for comparison script)
- A patched llama.cpp (see below)

## Building the llama.cpp helper

The helper at `tools/dump_llamacpp_layers.cpp` requires a patched llama.cpp with
per-layer state capture enabled. Without the patches, it falls back to dumping
only the final hidden state.

Three source files must be modified in the llama.cpp checkout:

### Patch 1: `src/llama-graph.h`

In `llm_graph_result`, after `t_h_nextn`, add:

```cpp
std::vector<ggml_tensor*> t_all_layers;
```

### Patch 2: `src/llama-graph.cpp`

In `llm_graph_result::set_outputs()`, before the closing brace, add:

```cpp
for (auto t : t_all_layers) {
    if (t) ggml_set_output(t);
}
```

### Patch 3: `src/llama-context.cpp`

In the decode path, after the `t_h_nextn` extraction block, add:

```cpp
if (!res->t_all_layers.empty()) {
    synchronize();
    FILE * fp = fopen("/tmp/llama_layers.bin", "wb");
    if (fp) {
        for (auto t : res->t_all_layers) {
            uint32_t n = t->ne[0];
            std::vector<float> buf(n);
            ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), t);
            ggml_backend_tensor_get_async(be, t, buf.data(), 0, n * sizeof(float));
            synchronize();
            fwrite(buf.data(), sizeof(float), n, fp);
        }
        fclose(fp);
    }
}
```

### Patch 4: `src/models/gemma4.cpp` (or target model)

At the per-layer block output point (after `build_cvec`, before `inpL = cur`), add:

```cpp
res->t_all_layers.push_back(cur);
```

### Build commands

```sh
cd /path/to/llama.cpp
cmake -B build -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama -j$(nproc)

g++ -std=c++17 -I./include -I./ggml/include -I./src \
    path/to/ember/tools/dump_llamacpp_layers.cpp \
    ./build/src/libllama.a \
    ./build/ggml/src/libggml.a \
    ./build/ggml/src/libggml-base.a \
    ./build/ggml/src/libggml-cpu.a \
    -lpthread -ldl -lm \
    -o dump_llamacpp_layers
```

## Running the comparison

### Manual steps

```sh
# 1. Dump llama.cpp layers
./dump_llamacpp_layers gemma-4-E2B-it-Q8_0.gguf "" llama_layers.bin 16

# 2. Dump Ember layers
target/release/ember --model gemma-4-E2B-it-Q8_0.gguf --arch gemma4 \
    --prompt "" --max-seq-len 16 --temperature 0 \
    --dump-layers ember_layers.bin

# 3. Compare
python3 scripts/compare_layer_dumps.py \
    --ember ember_layers.bin \
    --reference llama_layers.bin \
    --layers 35 --hidden-size 1536 \
    --out-md report.md --out-json report.json
```

### Automated wrapper

```sh
GEMMA_MODEL=gemma-4-E2B-it-Q8_0.gguf \
LLAMACPP_DIR=/path/to/llama.cpp \
    fish scripts/run_gemma_layer_compare.fish
```

## Expected output files

| File | Description |
|------|-------------|
| `artifacts/layer_compare_gemma/ember_layers.bin` | Ember per-layer states (f32 flat) |
| `artifacts/layer_compare_gemma/llama_layers.bin` | llama.cpp per-layer states (f32 flat) |
| `artifacts/layer_compare_gemma/report.md` | Markdown comparison table |
| `artifacts/layer_compare_gemma/report.json` | Machine-readable metrics |

## Binary format

Both Ember and llama.cpp dump files use the same format:

- dtype: f32, native endian
- shape: `[n_layers * hidden_size]` flat, layer-major
- layer 0 first, layer (n_layers - 1) last
- each layer: `hidden_size` consecutive f32 values

For Gemma 4: n_layers = 35, hidden_size = 1536, total 53,760 floats (215,040 bytes).

### Semantic boundary

Per-layer states are captured at the block output after the final residual add and
`layer_output_scale`. In llama.cpp this is `cur` after `build_cvec` in the gemma4
graph. In Ember this is the return value of `forward_with_cache` after step 4
(layer output scaling).

## Interpreting the metrics

- **cosine**: 1.0 = identical direction. Drops below 0.99 indicate divergence.
- **L2 norm**: magnitude of the hidden-state vector. Large mismatches suggest
  scaling or weight-loading issues.
- **mean abs diff**: average per-element difference. Sub-1e-3 is tight; >0.01
  indicates meaningful drift.
- **max abs diff**: worst-case element difference. Spikes suggest a single
  dimension or operation is off.

**Warning:** cosine similarity alone is not generation parity. A model can
produce different token predictions even at cosine 0.99. Use golden-logit
comparison (`--dump-logits`) for output-level validation.

## Troubleshooting

- **Shape mismatch error**: check `--layers` and `--hidden-size` against the
  model config. For Gemma 4 E2B: 35 layers, 1536 hidden.
- **llama.cpp binary hangs**: ensure `-DGGML_NATIVE=ON` was passed to cmake.
  Some CPU feature detection fails without it.
- **llama.cpp dump contains no per-layer data**: the source patches were not
  applied. The tool falls back to final hidden state only.
- **Ember `--dump-layers` not recognized**: update to a build that includes the
  flag (added during Gemma 4 parity work).
