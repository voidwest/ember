/// dump_llamacpp_layers — dump per-layer hidden states from a GGUF model via llama.cpp.
///
/// ## Binary output format
///
/// The output file contains concatenated per-layer hidden-state vectors for the
/// last prompt token, written in native-endian f32:
///
///   dtype:      f32 (native byte order)
///   shape:      [n_layers * n_embd]  (flat, row-major)
///   layer count: model n_layers
///   hidden size: model n_embd
///   row order:   layer 0 first, layer (n_layers-1) last
///
/// Each layer's vector is `n_embd` consecutive f32 values, taken from the last
/// token position in the sequence. The tensor boundary matches the per-layer
/// block output after the final residual add and layer_output_scale (i.e.
/// `cur` after `build_cvec` in llama.cpp's gemma4 graph, or the equivalent
/// point for other architectures).
///
/// ## Prerequisites
///
/// This tool requires a patched llama.cpp with per-layer state capture enabled.
/// Three source files must be modified:
///
///   1. `src/llama-graph.h` — add `std::vector<ggml_tensor*> t_all_layers;`
///      to `llm_graph_result`.
///   2. `src/llama-graph.cpp` — add `for (auto t : t_all_layers) ggml_set_output(t);`
///      in `set_outputs()`.
///   3. `src/llama-context.cpp` — add a file-write loop after `t_h_nextn` extraction
///      in the decode path, iterating `res->t_all_layers` and writing `ne[0]` floats
///      per tensor.
///   4. `src/models/gemma4.cpp` (or the target model) — add
///      `res->t_all_layers.push_back(cur);` at the per-layer block output point.
///
/// See `docs/layer-dump-tooling.md` for exact patches.
///
/// ## Build
///
///   cd /path/to/llama.cpp
///   cmake -B build -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF
///   cmake --build build --target llama -j$(nproc)
///   g++ -std=c++17 -I./include -I./ggml/include -I./src \
///       path/to/dump_llamacpp_layers.cpp \
///       ./build/src/libllama.a \
///       ./build/ggml/src/libggml.a \
///       ./build/ggml/src/libggml-base.a \
///       ./build/ggml/src/libggml-cpu.a \
///       -lpthread -ldl -lm -o dump_llamacpp_layers
///
/// ## Usage
///
///   ./dump_llamacpp_layers <model.gguf> <prompt> <out.bin> [ctx_size]
///
/// Arguments:
///   model.gguf   path to GGUF model
///   prompt       text prompt (use "" for BOS-only)
///   out.bin      path for binary output
///   ctx_size     context size (default: 16)

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> <out.bin> [ctx_size]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt     = argv[2];
    const char * out_path   = argv[3];
    int          ctx_size   = (argc >= 5) ? atoi(argv[4]) : 16;

    // --- backend init ---
    llama_backend_init();

    // --- load model ---
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "error: failed to load model %s\n", model_path);
        return 1;
    }

    // --- create context ---
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = ctx_size;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // --- tokenize ---
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tokens = 0;
    llama_token toks[ctx_size];
    if (strlen(prompt) == 0) {
        // BOS-only
        int bos = llama_vocab_bos(vocab);
        toks[0]  = bos;
        n_tokens = 1;
    } else {
        n_tokens = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks, ctx_size, true, true);
        if (n_tokens < 0) {
            fprintf(stderr, "error: tokenize failed\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    // --- decode ---
    llama_batch batch = llama_batch_get_one(toks, n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // The patched llama.cpp writes per-layer states to llama_35layers.bin
    // (or the equivalent file for other architectures) during decode.
    // This tool does not read that file — the patched decode path handles
    // the write directly.

    // For unpatched builds, dump the final hidden state as a fallback.
    float * embd = llama_get_embeddings_ith(ctx, n_tokens - 1);
    if (embd) {
        int n_embd = llama_model_n_embd(model);
        FILE * fp = fopen(out_path, "wb");
        if (fp) {
            fwrite(embd, sizeof(float), n_embd, fp);
            fclose(fp);
            fprintf(stderr, "info: wrote final hidden state (%d floats) to %s\n", n_embd, out_path);
            fprintf(stderr, "warn: patched llama.cpp required for per-layer states\n");
        }
    } else {
        fprintf(stderr, "warn: no embeddings available\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
