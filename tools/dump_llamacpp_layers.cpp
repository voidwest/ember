/// dump_llamacpp_layers — dump per-layer hidden states from a GGUF model via llama.cpp.
///
/// ## Binary output format
///
/// The output file contains concatenated per-layer hidden-state vectors for the
/// last prompt token. The patched llama.cpp source is native-endian f32; this
/// helper validates each value and publishes canonical little-endian f32:
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
///   g++ -std=c++17 -I./include -I./ggml/include -I./src
///       path/to/dump_llamacpp_layers.cpp ./build/src/libllama.a
///       ./build/ggml/src/libggml.a ./build/ggml/src/libggml-base.a
///       ./build/ggml/src/libggml-cpu.a -lpthread -ldl -lm
///       -o dump_llamacpp_layers
///
/// ## Usage
///
///   ./dump_llamacpp_layers <model.gguf> <prompt> <out.bin> [ctx_size] [patched_dump]
///
/// Arguments:
///   model.gguf   path to GGUF model
///   prompt       text prompt (use "" for BOS-only)
///   out.bin      path for binary output
///   ctx_size     context size (default: 16)
///   patched_dump path written by the patched llama.cpp decode path
///                (default: llama_35layers.bin in the current directory)
///
/// The tool deletes `patched_dump` before decode, then requires a newly created
/// file whose byte length is exactly n_layers * n_embd * sizeof(float). An
/// unpatched build is therefore a hard failure, never a final-hidden-state
/// fallback.

#include "llama.h"
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

static bool host_is_little_endian() {
    const uint32_t value = 1;
    return *reinterpret_cast<const unsigned char *>(&value) == 1;
}

static bool install_atomically(const fs::path & source, const fs::path & destination) {
    std::error_code ec;
    fs::path parent = destination.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) {
            fprintf(stderr, "error: cannot create output directory %s: %s\n",
                    parent.c_str(), ec.message().c_str());
            return false;
        }
    } else {
        parent = ".";
    }

    const fs::path absolute_source = fs::absolute(source, ec).lexically_normal();
    if (ec) {
        fprintf(stderr, "error: cannot resolve source path %s: %s\n",
                source.c_str(), ec.message().c_str());
        return false;
    }
    const fs::path absolute_destination = fs::absolute(destination, ec).lexically_normal();
    if (ec) {
        fprintf(stderr, "error: cannot resolve destination path %s: %s\n",
                destination.c_str(), ec.message().c_str());
        return false;
    }
    const bool source_is_destination = absolute_source == absolute_destination;

    const std::string pattern_string =
        (parent / ("." + destination.filename().string() + ".XXXXXX")).string();
    std::vector<char> pattern(pattern_string.begin(), pattern_string.end());
    pattern.push_back('\0');
    const int fd = mkstemp(pattern.data());
    if (fd < 0) {
        fprintf(stderr, "error: cannot create temporary output near %s: %s\n",
                destination.c_str(), strerror(errno));
        return false;
    }

    FILE * input = fopen(source.c_str(), "rb");
    FILE * output = fdopen(fd, "wb");
    bool ok = input != nullptr && output != nullptr;
    if (!input) {
        fprintf(stderr, "error: cannot open patched dump %s: %s\n",
                source.c_str(), strerror(errno));
    }
    if (!output) {
        fprintf(stderr, "error: cannot open temporary output: %s\n", strerror(errno));
        close(fd);
    }

    unsigned char buffer[64 * 1024];
    uintmax_t processed_bytes = 0;
    while (ok) {
        const size_t count = fread(buffer, 1, sizeof(buffer), input);
        if (count % sizeof(float) != 0) {
            fprintf(stderr, "error: patched dump ended on a partial float\n");
            ok = false;
        }
        for (size_t offset = 0; ok && offset < count; offset += sizeof(float)) {
            float value;
            memcpy(&value, buffer + offset, sizeof(value));
            if (!std::isfinite(value)) {
                fprintf(stderr,
                        "error: patched dump contains a non-finite float at byte %ju\n",
                        processed_bytes + offset);
                ok = false;
                break;
            }
            if (!host_is_little_endian()) {
                const unsigned char first = buffer[offset];
                const unsigned char second = buffer[offset + 1];
                buffer[offset] = buffer[offset + 3];
                buffer[offset + 1] = buffer[offset + 2];
                buffer[offset + 2] = second;
                buffer[offset + 3] = first;
            }
        }
        if (ok && count > 0 && fwrite(buffer, 1, count, output) != count) {
            fprintf(stderr, "error: failed writing temporary output: %s\n", strerror(errno));
            ok = false;
        }
        if (count < sizeof(buffer)) {
            if (ferror(input)) {
                fprintf(stderr, "error: failed reading patched dump: %s\n", strerror(errno));
                ok = false;
            }
            break;
        }
        processed_bytes += count;
    }
    if (input) {
        fclose(input);
    }
    if (output) {
        if (fflush(output) != 0) {
            fprintf(stderr, "error: failed to flush temporary output: %s\n", strerror(errno));
            ok = false;
        }
        if (fsync(fileno(output)) != 0) {
            fprintf(stderr, "error: failed to sync temporary output: %s\n", strerror(errno));
            ok = false;
        }
        if (fclose(output) != 0) {
            fprintf(stderr, "error: failed to close temporary output: %s\n", strerror(errno));
            ok = false;
        }
    }

    const fs::path temporary(pattern.data());
    if (ok && std::rename(temporary.c_str(), destination.c_str()) != 0) {
        fprintf(stderr, "error: cannot install %s: %s\n",
                destination.c_str(), strerror(errno));
        ok = false;
    }
    if (!ok) {
        std::remove(temporary.c_str());
        return false;
    }
    if (!source_is_destination) {
        fs::remove(source, ec);
        if (ec) {
            fprintf(stderr, "warning: could not remove staged patched dump %s: %s\n",
                    source.c_str(), ec.message().c_str());
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <model.gguf> <prompt> <out.bin> [ctx_size] [patched_dump]\n",
                argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt     = argv[2];
    const char * out_path   = argv[3];
    const char * patched_dump_path = (argc >= 6) ? argv[5] : "llama_35layers.bin";
    long ctx_size_long = 16;
    if (argc >= 5) {
        char * end = nullptr;
        errno = 0;
        ctx_size_long = strtol(argv[4], &end, 10);
        if (errno != 0 || end == argv[4] || *end != '\0' ||
            ctx_size_long < 1 || ctx_size_long > INT_MAX) {
            fprintf(stderr, "error: ctx_size must be an integer in [1, %d]\n", INT_MAX);
            return 1;
        }
    }
    const int ctx_size = static_cast<int>(ctx_size_long);

    if (sizeof(float) != 4 || CHAR_BIT != 8 ||
        !std::numeric_limits<float>::is_iec559) {
        fprintf(stderr, "error: this binary requires IEEE-754 32-bit float storage\n");
        return 1;
    }

    std::error_code file_error;
    fs::remove(patched_dump_path, file_error);
    if (file_error) {
        fprintf(stderr, "error: cannot remove stale patched dump %s: %s\n",
                patched_dump_path, file_error.message().c_str());
        return 1;
    }

    // --- backend init ---
    llama_backend_init();

    // --- load model ---
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "error: failed to load model %s\n", model_path);
        llama_backend_free();
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
        llama_backend_free();
        return 1;
    }

    // --- tokenize ---
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tokens = 0;
    std::vector<llama_token> toks;
    if (strlen(prompt) == 0) {
        // BOS-only
        const llama_token bos = llama_vocab_bos(vocab);
        if (bos == LLAMA_TOKEN_NULL) {
            fprintf(stderr, "error: model vocabulary has no BOS token for an empty prompt\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        toks.push_back(bos);
        n_tokens = 1;
    } else {
        if (strlen(prompt) > static_cast<size_t>(INT_MAX)) {
            fprintf(stderr, "error: prompt is too long for llama.cpp tokenization\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const int required = llama_tokenize(
            vocab, prompt, static_cast<int>(strlen(prompt)), nullptr, 0, true, true);
        if (required == INT_MIN) {
            fprintf(stderr, "error: tokenizer size requirement overflowed\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const int capacity = required < 0 ? -required : required;
        if (capacity <= 0) {
            fprintf(stderr, "error: tokenizer produced no tokens\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        toks.resize(static_cast<size_t>(capacity));
        n_tokens = llama_tokenize(
            vocab, prompt, static_cast<int>(strlen(prompt)), toks.data(), capacity, true, true);
        if (n_tokens <= 0 || n_tokens > capacity) {
            fprintf(stderr, "error: tokenize failed (result %d, capacity %d)\n",
                    n_tokens, capacity);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        toks.resize(static_cast<size_t>(n_tokens));
    }
    if (n_tokens > ctx_size) {
        fprintf(stderr, "error: tokenized prompt has %d tokens but ctx_size is %d\n",
                n_tokens, ctx_size);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // --- decode ---
    llama_batch batch = llama_batch_get_one(toks.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int n_layers = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);
    if (n_layers <= 0 || n_embd <= 0) {
        fprintf(stderr, "error: model reported invalid dimensions: %d layers x %d hidden\n",
                n_layers, n_embd);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const uintmax_t expected_bytes = static_cast<uintmax_t>(n_layers) *
                                     static_cast<uintmax_t>(n_embd) * sizeof(float);

    const fs::path patched_dump(patched_dump_path);
    if (!fs::is_regular_file(patched_dump, file_error)) {
        fprintf(stderr,
                "error: patched llama.cpp did not create %s; per-layer capture is required\n",
                patched_dump_path);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const uintmax_t actual_bytes = fs::file_size(patched_dump, file_error);
    if (file_error || actual_bytes != expected_bytes) {
        fprintf(stderr,
                "error: patched dump has %ju bytes; expected %ju (%d layers x %d floats)\n",
                actual_bytes, expected_bytes, n_layers, n_embd);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (!install_atomically(patched_dump, fs::path(out_path))) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    fprintf(stderr, "info: wrote %d layers x %d floats to %s\n",
            n_layers, n_embd, out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
