// Diagnostic greedy decode against the pinned llama.cpp C API.
// Prints input IDs and the eight largest logits at every step. This does not
// replace the frozen numerical gates; it explains a specific token divergence.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Optional, owned last-token rows at one evaluation. The graph callback avoids
// patching the pinned reference. Capture-on/off equivalence must be checked.
struct Capture {
    FILE *out = nullptr;
    int target_step = 0;
    int current_step = 0;
    bool failed = false;
    bool ffn_detail = false;
    bool attention_detail = false;
    bool qkv_detail = false;
    bool rope_detail = false;
    bool softmax_detail = false;
    int prompt_tokens = 0;
    std::set<std::string> names;
};

static bool selected_name(const char *name, bool detail, bool attention_detail) {
    if (attention_detail && !std::strcmp(name, "kqv_out-0")) return true;
    if (detail) {
        for (const char *extra : {"ffn_inp-0", "ffn_norm-0", "ffn_up-0", "ffn_gate-0", "ffn_swiglu-0"}) {
            if (!std::strcmp(name, extra)) return true;
        }
    }
    if (!std::strcmp(name, "result_norm") || !std::strcmp(name, "result_output")) return true;
    for (const char *prefix : {"attn_out-", "ffn_out-", "l_out-"}) {
        const size_t n = std::strlen(prefix);
        if (std::strncmp(name, prefix, n)) continue;
        const char *suffix = name + n;
        if (!*suffix) return false;
        for (; *suffix; ++suffix) if (*suffix < '0' || *suffix > '9') return false;
        return true;
    }
    return false;
}

static bool capture_tensor(ggml_tensor *tensor, bool ask, void *data) {
    auto &capture = *static_cast<Capture *>(data);
    const char *capture_name = tensor->name;
    bool qkv_selected = capture.qkv_detail && !std::strcmp(tensor->name, "attn_norm-0");
    if (capture.qkv_detail && tensor->op == GGML_OP_MUL_MAT) {
        for (const auto &entry : {std::make_pair("Qcur-0", "q_projection-0"),
                                  std::make_pair("Kcur-0", "k_projection-0"),
                                  std::make_pair("Vcur-0", "v_projection-0")}) {
            if (!std::strcmp(tensor->name, entry.first)) {
                capture_name = entry.second;
                qkv_selected = true;
            }
        }
    }
    bool rope_selected = false;
    if (capture.rope_detail && tensor->op == GGML_OP_ROPE) {
        if (!std::strcmp(tensor->name, "Qcur-0")) {
            capture_name = "q_rope-0";
            rope_selected = true;
        } else if (!std::strcmp(tensor->name, "Kcur-0")) {
            capture_name = "k_rope-0";
            rope_selected = true;
        }
    }
    const bool score_selected = capture.softmax_detail &&
        ((!std::strcmp(tensor->name, "kq-0") && tensor->op == GGML_OP_MUL_MAT) ||
         (!std::strcmp(tensor->name, "kq_soft_max-0") && tensor->op == GGML_OP_SOFT_MAX));
    const bool selected = capture.current_step == capture.target_step &&
        (score_selected || rope_selected || qkv_selected || selected_name(tensor->name, capture.ffn_detail, capture.attention_detail));
    if (ask) return selected;
    if (!selected) return true;
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[0] > 1000000 ||
        tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || (!rope_selected && !score_selected && tensor->ne[2] != 1) || tensor->ne[3] != 1 ||
        tensor->nb[0] != sizeof(float) || !capture.names.insert(capture_name).second) {
        capture.failed = true;
        return false;
    }
    if (rope_selected && (tensor->ne[1] > 1000000 / tensor->ne[0] ||
        tensor->nb[1] != static_cast<size_t>(tensor->ne[0]) * sizeof(float))) {
        capture.failed = true;
        return false;
    }
    const size_t score_width = static_cast<size_t>(capture.prompt_tokens + capture.current_step - 1);
    if (score_selected && (capture.current_step <= 1 || tensor->ne[1] != 1 || score_width > static_cast<size_t>(tensor->ne[0]))) {
        capture.failed = true;
        return false;
    }
    const size_t width = score_selected ? score_width : static_cast<size_t>(tensor->ne[0]) * (rope_selected ? static_cast<size_t>(tensor->ne[1]) : 1);
    const size_t row = static_cast<size_t>(tensor->ne[rope_selected ? 2 : 1] - 1);
    const size_t stride = tensor->nb[rope_selected ? 2 : 1];
    const size_t bytes = ggml_nbytes(tensor);
    if (!stride || width * sizeof(float) > bytes ||
        row > (bytes - width * sizeof(float)) / stride) {
        capture.failed = true;
        return false;
    }
    std::vector<float> values(width);
    ggml_backend_tensor_get(tensor, values.data(), row * stride, width * sizeof(float));
    for (float value : values) if (!std::isfinite(value)) {
        capture.failed = true;
        return false;
    }
    std::fprintf(capture.out, "%s{\"name\":\"%s\",\"step\":%d,\"row\":%zu,\"width\":%zu,\"values\":[",
                 capture.names.size() == 1 ? "" : ",\n", capture_name, capture.current_step, row, width);
    for (size_t i = 0; i < width; ++i) std::fprintf(capture.out, "%s%.9g", i ? "," : "", values[i]);
    std::fprintf(capture.out, "]}");
    if (std::ferror(capture.out)) {
        capture.failed = true;
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: diagnose_decode MODEL PROMPT STEPS OUTPUT_JSON\n");
        return 2;
    }
    int steps = std::atoi(argv[3]);
    if (steps < 1 || steps > 64) return 2;
    int full_logits_step = 0;
    if (const char *step = std::getenv("EMBER_REFERENCE_FULL_LOGITS_STEP")) {
        char *end = nullptr;
        const long parsed = std::strtol(step, &end, 10);
        if (!*step || *end || parsed < 1 || parsed > steps) return 2;
        full_logits_step = static_cast<int>(parsed);
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> input(256);
    int n = llama_tokenize(vocab, argv[2], (int)std::strlen(argv[2]), input.data(), (int)input.size(), true, false);
    if (n <= 0 || n + steps > 256) return 1;
    input.resize(n);
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    bool f32_cache = std::getenv("EMBER_REFERENCE_F32_KV") != nullptr;
    if (f32_cache) {
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
    }
    bool no_flash = std::getenv("EMBER_REFERENCE_NO_FLASH") != nullptr;
    if (no_flash) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    Capture capture;
    capture.ffn_detail = std::getenv("EMBER_REFERENCE_FFN_DETAIL") != nullptr;
    capture.attention_detail = std::getenv("EMBER_REFERENCE_ATTENTION_DETAIL") != nullptr;
    capture.qkv_detail = std::getenv("EMBER_REFERENCE_QKV_DETAIL") != nullptr;
    capture.rope_detail = std::getenv("EMBER_REFERENCE_ROPE_DETAIL") != nullptr;
    capture.softmax_detail = std::getenv("EMBER_REFERENCE_SOFTMAX_DETAIL") != nullptr;
    capture.prompt_tokens = n;
    if (const char *path = std::getenv("EMBER_REFERENCE_CAPTURE")) {
        const char *step = std::getenv("EMBER_REFERENCE_CAPTURE_STEP");
        if (!step || !*step) return 2;
        char *end = nullptr;
        const long parsed = std::strtol(step, &end, 10);
        if (*end || parsed < 1 || parsed > steps) return 2;
        capture.target_step = static_cast<int>(parsed);
        capture.out = std::fopen(path, "wx");
        if (!capture.out) return 1;
        std::fprintf(capture.out, "{\"schema\":\"reference-intermediates-diagnostic-v1\",\"tensors\":[\n");
        cp.cb_eval = capture_tensor;
        cp.cb_eval_user_data = &capture;
    }
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    auto batch = llama_batch_init(256, 0, 1);
    FILE *out = std::fopen(argv[4], "wb");
    if (!out) return 1;
    std::fprintf(out, "{\"f32_cache\":%s,\"flash_disabled\":%s,\"input_token_ids\":[", f32_cache ? "true" : "false", no_flash ? "true" : "false");
    for (int i = 0; i < n; ++i) std::fprintf(out, "%s%d", i ? "," : "", input[i]);
    std::fprintf(out, "],\"steps\":[\n");
    std::vector<llama_token> current = input;
    int position = 0;
    for (int step = 0; step < steps; ++step) {
        capture.current_step = step + 1;
        batch.n_tokens = (int)current.size();
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = current[i];
            batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == batch.n_tokens - 1;
        }
        if (llama_decode(ctx, batch) != 0) return 1;
        float *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) return 1;
        std::vector<std::pair<int, float>> ranked;
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
            if (!std::isfinite(logits[i])) return 1;
            ranked.emplace_back(i, logits[i]);
        }
        std::partial_sort(ranked.begin(), ranked.begin() + 8, ranked.end(), [](auto a, auto b) {
            return a.second == b.second ? a.first < b.first : a.second > b.second;
        });
        std::fprintf(out, "%s{\"step\":%d,\"token\":%d,\"top_logits\":[", step ? ",\n" : "", step + 1, ranked[0].first);
        for (int i = 0; i < 8; ++i) std::fprintf(out, "%s[%d,%.9g]", i ? "," : "", ranked[i].first, ranked[i].second);
        std::fprintf(out, "]");
        if (step + 1 == full_logits_step) {
            static_assert(sizeof(float) == sizeof(uint32_t));
            std::fprintf(out, ",\"full_logit_bits\":[");
            for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
                uint32_t bits;
                std::memcpy(&bits, logits + i, sizeof(bits));
                std::fprintf(out, "%s\"%08x\"", i ? "," : "", static_cast<unsigned>(bits));
            }
            std::fprintf(out, "]");
        }
        std::fprintf(out, "}");
        position += batch.n_tokens;
        current.assign(1, ranked[0].first);
    }
    std::fprintf(out, "\n]}\n");
    int close_result = std::fclose(out);
    if (capture.out) {
        std::fprintf(capture.out, "\n]}\n");
        bool missing = !capture.names.count("result_norm") || !capture.names.count("result_output");
        for (int layer = 0; layer < llama_model_n_layer(model); ++layer) {
            for (const char *prefix : {"attn_out-", "ffn_out-", "l_out-"}) {
                missing = missing || !capture.names.count(std::string(prefix) + std::to_string(layer));
            }
        }
        if (capture.softmax_detail) {
            missing = missing || !capture.names.count("kq-0") || !capture.names.count("kq_soft_max-0");
        }
        if (capture.rope_detail) {
            missing = missing || !capture.names.count("q_rope-0") || !capture.names.count("k_rope-0");
        }
        if (capture.qkv_detail) {
            for (const char *name : {"attn_norm-0", "q_projection-0", "k_projection-0", "v_projection-0"}) {
                missing = missing || !capture.names.count(name);
            }
        }
        if (capture.attention_detail) missing = missing || !capture.names.count("kqv_out-0");
        if (capture.ffn_detail) {
            for (const char *name : {"ffn_inp-0", "ffn_norm-0", "ffn_up-0", "ffn_gate-0", "ffn_swiglu-0"}) {
                missing = missing || !capture.names.count(name);
            }
        }
        if (std::fclose(capture.out) != 0 || capture.failed || missing) close_result = 1;
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return close_result == 0 ? 0 : 1;
}
