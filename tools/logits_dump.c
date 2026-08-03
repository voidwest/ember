// logits_dump: final-position logits from the pinned llama.cpp (b9999).
//
// Minimal C harness against the built libllama: load the model, tokenize
// each prompt line from the input file (one prompt per line), decode with
// the pinned context, and write the final-position logits row as f32 to
// one .npy-free binary file per prompt (named by output prefix + index).
//
// Built and invoked by scripts/validate_golden_ladder.sh; uses the
// deprecated-but-present load path (llama_load_model_from_file) which is
// still exported at the pinned tag.
//
// Usage: logits_dump MODEL INPUT_TXT OUT_PREFIX N_CTX

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cerrno>

static char *read_line(FILE *file) {
    size_t capacity = 256, length = 0;
    char *buffer = (char *)malloc(capacity);
    if (!buffer) return NULL;
    int c;
    while ((c = fgetc(file)) != EOF && c != '\n') {
        if (length + 1 >= capacity) {
            capacity *= 2;
            char *grown = (char *)realloc(buffer, capacity);
            if (!grown) { free(buffer); return NULL; }
            buffer = grown;
        }
        buffer[length++] = (char)c;
    }
    if (c == EOF && length == 0) { free(buffer); return NULL; }
    buffer[length] = '\0';
    return buffer;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL INPUT_TXT OUT_PREFIX N_CTX\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *input_path = argv[2];
    const char *out_prefix = argv[3];
    int n_ctx = atoi(argv[4]);
    if (n_ctx <= 0) { fprintf(stderr, "invalid n_ctx\n"); return 2; }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    struct llama_model *model = llama_load_model_from_file(model_path, model_params);
    if (!model) { fprintf(stderr, "failed to load model %s\n", model_path); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "n_vocab=%d\n", n_vocab);

    // The pinned tag's public header exposes no KV-cache API, so each
    // prompt gets a fresh context (128-token KV is cheap).
    FILE *input = fopen(input_path, "r");
    if (!input) { fprintf(stderr, "failed to open %s\n", input_path); return 1; }

    char *prompt;
    int index = 0;
    while ((prompt = read_line(input)) != NULL) {
        int max_tokens = n_ctx;
        llama_token *tokens = (llama_token *)malloc((size_t)max_tokens * sizeof(llama_token));
        int n_tokens = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                      tokens, max_tokens, true, false);
        if (n_tokens <= 0) {
            fprintf(stderr, "prompt %d failed to tokenize\n", index);
            free(tokens);
            free(prompt);
            fclose(input);
            return 1;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx;
        ctx_params.n_threads = 8;
        ctx_params.n_threads_batch = 8;
        struct llama_context *ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prompt %d decode failed\n", index);
            llama_batch_free(batch);
            llama_free(ctx);
            free(tokens);
            free(prompt);
            fclose(input);
            return 1;
        }

        // Only the final token requests logits, so the (compacted) output
        // buffer holds exactly one row at the pinned tag.
        float *logits = llama_get_logits_ith(ctx, n_tokens - 1);
        if (!logits) {
            fprintf(stderr, "prompt %d: null logits buffer\n", index);
            return 1;
        }
        const float *final_row = logits;
        fprintf(stderr, "logits[0]=%f logits[1]=%f\n", logits[0], logits[1]);

        char out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s.%d.bin", out_prefix, index);
        FILE *out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "failed to open %s\n", out_path); return 1; }
        size_t written = fwrite(final_row, sizeof(float), (size_t)n_vocab, out);
        fclose(out);
        if (written != (size_t)n_vocab) {
            fprintf(stderr, "short write for prompt %d: %zu of %d (errno=%d)\n",
                    index, written, n_vocab, ferror(out) ? errno : 0);
            return 1;
        }
        fprintf(stderr, "prompt %d: %d tokens -> %s\n", index, n_tokens, out_path);

        llama_batch_free(batch);
        llama_free(ctx);
        free(tokens);
        free(prompt);
        index++;
    }
    fclose(input);
    free(prompt);

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
