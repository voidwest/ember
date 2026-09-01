#include "llama.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    const llama_model_params params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr) {
        std::fprintf(stderr, "HARNESS: LOAD_REJECT\n");
        llama_backend_free();
        return 1;
    }

    std::fprintf(stderr, "HARNESS: LOAD_OK\n");
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
