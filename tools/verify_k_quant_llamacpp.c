// Independent Q4_K/Q6_K x Q8_K known-answer generator.
//
// This file is compiled against the pinned llama.cpp tree by
// tools/verify_k_quant_llamacpp.sh. Keep its formulas synchronized with the
// Rust fixture in k_quant_matmul::tests::pinned_llama_cpp_known_answer_vector.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s Q8_OUTPUT\n", argv[0]);
        return 64;
    }
    ggml_cpu_init();

    enum { N = 512, NB = N / QK_K };
    float activation[N];
    for (int index = 0; index < N; ++index) {
        activation[index] = (float) ((index * 37) % 257 - 128) / 16.0f;
    }
    block_q8_K q8[NB];
    quantize_row_q8_K_ref(activation, q8, N);

    block_q4_K q4[NB];
    block_q6_K q6[NB];
    memset(q4, 0, sizeof(q4));
    memset(q6, 0, sizeof(q6));
    for (int block = 0; block < NB; ++block) {
        q4[block].d = block == 0 ? 0x3400 : 0xb800;    // f16: 0.25, -0.5
        q4[block].dmin = block == 0 ? 0x3000 : 0x3800; // f16: 0.125, 0.5
        for (int index = 0; index < K_SCALE_SIZE; ++index) {
            q4[block].scales[index] = (uint8_t) (block * 53 + index * 37 + 11);
        }
        for (int index = 0; index < QK_K / 2; ++index) {
            q4[block].qs[index] = (uint8_t) (block * 29 + index * 73 + 19);
            q6[block].ql[index] = (uint8_t) (block * 31 + index * 67 + 7);
        }
        for (int index = 0; index < QK_K / 4; ++index) {
            q6[block].qh[index] = (uint8_t) (block * 47 + index * 43 + 23);
        }
        for (int index = 0; index < QK_K / 16; ++index) {
            q6[block].scales[index] = (int8_t) (block * 59 + index * 41 - 121);
        }
        q6[block].d = block == 0 ? 0x3800 : 0xb400; // f16: 0.5, -0.25
    }

    float q4_generic = 0.0f;
    float q6_generic = 0.0f;
    float q4_dispatch = 0.0f;
    float q6_dispatch = 0.0f;
    ggml_vec_dot_q4_K_q8_K_generic(N, &q4_generic, 0, q4, 0, q8, 0, 1);
    ggml_vec_dot_q6_K_q8_K_generic(N, &q6_generic, 0, q6, 0, q8, 0, 1);
    ggml_vec_dot_q4_K_q8_K(N, &q4_dispatch, 0, q4, 0, q8, 0, 1);
    ggml_vec_dot_q6_K_q8_K(N, &q6_dispatch, 0, q6, 0, q8, 0, 1);

    printf("q4_generic=%08x\n", float_bits(q4_generic));
    printf("q6_generic=%08x\n", float_bits(q6_generic));
    printf("q4_dispatch=%08x\n", float_bits(q4_dispatch));
    printf("q6_dispatch=%08x\n", float_bits(q6_dispatch));

    FILE * output = fopen(argv[1], "wb");
    if (output == NULL || fwrite(q8, 1, sizeof(q8), output) != sizeof(q8) || fclose(output) != 0) {
        fprintf(stderr, "could not write Q8_K fixture\n");
        return 74;
    }
    return 0;
}
