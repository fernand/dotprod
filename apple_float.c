// clang -O2 -ffast-math -mcpu=apple-m1 float.c -o float && ./float
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

// AMX stuff
#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")
#define AMX_OP_GPR(op, gpr) \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")
#define AMX_LDX(gpr)    AMX_OP_GPR( 0, gpr)
#define AMX_LDY(gpr)    AMX_OP_GPR( 1, gpr)
#define AMX_STX(gpr)    AMX_OP_GPR( 2, gpr)
#define AMX_STY(gpr)    AMX_OP_GPR( 3, gpr)
#define AMX_LDZ(gpr)    AMX_OP_GPR( 4, gpr)
#define AMX_STZ(gpr)    AMX_OP_GPR( 5, gpr)
#define AMX_FMA16(gpr)  AMX_OP_GPR(15, gpr)
#define AMX_FMA32(gpr)  AMX_OP_GPR(12, gpr)
#define AMX_SET()       AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()       AMX_NOP_OP_IMM5(17, 1)

#define FLOAT _Float16
#define NUM_PER_CHUNK 256
#define FMA AMX_FMA16

// #define FLOAT float
// #define NUM_PER_CHUNK 128
// #define FMA AMX_FMA32

#define SZ 1536
#define NUM_VECS 1000000
#define NUM_ITER 20

const uint64_t load_store_2 = 1ull << 62;
const uint64_t load_store_width = 128;
const uint64_t vec_mode = 1ull << 63;
const uint64_t first_z_row = 0ull << 20;

FLOAT dot(FLOAT *a, FLOAT *b) {
    FLOAT result = 0;
    for(int i=0; i<SZ; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

FLOAT opt_dot(FLOAT *a, FLOAT *b) {
    uint64_t reset_z = 1ull << 27;
    for (int i=0; i<(SZ/NUM_PER_CHUNK); ++i) {
        AMX_LDX(load_store_2 | 0ull << 56 | (uint64_t)a);
        AMX_LDY(load_store_2 | 0ull << 56 | (uint64_t)b);
        AMX_LDX(load_store_2 | 2ull << 56 | (uint64_t)a + load_store_width);
        AMX_LDY(load_store_2 | 2ull << 56 | (uint64_t)b + load_store_width);
        AMX_LDX(load_store_2 | 4ull << 56 | (uint64_t)a + 2 * load_store_width);
        AMX_LDY(load_store_2 | 4ull << 56 | (uint64_t)b + 2 * load_store_width);
        AMX_LDX(load_store_2 | 6ull << 56 | (uint64_t)a + 3 * load_store_width);
        AMX_LDY(load_store_2 | 6ull << 56 | (uint64_t)b + 3 * load_store_width);

        FMA(vec_mode | reset_z | first_z_row | 0ull << 10 | 0ull << 0);
        FMA(vec_mode | first_z_row | (128ull * 0 + 64ull) << 10 | (128ull * 0 + 64ull) << 0);
        FMA(vec_mode | first_z_row | (128ull * 1) << 10 | (128ull * 1) << 0);
        FMA(vec_mode | first_z_row | (128ull * 1 + 64ull) << 10 | (128ull * 1 + 64ull) << 0);
        FMA(vec_mode | first_z_row | (128ull * 2) << 10 | (128ull * 2) << 0);
        FMA(vec_mode | first_z_row | (128ull * 2 + 64ull) << 10 | (128ull * 2 + 64ull) << 0);
        FMA(vec_mode | first_z_row | (128ull * 3) << 10 | (128ull * 3) << 0);
        FMA(vec_mode | first_z_row | (128ull * 3 + 64ull) << 10 | (128ull * 3 + 64ull) << 0);

        a += NUM_PER_CHUNK;
        b += NUM_PER_CHUNK;
        reset_z = 0;
    }

    FLOAT acc[NUM_PER_CHUNK];
    AMX_STZ(load_store_2 | 0ull << 56 | (uint64_t)acc);
    AMX_STZ(load_store_2 | 1ull << 56 | (uint64_t)acc + load_store_width);
    AMX_STZ(load_store_2 | 2ull << 56 | (uint64_t)acc + 2 * load_store_width);
    AMX_STZ(load_store_2 | 3ull << 56 | (uint64_t)acc + 3 * load_store_width);
    FLOAT result = 0;
    for (int i=0; i<NUM_PER_CHUNK; ++i) {
        result += acc[i];
    }

    return result;
}

typedef FLOAT (*DotType)(FLOAT* a, FLOAT* b);
void benchmark(FLOAT* embeds, FLOAT* v, DotType func) {
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    FLOAT last_dot = 0;
    for (int j=0; j<NUM_ITER; j++) {
        FLOAT* current_v = embeds;
        for(int i=0; i<NUM_VECS; ++i) {
            last_dot = func(current_v, v);
            current_v += SZ;
        }
    }
    uint64_t t2 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    printf("last_dot: %.3f\n", (float)last_dot);
    printf("time ms: %.2f\n", (float)(t2 - t1) / (NUM_ITER * 1e6));
}

int main() {
    FLOAT* embeds = malloc(SZ*NUM_VECS*sizeof(FLOAT));
    FLOAT* v = malloc(SZ*sizeof(FLOAT));
    srand(time(NULL));
    for (int i=0; i<SZ*NUM_VECS; ++i) {
        embeds[i] = (FLOAT)((float)rand()/(float)RAND_MAX);
    }
    for (int i=0; i<SZ; ++i) {
        v[i] = (FLOAT)((float)rand()/(float)RAND_MAX);
    }

    printf("Float implementation, SIMD optimal Clang code gen using FMA intrinsics:\n");
    benchmark(embeds, v, dot);
    AMX_SET();
    printf("Float implementation, AMX silicon:\n");
    benchmark(embeds, v, opt_dot);
    AMX_CLR();
}