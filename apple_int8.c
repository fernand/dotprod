// clang -O2 -ffast-math -mcpu=apple-m1 int8.c -o int8 && ./int8
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <arm_neon.h>

#define SZ 1536
#define NUM_VECS 1000000
#define NUM_ITER 20

int32_t dot(int8_t *a, int8_t *b) {
    int32_t result = 0;
    for(int i=0; i<SZ; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

int32_t dot_opt(int8_t *a, int8_t *b) {
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    for (int i=0; i<SZ/64; ++i) {
        sum0 = vdotq_s32(sum0, vld1q_s8(a), vld1q_s8(b));
        sum1 = vdotq_s32(sum1, vld1q_s8(a+16), vld1q_s8(b+16));
        sum2 = vdotq_s32(sum2, vld1q_s8(a+32), vld1q_s8(b+32));
        sum3 = vdotq_s32(sum3, vld1q_s8(a+48), vld1q_s8(b+48));
        a += 64;
        b += 64;
    }
    return
        vgetq_lane_s32(sum0, 0) + vgetq_lane_s32(sum0, 1) + vgetq_lane_s32(sum0, 2) + vgetq_lane_s32(sum0, 3) +
        vgetq_lane_s32(sum1, 0) + vgetq_lane_s32(sum1, 1) + vgetq_lane_s32(sum1, 2) + vgetq_lane_s32(sum1, 3) +
        vgetq_lane_s32(sum2, 0) + vgetq_lane_s32(sum2, 1) + vgetq_lane_s32(sum2, 2) + vgetq_lane_s32(sum2, 3) +
        vgetq_lane_s32(sum3, 0) + vgetq_lane_s32(sum3, 1) + vgetq_lane_s32(sum3, 2) + vgetq_lane_s32(sum3, 3);
}

typedef int32_t (*DotFunc)(int8_t* a, int8_t* b);
void benchmark(int8_t* embeds, int8_t* v, DotFunc func) {
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    int8_t last_dot = 0;
    for (int j=0; j<NUM_ITER; j++) {
        int8_t* current_v = embeds;
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
    int8_t* embeds = malloc(SZ*NUM_VECS*sizeof(int8_t));
    int8_t* v = malloc(SZ*sizeof(int8_t));
    srand(time(NULL));
    for (int i=0; i<SZ*NUM_VECS; ++i) {
        embeds[i] = (int8_t)(127*(float)rand()/(float)RAND_MAX);
    }
    for (int i=0; i<SZ; ++i) {
        v[i] = (int8_t)(127*(float)rand()/(float)RAND_MAX);
    }

    printf("int8 implementation with Clang codegen:\n");
    benchmark(embeds, v, dot);
    printf("int8 implementation with handwritten dotprot intrinsics:\n");
    benchmark(embeds, v, dot_opt);
}