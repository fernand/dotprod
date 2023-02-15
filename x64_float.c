// clang -O2 -ffast-math -march=native x64_float.c -o float && ./float
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define SZ 1536
#define NUM_VECS 1000000
#define NUM_ITER 20

float dot(float *a, float *b) {
    float result = 0;
    for(int i=0; i<SZ; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

typedef float (*DotFunc)(float* a, float* b);
void benchmark(float* embeds, float* v, DotFunc func) {
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    float last_dot = 0;
    for (int j=0; j<NUM_ITER; j++) {
        float* current_v = embeds;
        for(int i=0; i<NUM_VECS; ++i) {
            last_dot = func(current_v, v);
            current_v += SZ;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("last_dot: %.3f\n", (float)last_dot);
    float time_taken = (float)(t2.tv_nsec - t1.tv_nsec) / (NUM_ITER * 1e6) + (float)(t2.tv_sec - t1.tv_sec) / (NUM_ITER * 1e-3);
    printf("time ms: %.2f\n", time_taken);
}

int main() {
    float* embeds = malloc(SZ*NUM_VECS*sizeof(float));
    float* v = malloc(SZ*sizeof(float));
    srand(time(NULL));
    for (int i=0; i<SZ*NUM_VECS; ++i) {
        embeds[i] = (float)((float)rand()/(float)RAND_MAX);
    }
    for (int i=0; i<SZ; ++i) {
        v[i] = (float)((float)rand()/(float)RAND_MAX);
    }

    printf("Float implementation, SIMD optimal Clang code gen using FMA intrinsics:\n");
    benchmark(embeds, v, dot);
}