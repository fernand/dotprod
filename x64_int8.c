// clang -O2 -ffast-math -march=native x64_int8.c -o int8 && ./int8
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

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

typedef int32_t (*DotFunc)(int8_t* a, int8_t* b);
void benchmark(int8_t* embeds, int8_t* v, DotFunc func) {
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    int8_t last_dot = 0;
    for (int j=0; j<NUM_ITER; j++) {
        int8_t* current_v = embeds;
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
}