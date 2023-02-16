// clang -O2 -march=native x64_int8.c -o int8 && ./int8
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

#define SZ 1536
#define NUM_VECS 1000000
//#define NUM_VECS 2560
#define NUM_ITER 20

int16_t dot(int8_t *a, int8_t *b) {
    int16_t result = 0;
    for(int i=0; i<SZ; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

int16_t dot_opt(int8_t *a, int8_t *b) {
    __m256i sum0 = _mm256_set1_epi16(0);
    __m256i sum1 = _mm256_set1_epi16(0);
    __m256i sum2 = _mm256_set1_epi16(0);
    __m256i sum3 = _mm256_set1_epi16(0);
    for(int i=0; i<SZ/64; ++i) {
        __m256 a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)a));
        __m256 a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(a+16)));
        __m256 a2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(a+32)));
        __m256 a3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(a+48)));
        __m256 b0 = _mm256_cvtepi8_epi16(_mm_stream_load_si128((__m128i*)b));
        __m256 b1 = _mm256_cvtepi8_epi16(_mm_stream_load_si128((__m128i*)(b+16)));
        __m256 b2 = _mm256_cvtepi8_epi16(_mm_stream_load_si128((__m128i*)(b+32)));
        __m256 b3 = _mm256_cvtepi8_epi16(_mm_stream_load_si128((__m128i*)(b+48)));
        sum0 = _mm256_add_epi16(sum0, _mm256_mullo_epi16(a0, b0));
        sum1 = _mm256_add_epi16(sum1, _mm256_mullo_epi16(a1, b1));
        sum2 = _mm256_add_epi16(sum2, _mm256_mullo_epi16(a2, b2));
        sum3 = _mm256_add_epi16(sum3, _mm256_mullo_epi16(a3, b3));
        a += 64;
        b += 64;
    }
    __m256 sum = _mm256_add_epi16(sum0, sum1);
    sum = _mm256_add_epi16(sum, sum2);
    sum = _mm256_add_epi16(sum, sum3);
    int16_t result = 0;
    for(int i=0; i<16; ++i) {
        result += ((int16_t*)&sum)[i];
    }
    return result;
}

typedef int16_t (*DotFunc)(int8_t* a, int8_t* b);
void benchmark(int8_t* embeds, int8_t* v, DotFunc func) {
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    int8_t last_dot = 0;
    for (int j=0; j<NUM_ITER; j++) {
        for(int i=0; i<NUM_VECS; ++i) {
            last_dot = func(&embeds[i*SZ], v);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("last_dot: %i\n", (int)last_dot);
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
    //printf("int8 implementation with AVX2:\n");
    //benchmark(embeds, v, dot_opt);
}
