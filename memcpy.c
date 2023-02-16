#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define SZ 1536
#define NUM_VECS 1000000

int main() {
    int8_t* src = malloc(SZ*NUM_VECS);
    int8_t* dst = malloc(SZ*NUM_VECS);
    for(int i=0; i<SZ*NUM_VECS; ++i) {
        src[i] = (int8_t)i;
        dst[i] = (int8_t)(-i);
    }
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    for(int i=0; i<200; ++i) {
        memcpy(src, dst, SZ*NUM_VECS);
    }
    printf("%i,%i\n", (int)src[1000], (int)dst[1000]);
    uint64_t t2 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    printf("time ms: %.2f\n", (float)(t2 - t1) / (200 * 1e6));
}