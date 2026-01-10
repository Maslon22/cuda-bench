#pragma once
#include <curand_kernel.h>

__global__ void init_rng(
    curandStatePhilox4_32_10_t* states,
    int n
);

__global__ void montecarlo_gpu(
    curandStatePhilox4_32_10_t* states,
    unsigned long long* inside,
    int N
);
