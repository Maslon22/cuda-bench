#include "../common/precision.hpp"
#include "../gpu/montecarlo.cuh"
#include <curand_kernel.h>

__global__ void init_rng(curandStatePhilox4_32_10_t* states, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        curand_init(1234, i, 0, &states[i]);
}

__global__ void montecarlo_gpu(
    curandStatePhilox4_32_10_t* states,
    unsigned long long* inside,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
#if defined(USE_FP32)
        real x = curand_uniform(&states[i]);
        real y = curand_uniform(&states[i]);
#else
        real x = curand_uniform_double(&states[i]);
        real y = curand_uniform_double(&states[i]);
#endif
        if(x*x + y*y <= real(1))
            atomicAdd(inside, 1ULL);
    }
}
