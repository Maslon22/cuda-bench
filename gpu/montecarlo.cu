#include "../common/precision.hpp"
#include <curand_kernel.h>

__global__ void montecarlo_kernel(
    unsigned long long* global_inside,
    unsigned long long seed,
    int total_samples
){
    extern __shared__ unsigned int sdata[];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int gdim = gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, gid, 0, &state);

    unsigned int local = 0;

    for(int i = gid; i < total_samples; i += gdim){
#if defined(USE_FP32)
        real x = curand_uniform(&state);
        real y = curand_uniform(&state);
#else   // USE_FP64
        real x = curand_uniform_double(&state);
        real y = curand_uniform_double(&state);
#endif
        if(x*x + y*y <= real(1))
            local++;
    }

    sdata[tid] = local;
    __syncthreads();

    // redukcja w bloku
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
        if(tid < offset)
            sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }

    if(tid == 0)
        atomicAdd(global_inside, (unsigned long long)sdata[0]);
}
