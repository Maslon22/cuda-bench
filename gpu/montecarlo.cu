#include <curand_kernel.h>

__global__ void init(curandState*s,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) curand_init(1234,i,0,&s[i]);
}

__global__ void monte(curandState*s,int*n,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){
        float x=curand_uniform(&s[i]);
        float y=curand_uniform(&s[i]);
        if(x*x+y*y<=1) atomicAdd(n,1);
    }
}