#include "../common/precision.hpp"

#define TILE 16

__global__ void matmul_naive(const real*A,const real*B,real*C,int N){
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if(r<N && c<N){
        real s = 0;
        for(int k=0;k<N;k++) s += A[r*N+k]*B[k*N+c];
        C[r*N+c] = s;
    }
}

__global__ void matmul_tiled(const real*A,const real*B,real*C,int N){
    __shared__ real As[TILE][TILE], Bs[TILE][TILE];
    int r = blockIdx.y*TILE + threadIdx.y;
    int c = blockIdx.x*TILE + threadIdx.x;
    real s = 0;
    for(int t=0;t<(N+TILE-1)/TILE;t++){
        As[threadIdx.y][threadIdx.x] =
            (r<N && t*TILE+threadIdx.x<N) ? A[r*N+t*TILE+threadIdx.x] : 0;
        Bs[threadIdx.y][threadIdx.x] =
            (c<N && t*TILE+threadIdx.y<N) ? B[(t*TILE+threadIdx.y)*N+c] : 0;
        __syncthreads();
        for(int k=0;k<TILE;k++) s += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(r<N && c<N) C[r*N+c] = s;
}
