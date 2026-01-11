#include "../common/precision.hpp"

// ================= PARAMETRY =================
#define T 16   // wielkość bloku w X i Y
#define K 3    // rozmiar kernela (3x3)
#define R (K/2) // halo / padding dla shared memory

// ================= NAIVE CONV2D =================
__global__ void conv2d(const real* __restrict__ i,
                       real* __restrict__ o,
                       const real* __restrict__ k,
                       int W, int H)
{
    int x = blockIdx.x*T + threadIdx.x;
    int y = blockIdx.y*T + threadIdx.y;

    if(x >= R && y >= R && x < W-R && y < H-R){
        real s = 0;
        for(int ky=-R; ky<=R; ky++)
            for(int kx=-R; kx<=R; kx++)
                s += i[(y+ky)*W + (x+kx)] * k[(ky+R)*K + (kx+R)];
        o[y*W + x] = s;
    }
}

// ================= SHARED MEMORY CONV2D =================
__global__ void conv2d_shared(const real* __restrict__ i,
                              real* __restrict__ o,
                              const real* __restrict__ k,
                              int W, int H)
{
    __shared__ real tile[T + 2*R][T + 2*R]; // kafelek + halo

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x*T + tx;
    int y = blockIdx.y*T + ty;

    // ------------------- Wczytanie danych do shared memory -------------------
    // wczytanie środka
if(x < W && y < H)
    tile[ty+R][tx+R] = i[y*W + x];

// halo X
if(tx < R && x >= R)
    tile[ty+R][tx] = i[y*W + (x-R)];

if(tx >= T-R && x+R < W)
    tile[ty+R][tx+2*R] = i[y*W + (x+R)];

// halo Y
if(ty < R && y >= R)
    tile[ty][tx+R] = i[(y-R)*W + x];

if(ty >= T-R && y+R < H)
    tile[ty+2*R][tx+R] = i[(y+R)*W + x];

// wczytanie rogów
if(tx < R && ty < R && x >= R && y >= R)
    tile[ty][tx] = i[(y-R)*W + (x-R)];

if(tx >= T-R && ty < R && x+R < W && y >= R)
    tile[ty][tx+2*R] = i[(y-R)*W + (x+R)];

if(tx < R && ty >= T-R && x >= R && y+R < H)
    tile[ty+2*R][tx] = i[(y+R)*W + (x-R)];

if(tx >= T-R && ty >= T-R && x+R < W && y+R < H)
    tile[ty+2*R][tx+2*R] = i[(y+R)*W + (x+R)];

    __syncthreads();

    // ------------------- Obliczenia konwolucji -------------------
    if(x >= R && y >= R && x < W-R && y < H-R){
        real s = 0;
        for(int ky=-R; ky<=R; ky++)
            for(int kx=-R; kx<=R; kx++)
                s += tile[ty+R+ky][tx+R+kx] * k[(ky+R)*K + (kx+R)];
        o[y*W + x] = s;
    }
}
