#include "../common/precision.hpp"

#define T 16
__global__ void conv2d(const real* i, real* o, const real* k, int W, int H){
    int x = blockIdx.x*T + threadIdx.x;
    int y = blockIdx.y*T + threadIdx.y;
    if(x>0 && y>0 && x<W-1 && y<H-1){
        real s = 0;
        for(int ky=-1;ky<=1;ky++)
            for(int kx=-1;kx<=1;kx++)
                s += i[(y+ky)*W+x+kx] * k[(ky+1)*3+(kx+1)];
        o[y*W+x] = s;
    }
}
