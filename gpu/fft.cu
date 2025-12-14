#include <cufft.h>
void fft_gpu(cufftComplex*d,int N){
    cufftHandle p;
    cufftPlan1d(&p,N,CUFFT_C2C,1);
    cufftExecC2C(p,d,d,CUFFT_FORWARD);
    cufftDestroy(p);
}