#include "../common/precision.hpp"
#include <cufft.h>

void fft_gpu(void* d, int N){
#if defined(USE_FP32)
    cufftHandle p;
    cufftPlan1d(&p, N, CUFFT_C2C, 1);
    cufftExecC2C(p, (cufftComplex*)d, (cufftComplex*)d, CUFFT_FORWARD);
#elif defined(USE_FP64)
    cufftHandle p;
    cufftPlan1d(&p, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(p, (cufftDoubleComplex*)d, (cufftDoubleComplex*)d, CUFFT_FORWARD);
#endif
    cufftDestroy(p);
}
