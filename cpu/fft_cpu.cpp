#include <cmath>
#include "../common/precision.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


void dft_cpu(const complex_t* in, complex_t* out, int N){
    for(int k=0;k<N;k++){
        complex_t s(0,0);
        for(int n=0;n<N;n++){
            real a = -2 * M_PI * k * n / N;
            s += in[n] * complex_t(cos(a), sin(a));
        }
        out[k] = s;
    }
}
