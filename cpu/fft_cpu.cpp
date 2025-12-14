#include <complex>
void dft_cpu(const std::complex<double>*in,std::complex<double>*out,int N){
    for(int k=0;k<N;k++){
        std::complex<double>s(0,0);
        for(int n=0;n<N;n++){
            double a=-2*3.14*k*n/N;
            s+=in[n]*std::complex<double>(cos(a),sin(a));
        }
        out[k]=s;
    }
}