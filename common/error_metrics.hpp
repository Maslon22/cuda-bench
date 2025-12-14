#pragma once
#include <cmath>

double abs_error(double a,double b){ return std::abs(a-b); }
double rel_error(double a,double b){ return std::abs(a-b)/(std::abs(a)+1e-12); }

double rmse(const double* r,const double* t,int n){
    double s=0;
    for(int i=0;i<n;i++){ double d=r[i]-t[i]; s+=d*d; }
    return std::sqrt(s/n);
}

inline double rmse_complex(
    const std::complex<double>* a,
    const std::complex<double>* b,
    int n
) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double dr = a[i].real() - b[i].real();
        double di = a[i].imag() - b[i].imag();
        sum += dr*dr + di*di;
    }
    return std::sqrt(sum / n);
}