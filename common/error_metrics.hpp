#pragma once
#include <cmath>
#include <complex>

// ===================== RMSE (real) =====================
template <typename T>
double rmse(const T* a, const T* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = double(a[i]) - double(b[i]);
        s += d * d;
    }
    return std::sqrt(s / n);
}

// ===================== RMSE (complex) =====================
template <typename T>
double rmse_complex(const std::complex<T>* a,
                    const std::complex<T>* b,
                    int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double dr = double(a[i].real()) - double(b[i].real());
        double di = double(a[i].imag()) - double(b[i].imag());
        s += dr * dr + di * di;
    }
    return std::sqrt(s / n);
}

// ===================== ABS ERROR =====================
inline double abs_error(double ref, double val) {
    return std::abs(ref - val);
}

// ===================== REL ERROR =====================
inline double rel_error(double ref, double val) {
    return std::abs(ref - val) / std::abs(ref);
}
