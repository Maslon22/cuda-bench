#pragma once
#include <complex>

#if defined(USE_FP32)
using real = float;
using complex_t = std::complex<float>;
#elif defined(USE_FP64)
using real = double;
using complex_t = std::complex<double>;
#else
#error "Define USE_FP32 or USE_FP64"
#endif
