#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>

#include "common/precision.hpp"
#include "common/timer.hpp"
#include "common/csv.hpp"
#include "gpu/montecarlo.cuh"

constexpr double PI = 3.141592653589793238462643383279502884;
__global__ void matmul_naive(const real*, const real*, real*, int);
__global__ void matmul_tiled(const real*, const real*, real*, int);
__global__ void conv2d(const real*, real*, const real*, int, int);
__global__ void init(curandState*, int);
__global__ void monte(curandState*, int*, int);

// ===================== PRECISION =====================
#if defined(USE_FP32)
    #define CUFFT_EXEC cufftExecC2C
    #define CUFFT_TYPE CUFFT_C2C
    using cufft_complex = cufftComplex;
    constexpr const char* PREC = "float";
#elif defined(USE_FP64)
    #define CUFFT_EXEC cufftExecZ2Z
    #define CUFFT_TYPE CUFFT_Z2Z
    using cufft_complex = cufftDoubleComplex;
    constexpr const char* PREC = "double";
#else
    #error "Define USE_FP32 or USE_FP64"
#endif

// ===================== UTILS =====================
inline void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

std::string gpu_name() {
    cudaDeviceProp p{};
    int dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&p, dev);
    return p.name;
}

template <typename T>
void fill_random(std::vector<T>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (auto& x : v) x = static_cast<T>(d(gen));
}

// ====================================================
// ===================== MAIN ==========================
// ====================================================

int main() {

    std::string gpu = gpu_name();
    std::cout << "GPU-only benchmark on: " << gpu
              << " (" << PREC << ")\n";

    // ====================================================
    // ================= MATMUL ===========================
    // ====================================================
    csv_header("results_gpu_matmul.csv");
    std::vector<int> sizes_mm = {4096, 6144, 8192, 16384};

    for (int N : sizes_mm) {
        size_t bytes = N * N * sizeof(real);

        std::vector<real> A(N*N), B(N*N);
        fill_random(A, 0);
        fill_random(B, 1);

        real *dA,*dB,*dC;
        check(cudaMalloc(&dA,bytes));
        check(cudaMalloc(&dB,bytes));
        check(cudaMalloc(&dC,bytes));
        check(cudaMemcpy(dA,A.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dB,B.data(),bytes,cudaMemcpyHostToDevice));

        dim3 block(16,16);
        dim3 grid((N+15)/16,(N+15)/16);

        GpuTimer tg;

        // --- NAIVE ---
        tg.tic();
        for(int i=0;i<5;i++)
            matmul_naive<<<grid,block>>>(dA,dB,dC,N);
        check(cudaDeviceSynchronize());
        double t = tg.toc()/5.0;

        double gflops = (2.0 * N * N * N) / (t * 1e6);

        csv_add("results_gpu_matmul.csv",
                "matmul","naive",gpu,PREC,
                N,t,gflops,0,0,0,0);

        // --- TILED ---
        tg.tic();
        for(int i=0;i<5;i++)
            matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
        check(cudaDeviceSynchronize());
        t = tg.toc()/5.0;

        gflops = (2.0 * N * N * N) / (t * 1e6);

        csv_add("results_gpu_matmul.csv",
                "matmul","tiled",gpu,PREC,
                N,t,gflops,0,0,0,0);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ====================================================
    // ================= FFT ==============================
    // ====================================================
    csv_header("results_gpu_fft.csv");
    std::vector<int> fft_sizes = {1<<19,1<<20,1<<21,1<<22};

    for (int N : fft_sizes) {
        cufft_complex* d;
        check(cudaMalloc(&d, N*sizeof(cufft_complex)));
        check(cudaMemset(d, 0, N*sizeof(cufft_complex)));

        cufftHandle plan;
        cufftPlan1d(&plan, N, CUFFT_TYPE, 1);

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            CUFFT_EXEC(plan, d, d, CUFFT_FORWARD);
        check(cudaDeviceSynchronize());
        double t = tg.toc()/10.0;

        double gflops = 5.0 * N * std::log2(double(N)) / (t * 1e6);

        csv_add("results_gpu_fft.csv",
                "fft","cufft",gpu,PREC,
                N,t,gflops,0,0,0,0);

        cufftDestroy(plan);
        cudaFree(d);
    }

    // ====================================================
    // ================= MONTE CARLO ======================
    // ====================================================
    csv_header("results_gpu_montecarlo.csv");
    std::vector<int> mc_sizes = {10'000'000, 30'000'000, 50'000'000};

    for (int N : mc_sizes) {

        curandStatePhilox4_32_10_t* d_states;
        unsigned long long* d_inside;
        unsigned long long h_inside = 0;

        check(cudaMalloc(&d_states,
            N * sizeof(curandStatePhilox4_32_10_t)));
        check(cudaMalloc(&d_inside,
            sizeof(unsigned long long)));

        check(cudaMemcpy(d_inside, &h_inside,
            sizeof(h_inside), cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        init_rng<<<grid, block>>>(d_states, N);

        GpuTimer tg;
        tg.tic();
        montecarlo_gpu<<<grid, block>>>(d_states, d_inside, N);
        check(cudaDeviceSynchronize());
        double t = tg.toc();

        check(cudaMemcpy(&h_inside, d_inside,
            sizeof(h_inside), cudaMemcpyDeviceToHost));

        real pi = real(4) * h_inside / N;

        double abs = std::abs(double(pi) - PI);
        double rel = abs / PI;

        csv_add("results_gpu_montecarlo.csv",
                "montecarlo", "pi",
                gpu_name(), PREC,
                N,
                t,
                N / t,     // throughput
                0,         // speedup (GPU-only)
                0,         // rmse (brak referencji)
                abs,
                rel);

        cudaFree(d_states);
        cudaFree(d_inside);
    }

    // ====================================================
    // ================= CONV2D ===========================
    // ====================================================
    csv_header("results_gpu_conv2d.csv");
    std::vector<int> conv_sizes = {4096, 6144, 8192, 16384, 32768};

    for (int S : conv_sizes) {
        size_t bytes = S*S*sizeof(real);

        std::vector<real> img(S*S);
        fill_random(img, 0);

        real kernel[9];
        std::mt19937 gen(123);
        std::uniform_real_distribution<double> d(-1.0,1.0);
        for(int i=0;i<9;i++) kernel[i]=static_cast<real>(d(gen));

        real *dI,*dO,*dK;
        check(cudaMalloc(&dI,bytes));
        check(cudaMalloc(&dO,bytes));
        check(cudaMalloc(&dK,9*sizeof(real)));

        check(cudaMemcpy(dI,img.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dK,kernel,9*sizeof(real),cudaMemcpyHostToDevice));

        dim3 block(16,16);
        dim3 grid((S+15)/16,(S+15)/16);

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            conv2d<<<grid,block>>>(dI,dO,dK,S,S);
        check(cudaDeviceSynchronize());
        double t = tg.toc()/10.0;

        double gflops = (2.0 * S * S * 9) / (t * 1e6);

        csv_add("results_gpu_conv2d.csv",
                "conv2d","3x3",gpu,PREC,
                S,t,gflops,0,0,0,0);

        cudaFree(dI); cudaFree(dO); cudaFree(dK);
    }

    std::cout << "GPU-only benchmark DONE (" << PREC << ")\n";
    return 0;
}
