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
__global__ void conv2d(const real* __restrict__, real* __restrict__, const real* __restrict__, int, int);
__global__ void conv2d_shared(const real* __restrict__, real* __restrict__, const real* __restrict__, int, int);
__global__ void init(curandState*, int);
__global__ void montecarlo_kernel(
    unsigned long long* global_inside,
    unsigned long long seed,
    int total_samples
);
__global__ void reset_counter(unsigned long long* x){
    if(threadIdx.x == 0 && blockIdx.x == 0)
        *x = 0ULL;
}

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

        matmul_naive<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            matmul_naive<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize(); 
        double t_n = tg.toc() / 10;

        double gflops = (2.0 * N * N * N) / (t_n * 1e6);

        csv_add("results_gpu_matmul.csv",
                "matmul","naive",gpu,PREC,
                N,t_n,gflops,0,0,0,0);

        matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize();

        tg.tic();
        for(int i=0;i<10;i++)
            matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize(); 
        double t_t = tg.toc() / 10;

        gflops = (2.0 * N * N * N) / (t_t * 1e6);

        csv_add("results_gpu_matmul.csv",
                "matmul","tiled",gpu,PREC,
                N,t_t,gflops,0,0,0,0);

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
        CUFFT_EXEC(plan, d, d, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            CUFFT_EXEC(plan, d, d, CUFFT_FORWARD);
        cudaDeviceSynchronize();
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

        unsigned long long* d_inside;
        cudaMalloc(&d_inside, sizeof(unsigned long long));
        cudaMemset(d_inside, 0, sizeof(unsigned long long));
        
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        
        size_t shmem = block.x * sizeof(unsigned int);
        
        GpuTimer tg;
        int repeats = 1000;
        cudaDeviceSynchronize();
        tg.tic();
        for(int r = 0; r < repeats; r++){
            montecarlo_kernel<<<grid, block, shmem>>>(d_inside, 123456ULL, N);
        }
        cudaDeviceSynchronize();
        double t_ms = tg.toc();
        unsigned long long h_inside = 0;
        cudaMemcpy(&h_inside, d_inside,
            sizeof(h_inside), cudaMemcpyDeviceToHost);
        double t = t_ms/repeats;
        real pi = real(4) * h_inside / N;
        double time_s = t * 1e-3;
        double abs = std::abs(double(pi) - PI);
        double rel = abs / PI;

        csv_add("results_gpu_montecarlo.csv",
                "montecarlo", "pi",
                gpu_name(), PREC,
                N,
                t,
                N / time_s,     // throughput
                0,         // speedup (GPU-only)
                0,         // rmse (brak referencji)
                abs,
                rel);

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
        cudaMemset(dO, 0, bytes);
        conv2d<<<grid,block>>>(dI,dO,dK,S,S);
        cudaDeviceSynchronize();

        
        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            conv2d<<<grid, block>>>(dI, dO, dK, S, S);
        cudaDeviceSynchronize();
        double t_gpu_naive = tg.toc()/10;


        csv_add("results_gpu_conv2d.csv", "conv2d", "3x3_naive", "GPU", PREC,
                S, t_gpu_naive, (2.0*S*S*9)/(t_gpu_naive*1e6),
                0, 0, 0, 0);
        conv2d_shared<<<grid, block>>>(dI, dO, dK, S, S);
        cudaMemset(dO, 0, bytes);
        cudaDeviceSynchronize();

        tg.tic();
        for(int i=0;i<10;i++)
            conv2d_shared<<<grid, block>>>(dI, dO, dK, S, S);
        cudaDeviceSynchronize();
        double t_gpu_shared = tg.toc()/10;


        csv_add("results_gpu_conv2d.csv", "conv2d", "3x3_shared", "GPU", PREC,
                S, t_gpu_shared, (2.0*S*S*9)/(t_gpu_shared*1e6),
                0, 0, 0, 0);
        cudaFree(dI); cudaFree(dO); cudaFree(dK);
    }

    std::cout << "GPU-only benchmark DONE (" << PREC << ")\n";
    return 0;
}








