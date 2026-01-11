#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <random>
#include "common/precision.hpp"
#include "gpu/montecarlo.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>

#include "common/timer.hpp"
#include "common/csv.hpp"
#include "common/error_metrics.hpp"


template <typename T>
void fill_random(std::vector<T>& v, unsigned seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto& x : v)
        x = static_cast<T>(dist(gen));
}

template <typename T>
void fill_random_complex(std::vector<std::complex<T>>& v, unsigned seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto& x : v)
        x = std::complex<T>(
            static_cast<T>(dist(gen)),
            static_cast<T>(dist(gen))
        );
}
// ============================================================
// =================== PRECYZJA ================================
// ============================================================

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

// ============================================================
// ===================== CPU =================================
// ============================================================

void matmul_cpu(const real*, const real*, real*, int);
void matmul_cpu_blocked(const real*, const real*, real*, int);
real montecarlo_cpu(int);
void dft_cpu(const complex_t*, complex_t*, int);
void conv2d_cpu(const real*, real*, const real*, int, int);

// ============================================================
// ===================== GPU =================================
// ============================================================

__global__ void matmul_naive(const real*, const real*, real*, int);
__global__ void matmul_tiled(const real*, const real*, real*, int);
__global__ void init(curandState*, int);
__global__ void monte(curandState*, int*, int);
__global__ void conv2d(const real* __restrict__, real* __restrict__, const real* __restrict__, int, int);
__global__ void conv2d_shared(const real* __restrict__, real* __restrict__, const real* __restrict__, int, int);
__global__ void reset_counter(unsigned long long* x){
    if(threadIdx.x == 0 && blockIdx.x == 0)
        *x = 0ULL;
}


// ============================================================

inline void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// ============================================================
// ===================== MAIN =================================
// ============================================================

int main() {

    // ============================================================
    // ================= MATRIX MULTIPLICATION ===================
    // ============================================================

    csv_header("results_matmul.csv");
    std::vector<int> sizes = {256, 512, 1024, 2048};

    for (int N : sizes) {
        size_t bytes = N * N * sizeof(real);

        std::vector<real> A(N*N);
        std::vector<real> B(N*N);

        fill_random(A, 0);
        fill_random(B, 1);

        std::vector<real> Ccpu1(N*N), Ccpu2(N*N), Cgpu(N*N);

        CpuTimer tc;

        tc.tic();
        matmul_cpu(A.data(), B.data(), Ccpu1.data(), N);
        double t_cpu_naive = tc.toc();

        tc.tic();
        matmul_cpu_blocked(A.data(), B.data(), Ccpu2.data(), N);
        double t_cpu_blocked = tc.toc();

        double t_cpu_ref = std::min(t_cpu_naive, t_cpu_blocked);

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
        double t_gpu_naive = tg.toc() / 10;

        check(cudaMemcpy(Cgpu.data(),dC,bytes,cudaMemcpyDeviceToHost));
        double err_naive = rmse(Ccpu2.data(), Cgpu.data(), N*N);

        matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize();

        tg.tic();
        for(int i=0;i<10;i++)
            matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize(); 
        double t_gpu_tiled = tg.toc() / 10;

        check(cudaMemcpy(Cgpu.data(),dC,bytes,cudaMemcpyDeviceToHost));
        double err_tiled = rmse(Ccpu2.data(), Cgpu.data(), N*N);

        double flops = 2.0 * N * N * N;

        csv_add("results_matmul.csv","matmul","naive","CPU",PREC,N,
                t_cpu_naive, flops/(t_cpu_naive*1e6), 1.0, 0,0,0);

        csv_add("results_matmul.csv","matmul","blocked","CPU",PREC,N,
                t_cpu_blocked, flops/(t_cpu_blocked*1e6),
                t_cpu_naive/t_cpu_blocked, 0,0,0);

        csv_add("results_matmul.csv","matmul","naive","GPU",PREC,N,
                t_gpu_naive, flops/(t_gpu_naive*1e6),
                t_cpu_ref/t_gpu_naive, err_naive,0,0);

        csv_add("results_matmul.csv","matmul","tiled","GPU",PREC,N,
                t_gpu_tiled, flops/(t_gpu_tiled*1e6),
                t_cpu_ref/t_gpu_tiled, err_tiled,0,0);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ============================================================
    // ================= MONTE CARLO ==============================
    // ============================================================

    csv_header("results_montecarlo.csv");
    std::vector<int> mc_sizes = {
    int(1e6), int(5e6), int(1e7), int(5e7)
};


    for (int N : mc_sizes) {

        CpuTimer tc;
        tc.tic();
        real pi_cpu = montecarlo_cpu(N);
        double t_cpu = tc.toc();

        double thr_cpu = N / t_cpu;

        curandStatePhilox4_32_10_t* d_states;
        unsigned long long* d_inside;
        unsigned long long h_inside = 0;

        cudaMalloc(&d_states, N * sizeof(curandStatePhilox4_32_10_t));
        cudaMalloc(&d_inside, sizeof(unsigned long long));
        cudaMemcpy(d_inside, &h_inside, sizeof(h_inside), cudaMemcpyHostToDevice);

        init_rng<<<(N+255)/256,256>>>(d_states, N);
        montecarlo_gpu<<<(N+255)/256,256>>>(d_states, d_inside, N);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<5;i++){
            reset_counter<<<1,1>>>(d_inside);
            montecarlo_gpu<<<(N+255)/256,256>>>(d_states, d_inside, N);
        }
        cudaDeviceSynchronize();
        double t_gpu = tg.toc()/5.0;

        cudaMemcpy(&h_inside, d_inside, sizeof(h_inside), cudaMemcpyDeviceToHost);

        real pi_gpu = real(4) * h_inside / N;

        double abs = abs_error(pi_cpu, pi_gpu);
        double rel = rel_error(pi_cpu, pi_gpu);

        csv_add("results_montecarlo.csv","montecarlo","pi","CPU",PREC,
                N, t_cpu, thr_cpu, 1.0, 0, abs, rel);

        csv_add("results_montecarlo.csv","montecarlo","pi","GPU",PREC,
                N, t_gpu, N/t_gpu, t_cpu/t_gpu, 0, abs, rel);

        cudaFree(d_states);
        cudaFree(d_inside);
    }

    // ============================================================
    // ================= FFT =====================================
    // ============================================================

    csv_header("results_fft.csv");
    std::vector<int> fft_sizes = {512, 1024, 2048, 4096};

    for (int N : fft_sizes) {

        std::vector<complex_t> in(N), out_cpu(N), out_gpu(N);
        fill_random_complex(in, 0);

        CpuTimer tc;
        tc.tic();
        dft_cpu(in.data(), out_cpu.data(), N);
        double t_cpu = tc.toc();

        cufft_complex* d_fft;
        check(cudaMalloc(&d_fft, N*sizeof(cufft_complex)));
        check(cudaMemcpy(d_fft, in.data(), N*sizeof(cufft_complex),
                         cudaMemcpyHostToDevice));

        cufftHandle plan;
        cufftPlan1d(&plan, N, CUFFT_TYPE, 1);
        CUFFT_EXEC(plan, d_fft, d_fft, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            CUFFT_EXEC(plan, d_fft, d_fft, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        double t_gpu = tg.toc()/10.0;

        check(cudaMemcpy(out_gpu.data(), d_fft,
                         N*sizeof(complex_t),
                         cudaMemcpyDeviceToHost));

        double flops = 5.0 * N * std::log2(double(N));
        double rmse_fft = rmse_complex(out_cpu.data(), out_gpu.data(), N);

        csv_add("results_fft.csv","fft","dft","CPU",PREC,
                N, t_cpu, flops/(t_cpu*1e6), 1.0, 0,0,0);

        csv_add("results_fft.csv","fft","cufft","GPU",PREC,
                N, t_gpu, flops/(t_gpu*1e6),
                t_cpu/t_gpu, rmse_fft,0,0);

        cufftDestroy(plan);
        cudaFree(d_fft);
    }

    // ============================================================
    // ================= CONV2D ==================================
    // ============================================================

    csv_header("results_conv2d.csv");
    std::vector<int> conv_sizes = {512, 1024, 2048, 4096};


    for (int S : conv_sizes) {
        size_t bytes = S*S*sizeof(real);

        std::vector<real> img(S*S);
        fill_random(img, 0);

        real kernel[9];
        {
            std::mt19937 gen(123);
            std::uniform_real_distribution<double> d(-1.0, 1.0);
            for (int i = 0; i < 9; i++)
                kernel[i] = static_cast<real>(d(gen));
        }

        std::vector<real> out_cpu(S*S), out_gpu(S*S);

        CpuTimer tc;
        tc.tic();
        conv2d_cpu(img.data(), out_cpu.data(), kernel, S, S);
        double t_cpu = tc.toc();

        real *dI,*dO,*dK;
        check(cudaMalloc(&dI,bytes));
        check(cudaMalloc(&dO,bytes));
        check(cudaMalloc(&dK,9*sizeof(real)));

        check(cudaMemcpy(dI,img.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dK,kernel,9*sizeof(real),cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid(
            (S + block.x - 1) / block.x,
            (S + block.y - 1) / block.y);
        cudaMemset(dO, 0, bytes);
        conv2d<<<grid,block>>>(dI,dO,dK,S,S);
        cudaDeviceSynchronize();

        
        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            conv2d<<<grid, block>>>(dI, dO, dK, S, S);
        cudaDeviceSynchronize();
        double t_gpu_naive = tg.toc()/10;

        check(cudaMemcpy(out_gpu.data(), dO, bytes, cudaMemcpyDeviceToHost));
        double rmse_naive = rmse(out_cpu.data(), out_gpu.data(), S*S);

        csv_add("results_conv2d.csv", "conv2d", "3x3_naive", "GPU", PREC,
                S, t_gpu_naive, (2.0*S*S*9)/(t_gpu_naive*1e6),
                t_cpu/t_gpu_naive, rmse_naive, 0, 0);
        conv2d_shared<<<grid, block>>>(dI, dO, dK, S, S);
        cudaMemset(dO, 0, bytes);
        cudaDeviceSynchronize();

        tg.tic();
        for(int i=0;i<10;i++)
            conv2d_shared<<<grid, block>>>(dI, dO, dK, S, S);
        cudaDeviceSynchronize();
        double t_gpu_shared = tg.toc()/10;

        check(cudaMemcpy(out_gpu.data(), dO, bytes, cudaMemcpyDeviceToHost));
        double rmse_shared = rmse(out_cpu.data(), out_gpu.data(), S*S);

        csv_add("results_conv2d.csv", "conv2d", "3x3_shared", "GPU", PREC,
                S, t_gpu_shared, (2.0*S*S*9)/(t_gpu_shared*1e6),
                t_cpu/t_gpu_shared, rmse_shared, 0, 0);

        // ================= CPU CSV =================
        csv_add("results_conv2d.csv", "conv2d", "3x3", "CPU", PREC,
                S, t_cpu, (2.0*S*S*9)/(t_cpu*1e6), 1.0, 0, 0, 0);

        // ================= Cleanup =================
        cudaFree(dI); cudaFree(dO); cudaFree(dK);
    }

    std::cout << "BENCHMARK DONE (" << PREC << ")\n";
    return 0;
}
