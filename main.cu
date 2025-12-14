#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <cstring>

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>

#include "common/timer.hpp"
#include "common/csv.hpp"
#include "common/error_metrics.hpp"

// ===================== CPU =====================
void matmul_cpu(const float*, const float*, float*, int);
void matmul_cpu_blocked(const float*, const float*, float*, int);
double montecarlo_cpu(int);
void dft_cpu(const std::complex<double>*, std::complex<double>*, int);
void conv2d_cpu(const float*, float*, const float*, int, int);

// ===================== GPU =====================
__global__ void matmul_naive(const float*, const float*, float*, int);
__global__ void matmul_tiled(const float*, const float*, float*, int);
__global__ void init(curandState*, int);
__global__ void monte(curandState*, int*, int);
__global__ void conv2d(const float*, float*, const float*, int, int);

// ===================== HELPERS =====================
void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(e) << "\n";
        exit(1);
    }
}

double rmse(const float* a, const float* b, int n){
    double s=0;
    for(int i=0;i<n;i++){
        double d = double(a[i]) - double(b[i]);
        s += d*d;
    }
    return std::sqrt(s/n);
}

// =================================================
// ===================== MAIN ======================
// =================================================
int main() {

    // =================================================
    // MATRIX MULTIPLICATION
    // =================================================
    csv_header("results_matmul.csv");
    std::vector<int> mat_sizes = {128, 256, 512, 1024};

    for (int N : mat_sizes) {
    size_t bytes = N*N*sizeof(float);

    std::vector<float> A(N*N, 1.0f);
    std::vector<float> B(N*N, 1.0f);

    std::vector<float> Ccpu_naive(N*N);
    std::vector<float> Ccpu_blocked(N*N);
    std::vector<float> Cgpu(N*N);

    CpuTimer tc;

    // CPU naive
    tc.tic();
    matmul_cpu(A.data(), B.data(), Ccpu_naive.data(), N);
    double t_cpu_naive = tc.toc();

    // CPU blocked
    tc.tic();
    matmul_cpu_blocked(A.data(), B.data(), Ccpu_blocked.data(), N);
    double t_cpu_blocked = tc.toc();

    float *dA,*dB,*dC;
    check(cudaMalloc(&dA,bytes));
    check(cudaMalloc(&dB,bytes));
    check(cudaMalloc(&dC,bytes));

    check(cudaMemcpy(dA,A.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dB,B.data(),bytes,cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N+15)/16,(N+15)/16);

    GpuTimer tg;

    // GPU naive
    tg.tic();
    matmul_naive<<<grid,block>>>(dA,dB,dC,N);
    check(cudaDeviceSynchronize());
    double t_gpu_naive = tg.toc();

    check(cudaMemcpy(Cgpu.data(),dC,bytes,cudaMemcpyDeviceToHost));
    double err_naive = rmse(Ccpu_naive.data(), Cgpu.data(), N*N);

    // GPU tiled
    tg.tic();
    matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
    check(cudaDeviceSynchronize());
    double t_gpu_tiled = tg.toc();

    check(cudaMemcpy(Cgpu.data(),dC,bytes,cudaMemcpyDeviceToHost));
    double err_tiled = rmse(Ccpu_naive.data(), Cgpu.data(), N*N);

    double flops = 2.0 * N * N * N;

    csv_add("results_matmul.csv","matmul","naive","CPU","float",N,
            t_cpu_naive, flops/(t_cpu_naive*1e6), 1.0, 0,0,0);

    csv_add("results_matmul.csv","matmul","blocked","CPU","float",N,
            t_cpu_blocked, flops/(t_cpu_blocked*1e6),
            t_cpu_naive/t_cpu_blocked, 0,0,0);

    csv_add("results_matmul.csv","matmul","naive","GPU","float",N,
            t_gpu_naive, flops/(t_gpu_naive*1e6),
            t_cpu_naive/t_gpu_naive, err_naive,0,0);

    csv_add("results_matmul.csv","matmul","tiled","GPU","float",N,
            t_gpu_tiled, flops/(t_gpu_tiled*1e6),
            t_cpu_naive/t_gpu_tiled, err_tiled,0,0);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

    // =================================================
    // MONTE CARLO
    // =================================================
    csv_header("results_montecarlo.csv");
    std::vector<int> mc_sizes = {
    int(1e6),
    int(5e6),
    int(1e7),
    int(5e7)
    };

    for (int N : mc_sizes) {
        CpuTimer tc;
        tc.tic();
        double pi_cpu = montecarlo_cpu(N);
        double t_cpu = tc.toc();

        double thr_cpu = N / t_cpu;

        curandState* states;
        int* d_in;
        int h_in = 0;

        check(cudaMalloc(&states,N*sizeof(curandState)));
        check(cudaMalloc(&d_in,sizeof(int)));
        check(cudaMemcpy(d_in,&h_in,sizeof(int),cudaMemcpyHostToDevice));

        init<<<(N+255)/256,256>>>(states,N);

        GpuTimer tg;
        tg.tic();
         monte<<<(N+255)/256,256>>>(states,d_in,N);
        cudaDeviceSynchronize();
        double t_gpu = tg.toc();

        double thr_gpu = N / t_gpu;

        check(cudaMemcpy(&h_in,d_in,sizeof(int),cudaMemcpyDeviceToHost));
        double pi_gpu = 4.0 * h_in / N;
        double abs = abs_error(pi_cpu, pi_gpu);
        double rel = rel_error(pi_cpu, pi_gpu);

        csv_add("results_montecarlo.csv",
            "montecarlo","pi","CPU","double",N,
            t_cpu, thr_cpu, 1.0, 0, abs, rel);

        csv_add("results_montecarlo.csv",
            "montecarlo","pi","GPU","double",N,
            t_gpu, thr_gpu, t_cpu/t_gpu, 0, abs, rel);

        cudaFree(states); cudaFree(d_in);
    }

    // =================================================
    // FFT
    // =================================================
    csv_header("results_fft.csv");

    std::vector<int> fft_sizes = {256, 512, 1024, 2048};

    for (int N : fft_sizes) {

        // ======== INPUT ========
        std::vector<std::complex<double>> in(N);
        std::vector<std::complex<double>> out_cpu(N);
        std::vector<std::complex<double>> out_gpu(N);

        for (int i = 0; i < N; i++)
            in[i] = { double(i), 0.0 };

        // ======== CPU: DFT ========
        CpuTimer tc;
        tc.tic();
        dft_cpu(in.data(), out_cpu.data(), N);
        double t_cpu = tc.toc();

        // ======== GPU: cuFFT ========
        cufftDoubleComplex* d_fft;
        check(cudaMalloc(&d_fft, N * sizeof(cufftDoubleComplex)));
        check(cudaMemcpy(
            d_fft,
            in.data(),
            N * sizeof(cufftDoubleComplex),
            cudaMemcpyHostToDevice
        ));

        cufftHandle plan;
        cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);

        GpuTimer tg;
        tg.tic();
        cufftExecZ2Z(plan, d_fft, d_fft, CUFFT_FORWARD);
        check(cudaDeviceSynchronize());
        double t_gpu = tg.toc();

        // ======== COPY BACK ========
        std::vector<cufftDoubleComplex> h_fft(N);
        check(cudaMemcpy(
            h_fft.data(),
            d_fft,
            N * sizeof(cufftDoubleComplex),
            cudaMemcpyDeviceToHost
        ));

        for (int i = 0; i < N; i++) {
            out_gpu[i] = {
                h_fft[i].x,
                h_fft[i].y
            };
        }

        // ======== METRICS ========
        double flops = 5.0 * N * std::log2((double)N);
        double gflops_cpu = flops / (t_cpu * 1e6);
        double gflops_gpu = flops / (t_gpu * 1e6);
        double speedup = t_cpu / t_gpu;
        double rmse_fft = rmse_complex(
            out_cpu.data(),
            out_gpu.data(),
            N
        );

        // ======== CSV ========
        csv_add(
            "results_fft.csv",
            "fft", "dft", "CPU", "double", N,
            t_cpu, gflops_cpu, 1.0,
            0.0, 0.0, 0.0
        );

        csv_add(
            "results_fft.csv",
            "fft", "cufft", "GPU", "double", N,
            t_gpu, gflops_gpu, speedup,
            rmse_fft, 0.0, 0.0
        );

        // ======== CLEANUP ========
        cufftDestroy(plan);
        cudaFree(d_fft);
    }

    // =================================================
    // CONVOLUTION
    // =================================================
    csv_header("results_conv2d.csv");
    std::vector<int> conv_sizes = {512, 1024, 2048};

    for (int S : conv_sizes) {
        int W=S,H=S;
        size_t bytes=W*H*sizeof(float);

        std::vector<float> img(W*H,1.0f), out_cpu(W*H), out_gpu(W*H);
        float kernel[9]={1,1,1,1,1,1,1,1,1};

        CpuTimer tc;
        tc.tic();
        conv2d_cpu(img.data(),out_cpu.data(),kernel,W,H);
        double t_cpu = tc.toc();

        float *dI,*dO,*dK;
        check(cudaMalloc(&dI,bytes));
        check(cudaMalloc(&dO,bytes));
        check(cudaMalloc(&dK,9*sizeof(float)));

        check(cudaMemcpy(dI,img.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dK,kernel,9*sizeof(float),cudaMemcpyHostToDevice));

        dim3 block(16,16);
        dim3 grid((W+15)/16,(H+15)/16);

        GpuTimer tg;
        tg.tic();
        conv2d<<<grid,block>>>(dI,dO,dK,W,H);
        check(cudaDeviceSynchronize());
        double t_gpu = tg.toc();
        double flops = W * H * 9 * 2;
        double gflops_cpu = flops / (t_cpu * 1e6);
        double gflops_gpu = flops / (t_gpu * 1e6);
        double rmse_conv = rmse(out_cpu.data(), out_gpu.data(), W*H);

        csv_add("results_conv2d.csv",
            "conv2d","3x3","CPU","float",W,
            t_cpu, gflops_cpu, 1.0, 0,0,0);

        csv_add("results_conv2d.csv",
            "conv2d","3x3","GPU","float",W,
            t_gpu, gflops_gpu, t_cpu/t_gpu, rmse_conv,0,0);

        cudaFree(dI); cudaFree(dO); cudaFree(dK);
    }

    std::cout << "ALL BENCHMARKS DONE\n";
    return 0;
}
