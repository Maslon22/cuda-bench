#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>

#include "common/timer.hpp"
#include "common/csv.hpp"

#define CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error: "<<cudaGetErrorString(x)<<"\n"; exit(1); }

// ===================== KERNELE =====================
__global__ void matmul_naive(const float* A, const float* B, float* C, int N);
__global__ void matmul_tiled(const float*, const float*, float*, int);
__global__ void init(curandState*, int);
__global__ void monte(curandState*, int*, int);
__global__ void conv2d(const float*, float*, const float*, int, int);

// ===================== GPU NAME =====================
std::string get_gpu_name() {
    cudaDeviceProp prop{};
    int dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    return std::string(prop.name);
}

// ===================== MAIN =====================
int main() {
    std::string gpu = get_gpu_name();
    std::cout << "Running on GPU: " << gpu << "\n";

    // ===================== MATMUL =====================
    csv_header("results_gpu_matmul.csv");
    std::vector<int> sizes_mm = {4096, 6144, 8192};

    for (int N : sizes_mm) {
        size_t bytes = N * N * sizeof(float);
        std::vector<float> A(N*N,1.f), B(N*N,1.f);
        float *dA,*dB,*dC;

        CHECK(cudaMalloc(&dA,bytes));
        CHECK(cudaMalloc(&dB,bytes));
        CHECK(cudaMalloc(&dC,bytes));
        CHECK(cudaMemcpy(dA,A.data(),bytes,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB,B.data(),bytes,cudaMemcpyHostToDevice));

        dim3 block(16,16), grid((N+15)/16,(N+15)/16);

        // ---- NAIVE ----
        matmul_naive<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            matmul_naive<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize();
        double t = tg.toc()/10.0;

        double gflops = (2.0*N*N*N)/(t*1e6);

        csv_add("results_gpu_matmul.csv","matmul","naive",gpu,"float",
                N,t,gflops,0,0,0,0);

        // ---- TILED ----
        matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize();

        tg.tic();
        for(int i=0;i<10;i++)
            matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
        cudaDeviceSynchronize();
        t = tg.toc()/10.0;
        gflops = (2.0*N*N*N)/(t*1e6);

        csv_add("results_gpu_matmul.csv","matmul","tiled",gpu,"float",
                N,t,gflops,0,0,0,0);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ===================== FFT =====================
    csv_header("results_gpu_fft.csv");
    std::vector<int> fft_sizes = {1<<19,1<<20,1<<21,1<<22};

    for (int N : fft_sizes) {
        cufftDoubleComplex* d;
        CHECK(cudaMalloc(&d,N*sizeof(cufftDoubleComplex)));
        CHECK(cudaMemset(d,0,N*sizeof(cufftDoubleComplex)));

        cufftHandle plan;
        cufftPlan1d(&plan,N,CUFFT_Z2Z,1);

        cufftExecZ2Z(plan,d,d,CUFFT_FORWARD);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            cufftExecZ2Z(plan,d,d,CUFFT_FORWARD);
        cudaDeviceSynchronize();
        double t = tg.toc()/10.0;

        csv_add("results_gpu_fft.csv","fft","cufft",gpu,"double",
                N,t,0,0,0,0,0);

        cufftDestroy(plan);
        cudaFree(d);
    }

    // ===================== MONTE CARLO =====================
    csv_header("results_gpu_monte.csv");
    std::vector<long long> mc_sizes = {30000000LL,50000000LL,80000000LL};

    for (auto N : mc_sizes) {
        curandState* s;
        int* d_in;
        CHECK(cudaMalloc(&s,N*sizeof(curandState)));
        CHECK(cudaMalloc(&d_in,sizeof(int)));
        CHECK(cudaMemset(d_in,0,sizeof(int)));

        init<<<(N+255)/256,256>>>(s,N);
        cudaDeviceSynchronize();

        monte<<<(N+255)/256,256>>>(s,d_in,N);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<5;i++)
            monte<<<(N+255)/256,256>>>(s,d_in,N);
        cudaDeviceSynchronize();
        double t = tg.toc()/5.0;

        csv_add("results_gpu_monte.csv","montecarlo","pi",gpu,"double",
                (int)N,t,0,0,0,0,0);

        cudaFree(s); cudaFree(d_in);
    }

    // ===================== CONV2D =====================
    csv_header("results_gpu_conv.csv");
    std::vector<int> img_sizes = {4096,6144,8192};
    float kernel[9] = {1,1,1,1,1,1,1,1,1};

    for (int S : img_sizes) {
        size_t bytes = S*S*sizeof(float);
        float *dI,*dO,*dK;
        CHECK(cudaMalloc(&dI,bytes));
        CHECK(cudaMalloc(&dO,bytes));
        CHECK(cudaMalloc(&dK,9*sizeof(float)));
        CHECK(cudaMemset(dI,1,bytes));
        CHECK(cudaMemcpy(dK,kernel,9*sizeof(float),cudaMemcpyHostToDevice));

        dim3 block(16,16), grid((S+15)/16,(S+15)/16);

        conv2d<<<grid,block>>>(dI,dO,dK,S,S);
        cudaDeviceSynchronize();

        GpuTimer tg;
        tg.tic();
        for(int i=0;i<10;i++)
            conv2d<<<grid,block>>>(dI,dO,dK,S,S);
        cudaDeviceSynchronize();
        double t = tg.toc()/10.0;

        csv_add("results_gpu_conv.csv","conv2d","3x3",gpu,"float",
                S,t,0,0,0,0,0);

        cudaFree(dI); cudaFree(dO); cudaFree(dK);
    }

    std::cout << "GPU-only benchmarks DONE\n";
    return 0;
}
