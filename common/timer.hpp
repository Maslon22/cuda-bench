#pragma once
#include <chrono>

struct CpuTimer {
    std::chrono::high_resolution_clock::time_point start;
    void tic() { start = std::chrono::high_resolution_clock::now(); }
    double toc() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    void tic() { cudaEventRecord(start); }
    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};