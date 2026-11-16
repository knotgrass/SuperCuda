#pragma once
#include <cuda_runtime.h>

cudaError_t launch_naive_sgemm(
    int M, int N, int K,
    float alpha, const float *dA, const float *dB,
    float beta, float *dC, cudaStream_t stream = nullptr);
