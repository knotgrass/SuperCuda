#include <cmath>
#include "common.cuh"
#include "naive.cuh"


void initialize_tensor_value(int M, int N, int K, float *hA, float *hB, float *hC){
    for (int i=0; i< M*K ; i++){
        hA[i] = 0.1f * std::pow(-1, i);
    }

    for (int i=0; i< K*N ; i++){
        hB[i] = 0.2f * std::pow(-1, i + 1);
    }

    for (int i=0; i< M*N ; i++){
        hC[i] = 0.0f;
    }
}

int main(int argc, char** argv) {
    uint M = 4096;
    uint N = 2048;
    uint K = 8192;
    float alpha = 0.5;
    float beta = -0.1;
    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    hA = (float *)malloc(sizeof(float) * M * K);
    hB = (float *)malloc(sizeof(float) * K * N);
    hC = (float *)malloc(sizeof(float) * M * N);

    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&dB,sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));

    initialize_tensor_value (M, N, K, hA, hB, hC);

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    CUDA_CHECK(launch_naive_sgemm(M, N, K, alpha, dA, dB, beta, dC));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA), free(hB), free(hC);

    return 0;
}
