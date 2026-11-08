#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cmath>

/**
 * @brief Single GEneral Matrix Multiply
 *        C = alpha * A @ B + beta * C, row-major
 *        A: M x K, B: K x N, C: M x N
 *        One thread per element of C: x -> col, y -> row
 *
 * @param[in, out] C overwritten with the result
 */
__global__ void naive_sgemm(
    const int M, const int N, const int K,
    float alpha, const float *A, const float *B,
    float beta, float *C
) {
    const uint rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const uint colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx < M && colIdx < N){
        /**
         * k: is colIdx of A and rowIdx of B
         * K: rowDim(num of cols) of matrix A
         * K: colDim(num of rows) of matrix B
         *
         * n & N is index & rowDim(num of col) of matrix B and C
         *
         * A: M x K    A[rowIdx][k]      = A[rowIdx * K + k]
         * B: K x N    B[k][colIdx]      = B[k * N + colIdx]
         * C: M x N    C[rowIdx][colIdx] = C[rowIdx * N + colIdx]
         */

        float c = beta * C[rowIdx * N + colIdx];

        for (int k = 0; k < K; k++) {
            c += alpha * (A[rowIdx * K + k] * B[k * N + colIdx]);
        }

        C[rowIdx * N + colIdx] = c;
    }
}


#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


/**
 * @brief const int + device pointer + stream -> cudaError_t
 */
cudaError_t launch_naive_sgemm(
    const int M, const int N, const int K,
    float alpha, const float *dA, const float *dB,
    float beta, float *dC, cudaStream_t stream=nullptr) {

    if (M == 0 || N == 0)
        return cudaSuccess;

    constexpr int BLOCK = 32;
    dim3 block = {BLOCK, BLOCK}; // 1024 threads
    dim3 grid = {CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK)}; // x -> cols, y -> rows
    naive_sgemm<<<grid, block, 0, stream>>>(M, N, K, alpha, dA, dB, beta, dC);

    return cudaGetLastError();
}

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

    cudaMalloc(&dA, sizeof(float) * M * K);
    cudaMalloc(&dB,sizeof(float) * K * N);
    cudaMalloc(&dC, sizeof(float) * M * N);

    initialize_tensor_value (M, N, K, hA, hB, hC);

    cudaMemcpy(dA, hA, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    launch_naive_sgemm(M, N, K, alpha, dA, dB, beta, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA), free(hB), free(hC);

    return 0;
}
