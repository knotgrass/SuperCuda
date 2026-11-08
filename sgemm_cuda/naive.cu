#include <cuda_runtime.h>

#include "common.cuh"
#include "naive.cuh"

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

/**
 * @brief const int + device pointer + stream -> cudaError_t
 */
cudaError_t launch_naive_sgemm(
    const int M, const int N, const int K,
    float alpha, const float *dA, const float *dB,
    float beta, float *dC, cudaStream_t stream) {

    if (M == 0 || N == 0)
        return cudaSuccess;

    constexpr int BLOCK = 32;
    const dim3 block(BLOCK, BLOCK);                           // 1024 threads
    const dim3 grid(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));  // x -> cols, y -> rows
    naive_sgemm<<<grid, block, 0, stream>>>(M, N, K, alpha, dA, dB, beta, dC);

    return cudaGetLastError();
}
