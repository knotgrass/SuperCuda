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


int main(int argc, char** argv) {
    uint M = 4096;
    uint N = 4096;
    uint K = 4096;
    float alpha = 0.5;
    float beta = -0.1;
    float *A, *B, *C;

    A = (float *)malloc(sizeof(float) * M * K);
    B = (float *)malloc(sizeof(float) * K * N);
    C = (float *)malloc(sizeof(float) * M * N);

    
    dim3 gridDim = {};
    dim3 blockDim = {};
    naive_sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    return 0;
}
