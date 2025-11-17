#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

using namespace std::chrono;

__global__ void vector_sum(const float *d_vec, const size_t n, double *d_result) {
    const size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid < n) {
        atomicAdd(d_result, d_vec[global_tid]);
    }
}
/**
 * @brief
 * Per-block accumulation using shared-memory atomics with a single global update per block.
 */
__global__ void vector_sum2(const float *d_vec, const size_t n, double *d_result) {
    const size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate a block-wide accumulator in shared memory.
    __shared__ double s_result;

    // Zero the shared accumulator once per block.
    if (threadIdx.x == 0) {
        s_result = 0;
    }
    __syncthreads(); // Ensure all threads see the initialized accumulator.

    // Accumulate contributions via shared-memory atomicAdd.
    if (global_tid < n) {
        atomicAdd(&s_result, d_vec[global_tid]);
    }
    __syncthreads(); // Ensure the block-wide sum is complete before flushing.

    // The block leader commits the partial sum to global memory.
    if (threadIdx.x == 0) {
        atomicAdd(d_result, s_result);
    }
}

/**
 * @brief
 * tree-based reduction with shared memory in one block.
 */
__global__ void vector_sum3(const float *d_vec, const size_t n, double *d_result) {
    const size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // Allocate a block-wide accumulator in shared memory.
    extern __shared__ double sdata[];
    sdata[threadIdx.x] = (gtid < n ? d_vec[gtid] : 0.0);

    for (size_t s = blockDim.x / 2; s > 0; s /= 2) {
        __syncthreads();
        if (threadIdx.x < s) { // haft of all threads in block perform
            sdata[threadIdx.x] += sdata[threadIdx.x + s]; // use add instead of atomicAdd
        }
    }

    __syncthreads(); // Ensure the block-wide sum is complete before flushing.
    if (threadIdx.x == 0) {
        atomicAdd(d_result, sdata[0]);
    }
}

/**
 * @brief
 * use register memory
 */
__global__ void vector_sum4(const float *d_vec, const size_t n, double *d_result) {
    const size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    //           ↑         ↑            ↑             ↑
    //       Register   Register     Register      Register
    //                    (built-in variables)
    extern __shared__ double sdata[];
    sdata[threadIdx.x] = (gtid < n ? d_vec[gtid] : 0.0);

    for (size_t s = blockDim.x / 2; s > 16; s /= 2) {
        __syncthreads();
        if (threadIdx.x < s) { // haft of all threads in block perform
            sdata[threadIdx.x] += sdata[threadIdx.x + s]; // use add instead of atomicAdd
        }
    }

    float x = sdata[threadIdx.x]; // ✅ Copy vào register 'x'
    //    ↑          ↑
    //  register    shared memory

    if (threadIdx.x < 32) {
        x += __shfl_sync(0xffffffff, x, threadIdx.x + 16); // ✅ Copy từ register của thread khác vào register 'x'
        x += __shfl_sync(0xffffffff, x, threadIdx.x + 8);
        x += __shfl_sync(0xffffffff, x, threadIdx.x + 4);
        x += __shfl_sync(0xffffffff, x, threadIdx.x + 2);
        x += __shfl_sync(0xffffffff, x, threadIdx.x + 1);
    }

    if (threadIdx.x == 0){
        atomicAdd(d_result, x);
    }
}

int main(int argc, char **argv) {
    const size_t size = 100'000'000;

    // allocate host memory and init data
    double result = 0;
    float *vec = new float[size];
    for (size_t i = 0; i < size; ++i) {
        vec[i] = 1.0;
    }

    // allocate device memory
    float *d_vec;
    double *d_result;
    cudaMalloc((void **)&d_vec, sizeof(float) * size);
    cudaMalloc((void **)&d_result, sizeof(double));
    cudaMemcpy(d_vec, vec, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(double), cudaMemcpyHostToDevice);

    // launch & benchmark the kernel
    const unsigned int n_threads = 256;
    const unsigned int n_blocks = (size + n_threads - 1) / n_threads;
    const system_clock::time_point before = system_clock::now();
    // vector_sum<<<n_blocks, n_threads>>>(d_vec, size, d_result);
    // vector_sum2<<<n_blocks, n_threads>>>(d_vec, size, d_result);
    // vector_sum3<<<n_blocks, n_threads, sizeof(double) * n_threads>>>(d_vec, size, d_result);
    vector_sum4<<<n_blocks, n_threads, sizeof(double) * n_threads>>>(d_vec, size, d_result);
    cudaDeviceSynchronize();
    const system_clock::time_point after = system_clock::now();
    const duration<double> elapsed = after - before;

    // get result & print elapsed time
    std::cout << "time elapsed = " << elapsed.count() << " seconds" << std::endl;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "sum of vector = " << result << std::endl;

    // release memory
    cudaFree(d_vec);
    cudaFree(d_result);
    delete[] vec;

    return 0;
}
