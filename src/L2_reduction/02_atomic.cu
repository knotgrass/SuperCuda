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
 * Compute block results in fast, shared memory, update global at end
 */
__global__ void vector_sum2(const float *d_vec, const size_t n, double *d_result) {
    const size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // allocate shared memory
    __shared__ double s_result;

    // init shared memory data in thread 0
    if (threadIdx.x == 0) {
        s_result = 0;
    }
    __syncthreads(); // make sure all threads have init shared memory data

    // compute block result, add result to fast shared memory
    if (global_tid < n) {
        atomicAdd(&s_result, d_vec[global_tid]);
    }
    __syncthreads(); // make sure all threads have updated shared memory data

    // transfer shared memory data to global memory
    if (threadIdx.x == 0) {
        atomicAdd(d_result, s_result);
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
    vector_sum<<<n_blocks, n_threads>>>(d_vec, size, d_result);
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
