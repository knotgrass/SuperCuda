#include <iostream>
#include <cuda_runtime.h>


__global__ void vector_sum(const float* d_vec, const size_t n, double* d_result){
    const size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid < n){
        atomicAdd(d_result, d_vec[global_tid]);
    }
}

int main(int argc, char** argv){
    const size_t size = 100'000'000;

    // allocate host memory and init data
    double result = 0;
    float *vec = new float[size];
    for (size_t i = 0; i < size; ++i){
        vec[i] = 1.0;
    }

    // allocate device memory
    float *d_vec;
    double *d_result;
    cudaMalloc((void**)&d_vec, sizeof(float) * size);
    cudaMalloc((void**)&d_result, sizeof(double));
    cudaMemcpy(d_vec, vec, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(double), cudaMemcpyHostToDevice);

    // launch the kernel
    const unsigned int n_threads = 256;
    const unsigned int n_blocks = (size + n_threads - 1) / n_threads;
    vector_sum<<<n_blocks, n_threads>>>(d_vec, size, d_result);
    // cudaDeviceSynchronize();

    // get result
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "total = " << result << std::endl;

    // release memory
    cudaFree(d_vec);
    cudaFree(d_result);
    delete[] vec;

    return 0;
}
