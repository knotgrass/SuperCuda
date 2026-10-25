#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <utility>
#include <numeric>
#include <iomanip>
#include <assert.h>


__global__ void multi_element_vector(
    float* out, float* vec1, float* vec2, const int size
) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid <= size) {
        for(int i = tid; i < size; i += stride){
            out[i] = vec1[i] * vec2[i];
        }
    }
    // else
    //     printf("elem = %f\n", out[size-1]);
}

__device__ float dResult;

__global__ void reduceAtomicGlobal(
    const float* __restrict input, int N
) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    /*
     * Since all blocks must have the same number of threads,
     * we may have to launch more threads than there are
     * inputs. Superfluous threads should not try to read
     * from the input (out of bounds access!)
     **/
    if (id < N)
        atomicAdd(&dResult, input[id]);

}


int main(int argc, char** argv) {
    const int N = 2000;

    float *vec1, *vec2;
    float *d_vec1, *d_vec2, *d_product;


    vec1 = (float*)malloc(sizeof(float) * N);
    vec2 = (float*)malloc(sizeof(float) * N);

    for (int i=0; i<N; i++){
        vec1[i] = 2.0f;
        vec2[i] = 0.5f;
    }

    cudaMalloc((void**) &d_vec1, sizeof(float) *N);
    cudaMalloc((void**) &d_vec2, sizeof(float) *N);
    cudaMalloc((void**) &d_product, sizeof(float) *N);

    cudaMemcpy(d_vec1, vec1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2, sizeof(float) * N, cudaMemcpyHostToDevice);
    delete[] vec1, vec2;

    int n_block = (N + 256) / 256;
    int n_thread = 256;
    multi_element_vector<<<n_block, n_thread>>>(d_product, d_vec1, d_vec2, N);
    cudaFree(d_vec1);
    cudaFree(d_vec2);

    /*
        verify multi_element_vector
    float* product;
    product = (float* )malloc(sizeof(float) * N);
    cudaMemcpy(product, d_product, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for (int i=0;i<N;i++){
        assert(std::fabs(product[i] - 1.0) <= 1e-6);
    }
    **/

    reduceAtomicGlobal<<<n_block, n_thread>>>(d_product, N);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&dResult, dResult, sizeof(float));
    std::cout << "dot Product = " << dResult << std::endl;
    assert(fabs(float(N) - dResult) <= 1e-6);
    
    cudaFree(d_product);

    return 0;
}
