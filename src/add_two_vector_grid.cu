#include <iostream>
#include <cmath>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void vector_add(float *out, float *a, float *b, int n) {
    // https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/#exercise-2-adding-more-thread-blocks
    // https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/solutions/vector_add_grid.cu
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid <= n) {
        for(int i = tid; i < n; i += stride){
            out[i] = a[i] + b[i];
        }
    }

}

void print_array(int size, float* arr){
    for (int i=0; i < size; i++){
        std::cout << arr[i] << ", ";
    } std::cout << std::endl;
}

int main(int argc, char** argv) {
    int    N = 10000000;
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    //step 1.1: Allocate host memory
    a = (float* )malloc(sizeof(float) * N);
    b = (float* )malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    //Step 1.2: initialized host data
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }


    //Step 2: Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);


    //Step3: Transfer input data from host memory to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


    //Step4: Executing kernel
    int n_block = 1;
    int n_thread = 1024;
    vector_add <<<n_block, n_thread>>>(d_out, d_a, d_b, N);
    // cudaDeviceSynchronize();

    //Step5: Transfer output from device memory to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    //Verification and print result
    for (int i = 0; i < N; i++) {
        assert(std::fabs(out[i] - a[i] - b[i]) <= 1e-6);
        // std::cout << out[i] << ", ";
    }
    std::cout << out[N-1] << "\n\n";


    //Step6: free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    delete[] a;
    delete[] b;
    delete[] out;

    return 0;
}
