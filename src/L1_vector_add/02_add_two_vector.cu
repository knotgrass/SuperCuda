#include <iostream>
#include <cmath>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void vector_add(float *out, float *a, float *b, const int n) {
    // printf("threadIdx.x = %d\n", threadIdx.x); // index of thread inside the block
    // printf("blockIdx.x = %d\n", blockIdx.x);  // index of block in the grid
    // printf("blockDim.x = %d\n", blockDim.x); // number of threads in the block
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void print_array(int size, float* arr){
    for (int i=0; i < size; i++){
        std::cout << arr[i] << ", ";
    } std::cout << std::endl;
}

int main(int argc, char** argv) {
    int    N = 100;
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    //step 1: Allocate host memory
    a = (float* )malloc(sizeof(float) * N);
    b = (float* )malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    //Step 2: initialized host data
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }


    //Step 3: Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);


    //Step 4: Transfer input data from host memory to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


    //Step 5: Executing kernel
    vector_add <<<1, 1>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();

    //Step 6: Transfer output from device memory to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    //Verification and print result
    for (int i = 0; i < N; i++) {
        assert(std::fabs(out[i] - a[i] - b[i]) <= 1e-6);
        std::cout << out[i] << ", ";
    }   std::cout << '\n' <<std::endl;


    //Step 7: free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    delete[] a;
    delete[] b;
    delete[] out;

    return 0;
}
