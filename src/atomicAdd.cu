#include <iostream>
#include <vector>
#include <numeric>
#include <assert.h>
#include <cuda_runtime.h>


__device__ float devResult = 0.0f;

__global__ void vectorSum(const float* input, int size){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        atomicAdd(&devResult, input[tid]);
}

__host__ void vectorSum_cpu(float* arr, int size){
    // tính tổng bằng std::accumulate
    std::vector<float> vals;
    for(int i=0; i<size; ++i)
        vals.push_back(arr[i]);
    system("clear");

    float cpu_sum = std::accumulate(vals.cbegin(), vals.cend(), 0.0f);
    std::cout << "Computed CPU value= "<< cpu_sum << std::endl;
    assert(fabs(cpu_sum - float(size) <= 1e-6));
}


int main(int argc, char** argv) {
    const int size = 16'777'217;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    float* input = new float[size];

    // Khởi tạo vector đầu vào
    for (int i = 0; i < size; ++i) {
        input[i] = 1.0f;
    }

    // Cấp phát bộ nhớ trên thiết bị
    float* devInput;
    cudaMalloc((void**)&devInput, sizeof(float) * size);

    // Sao chép dữ liệu từ host sang device
    cudaMemcpy(devInput, input, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(devResult, &devResult, sizeof(float));

    // Gọi kernel
    vectorSum<<<blocksPerGrid, threadsPerBlock>>>(devInput, size);

    // Sao chép kết quả từ device sang host
    cudaMemcpyFromSymbol(&devResult, devResult, sizeof(float));

    vectorSum_cpu(input, size);
    // In kết quả
    std::cout << "Tổng của vector: " << devResult << std::endl;
    assert(fabs(float(size) - devResult) <= 1e-6);

    // Release memory
    cudaFree(devInput);
    delete[] input;

    return 0;
}
