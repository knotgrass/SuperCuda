#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


int _ConvertSMVer2Cores(int major, int minor) {
    // Returns the number of streaming processors (CUDA cores) per SM for a given compute capability version
    switch ((major << 4) + minor) {
    case 0x10:
        return 8;
    case 0x11:
    case 0x12:
        return 8;
    case 0x13:
        return 8;
    case 0x20:
        return 32;
    case 0x21:
    case 0x30:
        return 192;
    case 0x35:
    case 0x37:
        return 192;
    case 0x50:
        return 128;
    case 0x52:
    case 0x53:
        return 128;
    case 0x60:
        return 64;
    case 0x61:
    case 0x62:
        return 128;
    case 0x70:
    case 0x72:
    case 0x75:
        return 64;
    case 0x80:
    case 0x86:
        return 64;
    default:
        printf("Unknown device type\n");
        return -1;
    }
}


//get cuda card properties
cudaError_t cardProperties()
{
    cudaError_t cudaStatus = cudaSuccess;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: %s\n", dev, deviceProp.name);
        printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
        printf("Number of SP per SM: %d\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
        printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Total registers: %d\n", deviceProp.regsPerBlock * deviceProp.warpSize);
        printf("Total shared memory: %ld bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
        printf("Total constant memory: %ld bytes\n", deviceProp.totalConstMem);
        printf("Global memory bandwidth (GB/s): %f\n", 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
    }
    return cudaStatus;
}

int main(int argc, char** argv)
{
    cudaError_t cudaStatus = cardProperties();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
