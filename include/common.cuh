#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err_ = (call);                                    \
        if (err_ != cudaSuccess) {                                    \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",    \
                         __FILE__, __LINE__,                          \
                         cudaGetErrorString(err_), cudaGetErrorName(err_)); \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)
