#include "torch_binding.h"

#include <torch/all.h>


TORCH_LIBRARY(gemm, ops) {
    ops.def("naive(Tensor A, Tensor B, Tensor C, float alpha, float beta) -> Tensor");
#if defined(CUDA_KERNEL)
    ops.impl("naive", torch::kCUDA, &naive_cuda);
#endif
}
