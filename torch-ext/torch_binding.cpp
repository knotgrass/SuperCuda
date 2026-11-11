#include <torch/library.h>

#include "registration_dev.h"
#include "torch_binding.h"

// Every op is registered here, in the namespace TORCH_EXTENSION_NAME (= "matmul" in the dev build),
// so Python sees torch.ops.matmul.<op>. One op per kernel, all with the same schema
// (`float` in a schema is double in C++). To add a kernel, copy the def line and the impl line.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("naive(Tensor A, Tensor B, Tensor C, float alpha, float beta) -> Tensor");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("naive", torch::kCUDA, &naive_cuda);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
