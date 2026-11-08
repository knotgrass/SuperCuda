#include <torch/library.h>

#include "registration_dev.h"
#include "torch_binding.h"

// Every op is registered here, in the namespace TORCH_EXTENSION_NAME (= "supercuda" in the dev build),
// so Python sees torch.ops.supercuda.<op>.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // The schema uses `float` for double; shapes and dtypes are checked in sgemm_cuda.
    ops.def("sgemm(Tensor A, Tensor B, Tensor C, float alpha, float beta) -> Tensor");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("sgemm", torch::kCUDA, &sgemm_cuda);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
