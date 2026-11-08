#pragma once
#include <torch/torch.h>

// Declarations of every Torch-facing function (one per op). Definitions live in <name>_cuda/.

// sgemm_cuda/sgemm.cu
// out = alpha * A @ B + beta * C. Returns a new tensor; C is not modified.
// A: (M, K), B: (K, N), C: (M, N); float32, contiguous, same CUDA device.
torch::Tensor sgemm_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                         double alpha, double beta);
