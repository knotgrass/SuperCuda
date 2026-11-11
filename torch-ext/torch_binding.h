#pragma once
#include <torch/torch.h>

// Declarations of every Torch op function. Definitions: matmul_cuda/ops.cu.
//
// All ops share the SGEMM signature:
//     out = alpha * A @ B + beta * C. Returns a new tensor; C is not modified.
//     A: (M, K), B: (K, N), C: (M, N); float32, contiguous, same CUDA device.
// To add a kernel, copy the declaration below with the new name.

torch::Tensor naive_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                         double alpha, double beta);
