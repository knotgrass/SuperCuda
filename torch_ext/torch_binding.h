#pragma once
#include <torch/all.h>


torch::Tensor naive_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    double alpha, double beta
);
