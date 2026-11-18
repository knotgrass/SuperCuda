#include "../csrc/gemm/naive.cuh"

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>      // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>         // c10::cuda::CUDAGuard
#include <c10/cuda/CUDAException.h>     // C10_CUDA_CHECK

#include <limits>

namespace {
// All GEMM launchers have this signature,
// so run_gemm takes any launcher as one parameter.
// As a result, the code that checks the input tensors exists only one time, in run_gemm.
using GemmLauncher = cudaError_t (*)(
    int M, int N, int K,
    float alpha, const float *dA, const float *dB,
    float beta, float *dC,
    cudaStream_t stream);

inline torch::Tensor run_gemm(const char *op_name,
                       GemmLauncher launch,
                       const torch::Tensor &A,
                       const torch::Tensor &B,
                       const torch::Tensor &C,
                       double alpha,
                       double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
                "gemm.", op_name, ": A, B, C must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device() && A.device() == C.device(),
                "gemm.", op_name, ": A, B, C must be on the same device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32 && C.scalar_type() == torch::kFloat32,
                "gemm.", op_name, ": A, B, C must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "gemm.", op_name,": A, B, C must be contiguous (row-major)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2,
                "gemm.", op_name, ": A, B, C must be 2-D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    TORCH_CHECK(B.size(0) == K,
                "gemm.", op_name, ": A is ", M, "x", K, " but B is ", B.size(0), "x", N);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N,
                "gemm.", op_name, ": C must be ", M, "x", N, ", got ", C.size(0), "x", C.size(1));
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    TORCH_CHECK(M <= kIntMax && N <= kIntMax && K <= kIntMax,
                "gemm.", op_name, ": M, N, K must fit in int");

    const c10::cuda::CUDAGuard guard (A.device());
    torch::Tensor out = C.clone();

    if (out.numel() == 0) return out;

    C10_CUDA_CHECK(launch(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<float>(alpha), A.data_ptr<float>(), B.data_ptr<float>(),
        static_cast<float>(beta), out.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()
    ));

    return out;
}
}  // namespace


torch::Tensor naive_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    double alpha, double beta
) {
    return run_gemm("naive", launch_naive_gemm, A, B, C, alpha, beta);
}
