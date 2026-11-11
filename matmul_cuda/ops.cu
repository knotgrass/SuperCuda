#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>      // c10::cuda::CUDAGuard
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK

#include <limits>

#include "naive.cuh"

// Torch ops of the matmul extension: one op per kernel, all with the SGEMM signature
//     out = alpha * A @ B + beta * C        A: (M, K), B: (K, N), C: (M, N)
// An op is one line: it names its launcher and calls run_sgemm, which does everything the kernels share.
// To add a kernel: include its header, then copy naive_cuda at the bottom of this file.

namespace {

// Every launcher in this directory has this signature (see naive.cuh).
using SgemmLauncher = cudaError_t (*)(int M, int N, int K,
                                      float alpha, const float* dA, const float* dB,
                                      float beta, float* dC, cudaStream_t stream);

/**
 * @brief Adapter between torch::Tensor and a raw launcher.
 *
 * It checks the tensors, picks the device and the current stream, and hands device pointers to
 * `launch`. It does not launch, sync, or copy to the host.
 *
 * @param op     op name, for error messages ("matmul.<op>: ...")
 * @param launch raw launcher of the kernel
 * @return a new (M, N) tensor. C is not modified: the kernel reads beta * C from a copy and
 *         overwrites that copy in place.
 */
torch::Tensor run_sgemm(const char* op, SgemmLauncher launch,
                        const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                        double alpha, double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "matmul.", op, ": A, B, C must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device() && A.device() == C.device(),
                "matmul.", op, ": A, B, C must be on the same device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32 &&
                    C.scalar_type() == torch::kFloat32,
                "matmul.", op, ": A, B, C must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "matmul.", op, ": A, B, C must be contiguous (row-major)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "matmul.", op, ": A, B, C must be 2-D");

    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "matmul.", op, ": A is ", M, "x", K, " but B is ", B.size(0), "x", N);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N,
                "matmul.", op, ": C must be ", M, "x", N, ", got ", C.size(0), "x", C.size(1));
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    TORCH_CHECK(M <= kIntMax && N <= kIntMax && K <= kIntMax, "matmul.", op, ": M, N, K must fit in int");

    // All CUDA calls below (clone, launch) go to A's device and to the stream PyTorch is using there.
    const c10::cuda::CUDAGuard guard(A.device());

    torch::Tensor out = C.clone();     // the kernel reads beta * C and overwrites it in place
    if (out.numel() == 0) return out;  // a grid of 0 blocks is an invalid launch

    // A launcher returns cudaGetLastError() (launch errors only). C10_CUDA_CHECK turns a failure
    // into a c10::Error, which Python receives as RuntimeError. This replaces
    // C10_CUDA_KERNEL_LAUNCH_CHECK(), which would read an error the launcher already consumed.
    C10_CUDA_CHECK(launch(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<float>(alpha), A.data_ptr<float>(), B.data_ptr<float>(),
        static_cast<float>(beta), out.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()));
    return out;
}

}  // namespace

// matmul.naive: one thread per element of C (naive.cu).
torch::Tensor naive_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                         double alpha, double beta) {
    return run_sgemm("naive", launch_naive_sgemm, A, B, C, alpha, beta);
}
