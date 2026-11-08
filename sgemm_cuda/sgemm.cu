#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>      // c10::cuda::CUDAGuard
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK

#include <limits>

#include "naive.cuh"

/**
 * @brief Torch op for naive_sgemm: out = alpha * A @ B + beta * C.
 *
 * This is the adapter layer: it checks the tensors, picks the device and the current stream,
 * and hands device pointers to launch_naive_sgemm. It does not launch, allocate on its own, or sync.
 *
 * @return a new tensor (M x N). C is not modified: the kernel reads beta * C from a copy and
 *         overwrites that copy in place.
 */
torch::Tensor sgemm_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                         double alpha, double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "sgemm: A, B, C must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device() && A.device() == C.device(),
                "sgemm: A, B, C must be on the same device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32 &&
                    C.scalar_type() == torch::kFloat32,
                "sgemm: A, B, C must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "sgemm: A, B, C must be contiguous (row-major)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "sgemm: A, B, C must be 2-D");

    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "sgemm: A is ", M, "x", K, " but B is ", B.size(0), "x", N);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N,
                "sgemm: C must be ", M, "x", N, ", got ", C.size(0), "x", C.size(1));
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    TORCH_CHECK(M <= kIntMax && N <= kIntMax && K <= kIntMax, "sgemm: M, N, K must fit in int");

    // All CUDA calls below (clone, launch) go to A's device and to the stream PyTorch is using there.
    const c10::cuda::CUDAGuard guard(A.device());

    torch::Tensor out = C.clone();  // the kernel reads beta * C and overwrites it in place
    if (out.numel() == 0) return out;  // a grid of 0 blocks is an invalid launch

    // launch_naive_sgemm returns cudaGetLastError() (launch errors only). C10_CUDA_CHECK turns a
    // failure into a c10::Error, which Python receives as RuntimeError. This replaces
    // C10_CUDA_KERNEL_LAUNCH_CHECK(), which would read an error the launcher already consumed.
    C10_CUDA_CHECK(launch_naive_sgemm(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<float>(alpha), A.data_ptr<float>(), B.data_ptr<float>(),
        static_cast<float>(beta), out.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()));
    return out;
}
