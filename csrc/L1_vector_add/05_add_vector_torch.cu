#include <torch/extension.h>


__global__ void vector_add_kernel(const float* vec1, const float* vec2,
                                  float* output, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = vec1[idx] + vec2[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor vec1, torch::Tensor vec2){
    TORCH_CHECK(vec1.device().is_cuda(), "vec1 must be a CUDA tensor");
    TORCH_CHECK(vec2.device().is_cuda(), "vec2 must be a CUDA tensor");
    TORCH_CHECK(vec1.device() == vec2.device(), "vec1 and vec2 must be on the same CUDA device");

    TORCH_CHECK(vec1.dim() == 1, "vec1 must be a 1D tensor");
    TORCH_CHECK(vec2.dim() == 1, "vec2 must be a 1D tensor");
    TORCH_CHECK(vec1.size(0) == vec2.size(0), "vec1 and vec2 must have the same size");

    if (not vec1.is_contiguous())
        vec1 = vec1.contiguous();

    if (not vec2.is_contiguous())
        vec2 = vec2.contiguous();

    const int64_t size = vec1.size(0);
    torch::Tensor output = torch::zeros_like(vec1);

    // Launch the kernel
    const unsigned int n_threads = 256; // block size
    const unsigned int n_blocks = (size + n_threads - 1) / n_threads; // grid size
    vector_add_kernel<<<n_blocks, n_threads>>>(
        vec1.data_ptr<float>(),
        vec2.data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        size
    );

    return output;
}

PYBIND11_MODULE(vector_add_torch_ext, m, py::mod_gil_not_used()) {
    m.doc() = "Vector addition with CUDA"; // optional module docstring
    m.def("vector_add", &vector_add_cuda, "Add two vectors", py::arg("vec1"), py::arg("vec2"));
}
