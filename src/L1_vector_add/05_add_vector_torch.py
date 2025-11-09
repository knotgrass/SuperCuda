import os, sys

os.chdir(os.path.dirname(__file__))

import torch
import torch.utils.cpp_extension


vector_add_torch_ext = torch.utils.cpp_extension.load(
    "vector_add_torch_ext",
    sources=["05_add_vector_torch.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# Example usage
input1 = torch.randn(1000, device="cuda")
input2 = torch.randn(1000, device="cuda")

output = vector_add_torch_ext.vector_add(input1, input2)

print(output)
torch.testing.assert_close(output, input1 + input2)
