# CLAUDE.md

CUDA learning repo. Kernels are Torch ops in the Hugging Face kernel-builder layout.
The owner writes the kernel bodies (SGEMM series next: naive, then Strassen, Winograd, tensor cores).
Do not write a kernel body unless asked; write the launcher, test, and benchmark around it.

## Layout

- `build.toml`: kernel-builder config (edition 6). It is the **only** source list: `torch-ext/matmul/_ops.py` reads it.
  Extension name: `matmul` (`import matmul`, `torch.ops.matmul.*`, `build/<mode>/matmul.so`).
- `matmul_cuda/<kernel>.cu` + `<kernel>.cuh`: one kernel and its raw launcher (no Torch headers).
  `matmul_cuda/ops.cu`: the Torch ops, one per kernel. `include/common.cuh`: `CEIL_DIV`, `CUDA_CHECK`.
- `torch-ext/torch_binding.{h,cpp}`: op declarations and op registration. `registration_dev.h` stands in for the
  `registration.h` that kernel-builder generates.
- `torch-ext/matmul/`: `__init__.py` (one thin wrapper per kernel + the `KERNELS` dict), `_ops.py` (dev JIT loader).
- `tests/`, `benchmarks/`: both loop over `matmul.KERNELS`. pytest vs. PyTorch; `triton.testing.do_bench` vs. cuBLAS;
  `benchmarks/profile_matmul.py` is the launch-only driver for ncu and nsys.
  `tests/conftest.py`: autouse fixture (TF32 off, `torch.cuda.synchronize()` after every test).
  Daily loop: `python -m pytest tests -q -m "not slow"`.
- `csrc/`: standalone programs with `main()`. Keep their logic unchanged. `csrc/sgemm/naive.{cu,cuh}` only include
  `matmul_cuda/naive.{cu,cuh}`, so `main.cu` and the Torch op share one kernel source.
- `script/`: build/profile wrappers, `roofline.py`; `script/setup/`: env install scripts (the user runs these, not Claude).

## Environment

- Conda env `cxx`. Run `conda activate cxx` first: scripts and `.vscode/` use `$CONDA_PREFIX`.
- `CUDA_HOME` must point to the env (`~/.zshrc` sets `/usr/local/cuda`, a different nvcc) and
  `TORCH_CUDA_ARCH_LIST=12.0` (8.6 on the A10). `script/setup/cuda.sh` sets both as conda env vars.
  If they are not set, pass them per command: `CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=12.0 script/build.sh`.
- `SUPERCUDA_BUILD=debug` selects the debug build for any Python command (default: release).
- C++20 everywhere: host, nvcc, IntelliSense.
- Do not install, upgrade, or remove packages. Write the command and ask the user to run it.
- Never `git add -A`: `libs/` holds a 1.4 GB local CUTLASS install (gitignored, but stage paths explicitly).

## Kernel, launcher, op conventions (`matmul_cuda/`)

Three layers. Every kernel computes `out = alpha * A @ B + beta * C` (row-major, float32).

- **Kernel** `<kernel>_sgemm` (`__global__`): keeps its lesson name. The owner writes the body.
- **Raw launcher** `launch_<kernel>_sgemm(M, N, K, alpha, dA, dB, beta, dC, stream)` in the same `.cu`, declared in
  `<kernel>.cuh` with the same signature as `naive.cuh`:
  - Takes device pointers and a stream. Computes grid/block, launches, returns `cudaGetLastError()`.
  - Returns `cudaSuccess` early when `M == 0 || N == 0` (a grid of 0 blocks is an invalid launch).
  - Does not allocate, copy to or from the host, synchronize, or include Torch headers (`csrc/` programs compile it).
- **Torch op** `<kernel>_cuda` in `ops.cu`: one line, `return run_sgemm("<kernel>", launch_<kernel>_sgemm, A, B, C, alpha, beta);`.
  `run_sgemm` is shared: `TORCH_CHECK` (CUDA device, same device, float32, `is_contiguous()`, shapes),
  `c10::cuda::CUDAGuard`, `out = C.clone()` (the kernel reads `beta * C`, so not `torch::zeros`), early return when
  `numel() == 0`, launch on `at::cuda::getCurrentCUDAStream()`, `C10_CUDA_CHECK(launch(...))`.
- No `PYBIND11_MODULE`. Register ops only in `torch-ext/torch_binding.cpp`. Op namespace: `matmul`.

## Add a new kernel

1. Create `matmul_cuda/<kernel>.cu` and `<kernel>.cuh`: the kernel and `launch_<kernel>_sgemm`.
2. Add both files to the `src` list of `[kernel.matmul]` in `build.toml`.
3. In `matmul_cuda/ops.cu`: include the header and add `<kernel>_cuda` (copy `naive_cuda`).
4. Declare `<kernel>_cuda` in `torch-ext/torch_binding.h`; register it in `torch-ext/torch_binding.cpp`
   (`ops.def` schema + `ops.impl(..., torch::kCUDA, ...)`, copy the two `naive` lines).
5. In `torch-ext/matmul/__init__.py`: add the wrapper `<kernel>` and its entry in `KERNELS`.
6. Tests and benchmarks loop over `matmul.KERNELS`: no change is needed. Add a test only for behaviour that is
   specific to the kernel.

Then run `script/build.sh release && python -m pytest tests -q`.

## Rules for performance work

- Never profile a `-G` (debug) build. Profile `script/build.sh release` (`-O3 -lineinfo`).
- Keep TF32 off in GEMM tests and benchmarks: `torch.backends.cuda.matmul.allow_tf32 = False`.
- Run `script/sanitizer.sh memcheck ...` on new kernels; it sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.
- `ncu` needs GPU counter access (`ERR_NVGPUCTRPERM`): see `script/ncu.sh`.
