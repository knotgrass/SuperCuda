# CLAUDE.md

CUDA learning repo. Kernels are Torch ops in the Hugging Face kernel-builder layout.
The owner writes the kernel bodies (SGEMM series next: naive, then Strassen, Winograd, tensor cores).
Do not write a kernel body unless asked; write the launcher, test, and benchmark around it.

## Layout

- `build.toml`: kernel-builder config (edition 6). It is the **only** source list: `torch-ext/supercuda/_ops.py` reads it.
- `<name>_cuda/*.cu`: kernel + launcher. `common/common.cuh`: `CEIL_DIV`, `CUDA_CHECK`, shared `__device__` helpers.
- `torch-ext/torch_binding.{h,cpp}`: launcher declarations and op registration. `registration_dev.h` stands in for the
  `registration.h` that kernel-builder generates.
- `torch-ext/supercuda/`: `__init__.py` (one thin wrapper per op), `_ops.py` (dev JIT loader).
- `tests/`, `benchmarks/`: pytest vs. PyTorch; `triton.testing.do_bench`.
- `csrc/`: old single-file programs with `main()`. Keep their logic unchanged.
- `script/`: build/profile wrappers; `script/setup/`: env install scripts (the user runs these, not Claude).

## Environment

- Conda env `cxx`. Run `conda activate cxx` first: scripts and `.vscode/` use `$CONDA_PREFIX`.
- `CUDA_HOME` must point to the env (`~/.zshrc` sets `/usr/local/cuda`, a different nvcc) and
  `TORCH_CUDA_ARCH_LIST=12.0` (8.6 on the A10). `script/setup/cuda.sh` sets both as conda env vars.
  If they are not set, pass them per command: `CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=12.0 script/build.sh`.
- C++20 everywhere: host, nvcc, IntelliSense.
- Do not install, upgrade, or remove packages. Write the command and ask the user to run it.
- Never `git add -A`: `libs/` holds a 1.4 GB local CUTLASS install (gitignored, but stage paths explicitly).

## Launcher conventions (every `.cu` in `*_cuda/`)

- Kernel keeps its lesson name; the launcher is `<op>_cuda` and takes/returns `torch::Tensor`.
- Check inputs with `TORCH_CHECK`: CUDA device, same device, dtype, `is_contiguous()`, shapes.
- `const c10::cuda::CUDAGuard guard(x.device());`
- Return early when `numel() == 0` (a grid of 0 blocks is an invalid launch).
- Launch on `at::cuda::getCurrentCUDAStream()`; call `C10_CUDA_KERNEL_LAUNCH_CHECK();` after each launch.
- Use `torch::zeros` for an output that the kernel reads or accumulates into (atomics, `beta * C`).
- No `PYBIND11_MODULE`. Register ops only in `torch-ext/torch_binding.cpp`. Op namespace: `supercuda`.

## Add a new kernel

1. Create `<name>_cuda/<name>.cu` with kernel + launcher.
2. Add the source to `build.toml` (a new `[kernel.<name>]` section, or the `src` list of an existing one).
3. Declare the launcher in `torch-ext/torch_binding.h`.
4. Register the op in `torch-ext/torch_binding.cpp` (`ops.def` schema + `ops.impl(..., torch::kCUDA, ...)`).
5. Add a Python wrapper in `torch-ext/supercuda/__init__.py` and add it to `__all__`.
6. Add a test in `tests/` (PyTorch reference, sizes that are not multiples of the block size) and a benchmark in `benchmarks/`.

Then run `script/build.sh release && python -m pytest tests -q`.

## Rules for performance work

- Never profile a `-G` (debug) build. Profile `script/build.sh release` (`-O3 -lineinfo`).
- Keep TF32 off in GEMM tests and benchmarks: `torch.backends.cuda.matmul.allow_tf32 = False`.
- Run `script/sanitizer.sh memcheck ...` on new kernels; it sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.
- `ncu` needs GPU counter access (`ERR_NVGPUCTRPERM`): see `script/ncu.sh`.
