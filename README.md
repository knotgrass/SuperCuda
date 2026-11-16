# Parallel

CUDA kernels for learning. The matrix-multiply (SGEMM) series is a Torch extension, called from Python:

```python
import torch, gemm

A = torch.randn(1000, 777, device="cuda")
B = torch.randn(777, 1500, device="cuda")
C = torch.randn(1000, 1500, device="cuda")
out = gemm.naive(A, B, C, alpha=0.5, beta=-0.1)   # alpha * A @ B + beta * C
```

One function per kernel: `gemm.naive`, then `gemm.coalescing`, ... All have the same signature.
`gemm.KERNELS` lists them; the tests and the benchmark loop over it.
The layout follows Hugging Face [kernel-builder](https://huggingface.co/docs/kernels/builder/writing-kernels),
with one difference: the kernel sources are in `csrc/gemm/`, not in a `gemm_cuda/` directory.

## Layout

```text
SuperCuda/
├── build.toml            # kernel-builder config; also the source list for the JIT build
├── include/common.cuh    # CEIL_DIV, CUDA_CHECK
├── csrc/                 # all CUDA sources; one directory per program
│   ├── gemm/
│   │   ├── naive.cu, naive.cuh   # one kernel + its raw launcher (device pointers + stream, no Torch)
│   │   └── main.cu               # standalone program that calls the launcher
│   └── L1_vector_add/, L2_reduction/, *.cu   # older lessons, each file has a main()
├── torch-ext/
│   ├── ops.cu            # the Torch ops: one per kernel, each calls its launcher
│   ├── torch_binding.h   # declarations of all op functions
│   ├── torch_binding.cpp # TORCH_LIBRARY registration of all ops
│   └── gemm/             # Python package: __init__.py (wrappers, KERNELS), _ops.py (dev JIT loader)
├── tests/                # pytest: every kernel vs. PyTorch
├── benchmarks/           # benchmark_matmul.py (do_bench vs. cuBLAS), profile_matmul.py (driver for ncu, nsys)
├── script/               # build, compdb, sanitizer, ncu, nsys, roofline, compile_n_run; setup/ = env install
├── third_party/          # third-party source (CUTLASS submodule)
└── tutorial/             # slides (PDF) and notes/
```

How a Python call reaches a kernel:

```mermaid
flowchart LR
    T[build.toml] --> O["torch-ext/gemm/_ops.py<br/>cpp_extension.load"]
    O --> S["build/release (or debug)/gemm.so"]
    S --> R["torch_binding.cpp<br/>TORCH_LIBRARY"]
    R --> N[torch.ops.gemm.naive]
    N --> W[gemm.naive]
    W --> X[tests/, benchmarks/]
    N --> P["torch-ext/ops.cu: naive_cuda"]
    P --> L["csrc/gemm/naive.cu: launch_naive_sgemm"]
    L --> K["naive_sgemm (GPU)"]
```

## Setup

Use the conda env `cxx`. Activate it before you start VS Code or run a script:
the scripts and `.vscode/` use `$CONDA_PREFIX`.

To install or upgrade the toolchain (CUDA 13.4.3 + torch 2.14.1 for cu132):

```bash
conda activate cxx
conda install --override-channels -c pkgs/main "cuda=13.4.3" -y
pip install "torch==2.14.1" torchvision --index-url https://download.pytorch.org/whl/cu132
conda env config vars set CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="12.0"   # 8.6 on A10
conda deactivate && conda activate cxx
nvcc --version | tail -2                                                 # 13.4
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.14.1 13.2
```

- `CUDA_HOME` must be the env: `~/.zshrc` sets it to `/usr/local/cuda` (a different nvcc).
- `TORCH_CUDA_ARCH_LIST` limits the JIT build to the local GPU. Without it, the conda nvcc
  activation script sets 9 architectures and every build is slow.
- nvcc 13.4 with torch built for 13.2 is a minor-version mismatch: `cpp_extension` prints a warning only.
- The full install scripts are in `script/setup/`.

## Workflow

All commands run from the repo root, in the `cxx` env.

| Task | Command |
|---|---|
| Build (release: `-O3 -lineinfo`) | `script/build.sh release` |
| Build (debug: `-O0 -g -G`) | `script/build.sh debug` |
| Test | `python -m pytest tests -q` |
| Benchmark (all kernels vs. cuBLAS) | `python benchmarks/benchmark_matmul.py` |
| Memory errors | `script/sanitizer.sh memcheck python -m pytest tests/test_matmul.py -q -x` |
| One kernel, all metrics | `script/ncu.sh naive_sgemm naive_1024 python benchmarks/profile_matmul.py --kernel naive` |
| Roofline of that report | `python script/roofline.py reports/naive_1024.ncu-repz --png` |
| Timeline | `script/nsys.sh naive_1024 python benchmarks/profile_matmul.py --kernel naive` |
| IntelliSense database | `script/compdb.sh` (after a release build) |
| Standalone program | `script/compile_n_run.sh csrc/helloworld.cu`, `script/compile_n_run.sh csrc/gemm` |

- The first `import gemm` compiles the extension into `build/<mode>/gemm.so`. Later imports rebuild only changed files.
  `SUPERCUDA_BUILD=debug` selects the debug build.
- Reports go to `reports/`. Open them with `ncu-ui` / `nsys-ui`.
- `ncu` fails with `ERR_NVGPUCTRPERM` until GPU counters are open to normal users: see the comment in `script/ncu.sh`.
- Profile only the release build. `-G` turns off device optimizations.

### Debug a kernel in VS Code

1. Set a breakpoint in a kernel file, for example `csrc/gemm/naive.cu`.
2. Open the test file that calls the kernel: `tests/test_matmul.py`.
3. Start **cuda-gdb: pytest current file**. It builds the debug extension first (`ext: build debug`).

For Python-side bugs, use **Python: pytest current file**.
For a standalone program, open its `.cu` file and start **C/C++: nvcc build and debug active file**.

### Add a kernel

See the checklist in [CLAUDE.md](CLAUDE.md#add-a-new-kernel).

## Resources

### nvidia cuda-c-programming-guide
- [ ] [nVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [ ] [ An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

### article
- [ ] [A Case Study in CUDA Kernel Fusion](https://arxiv.org/abs/2312.11918)

### blog
- [ ] [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
### cuda code sample
- [x] [tutorial-viblo](https://viblo.asia/s/lap-trinh-song-song-0gdJzv6kJz5)
- [x] [cuda-tutorial](https://cuda-tutorial.readthedocs.io/en/latest/)
- [x] [CoffeeBeforeArch](https://github.com/CoffeeBeforeArch/cuda_programming)
- [ ] [cuda-code-sample](https://cuda-tutorial.github.io/)
- [x] [vietnamen-wordpress](https://vietnamen.wordpress.com/2009/10/03/ph%e1%ba%a7n-2-cuda-programming-model/)
- [x] [supercomputingblog](http://supercomputingblog.com/category/cuda/)
- [ ] [https://modal.com/gpu-glossary](https://modal.com/gpu-glossary/device-hardware/special-function-unit)

### nvidia tool usage
- [ ] [NgoHuuGiaHuy](https://viblo.asia/s/cach-su-dung-nvidia-tools-PwlVmRAw45Z)
- [ ] [debug in Linux with VScode](https://viblo.asia/p/debug-cuda-code-tren-visual-studio-code-cung-nvidia-nsight-visual-studio-code-edition-obA46Ow9JKv)

### Technique
- [ ] tiled [nichijou](https://nichijou.co/cuda7-tiling/) , [blog/tiled-matrix-multiplication](https://penny-xu.github.io/blog/tiled-matrix-multiplication) , [tiling in cuda](https://www.google.com/search?q=tiling+in+cuda&oq=tiling+in+cuda&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIKCAEQABiGAxiKBTIKCAIQABiGAxiKBdIBCTIxNjUzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [ ] tuning
- [ ] fused kernel
- [ ] [Nvidia® Tensor Memory Accelerator(TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)
