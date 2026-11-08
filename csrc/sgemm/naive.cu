// The kernel and its launcher live in sgemm_cuda/naive.cu: the Torch extension builds that file.
// This forwarding file keeps `script/compile_n_run.sh csrc/sgemm` working (it compiles every .cu in
// this directory), so the standalone program in main.cu and the Torch op share one kernel source.
#include "../../sgemm_cuda/naive.cu"
