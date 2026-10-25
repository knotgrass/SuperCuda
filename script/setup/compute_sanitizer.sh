#! /bin/bash
# Install compute-sanitizer (memcheck, racecheck, synccheck, initcheck) into the conda env.
# Run it from an activated env: `conda activate cxx && bash script/setup/compute_sanitizer.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi

# cuda-sanitizer-api 13.4.92 is the compute-sanitizer in cuda=13.4.3 (cuda.sh). It has bin/compute-sanitizer.
conda install cuda-sanitizer-api=13.4.92 --override-channels -c pkgs/main --name "$CONDAENV" -y
command -v compute-sanitizer
compute-sanitizer --version
