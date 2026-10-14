#!/usr/bin/env bash
# Build a separate environment. Existing prefixes and source trees are preserved.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${1:-"$HOME/.conda/envs/pacemaker-cpu"}
JOBS=${PACEMAKER_CPU_BUILD_JOBS:-4}
BLAS=${PACEMAKER_CPU_BLAS:-auto}
TP_REV=1e44b2558356800ae070658c0bb856ff9bf74538
if [[ -e "$PREFIX" ]]; then
    echo "Refusing to overwrite existing prefix: $PREFIX" >&2
    exit 1
fi
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'PACEMAKER_CPU_BUILD_JOBS must be a positive integer' >&2
    exit 1
fi
if [[ "$BLAS" == auto ]]; then
    if grep -q GenuineIntel /proc/cpuinfo; then BLAS=mkl; else BLAS=openblas; fi
fi
if [[ "$BLAS" != mkl && "$BLAS" != openblas ]]; then
    echo 'PACEMAKER_CPU_BLAS must be auto, mkl or openblas' >&2
    exit 1
fi
CONDA=${CONDA_EXE:-conda}
"$CONDA" env create --yes --quiet --prefix "$PREFIX" --file "$ROOT/conda/cpu-environment.yml"
if [[ "$BLAS" == mkl ]]; then
    "$CONDA" install --yes --prefix "$PREFIX" --override-channels -c conda-forge \
        'mkl=2025.3.1' 'libblas=*=*mkl' 'libcblas=*=*mkl' 'liblapack=*=*mkl'
fi
# Activation also selects the environment's compiler and runtime libraries.
eval "$("$CONDA" shell.bash hook)"
conda activate "$PREFIX"
export PYTHONNOUSERSITE=1
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export CXXFLAGS="${CXXFLAGS:-} -include cstdint"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
SRC="$CONDA_PREFIX/share/pacemaker-cpu/sources"
mkdir -p "$SRC"
git init -q "$SRC/TensorPotential"
git -C "$SRC/TensorPotential" fetch --depth 1 https://github.com/ICAMS/TensorPotential.git "$TP_REV"
git -C "$SRC/TensorPotential" checkout -q --detach FETCH_HEAD
test "$(git -C "$SRC/TensorPotential" rev-parse HEAD)" = "$TP_REV"
python -m pip install --no-build-isolation --no-deps \
    "$ROOT/lib/maxvolpy" "$ROOT" "$SRC/TensorPotential"
pacemaker-cpu-build --isa both --jobs "$JOBS"
python -m pip check
pacemaker-cpu --cpu-info
printf '\nReady. Activate with: conda activate %q\n' "$CONDA_PREFIX"
echo 'Run pacemaker-cpu input.yaml, or pacemaker-cpu -p initial_potential.yaml'
