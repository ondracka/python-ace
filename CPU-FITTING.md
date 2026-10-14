# Opt-in Pacemaker CPU fitting

This branch adds `pacemaker-cpu`. Plain `pacemaker` keeps its upstream behavior.
The accelerated command uses the same ACE representation, float64 arithmetic,
fitting labels, global weights, regularization and analytic parameter gradients.
It retains full BFGS, replacing its cubic matrix update with an equivalent
rank-two update. Persistent processes evaluate whole batches and the parent
retains the original global summation order.

## Install a separate environment

On Linux x86-64, with Conda already available:

```bash
git clone --branch pacemaker-cpu --single-branch https://github.com/ondracka/python-ace.git
cd python-ace
bash conda/create-cpu-environment.sh "$HOME/.conda/envs/pacemaker-cpu"
conda activate "$HOME/.conda/envs/pacemaker-cpu"
pacemaker-cpu --cpu-info
```

The script refuses to overwrite an existing environment. It downloads the
dependencies, installs this checkout, fetches legacy TensorPotential at
`1e44b2558356800ae070658c0bb856ff9bf74538`, and compiles both AVX2 and AVX-512
kernel variants. The base Pacemaker source is ICAMS commit
`66c35ea6e5dbed2857d423952b922183206c140e` plus this branch's CPU commits.
No training data or fitted potential is included or downloaded.

The tested numerical stack is Python 3.10, TensorFlow 2.21.0, NumPy 1.26.4,
SciPy 1.13.1 and pandas 2.0.0. These pins are intentional. The BFGS source
guard and TensorPotential source guards reject incompatible implementations.
TensorFlow is an unmodified wheel, not a custom TensorFlow build.

By default, installation uses four compiler jobs, Conda's GCC compiler, MKL
on Intel and OpenBLAS otherwise. Explicit choices are available:

```bash
PACEMAKER_CPU_BUILD_JOBS=4 PACEMAKER_CPU_BLAS=mkl \
  bash conda/create-cpu-environment.sh "$HOME/.conda/envs/pacemaker-cpu"
```

`conda/cpu-environment.yml` is the dependency recipe. The shell entry point
also builds PyACE/maxvolpy and the native kernels, which `conda env create`
alone does not do. This is a source-install environment recipe, not a
published binary Conda channel package. Conda environment-file behavior is
documented in the [Conda environment guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Fit

Use the ordinary input and warm-start arguments:

```bash
PACEMAKER_CPU_THREADS=12 PACEMAKER_CPU_WORKERS=12 \
  pacemaker-cpu -p initial_potential.yaml
```

Keep `backend.evaluator: tensorpot`. A small batch size works well with many
workers, for example:

```yaml
fit:
  optimizer: BFGS
  maxiter: 1000
backend:
  evaluator: tensorpot
  batch_size: 2
```

Other potential, loss and data settings are unchanged. CPU acceleration does
not rewrite the input. Worker count must not exceed the number of batches.
The current parallel adapter supports BFGS, L-BFGS-B and CG in graph mode,
without a train/test split. It rejects unsupported modes explicitly.

Thread counts default to allowed physical cores, capped by
`SLURM_CPUS_PER_TASK` when present, with one worker per core. Set them
explicitly for reproducible benchmarks. `WBC_CPU_THREADS` and
`WBC_CPU_WORKERS` remain accepted aliases. Workers use single-thread BLAS,
while the parent BFGS can use the configured core count. Worker memory and
the full BFGS inverse Hessian are additional to the training dataset.

## CPU selection and portability

At launch, Linux CPU/OS flags are checked for all CPUs in the process affinity.
AVX-512F/DQ/VL selects packed-complex density kernels. Otherwise the AVX2/FMA
implementation is selected. Unsupported forced choices fail before loading
the native kernels. The build uses explicit ISA flags, never `-march=native`,
so the AVX2 variant can be built on an AVX-512 login node safely.

```bash
PACEMAKER_CPU_ISA=avx2 pacemaker-cpu --cpu-info
PACEMAKER_CPU_ISA=avx2 pacemaker-cpu input.yaml
```

Compact workspaces and the verified worker optimizations are enabled by
default on both ISAs. For diagnostics, `WBC_CPU_DENSITY_PACKED=0` selects the
non-packed AVX-512 path. Existing `WBC_CPU_NUMPY_RESULTS`,
`WBC_CPU_READY_RECEIVE`, `WBC_CPU_SET_COEFS_ONCE` and
`WBC_CPU_TENSOR_BATCHES` switches accept `0` to disable each worker change.

Native libraries are cached inside the environment under
`share/pacemaker-cpu/kernels`, keyed by compiler, ISA, TensorFlow build flags
and source hashes. `PACEMAKER_CPU_CACHE_DIR` overrides that path. No binary
is copied from another machine. Rebuild after changing compiler or TensorFlow:

```bash
pacemaker-cpu-build --isa both --jobs 4
```

Intel/Clang are also supported. To use Intel kernels, load the compiler module
and use the same `CXX=icpx` setting for building and fitting. Conda's default
GCC build requires no site compiler modules. AVX-512 compilation on an AVX2
machine is supported, but executing or performance-testing that path requires
actual AVX-512 hardware.

## Numerical smoke check

`tests/cpu/smoke.py` generates a small synthetic alloy dataset and compares
ordinary and accelerated two-step fits. It does not use project data or serve
as physical validation of a fitted model:

```bash
python tests/cpu/smoke.py --directory /tmp/pacemaker-cpu-smoke
```

The compact density kernel also has a synthetic derivative check. It covers
one, two and three embedding channels, empty and partial atom blocks,
repeated product factors, intermediate terms, graph output pruning and
structurally zero coefficient directions. It compares values, first
derivatives and mixed second derivatives with TensorFlow arithmetic and the
uncompressed native kernel:

```bash
python tests/cpu/density.py
python tests/cpu/density.py --threads 3
```

The CPU source patches were extracted from the original development history
without its private datasets, fitted models or project notes. Later compact,
packed and worker improvements are a separate follow-up commit. Upstream
license terms remain in [LICENSE.md](LICENSE.md), and the SciPy notice is
retained in `src/pyace_cpu/fast_bfgs.py`.

The fresh Rosat installation was checked on an AMD Ryzen 9 3900X. Both ISA
variants compiled, and automatic selection chose AVX2. The two-step,
two-worker smoke fit matched ordinary Pacemaker to approximately 1e-16 in
the reported loss/RMSEs and 2e-17 in fitted coefficients. AVX-512 selection
logic and rejection on an unsupported CPU were checked separately. AVX-512
execution was not retested on Rosat, which lacks that instruction set.
