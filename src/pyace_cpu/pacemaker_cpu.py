#!/usr/bin/env python3
"""Opt-in CPU Pacemaker launcher, keeping the installed upstream code untouched.

Use the ace-cpu environment. Normal Pacemaker arguments are forwarded unchanged.
The selected numerical rewrites apply only to this process, not to other jobs.
Build the native operations with pacemaker-cpu-build after installation or
changing the compiler/TensorFlow environment.
"""
import os
from pathlib import Path
import runpy
import sys
from .runtime import configure, information

def main():
    isa = configure()
    if sys.argv[1:] == ['--cpu-info']:
        import json
        print(json.dumps(information(), indent=2))
        return
    workers = int(os.environ.get('WBC_CPU_WORKERS', '1'))
    cores = int(os.environ.get('WBC_CPU_THREADS', '12' if workers > 1 else '8'))
    if workers < 1 or cores < 1 or cores % workers:
        raise ValueError('Positive WBC_CPU_THREADS must be divisible by WBC_CPU_WORKERS')
    compact = os.environ.get('WBC_CPU_COMPACT_DENSITY', '0')
    if compact not in ('0', '1'):
        raise ValueError('WBC_CPU_COMPACT_DENSITY must be 0 or 1')
    packed = os.environ.get('WBC_CPU_DENSITY_PACKED', '0')
    if packed not in ('0', '1'):
        raise ValueError('WBC_CPU_DENSITY_PACKED must be 0 or 1')
    if packed == '1' and compact != '1':
        raise ValueError('Packed AVX-512 requires WBC_CPU_COMPACT_DENSITY=1')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(cores // workers))
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(cores))
    os.environ.setdefault('MKL_NUM_THREADS', str(cores))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['WBC_CPU_EXPERIMENTS'] = ('native-bbasis-block4,reordered-shared-radial,'
                                       'native-angular-split,prune-moment-outputs,'
                                       'reuse-radial-derivatives,native-density,'
                                       'prune-density-outputs,radial-trig-recurrence,'
                                       'all-bond-moments')
    if compact == '1':
        os.environ['WBC_CPU_EXPERIMENTS'] += ',compact-density-workspace'

    from .fast_bfgs import enable
    enable()
    from .cpu_experiments import apply
    apply(os.environ['WBC_CPU_EXPERIMENTS'])
    if workers > 1:
        from .parallel_fit import enable as enable_workers
        enable_workers(workers, cores)

    pacemaker = Path(sys.executable).with_name('pacemaker')
    if not pacemaker.is_file():
        raise FileNotFoundError(f'No Pacemaker in this environment: {pacemaker}')
    print(f'CPU ISA: {isa}. CPU rewrites enabled: native SBessel and trigonometric recurrences with per-batch derivative '
          'reuse, shared identical radial primitives, atom-blocked float64 ACE '
          'densities with cross-rank product sharing and unused density work pruned, '
          'all-bond primitive moments and rank-one sums without bond-type '
          'geometry partitions, with unused derivative outputs pruned '
          'and separate real/imaginary angular functions with exact '
          'first/second derivatives, and rank-two full BFGS. '
          f'Batch workers: {workers}. Model, loss and trainable parameters '
          'are unchanged.', flush=True)
    if compact == '1':
        print('Compact density derivative workspaces enabled, terminal products '
              'are consumed without full-node scratch arrays.', flush=True)
    if packed == '1':
        print('AVX-512 packed-complex densities enabled, four atoms per block '
              'with unchanged float64 arithmetic.', flush=True)
    sys.argv[0] = str(pacemaker)
    # Keep this guarded, lightweight launcher as multiprocessing's main module.
    # Loading the CLI as __main__ would make spawned workers import its heavy
    # dependencies before their thread limits are configured.
    program = runpy.run_path(str(pacemaker), run_name='_wbc_pacemaker_cli')
    program['main'](sys.argv[1:])


if __name__ == '__main__':
    main()
