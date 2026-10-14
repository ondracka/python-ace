"""Compile CPU kernels for both ISAs without executing unsupported instructions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib
import json
import os
import subprocess
import sys

from .runtime import cache_root, select_isa


def compile_one(spec):
    source, compiler, flags, identity, library = spec
    library.parent.mkdir(parents=True, exist_ok=True)
    with library.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if library.is_file():
            print(f'Already built: {library.name}', flush=True)
            return
        temporary = library.with_suffix(f'.{os.getpid()}.tmp.so')
        command = [compiler, str(source), '-o', str(temporary), *flags]
        print(f'Building {library.name}', flush=True)
        subprocess.run(command, check=True)
        temporary.replace(library)
        library.with_suffix('.json').write_text(json.dumps(identity, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isa', choices=('auto', 'avx2', 'avx512', 'both'), default='both')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['PACEMAKER_CPU_BUILD_ONLY'] = '1'
    targets = ('avx2', 'avx512') if args.isa == 'both' else (select_isa(args.isa, cross_compile=True),)
    specs = []
    for isa in targets:
        os.environ['PACEMAKER_CPU_ISA'] = isa
        for name in ('basis', 'moments', 'angular', 'radial', 'all_moments', 'density'):
            module = importlib.import_module(f'pyace_cpu.native_{name}')
            variants = ('0', '1') if name == 'density' and isa == 'avx512' else ('0',)
            for packed in variants:
                os.environ['WBC_CPU_DENSITY_PACKED'] = packed
                specs.append((module.SOURCE, *module.build_spec()))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(compile_one, specs))
    print(f'Built {len(specs)} kernel variants in {cache_root()}', flush=True)


if __name__ == '__main__':
    main()
