"""Lightweight Linux CPU detection, before NumPy or TensorFlow is imported."""
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

AVX2 = frozenset(('avx2', 'fma'))
AVX512 = AVX2 | frozenset(('avx512f', 'avx512dq', 'avx512vl'))


def cpu_flags():
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('The native CPU fitter currently supports Linux x86-64 only')
    # Linux exposes AVX flags only when the kernel enables the corresponding
    # extended register state. Intersect flags across the allowed processors.
    allowed = os.sched_getaffinity(0)
    flags = []
    for block in Path('/proc/cpuinfo').read_text().split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if 'processor' in fields and int(fields['processor']) in allowed:
            flags.append(set(fields.get('flags', '').split()))
    if not flags:
        raise RuntimeError('Could not determine CPU flags for this process affinity')
    return set.intersection(*flags)


def select_isa(requested=None, *, flags=None, cross_compile=False):
    requested = requested or os.environ.get('PACEMAKER_CPU_ISA', 'auto')
    if requested not in ('auto', 'avx2', 'avx512'):
        raise ValueError('PACEMAKER_CPU_ISA must be auto, avx2 or avx512')
    flags = cpu_flags() if flags is None else set(flags)
    if requested == 'auto':
        requested = 'avx512' if AVX512 <= flags else 'avx2'
    needed = AVX512 if requested == 'avx512' else AVX2
    if not cross_compile and not needed <= flags:
        raise RuntimeError(f'{requested} is unavailable on this CPU/OS, missing {sorted(needed-flags)}')
    return requested


def physical_cpus():
    result, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((root/'physical_package_id').read_text().strip(),
               (root/'core_id').read_text().strip())
        if key not in seen:
            seen.add(key)
            result.append(cpu)
    return result


def cache_root():
    return Path(os.environ.get('PACEMAKER_CPU_CACHE_DIR',
                str(Path(sys.prefix)/'share/pacemaker-cpu/kernels'))).expanduser()


def configure():
    isa = select_isa()  # Reject unsupported forced ISA before importing native code.
    os.environ['PACEMAKER_CPU_ISA'] = isa
    available = physical_cpus()
    default_threads = min(len(available), int(os.environ.get('SLURM_CPUS_PER_TASK', len(available))))
    for name in ('THREADS', 'WORKERS'):
        new, old = f'PACEMAKER_CPU_{name}', f'WBC_CPU_{name}'
        if new in os.environ:
            if old in os.environ and os.environ[new] != os.environ[old]:
                raise ValueError(f'{new} and {old} disagree')
            os.environ[old] = os.environ[new]
    cores = int(os.environ.setdefault('WBC_CPU_THREADS', str(default_threads)))
    workers = int(os.environ.setdefault('WBC_CPU_WORKERS', str(cores)))
    if not 0 < workers <= cores <= len(available) or cores % workers:
        raise ValueError(f'Threads must divide into workers and fit within {len(available)} allowed physical cores')
    compact = os.environ.setdefault('WBC_CPU_COMPACT_DENSITY', '1')
    packed = os.environ.setdefault('WBC_CPU_DENSITY_PACKED',
                                   '1' if isa == 'avx512' and compact == '1' else '0')
    if packed == '1' and isa != 'avx512':
        raise ValueError('Packed density requires PACEMAKER_CPU_ISA=avx512 and a supported CPU')
    # These preserve the original global summation order and numerical model.
    for name in ('NUMPY_RESULTS', 'READY_RECEIVE', 'SET_COEFS_ONCE', 'TENSOR_BATCHES'):
        os.environ.setdefault(f'WBC_CPU_{name}', '1')
    return isa


def information():
    versions = {}
    for name in ('pyace', 'tensorpotential', 'tensorflow', 'numpy', 'scipy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(isa=select_isa(), physical_cpus=physical_cpus(),
                packed_density=os.environ.get('WBC_CPU_DENSITY_PACKED'),
                threads=os.environ.get('WBC_CPU_THREADS'),
                workers=os.environ.get('WBC_CPU_WORKERS'),
                cache=str(cache_root()), versions=versions)
