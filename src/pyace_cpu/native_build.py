"""Compiler flags and native-target identity for GCC and Intel/Clang builds."""
import os
import subprocess
from .runtime import cache_root, select_isa


def vector_width_flags(variable):
    """Opt-in per-operation compiler preference, preserving the default flags."""
    bits = os.environ.get(variable, '0')
    if bits not in ('0', '256', '512'):
        raise ValueError(f'{variable} must be 0, 256 or 512')
    return [] if bits == '0' else [f'-mprefer-vector-width={bits}']


def compiler_settings():
    compiler = os.environ.get('CXX', 'g++')
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    isa = select_isa(cross_compile=os.environ.get('PACEMAKER_CPU_BUILD_ONLY') == '1')
    arch = ['-march=x86-64', '-mavx2', '-mfma']
    if isa == 'avx512':
        arch += ['-mavx512f', '-mavx512dq', '-mavx512vl']
    flags = ['-std=c++17', '-shared', '-fPIC', '-O3', *arch, '-ffp-contract=off']
    if 'intel' in version.lower() or 'clang' in version.lower():
        # Intel's default FP model is not our numerical contract. Keep precise
        # operations and disable FMA contraction explicitly, as in the GCC build.
        if 'intel' in version.lower():
            flags.insert(-1, '-fp-model=precise')
            # TF 2.21's bundled Eigen AVX512-FP16 header specializes a removed
            # predux_half_dowto4 template. These operations are float64 only.
            # Skip that unused header, not AVX512F or float64 vectorization.
            # Intel's native AVX10 feature set keeps this macro defined even
            # with -mno-avx512fp16, so undefine the header-selection macro.
            flags.append('-U__AVX512FP16__')
    # Explicit ISA flags make the AVX2 build safe on an AVX2 compute node even
    # when installation ran on an AVX-512 login node. No -march=native binaries.
    target = subprocess.check_output(
        [compiler, *arch, '-dM', '-E', '-x', 'c++', '-'], input=b'')
    return compiler, flags, version, target
