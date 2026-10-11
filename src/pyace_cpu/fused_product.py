"""Load the opt-in, CPU-only complex product and its analytic derivatives.

Gradient expressions reuse the same differentiable op, so the mixed
derivatives required by force fitting remain available. All four operands
must have the same shape, as in the ACE higher-body-order products.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import tensorflow as tf

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'complex_product_op.cc'
_module = None


def build_spec():
    compiler = os.environ.get('CXX', 'g++')
    flags = ['-std=c++17', '-shared', '-fPIC', '-O3', '-march=native', '-ffp-contract=off']
    flags += tf.sysconfig.get_compile_flags()
    flags += tf.sysconfig.get_link_flags()
    compiler_version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    # '-march=native' alone is not a portable architecture identifier.
    native_options = subprocess.check_output([compiler, '-march=native', '-Q', '--help=target'])
    identity = dict(tensorflow=tf.__version__, compiler=compiler_version, flags=flags,
                    native_target_sha256=hashlib.sha256(native_options).hexdigest(),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    library = ROOT / 'compiled' / f'complex-product-{digest}.so'
    return compiler, flags, identity, library


def load():
    global _module
    if _module is not None:
        return _module
    _, _, _, library = build_spec()
    if not library.is_file():
        raise FileNotFoundError(f'Build explicitly first: python {Path(__file__).name} --build')
    _module = tf.load_op_library(str(library))

    @tf.RegisterGradient('WbcComplexProduct')
    def product_gradient(op, dr, di):
        ar, ai, br, bi = op.inputs
        if dr is None:
            dr = tf.zeros_like(ar)
        if di is None:
            di = tf.zeros_like(ar)
        dar, dai = _module.wbc_complex_product(dr, di, br, -bi)
        dbr, dbi = _module.wbc_complex_product(dr, di, ar, -ai)
        return dar, dai, dbr, dbi

    return _module


def enable():
    from tensorpotential.potentials.ace import ACE
    module = load()
    ACE.complexmul = staticmethod(module.wbc_complex_product)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', required=True)
    parser.parse_args()
    compiler, flags, identity, library = build_spec()
    if library.exists():
        raise FileExistsError(library)
    library.parent.mkdir(exist_ok=True)
    command = [compiler, str(SOURCE), '-o', str(library)] + flags
    print(json.dumps(dict(event='compile', command=command, **identity)), flush=True)
    subprocess.run(command, check=True)
    print(json.dumps(dict(event='compiled', library=str(library))), flush=True)
