"""Opt-in ordered ACE product DAG with exact first and second derivatives.

Only fixed basis plans and Clebsch-Gordan constants are stored. All potential
coefficients, including radial mixing, remain trainable through TensorFlow.
Third derivatives of the basis are deliberately unsupported by this prototype.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import tensorflow as tf

from .native_build import compiler_settings, cache_root

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'native_basis_op.cc'
_module = None
_block_size = 1


def build_spec():
    compiler, flags, compiler_version, native_options = compiler_settings()
    flags += tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    identity = dict(tensorflow=tf.__version__, compiler=compiler_version, flags=flags,
                    native_target_sha256=hashlib.sha256(native_options).hexdigest(),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return compiler, flags, identity, cache_root() / f'native-basis-{digest}.so'


def load():
    global _module
    if _module is not None:
        return _module
    _, _, _, library = build_spec()
    if not library.is_file():
        raise FileNotFoundError(f'Build explicitly first: python {Path(__file__).name} --build')
    _module = tf.load_op_library(str(library))

    @tf.RegisterGradient('WbcBasis')
    def basis_gradient(op, seed):
        dar, dai = _module.wbc_basis_grad(*op.inputs, seed, block_size=op.get_attr('block_size'))
        return [dar, dai, None, None, None, None, None]

    @tf.RegisterGradient('WbcBasisGrad')
    def basis_second_gradient(op, hr, hi):
        if hr is None:
            hr = tf.zeros_like(op.inputs[0])
        if hi is None:
            hi = tf.zeros_like(op.inputs[1])
        dar, dai, dseed = _module.wbc_basis_grad_grad(*op.inputs, hr, hi, block_size=op.get_attr('block_size'))
        return [dar, dai, None, None, None, None, None, dseed]

    return _module


def make_plan(potential, central, rank):
    config = potential.config.central_atom[central][rank - 2]
    factors = np.stack(config.munlm, axis=1)
    nf = potential.nelements * potential.nradmax * (potential.lmax + 1) ** 2
    left, right = [], []
    nn = nf
    previous_nodes = previous_inverse = None
    for k in range(rank):
        _, rows, inverse = np.unique(factors[:, :k + 1].reshape(len(factors), -1),
                                    axis=0, return_index=True, return_inverse=True)
        feature = factors[rows, k]
        flat = ((feature[:, 0] * potential.nradmax + feature[:, 1])
                * (potential.lmax + 1) ** 2 + feature[:, 2]).astype(np.int32)
        if k == 0:
            nodes = flat
        else:
            left.extend(previous_nodes[previous_inverse[rows]])
            right.extend(flat)
            nodes = np.arange(nn, nn + len(rows), dtype=np.int32)
            nn += len(rows)
        previous_nodes, previous_inverse = nodes, inverse
    terms = previous_nodes[previous_inverse]
    msum = np.asarray(config.msum, dtype=np.int32)
    if len(msum) and np.any(msum[1:] < msum[:-1]):
        raise ValueError('CG segment IDs must be ordered')
    nrows = int(msum[-1]) + 1 if len(msum) else 0
    row_ptr = np.r_[0, np.cumsum(np.bincount(msum, minlength=nrows))].astype(np.int32)
    return (np.array(left, dtype=np.int32), np.array(right, dtype=np.int32), row_ptr,
            np.asarray(terms, dtype=np.int32), np.asarray(config.genCG, dtype=np.float64).reshape(-1))


def native_bbasis(self, a_munlm_r, a_munlm_i, cntrl_at, rank):
    if not hasattr(self, '_native_bbasis_plans'):
        self._native_bbasis_plans = {}
    key = cntrl_at, rank
    if key not in self._native_bbasis_plans:
        self._native_bbasis_plans[key] = make_plan(self, cntrl_at, rank)
    nf = self.nelements * self.nradmax * (self.lmax + 1) ** 2
    natoms = tf.shape(a_munlm_r[cntrl_at])[0]
    ar = tf.reshape(tf.transpose(a_munlm_r[cntrl_at], [1, 2, 3, 0]), [nf, natoms])
    ai = tf.reshape(tf.transpose(a_munlm_i[cntrl_at], [1, 2, 3, 0]), [nf, natoms])
    result = load().wbc_basis(ar, ai, *self._native_bbasis_plans[key], block_size=_block_size)
    return tf.transpose(result, [1, 0])


def enable(block_size=1):
    global _block_size
    from tensorpotential.potentials.ace import ACE
    if block_size not in (1, 4, 8):
        raise ValueError('Supported ACE atom block sizes are 1, 4 and 8')
    _block_size = block_size
    load()
    ACE.compute_bbasis_funcs = native_bbasis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', required=True)
    parser.parse_args()
    compiler, flags, identity, library = build_spec()
    if library.exists():
        raise FileExistsError(library)
    library.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, str(SOURCE), '-o', str(library)] + flags
    print(json.dumps(dict(event='compile', command=command, **identity)), flush=True)
    subprocess.run(command, check=True)
    print(json.dumps(dict(event='compiled', library=str(library))), flush=True)
