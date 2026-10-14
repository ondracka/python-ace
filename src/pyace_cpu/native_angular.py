"""Sparse Cartesian polynomials matching the upstream harmonic recurrence.

First and second derivatives are exact for the polynomial expression, including
off the unit sphere. Normalization of neighbour vectors remains upstream.
"""
import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from numpy.polynomial import Polynomial
import tensorflow as tf

from .native_build import compiler_settings, cache_root

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'native_angular_op.cc'
_module = None


def build_spec():
    compiler, flags, version, target = compiler_settings()
    flags += tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    identity = dict(tensorflow=tf.__version__, compiler=version, flags=flags,
                    native_target_sha256=hashlib.sha256(target).hexdigest(),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return compiler, flags, identity, cache_root() / f'native-angular-{digest}.so'


def load():
    global _module
    if _module is not None:
        return _module
    _, _, _, library = build_spec()
    if not library.is_file():
        raise FileNotFoundError(f'Build explicitly first: python {Path(__file__).name} --build')
    _module = tf.load_op_library(str(library))

    @tf.RegisterGradient('WbcAngular')
    def grad(op, seed):
        return [_module.wbc_angular_grad(*op.inputs, seed), None, None, None]

    @tf.RegisterGradient('WbcAngularGrad')
    def second(op, direction):
        dx, ds = _module.wbc_angular_grad_grad(*op.inputs, direction)
        return [dx, None, None, None, ds]

    @tf.RegisterGradient('WbcAngularSplit')
    def split_grad(op, sr, si):
        if sr is None:
            sr = tf.zeros_like(op.outputs[0])
        if si is None:
            si = tf.zeros_like(op.outputs[1])
        return [_module.wbc_angular_split_grad(*op.inputs, sr, si), None, None, None]

    @tf.RegisterGradient('WbcAngularSplitGrad')
    def split_second(op, direction):
        dx, dsr, dsi = _module.wbc_angular_split_grad_grad(*op.inputs, direction)
        return [dx, None, None, None, dsr, dsi]

    return _module


@functools.lru_cache(maxsize=16)
def plan(lmax):
    z = Polynomial([0., 1.])
    plm = {(0, 0): Polynomial([np.sqrt(1. / (4. * np.pi))])}
    if lmax:
        plm[1, 0] = np.sqrt(3. / (4. * np.pi)) * z
        plm[1, 1] = Polynomial([-np.sqrt(3. / (8. * np.pi))])
    for l in range(2, lmax + 1):
        for m in range(l + 1):
            if m == l - 1:
                plm[l, m] = np.sqrt(2. * m + 3.) * z * plm[l - 1, l - 1]
            elif m == l:
                plm[l, m] = -np.sqrt(1. + 0.5 / m) * plm[l - 1, l - 1]
            else:
                a = np.sqrt((4. * l * l - 1.) / (l * l - m * m))
                b = -np.sqrt(((l - 1.) ** 2 - m * m) / (4. * (l - 1.) ** 2 - 1.))
                plm[l, m] = a * (z * plm[l - 1, m] + b * plm[l - 2, m])
    rows, powers, weights = [0], [], []
    for imaginary in (False, True):
        for l in range(lmax + 1):
            for signed_m in range(-l, l + 1):
                m = abs(signed_m)
                for py in range(m + 1):
                    phase = math.comb(m, py) * 1j ** py
                    if signed_m < 0:
                        phase = (-1.) ** m * phase.conjugate()
                    factor = phase.imag if imaginary else phase.real
                    for pz, value in enumerate(plm[l, m].coef):
                        weight = factor * value
                        if weight != 0.:
                            powers.append((m - py, py, pz))
                            weights.append(weight)
                rows.append(len(weights))
    return (np.asarray(rows, np.int32), np.asarray(powers, np.int32).reshape(-1, 3),
            np.asarray(weights, np.float64))


def compute_ylm(self, rhat):
    if self.prec.float != tf.float64:
        return _original(self, rhat)
    self.l_tile = tf.constant(np.repeat(np.arange(self._lmax + 1),
                                       2 * np.arange(self._lmax + 1) + 1), dtype=tf.int32)
    values = load().wbc_angular(tf.convert_to_tensor(rhat, tf.float64), *plan(self._lmax))
    nm = (self._lmax + 1) ** 2
    return values[:, :nm], values[:, nm:]


def compute_ylm_split(self, rhat):
    if self.prec.float != tf.float64:
        return _original(self, rhat)
    self.l_tile = tf.constant(np.repeat(np.arange(self._lmax + 1),
                                       2 * np.arange(self._lmax + 1) + 1), dtype=tf.int32)
    return load().wbc_angular_split(tf.convert_to_tensor(rhat, tf.float64), *plan(self._lmax))


_original = None


def enable(split=False):
    global _original
    from tensorpotential.functions.spherical_harmonics import SphericalHarmonics
    load()
    if _original is None:
        _original = SphericalHarmonics.compute_ylm
    SphericalHarmonics.compute_ylm = compute_ylm_split if split else compute_ylm


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
