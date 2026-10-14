"""Native SBessel recurrence with fixed inner/outer cutoffs and exact derivatives.

Non-SBessel families retain their upstream implementation. Sharing groups only
identical fixed primitive definitions. Radial mixing coefficients are untouched.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import tensorflow as tf

from .native_build import compiler_settings, cache_root

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'native_radial_op.cc'
_module = None
_reuse_derivative = False
_trig_recurrence = False


def build_spec():
    compiler, flags, version, target = compiler_settings()
    flags += tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    identity = dict(tensorflow=tf.__version__, compiler=version, flags=flags,
                    native_target_sha256=hashlib.sha256(target).hexdigest(),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return compiler, flags, identity, cache_root() / f'native-radial-{digest}.so'


def load():
    global _module
    if _module is not None:
        return _module
    _, _, _, library = build_spec()
    if not library.is_file():
        raise FileNotFoundError(f'Build explicitly first: python {Path(__file__).name} --build')
    _module = tf.load_op_library(str(library))

    @tf.RegisterGradient('WbcRadial')
    def grad(op, seed):
        return [_module.wbc_radial_grad(*op.inputs, seed,
                    trig_recurrence=op.get_attr('trig_recurrence')), None, None]

    @tf.RegisterGradient('WbcRadialGrad')
    def second(op, direction):
        dr, ds = _module.wbc_radial_grad_grad(*op.inputs, direction,
                    trig_recurrence=op.get_attr('trig_recurrence'))
        return [dr, None, None, ds]

    return _module


@functools.lru_cache(maxsize=16)
def plan(count):
    count = max(1, count)
    n = np.arange(count, dtype=np.float64)
    factors = ((-1.) ** n * np.sqrt(2.) * np.pi * (n + 1) * (n + 2)
               / np.sqrt((n + 1) ** 2 + (n + 2) ** 2))
    result = np.column_stack([factors, np.ones(count), np.zeros(count)])
    dprev = 1.
    for i in range(1, count):
        en = i ** 2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1)
        dn = 1. - en / dprev
        result[i, 1] = 1. / np.sqrt(dn)
        result[i, 2] = np.sqrt(en / dprev)
        dprev = dn
    return result


def enable_derivative_reuse():
    global _reuse_derivative
    _reuse_derivative = True


def enable_trig_recurrence():
    global _trig_recurrence
    _trig_recurrence = True


def _value_with_reused_derivative(distances, settings, recurrence, trig_recurrence=False):
    """Retain g' for this call and explicitly supply the complete second VJP.

    The auxiliary derivative is internal, not a second differentiable output.
    This avoids asking TF to differentiate a zero seed on that output and
    inadvertently constructing an unnecessary third coordinate derivative.
    """
    @tf.custom_gradient
    def value(r, fixed_settings, fixed_recurrence):
        result, derivative = load().wbc_radial_with_derivative(
            r, fixed_settings, fixed_recurrence, trig_recurrence=trig_recurrence)

        def first(seed):
            @tf.custom_gradient
            def contraction(x, incoming, cached_derivative):
                result = tf.reduce_sum(incoming * cached_derivative, axis=1, keepdims=True)

                def second(direction):
                    dr, _ = load().wbc_radial_grad_grad(
                        x, fixed_settings, fixed_recurrence, incoming, direction,
                        trig_recurrence=trig_recurrence)
                    return dr, direction * cached_derivative, None

                return result, second

            return contraction(r, seed, tf.stop_gradient(derivative)), None, None

        return result, first

    return value(distances, settings, recurrence)


def evaluate(distances, count, rcut, rin=0., delta=0., reuse_derivative=None,
             trig_recurrence=None):
    reuse = _reuse_derivative if reuse_derivative is None else reuse_derivative
    trig = _trig_recurrence if trig_recurrence is None else trig_recurrence
    inputs = (distances, tf.constant([rcut, rin, delta], tf.float64), plan(count))
    if reuse:
        return _value_with_reused_derivative(*inputs, trig_recurrence=trig)
    return load().wbc_radial(*inputs, trig_recurrence=trig)


def bond_radial(potential, distances, spec):
    if spec.radbasename in ('SBessel', 'TEST_SBessel'):
        return evaluate(distances, spec.nradbasemax, spec.rcut, spec.rcut_in, spec.dcut_in)
    from tensorpotential.functions.radial_functions import cutoff_func_poly
    values = potential.radial_function(distances, nfunc=spec.nradbasemax,
        ftype=spec.radbasename, cutoff=tf.constant(spec.rcut, tf.float64),
        lmbda=tf.constant(spec.radparameters[0], tf.float64))
    return values * (1 - cutoff_func_poly(distances, tf.constant(spec.rcut_in, tf.float64),
                                        tf.constant(spec.dcut_in, tf.float64)))


def partition_radial(potential, distances, part_distances):
    """Share the unpartitioned call only when every bond definition agrees."""
    specs = [potential.bond_specs[b] for b in potential.index.bond_combs]
    keys = [(s.radbasename, s.nradbasemax, s.rcut, s.rcut_in, s.dcut_in,
             tuple(s.radparameters)) for s in specs]
    if len(set(keys)) == 1 and specs[0].radbasename in ('SBessel', 'TEST_SBessel'):
        values = bond_radial(potential, distances, specs[0])
        return tf.dynamic_partition(values, potential.index.bond_partition, potential.index.ncombs)
    return [bond_radial(potential, d, s) for d, s in zip(part_distances, specs)]


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
