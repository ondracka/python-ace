"""All-bond primitive accumulation without partitioned radial/angular copies.

Applies only to complete bond catalogs with identical SBessel primitives.
Other models use the previously selected evaluator. Radial mixing, core
repulsion, embedding and all fitted coefficients retain their existing paths.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap

import tensorflow as tf

from native_build import compiler_settings, vector_width_flags

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'native_all_moments_op.cc'
_module = None


def build_spec():
    compiler, flags, version, target = compiler_settings()
    flags += vector_width_flags('WBC_CPU_MOMENTS_VECTOR_BITS')
    flags += tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    identity = dict(tensorflow=tf.__version__, compiler=version, flags=flags,
                    native_target_sha256=hashlib.sha256(target).hexdigest(),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return compiler, flags, identity, ROOT / 'compiled' / f'native-all-moments-{digest}.so'


def load():
    global _module
    if _module is not None:
        return _module
    _, _, _, library = build_spec()
    if not library.is_file():
        raise FileNotFoundError(f'Build explicitly first: python {Path(__file__).name} --build')
    _module = tf.load_op_library(str(library))

    @tf.RegisterGradient('WbcAllMoments')
    def gradient(op, sr, si, sg):
        seeds = [tf.zeros_like(op.outputs[i]) if value is None else value
                 for i, value in enumerate((sr, si, sg))]
        dr, dyr, dyi = _module.wbc_all_moments_grad(*op.inputs, *seeds)
        return [dr, dyr, dyi, None, None]

    @tf.RegisterGradient('WbcAllMomentsGrad')
    def second(op, hr, hyr, hyi):
        directions = [tf.zeros_like(op.inputs[i]) if value is None else value
                      for i, value in enumerate((hr, hyr, hyi))]
        dr, dyr, dyi, dsr, dsi, dsg = _module.wbc_all_moments_grad_grad(*op.inputs, *directions)
        return [dr, dyr, dyi, None, None, dsr, dsi, dsg]

    return _module


def compatible(potential):
    bonds = potential.index.bond_combs
    complete = {(i, j) for i in range(potential.nelements) for j in range(potential.nelements)}
    if potential.rankmax <= 1 or set(bonds) != complete:
        return False
    specs = [potential.bond_specs[b] for b in bonds]
    keys = [(s.radbasename, s.nradbasemax, s.rcut, s.rcut_in, s.dcut_in,
             tuple(s.radparameters)) for s in specs]
    return len(set(keys)) == 1 and specs[0].radbasename in ('SBessel', 'TEST_SBessel')


def moments(potential, distances, yr, yi):
    from native_radial import bond_radial
    spec = potential.bond_specs[potential.index.bond_combs[0]]
    radial = bond_radial(potential, distances, spec)
    nm = (potential.lmax + 1) ** 2
    segments = potential.index.ind_i * potential.nelements + potential.index.mu_j
    return load().wbc_all_moments(
        radial, tf.reshape(yr, [-1, nm]), tf.reshape(yi, [-1, nm]),
        segments, potential.index.nat * potential.nelements)


def enable():
    from tensorpotential.potentials.ace import ACE
    load()
    original = ACE.eval_atomic_energy
    source = getattr(original, '_wbc_source', None)
    if source is None:
        source = textwrap.dedent(inspect.getsource(original))
    namespace = dict(getattr(original, '_wbc_globals', original.__globals__))
    replacements = [
        ('def eval_atomic_energy(self, r_ij):\n',
         'def eval_atomic_energy(self, r_ij):\n'
         '    if not _wbc_all_compatible(self):\n'
         '        return _wbc_all_fallback(self, r_ij)\n'),
        ('    part_ylm_r = tf.dynamic_partition(ynlm_r, self.index.bond_partition, self.index.ncombs)\n', ''),
        ('    part_ylm_i = tf.dynamic_partition(ynlm_i, self.index.bond_partition, self.index.ncombs)\n', ''),
        ('    part_radial = _wbc_partition_radial(self, d_ij, part_d_ij)\n',
         '    pooled_r, pooled_i, pooled_g = _wbc_all_moments(self, d_ij, ynlm_r, ynlm_i)\n'),
        ('        comb_gjk = part_radial[c_i]\n'
         '        gk_i = tf.math.segment_sum(comb_gjk, self.index.bond_sum_ind[c_i])',
         '        pooled_ids = tf.reshape(self.index.atomic_stitching[c_i], [-1]) * self.nelements + c[1]\n'
         '        gk_i = tf.gather(pooled_g, pooled_ids)'),
        ('        primitive_r, primitive_i = _wbc_moments(\n'
         '            tf.expand_dims(comb_gjk, -1), part_ylm_r[c_i], part_ylm_i[c_i],\n'
         '            self.index.bond_sum_ind[c_i], nradout=nradbasei, l_dependent=False)',
         '        primitive_r = tf.gather(pooled_r, pooled_ids)\n'
         '        primitive_i = tf.gather(pooled_i, pooled_ids)'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('All-moment patch requires the reviewed reordered/shared-radial source: ' + old[:90])
        source = source.replace(old, new)
    namespace.update(_wbc_all_compatible=compatible, _wbc_all_fallback=original,
                     _wbc_all_moments=moments)
    exec(compile(source, '<wbc-all-moments>', 'exec'), namespace)
    ACE.eval_atomic_energy = tf.autograph.experimental.do_not_convert(namespace['eval_atomic_energy'])
    ACE.eval_atomic_energy._wbc_source = source
    ACE.eval_atomic_energy._wbc_globals = namespace


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
