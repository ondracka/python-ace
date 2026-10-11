"""Fused radial/angular moment sums, with all radial coefficients trainable.

The optional reordered variant sums primitive neighbour moments before radial
mixing, using linearity. No dataset-sized geometry or derivative cache is kept.
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

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'native_moments_op.cc'
_module = None
_prune_unused = False


def build_spec():
    compiler = os.environ.get('CXX', 'g++')
    flags = ['-std=c++17', '-shared', '-fPIC', '-O3', '-march=native', '-ffp-contract=off']
    flags += tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    target = subprocess.check_output([compiler, '-march=native', '-Q', '--help=target'])
    identity = dict(tensorflow=tf.__version__, compiler=version, flags=flags,
                    native_target_sha256=hashlib.sha256(target).hexdigest(),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return compiler, flags, identity, ROOT / 'compiled' / f'native-moments-{digest}.so'


def load():
    global _module
    if _module is not None:
        return _module
    _, _, _, library = build_spec()
    if not library.is_file():
        raise FileNotFoundError(f'Build explicitly first: python {Path(__file__).name} --build')
    _module = tf.load_op_library(str(library))

    def attrs(op):
        return dict(nradout=op.get_attr('nradout'), l_dependent=op.get_attr('l_dependent'),
                    prune_unused=op.get_attr('prune_unused'))

    @tf.RegisterGradient('WbcMoments')
    def grad(op, sr, si):
        if sr is None:
            sr = tf.zeros_like(op.outputs[0])
        if si is None:
            si = tf.zeros_like(op.outputs[1])
        result = _module.wbc_moments_grad(*op.inputs, sr, si, **attrs(op))
        return [*result, None]

    @tf.RegisterGradient('WbcMomentsGrad')
    def second(op, hr, hyr, hyi):
        directions = [tf.zeros_like(op.inputs[i]) if value is None else value
                      for i, value in enumerate((hr, hyr, hyi))]
        dr, dyr, dyi, dsr, dsi = _module.wbc_moments_grad_grad(*op.inputs, *directions, **attrs(op))
        return [dr, dyr, dyi, None, dsr, dsi]

    return _module


def moments(radial, yr, yi, segments, nradout, l_dependent=True):
    nm = tf.shape(yr)[-1]
    return load().wbc_moments(radial, tf.reshape(yr, [-1, nm]),
                             tf.reshape(yi, [-1, nm]), segments,
                             nradout=nradout, l_dependent=l_dependent,
                             prune_unused=_prune_unused)


def enable_output_pruning():
    """Prune only outputs the TensorFlow executor confirms are unused."""
    global _prune_unused
    _prune_unused = True


def enable(reorder=False, radial=False, share_radial=False):
    from tensorpotential.potentials.ace import ACE
    load()
    original = ACE.eval_atomic_energy
    source = textwrap.dedent(inspect.getsource(original))
    old = '''        rj_nl = tf.einsum('jk,nlk->jnl', comb_gjk, crad)
        rj_nl = tf.pad(rj_nl, [[0, 0], [0, self.nradmax - nradmaxi], [0, self.lmax - lmaxi]])
        rj_nlm = tf.gather(rj_nl, sh.l_tile, axis=2)
        aj_nlm_r = rj_nlm * part_ylm_r[c_i]
        aj_nlm_i = rj_nlm * part_ylm_i[c_i]

        a_nlm_r = tf.math.segment_sum(aj_nlm_r, self.index.bond_sum_ind[c_i])
        a_nlm_i = tf.math.segment_sum(aj_nlm_i, self.index.bond_sum_ind[c_i])'''
    new = '''        rj_nl = tf.einsum('jk,nlk->jnl', comb_gjk, crad)
        a_nlm_r, a_nlm_i = _wbc_moments(
            rj_nl, part_ylm_r[c_i], part_ylm_i[c_i],
            self.index.bond_sum_ind[c_i], nradout=self.nradmax)'''
    if reorder:
        new = '''        primitive_r, primitive_i = _wbc_moments(
            tf.expand_dims(comb_gjk, -1), part_ylm_r[c_i], part_ylm_i[c_i],
            self.index.bond_sum_ind[c_i], nradout=nradbasei, l_dependent=False)
        padded_crad = tf.pad(crad, [[0, self.nradmax - nradmaxi],
                                  [0, self.lmax - lmaxi], [0, 0]])
        crad_nmk = tf.gather(padded_crad, sh.l_tile, axis=1)
        a_nlm_r = tf.einsum('skm,nmk->snm', primitive_r, crad_nmk)
        a_nlm_i = tf.einsum('skm,nmk->snm', primitive_i, crad_nmk)'''
    if source.count(old) != 1:
        raise RuntimeError('Upstream ACE neighbour-moment block changed, review the patch')
    source = source.replace(old, new)
    namespace = dict(original.__globals__, _wbc_moments=moments)
    if radial:
        from native_radial import load as load_radial, bond_radial, partition_radial
        load_radial()
        radial_old = '''        comb_gjk = self.radial_function(part_d_ij[c_i],
                                        nfunc=nradbasei,
                                        ftype=bond_spec.radbasename,
                                        cutoff=tf.constant(bond_spec.rcut, dtype=tf.float64),
                                        lmbda=tf.constant(bond_spec.radparameters[0],
                                                          dtype=tf.float64))  # [None, nradbase]
        comb_gjk = comb_gjk * (1 - radial_functions.cutoff_func_poly(part_d_ij[c_i],
                                                                tf.constant(bond_spec.rcut_in, dtype=tf.float64),
                                                                tf.constant(bond_spec.dcut_in, dtype=tf.float64)))'''
        if source.count(radial_old) != 1:
            raise RuntimeError('Upstream radial block changed, review the patch')
        radial_new = '        comb_gjk = _wbc_bond_radial(self, part_d_ij[c_i], bond_spec)'
        if share_radial:
            anchor = '    radial_coefs = self.fit_coefs[:self.total_num_crad]'
            if source.count(anchor) != 1:
                raise RuntimeError('Upstream radial-coefficient anchor changed')
            source = source.replace(anchor, '    part_radial = _wbc_partition_radial(self, d_ij, part_d_ij)\n' + anchor)
            radial_new = '        comb_gjk = part_radial[c_i]'
        source = source.replace(radial_old, radial_new)
        namespace.update(_wbc_bond_radial=bond_radial, _wbc_partition_radial=partition_radial)
    exec(compile(source, '<wbc-native-moments>', 'exec'), namespace)
    # The existing ACE loops/branches are Python-static at tracing time.
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
