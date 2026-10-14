"""Fuse ACE products with trainable density coefficients, without a basis matrix.

Fixed Clebsch-Gordan weights are contracted with current fitting coefficients
on every call. Ordered product prefixes can be shared across body ranks.
All first and second derivatives remain analytic, including coefficient paths.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap

import numpy as np
import tensorflow as tf

from native_build import compiler_settings, vector_width_flags

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'native_density_op.cc'
_module = None
_share_products = True
_prune_unused = False
_reuse_products = False
_compact_workspace = False


def build_spec():
    compiler, flags, version, target = compiler_settings()
    block = os.environ.get('WBC_CPU_DENSITY_BLOCK', '4')
    if block not in ('4', '8'):
        raise ValueError('WBC_CPU_DENSITY_BLOCK must be 4 or 8')
    if block != '4':
        flags.append(f'-DWBC_DENSITY_BLOCK={block}')
    packed = os.environ.get('WBC_CPU_DENSITY_PACKED', '0')
    if packed not in ('0', '1') or (packed == '1' and block != '4'):
        raise ValueError('WBC_CPU_DENSITY_PACKED must be 0 or 1, and requires four-atom blocks')
    if packed == '1':
        flags.append('-DWBC_DENSITY_PACKED_AVX512=1')
    flags += vector_width_flags('WBC_CPU_DENSITY_VECTOR_BITS')
    flags += tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    identity = dict(tensorflow=tf.__version__, compiler=version, flags=flags,
                    native_target_sha256=hashlib.sha256(target).hexdigest(),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    compact_header_sha256=hashlib.sha256(
                        (ROOT / 'native_density_compact.h').read_bytes()).hexdigest())
    if packed == '1':
        identity['packed_header_sha256'] = hashlib.sha256(
            (ROOT / 'native_density_packed.h').read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return compiler, flags, identity, ROOT / 'compiled' / f'native-density-{digest}.so'


def load():
    global _module
    if _module is not None:
        return _module
    _, _, _, library = build_spec()
    if not library.is_file():
        raise FileNotFoundError(f'Build explicitly first: python {Path(__file__).name} --build')
    _module = tf.load_op_library(str(library))

    @tf.RegisterGradient('WbcDensity')
    def gradient(op, seed):
        ar, ai, weights = _module.wbc_density_grad(
            *op.inputs, seed, prune_unused=op.get_attr('prune_unused'),
            compact_workspace=op.get_attr('compact_workspace'))
        return [ar, ai, None, None, None, weights]

    @tf.RegisterGradient('WbcDensityGrad')
    def second_gradient(op, hr, hi, hw):
        prune = op.get_attr('prune_unused')
        directions = [tf.zeros_like(op.inputs[index]) if h is None else h
                      for index, h in zip((0, 1, 5), (hr, hi, hw))]
        ar, ai, weights, seed = _module.wbc_density_grad_grad(
            *op.inputs, *directions, prune_unused=prune,
            compact_workspace=op.get_attr('compact_workspace'),
            zero_weight_direction=prune and _structurally_zero(hw))
        return [ar, ai, None, None, None, weights, seed]

    return _module


def enable_output_pruning():
    global _prune_unused
    _prune_unused = True


def enable_product_reuse():
    global _reuse_products
    _reuse_products = True


def enable_compact_workspace():
    global _compact_workspace
    _compact_workspace = True


def _structurally_zero(value):
    """Recognize only graph-constant zero directions, never current fit values."""
    if value is None:
        return True
    if tf.executing_eagerly():
        return False
    op = value.op
    if op.type == 'ZerosLike':
        return True
    if op.type in ('Identity', 'StopGradient'):
        return _structurally_zero(op.inputs[0])
    if op.type == 'Fill':
        scalar = tf.get_static_value(op.inputs[1])
        return scalar is not None and np.ndim(scalar) == 0 and bool(scalar == 0)
    if op.type == 'Const':
        constant = tf.get_static_value(value)
        return constant is not None and bool(np.all(constant == 0))
    return False


def evaluate(ar, ai, left, right, terms, weights, prune_unused=None, reuse_products=None,
             compact_workspace=None):
    prune = _prune_unused if prune_unused is None else prune_unused
    reuse = _reuse_products if reuse_products is None else reuse_products
    compact = _compact_workspace if compact_workspace is None else compact_workspace
    if compact and reuse:
        raise ValueError('Compact density workspaces are incompatible with product caching')
    module = load()
    if not reuse:
        return module.wbc_density(ar, ai, left, right, terms, weights, prune_unused=prune,
                                  compact_workspace=compact)

    @tf.custom_gradient
    def value(a_r, a_i, coefficients):
        rho, products_r, products_i = module.wbc_density_cached(
            a_r, a_i, left, right, terms, coefficients, prune_unused=prune)

        def first(seed):
            @tf.custom_gradient
            def backward(x_r, x_i, c, incoming, saved_r, saved_i):
                derivatives = module.wbc_density_cached_grad(
                    x_r, x_i, left, right, terms, c, incoming, saved_r, saved_i,
                    prune_unused=prune)

                def second(hr, hi, hw):
                    directions = [tf.zeros_like(x) if h is None else h
                                  for x, h in zip((x_r, x_i, c), (hr, hi, hw))]
                    dr, di, dw, ds = module.wbc_density_cached_grad_grad(
                        x_r, x_i, left, right, terms, c, incoming, *directions,
                        saved_r, saved_i, prune_unused=prune,
                        zero_weight_direction=prune and _structurally_zero(hw))
                    # This op supplies the total Hessian-vector product, including
                    # product dependence on x_r/x_i. Do not differentiate the
                    # auxiliary cache separately and count that dependence twice.
                    return dr, di, dw, ds, None, None

                return tuple(derivatives), second

            return backward(a_r, a_i, coefficients, seed,
                            tf.stop_gradient(products_r), tf.stop_gradient(products_i))

        return rho, first

    return value(tf.convert_to_tensor(ar, tf.float64), tf.convert_to_tensor(ai, tf.float64),
                 tf.convert_to_tensor(weights, tf.float64))


def make_plan(potential, central, share_products=True):
    """Return a topological prefix DAG and its fixed linear CG contraction."""
    nf = potential.nelements * potential.nradmax * (potential.lmax + 1) ** 2
    left, right, term_nodes, basis_ids, cg = [], [], [], [], []
    prefixes = {}
    offset = 0
    for rank in range(2, potential.ranksmax[central] + 1):
        config = potential.config.central_atom[central][rank - 2]
        factors = np.stack(config.munlm, axis=1)
        flat = ((factors[:, :, 0] * potential.nradmax + factors[:, :, 1])
                * (potential.lmax + 1) ** 2 + factors[:, :, 2])
        for sequence in flat:
            sequence = tuple(int(f) for f in sequence)
            node = sequence[0]
            for length in range(2, rank + 1):
                key = sequence[:length] if share_products else (rank, sequence[:length])
                if key not in prefixes:
                    prefixes[key] = nf + len(left)
                    left.append(node)
                    right.append(sequence[length - 1])
                node = prefixes[key]
            term_nodes.append(node)
        msum = np.asarray(config.msum, dtype=np.int32)
        if len(msum) and np.any(msum[1:] < msum[:-1]):
            raise ValueError('CG segment IDs must be ordered')
        basis_ids.extend(msum + offset)
        cg.extend(np.asarray(config.genCG, dtype=np.float64).reshape(-1))
        offset += int(msum[-1]) + 1 if len(msum) else 0
    nodes, inverse = np.unique(term_nodes, return_inverse=True)
    return dict(left=np.asarray(left, np.int32), right=np.asarray(right, np.int32),
                nodes=nodes.astype(np.int32), segments=inverse.astype(np.int32),
                basis_ids=np.asarray(basis_ids, np.int32), cg=np.asarray(cg, np.float64),
                basis_count=offset)


def density(self, ar_parts, ai_parts, central, coefs):
    if not hasattr(self, '_wbc_density_plans'):
        self._wbc_density_plans = {}
    if central not in self._wbc_density_plans:
        self._wbc_density_plans[central] = make_plan(self, central, _share_products)
    plan = self._wbc_density_plans[central]
    nf = self.nelements * self.nradmax * (self.lmax + 1) ** 2
    natoms = tf.shape(ar_parts[central])[0]
    ar = tf.reshape(tf.transpose(ar_parts[central], [1, 2, 3, 0]), [nf, natoms])
    ai = tf.reshape(tf.transpose(ai_parts[central], [1, 2, 3, 0]), [nf, natoms])
    coefs = tf.ensure_shape(coefs, [plan['basis_count'], self.embed_spec[central].ndensity])
    weights = tf.math.unsorted_segment_sum(
        tf.gather(coefs, plan['basis_ids']) * tf.constant(plan['cg'][:, None], tf.float64),
        plan['segments'], len(plan['nodes']))
    result = evaluate(ar, ai, plan['left'], plan['right'], plan['nodes'], weights)
    return tf.transpose(result, [1, 0])


def enable(share_products=True):
    global _share_products
    from tensorpotential.potentials.ace import ACE
    load()
    _share_products = share_products
    original = ACE.eval_atomic_energy
    source = getattr(original, '_wbc_source', None)
    if source is None:
        source = textwrap.dedent(inspect.getsource(original))
    namespace = dict(getattr(original, '_wbc_globals', original.__globals__))
    old = '''        collect_basis = [[] for _ in range(self.nelements)]
        for i, cent_at in enumerate(collect_basis):
            for k in range(2, self.ranksmax[i] + 1):
                cent_at += [self.compute_bbasis_funcs(a_munlm_r, a_munlm_i, i, k)]
        basis = [tf.concat(cent_at, axis=1) for cent_at in collect_basis]
'''
    old_rho = '            rho = tf.matmul(basis[c_i], coefs) + rho_r1_part[c_i]'
    if source.count(old) != 1 or source.count(old_rho) != 1:
        raise RuntimeError('Upstream ACE basis/density block changed, review the patch')
    source = source.replace(old, '').replace(old_rho,
        '            rho = _wbc_density(self, a_munlm_r, a_munlm_i, c_i, coefs) + rho_r1_part[c_i]')
    namespace['_wbc_density'] = density
    exec(compile(source, '<wbc-native-density>', 'exec'), namespace)
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
