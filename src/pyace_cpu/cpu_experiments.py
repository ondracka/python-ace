"""Opt-in, process-local evaluator experiments. No installed defaults change."""

import numpy as np
import tensorflow as tf


def nested_grad(self):
    """Differentiate energy w.r.t. positions, then loss w.r.t. coefficients."""
    pos = self.compute_positions()
    self.fit_coefs = self.potential.fit_coefs
    with tf.GradientTape(watch_accessed_variables=False) as outer:
        outer.watch(self.fit_coefs)
        with tf.GradientTape(watch_accessed_variables=False) as inner:
            inner.watch(pos)
            energy = self.forward_pass(pos)
        forces = -tf.convert_to_tensor(inner.gradient(energy, pos))
        loss = self.compute_loss(energy, forces)
    gradient = outer.gradient(loss, self.fit_coefs)
    return loss, gradient, energy, forces


def directional_grad(self):
    """Force-loss parameter gradient via reverse-over-forward differentiation.

First obtain the actual forces. Their loss residual defines a fixed direction
in position space. Differentiating the energy JVP with respect to parameters
then supplies the mixed derivative, without differentiating a full reverse
position-gradient graph. This is the same gradient, not a Gauss-Newton or
finite-difference approximation.
"""
    from tensorpotential.constants import LOSS_FORCE_FACTOR
    pos = self.compute_positions()
    self.fit_coefs = self.potential.fit_coefs
    with tf.GradientTape(watch_accessed_variables=False) as positions_tape:
        positions_tape.watch(pos)
        energy = self.forward_pass(pos)
    forces = -tf.convert_to_tensor(positions_tape.gradient(energy, pos))
    tangent = tf.stop_gradient(-2 * self.loss_specs[LOSS_FORCE_FACTOR]
                              * self.fweights * (forces - self.forces))
    with tf.GradientTape(watch_accessed_variables=False) as parameters_tape:
        parameters_tape.watch(self.fit_coefs)
        with tf.autodiff.ForwardAccumulator(pos, tangent) as forward:
            energy_again = self.forward_pass(pos)
        # The force term is constant here. Its actual parameter derivative is
        # supplied by the directional-energy term instead.
        loss = self.compute_loss(energy_again, tf.stop_gradient(forces))
        surrogate = loss + tf.reduce_sum(forward.jvp(energy_again))
    gradient = parameters_tape.gradient(surrogate, self.fit_coefs)
    return loss, gradient, energy, forces


def vector_bessel(x, rc, deg):
    """Same SBessel family, vectorized sincs and constant recurrence matrix."""
    count = max(1, deg)
    n = np.arange(count, dtype=np.float64)
    transform = np.eye(count, dtype=np.float64)
    dprev = 1.0
    for i in range(1, count):
        en = i ** 2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1)
        dn = 1.0 - en / dprev
        transform[i, i] = 1.0 / np.sqrt(dn)
        transform[i] += np.sqrt(en / dprev / dn) * transform[i - 1]
        dprev = dn
    k = tf.constant(np.arange(1, count + 2, dtype=np.float64)[None, :])
    phase = x * k * tf.constant(np.pi, tf.float64) / rc
    sinc = tf.where(phase != 0, tf.sin(phase) / phase, tf.ones_like(phase))
    factors = ((-1.) ** n * np.sqrt(2.) * np.pi * (n + 1) * (n + 2)
               / np.sqrt((n + 1) ** 2 + (n + 2) ** 2))
    f = (sinc[:, :-1] + sinc[:, 1:]) * tf.constant(factors[None, :]) / rc ** 1.5
    return tf.matmul(f, tf.constant(transform.T))


def bbasis_unsorted(self, a_munlm_r, a_munlm_i, cntrl_at, rank):
    """Upstream basis products, with the equivalent XLA-supported reduction."""
    a_r = tf.transpose(a_munlm_r[cntrl_at], [1, 2, 3, 0])
    a_i = tf.transpose(a_munlm_i[cntrl_at], [1, 2, 3, 0])
    config = self.config.central_atom[cntrl_at][rank - 2]
    first_r = self.flat_gather_nd(a_r, config.munlm[0])
    first_i = self.flat_gather_nd(a_i, config.munlm[0])
    product_r, product_i = first_r, first_i
    for k in range(1, rank):
        next_r = self.flat_gather_nd(a_r, config.munlm[k])
        next_i = self.flat_gather_nd(a_i, config.munlm[k])
        product_r, product_i = self.complexmul(product_r, product_i, next_r, next_i)
    weighted = product_r * tf.convert_to_tensor(config.genCG, tf.float64)
    nsegments = int(config.msum[-1]) + 1 if len(config.msum) else 0
    return tf.transpose(tf.math.unsorted_segment_sum(weighted, config.msum, nsegments), [1, 0])


def shared_bbasis(self, a_munlm_r, a_munlm_i, cntrl_at, rank):
    """Compute each distinct ordered prefix product only once within a rank.

The factor order and Clebsch-Gordan summation are unchanged. Only identical
intermediate products are shared. Plans depend on the basis, not coefficients
or atomic geometry, so ordinary autodiff still supplies all derivatives.
"""
    config = self.config.central_atom[cntrl_at][rank - 2]
    if not hasattr(self, '_prefix_plans'):
        self._prefix_plans = {}
    key = (cntrl_at, rank)
    if key not in self._prefix_plans:
        factors = np.stack(config.munlm, axis=1)
        plans = []
        previous_inverse = None
        for k in range(rank):
            _, rows, inverse = np.unique(factors[:, :k + 1].reshape(len(factors), -1),
                                         axis=0, return_index=True, return_inverse=True)
            feature = factors[rows, k]
            flat = ((feature[:, 0] * self.nradmax + feature[:, 1])
                    * (self.lmax + 1) ** 2 + feature[:, 2]).astype(np.int32)
            previous = None if k == 0 else previous_inverse[rows].astype(np.int32)
            plans.append((previous, flat))
            previous_inverse = inverse
        self._prefix_plans[key] = (plans, previous_inverse.astype(np.int32))
    plans, inverse = self._prefix_plans[key]
    nfeatures = self.nelements * self.nradmax * (self.lmax + 1) ** 2
    natoms = tf.shape(a_munlm_r[cntrl_at])[0]
    a_r = tf.reshape(tf.transpose(a_munlm_r[cntrl_at], [1, 2, 3, 0]), [nfeatures, natoms])
    a_i = tf.reshape(tf.transpose(a_munlm_i[cntrl_at], [1, 2, 3, 0]), [nfeatures, natoms])
    for previous, flat in plans:
        next_r = tf.gather(a_r, flat)
        next_i = tf.gather(a_i, flat)
        if previous is None:
            product_r, product_i = next_r, next_i
        else:
            product_r, product_i = self.complexmul(tf.gather(product_r, previous),
                                                 tf.gather(product_i, previous), next_r, next_i)
    weighted = tf.gather(product_r, inverse) * tf.convert_to_tensor(config.genCG, tf.float64)
    return tf.transpose(tf.math.segment_sum(weighted, config.msum), [1, 0])


def apply(names):
    """Call before tracing, in each parent/worker process."""
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.functions import radial_functions
    from tensorpotential.potentials.ace import ACE
    def fused():
        from fused_product import enable
        enable()
    def native_basis(block_size=1):
        from native_basis import enable
        enable(block_size=block_size)
    def native_moments(reorder=False, radial=False, share_radial=False):
        from native_moments import enable
        enable(reorder=reorder, radial=radial, share_radial=share_radial)
    def native_angular(split=False):
        from native_angular import enable
        enable(split=split)
    def prune_moments():
        from native_moments import enable_output_pruning
        enable_output_pruning()
    def reuse_radial():
        from native_radial import enable_derivative_reuse
        enable_derivative_reuse()
    choices = {'nested': lambda: setattr(TensorPotential, 'compute_grad', nested_grad),
               'directional': lambda: setattr(TensorPotential, 'compute_grad', directional_grad),
               'shared-products': lambda: setattr(ACE, 'compute_bbasis_funcs', shared_bbasis),
               'fused-products': fused,
               'native-bbasis': native_basis,
               'native-bbasis-block4': lambda: native_basis(block_size=4),
               'native-bbasis-block8': lambda: native_basis(block_size=8),
               'native-moments': native_moments,
               'reordered-moments': lambda: native_moments(reorder=True),
               'reordered-native-radial': lambda: native_moments(reorder=True, radial=True),
               'reordered-shared-radial': lambda: native_moments(reorder=True, radial=True, share_radial=True),
               'native-angular': native_angular,
               'native-angular-split': lambda: native_angular(split=True),
               'prune-moment-outputs': prune_moments,
               'reuse-radial-derivatives': reuse_radial,
               'radial': lambda: setattr(radial_functions, 'simplified_bessel', vector_bessel),
               'radial-jit': lambda: setattr(radial_functions, 'simplified_bessel',
                   tf.function(vector_bessel, jit_compile=True, autograph=False,
                               reduce_retracing=True)),
               'bbasis-jit': lambda: setattr(ACE, 'compute_bbasis_funcs',
                   tf.function(ACE.compute_bbasis_funcs, jit_compile=True,
                               autograph=False, reduce_retracing=True)),
               'bbasis-jit-unsorted': lambda: setattr(ACE, 'compute_bbasis_funcs',
                   tf.function(bbasis_unsorted, jit_compile=True,
                               autograph=False, reduce_retracing=True))}
    for name in filter(None, names.split(',')):
        if name not in choices:
            raise ValueError(f'Unknown CPU experiment: {name}')
        choices[name]()
