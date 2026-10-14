#!/usr/bin/env python3
"""Check compact density derivatives and output pruning without private data."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(args.threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    import numpy as np
    import tensorflow as tf
    from pyace_cpu import native_density

    tf.get_logger().setLevel('ERROR')
    module = native_density.load()
    rng = np.random.default_rng(422)
    # Input terms, intermediate terms, repeated factors and terminal products.
    left = np.array([0, 1, 7, 8, 9, 10], np.int32)
    right = np.array([0, 2, 3, 8, 4, 5], np.int32)
    terms = np.array([0, 7, 8, 9, 10, 11, 12], np.int32)
    comparisons, maximum = 0, 0.

    def check(expected, actual):
        nonlocal comparisons, maximum
        for old, new in zip(expected, actual):
            old, new = np.asarray(old), np.asarray(new)
            np.testing.assert_allclose(new, old, rtol=2e-12, atol=2e-12)
            if old.size:
                maximum = max(maximum, float(np.max(np.abs(new - old))))
        comparisons += 1

    def reference(ar, ai, weights):
        real, imag = list(tf.unstack(ar)), list(tf.unstack(ai))
        for l, r in zip(left, right):
            real.append(real[l] * real[r] - imag[l] * imag[r])
            imag.append(imag[r] * real[l] + imag[l] * real[r])
        return tf.matmul(weights, tf.stack([real[t] for t in terms]), transpose_a=True)

    def derivatives(fn, ar, ai, weights, seed, hr, hi, hw):
        values = [ar, ai, weights]
        with tf.GradientTape() as outer:
            outer.watch(values + [seed])
            with tf.GradientTape() as inner:
                inner.watch(values)
                value = fn(ar, ai, weights)
                objective = tf.reduce_sum(value * seed)
            gradient = inner.gradient(objective, values)
            directional = sum(tf.reduce_sum(g * h)
                              for g, h in zip(gradient, (hr, hi, hw)))
        second = outer.gradient(directional, values + [seed],
                                unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return [value, *gradient, *second]

    # Exercise the usual two-channel embedding and other channel counts.
    for channels in (1, 2, 3):
        for atoms in (0, 1, 4, 5, 17):
            ar, ai, hr, hi = [tf.constant(rng.normal(scale=.2, size=(7, atoms)), tf.float64)
                              for _ in range(4)]
            weights, hw = [tf.constant(rng.normal(size=(7, channels)), tf.float64)
                           for _ in range(2)]
            seed = tf.constant(rng.normal(size=(channels, atoms)), tf.float64)

            def candidate(ar, ai, weights):
                return native_density.evaluate(ar, ai, left, right, terms, weights,
                                               compact_workspace=True)

            inputs = (ar, ai, weights, seed, hr, hi, hw)
            expected = derivatives(reference, *inputs)
            check(expected, derivatives(candidate, *inputs))
            graph = tf.function(lambda *x: derivatives(candidate, *x), autograph=False)
            check(expected, graph(*inputs))

            for mode in (1, 2):
                subsets = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
                if mode == 2:
                    subsets += [(3,), (2, 3), (0, 1, 2, 3)]
                for zero in ((False, True) if mode == 2 else (False,)):
                    direction = tf.zeros_like(hw) if zero else hw
                    inputs = (ar, ai, left, right, terms, weights, seed)
                    # Uncompressed native kernel is an independent reference.
                    expected = (module.wbc_density_grad(*inputs) if mode == 1 else
                                module.wbc_density_grad_grad(*inputs, hr, hi, direction))
                    for selected in subsets:
                        @tf.function(autograph=False)
                        def evaluate(ar, ai, weights, seed, hr, hi, hw):
                            inputs = (ar, ai, left, right, terms, weights, seed)
                            outputs = (module.wbc_density_grad(*inputs, prune_unused=True,
                                           compact_workspace=True) if mode == 1 else
                                       module.wbc_density_grad_grad(*inputs, hr, hi, hw,
                                           prune_unused=True, compact_workspace=True,
                                           zero_weight_direction=zero))
                            return tuple(outputs[i] for i in selected)

                        check([expected[i] for i in selected],
                              evaluate(ar, ai, weights, seed, hr, hi, direction))
    print(json.dumps(dict(status='passed', threads=args.threads,
                          comparisons=comparisons, max_absolute_difference=maximum)))


if __name__ == '__main__':
    main()
