"""Energy-only predictions for the frozen legacy TensorPotential backend.

Source-offset profiling needs energies, not position/cell derivatives. Reuse
the upstream forward pass and input signature without computing forces/stress
that are discarded by the caller. Do not patch the installed TensorPotential.
"""

import tensorflow as tf
from tensorpotential.utils.utilities import input2eval


def make_energy_only_evaluator(tensorpot):
    """Return a batch evaluator sharing the potential's live coefficients."""
    signature = tensorpot._evaluate.input_signature
    names = tuple(spec.name for spec in signature)

    def forward(*values):
        tensorpot.init_placeholders(**dict(zip(names, values)))
        # Legacy ACE appends auxiliary-loss tensors to a Python list. A
        # prediction trace must not leave its tensors for the fitting graph.
        # These coefficient-only penalties do not affect predicted energies.
        potential = tensorpot.potential
        smoothness = potential.compute_smoothness
        orthogonality = potential.compute_orthogonality
        auxiliary = potential.aux
        try:
            potential.compute_smoothness = False
            potential.compute_orthogonality = False
            return tensorpot.forward_pass(tensorpot.compute_positions())
        finally:
            potential.compute_smoothness = smoothness
            potential.compute_orthogonality = orthogonality
            potential.aux = auxiliary

    graph = tf.function(forward, input_signature=signature, autograph=False)

    def evaluate(batch, *, eager=False):
        return (forward if eager else graph)(*input2eval(batch))

    return evaluate
