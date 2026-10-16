"""Energy-only offset predictions match upstream without force/stress work."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.tensorpot


@pytest.mark.parametrize('eager', [False, True])
def test_energy_only_reuses_forward_pass_and_live_coefficients(eager):
    pytest.importorskip('tensorpotential')
    from pyace.basis import BBasisConfiguration
    from pyace.preparedata import get_reference_dataset
    from pyace.const import TENSORPOT_EVAL
    from pyace.utils.tensorpotential_energy import make_energy_only_evaluator
    from tensorpotential.potentials.ace import ACE
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.utils.utilities import batching_data

    root = Path(__file__).parent
    frame = get_reference_dataset(TENSORPOT_EVAL, str(root / 'representative_df.pckl.gzip'))
    batches = batching_data(frame.iloc[:4].copy(), batch_size=3)
    model = BBasisConfiguration(str(root / 'Al-r1l0.yaml'))
    rng = np.random.default_rng(149)
    model.set_func_coeffs(rng.normal(scale=.05, size=len(model.get_func_coeffs())))
    tensorpot = TensorPotential(ACE(model))
    energy_only = make_energy_only_evaluator(tensorpot)
    initial = tensorpot.potential.get_coefs().numpy().copy()
    for shift in (0., .003, -.002):
        tensorpot.potential.set_coefs(initial + shift)
        expected = [tensorpot.evaluate(batch)[0].numpy() for batch in batches]
        methods = ('compute_energy_forces_stress', 'evaluate', 'eager_evaluate', '_evaluate')
        originals = {name: getattr(tensorpot, name) for name in methods}
        def forbidden_derivatives():
            raise AssertionError('Energy-only predictions must not compute forces/stress')
        for name in methods:
            setattr(tensorpot, name, forbidden_derivatives)
        try:
            actual = [energy_only(batch, eager=eager).numpy() for batch in batches]
        finally:
            for name, method in originals.items():
                setattr(tensorpot, name, method)
        for result, reference in zip(actual, expected):
            np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('eager', [False, True])
def test_energy_prepass_preserves_smoothness_loss_and_gradients(eager):
    pytest.importorskip('tensorpotential')
    from pyace.basis import BBasisConfiguration
    from pyace.preparedata import get_reference_dataset
    from pyace.const import TENSORPOT_EVAL
    from pyace.utils.tensorpotential_energy import make_energy_only_evaluator
    from tensorpotential.potentials.ace import ACE
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.constants import (LOSS_TYPE, LOSS_ENERGY_FACTOR,
                                          LOSS_FORCE_FACTOR, AUX_LOSS_FACTOR)
    from tensorpotential.utils.utilities import batching_data

    root = Path(__file__).parent
    frame = get_reference_dataset(TENSORPOT_EVAL, str(root / 'representative_df.pckl.gzip'))
    batch = batching_data(frame.iloc[:2].copy(), batch_size=2)[0]
    model = BBasisConfiguration(str(root / 'Al-r1l0.yaml'))
    specifications = {LOSS_TYPE: 'per-atom', LOSS_ENERGY_FACTOR: .9,
                      LOSS_FORCE_FACTOR: .1, AUX_LOSS_FACTOR: [1e-5, 2e-5, 3e-5]}
    expected = TensorPotential(ACE(model, compute_smoothness=True), loss_specs=specifications)
    actual = TensorPotential(ACE(model, compute_smoothness=True), loss_specs=specifications)
    energy_only = make_energy_only_evaluator(actual)
    coefficients = actual.potential.get_coefs().numpy().copy()
    for shift in (0., .003):
        current = coefficients + shift
        actual.potential.set_coefs(current)
        auxiliary = actual.potential.aux
        predicted = energy_only(batch, eager=eager).numpy()
        assert actual.potential.aux is auxiliary
        assert actual.potential.compute_smoothness
        assert not actual.potential.compute_orthogonality
        reference = expected.external_fit(current, batch)
        result = actual.external_fit(current, batch)
        for candidate, target in zip(result, reference):
            np.testing.assert_allclose(candidate.numpy(), target.numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(predicted, result[2].numpy(), rtol=1e-12, atol=1e-12)
