"""Weighted source offsets, profiled gradients and legacy-backend integration."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyace.const import PYACE_EVAL, TENSORPOT_EVAL
from pyace.generalfit import GeneralACEFit
from pyace.preparedata import get_reference_dataset
from pyace.utils.source_energy_offsets import SourceEnergyOffsetManager


DATA = str(Path(__file__).with_name('representative_df.pckl.gzip'))
POTENTIAL = dict(
    deltaSplineBins=.001, element='Al', fs_parameters=[1, 1, 1, .5],
    npot='FinnisSinclairShiftedScaled', NameOfCutoffFunction='cos',
    nradmax_by_orders=[2], lmax_by_orders=[0], ndensity=2, rcut=8.7,
    dcut=.01, radparameters=[5.25], radbase='ChebExpCos',
)


def test_weighted_offsets_and_profiled_gradient():
    nat = np.array([2, 5, 4, 7, 3, 8])
    labels = np.array([0, 1, 2, 1, 0, 2])
    weights = np.array([.1, .2, .05, .3, .15, .2])
    design = np.random.default_rng(17).normal(size=(6, 4))
    target = np.array([1., 100., -37., 102., 2., -36.])
    frame = pd.DataFrame(dict(NUMBER_OF_ATOMS=nat, source=labels,
                              w_energy=weights, energy_corrected=target*nat))
    manager = SourceEnergyOffsetManager(column='source', reference_value=0, enabled=True)

    def objective(coefficients):
        prediction = design @ coefficients
        offsets = manager.fit_offsets(frame, prediction*nat)
        residual = prediction + manager.get_offsets_per_atom(frame) - target
        for label in (1, 2):
            mask = labels == label
            assert np.dot(weights[mask], residual[mask]) == pytest.approx(0., abs=1e-13)
        assert offsets[0] == 0.
        return np.dot(weights, residual**2), 2*design.T @ (weights*residual)

    coefficients = np.array([.2, -.1, .7, .3])
    _, gradient = objective(coefficients)
    h = 1e-4
    finite = np.array([(objective(coefficients+h*direction)[0]
                        - objective(coefficients-h*direction)[0])/(2*h)
                       for direction in np.eye(4)])
    np.testing.assert_allclose(gradient, finite, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('frame', [pd.DataFrame({'source': [None]}), pd.DataFrame()])
def test_missing_labels_fail(frame):
    manager = SourceEnergyOffsetManager(column='source', reference_value=0, enabled=True)
    with pytest.raises(ValueError):
        manager.validate_dataframe(frame)


def dataset(backend, shifted):
    original = get_reference_dataset(backend, DATA).copy()
    frames = []
    for source, shift in ((0, 0.), (1, 100.), (2, -37.5)):
        frame = original.copy()
        frame['source_id'] = source
        frame['energy_corrected'] = (
            frame['energy_corrected'].to_numpy(dtype=float)
            + (shift if shifted else 0.)*frame['NUMBER_OF_ATOMS'])
        frame['energy_corrected_per_atom'] = frame['energy_corrected']/frame['NUMBER_OF_ATOMS']
        # Same nonuniform weights on each source, independent of atom count.
        frame['w_energy'] = np.arange(1., len(frame)+1)
        frame['w_forces'] = frame['NUMBER_OF_ATOMS'].map(lambda n: np.full(n, .01))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fit(dataframe, backend, offsets=True):
    config = dict(optimizer='L-BFGS-B', maxiter=0, options={'gtol': 1e6},
                  loss={'kappa': .5})
    if offsets:
        config['source_energy_offsets'] = dict(column='source_id', reference=0)
    fitter = GeneralACEFit(
        potential_config=copy.deepcopy(POTENTIAL), data_config=dataframe,
        fit_config=config, backend_config=dict(evaluator=backend,
                                              parallel_mode='serial', batch_size=5), seed=42)
    fitter.fit()
    return fitter


@pytest.mark.parametrize('backend', [PYACE_EVAL, pytest.param(TENSORPOT_EVAL, marks=pytest.mark.tensorpot)])
def test_offsets_preserve_objective_under_source_energy_shifts(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    unshifted = fit(dataset(backend, False), backend)
    shifted = fit(dataset(backend, True), backend)
    np.testing.assert_allclose(shifted.fit_backend.res_opt.fun,
                               unshifted.fit_backend.res_opt.fun, rtol=1e-10, atol=1e-10)
    first = unshifted.fit_backend.get_last_source_energy_offsets()
    second = shifted.fit_backend.get_last_source_energy_offsets()
    assert second[0] == 0.
    assert second[1]-first[1] == pytest.approx(100., abs=1e-10)
    assert second[2]-first[2] == pytest.approx(-37.5, abs=1e-10)
    prediction = shifted.fit_backend.predict()
    np.testing.assert_allclose(
        prediction['energy_pred'], np.asarray(prediction['energy_pred_raw'])
        + np.asarray(prediction['energy_source_offset_fit']))
    np.testing.assert_allclose(
        shifted.fit_backend.last_fit_metric_data['rmse_epa'],
        np.sqrt(np.mean(((np.asarray(prediction['energy_pred'])
                         - shifted.fitting_data['energy_corrected'].to_numpy())
                        / shifted.fitting_data['NUMBER_OF_ATOMS'].to_numpy())**2)),
        rtol=1e-10, atol=1e-10)


@pytest.mark.tensorpot
def test_tensorpot_gradient_and_accepted_offsets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = fit(dataset(TENSORPOT_EVAL, True), TENSORPOT_EVAL)
    fitter = model.fit_backend.fitter
    from tensorpotential.utils.utilities import batching_data
    batches = batching_data(fitter.data_df, batch_size=5)
    x = fitter.fit_coefs.copy()
    _, gradient = fitter.fit_func(x, batches)
    direction = np.random.default_rng(67).normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    h = 1e-5
    plus = fitter.fit_func(x+h*direction, batches)[0]
    minus = fitter.fit_func(x-h*direction, batches)[0]
    assert np.dot(gradient, direction) == pytest.approx((plus-minus)/(2*h), rel=2e-5, abs=1e-6)
    fitter.update_last_metrics(x)
    assert model.fit_backend.get_last_source_energy_offsets() == pytest.approx(
        {int(key): value for key, value in fitter.last_fit_metric_data['source_energy_offsets'].items()})


@pytest.mark.tensorpot
def test_test_metrics_use_training_offsets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = fit(dataset(TENSORPOT_EVAL, True), TENSORPOT_EVAL)
    fitter = model.fit_backend.fitter
    from tensorpotential.fitmetrics import FitMetrics
    test = fitter.data_df.iloc[[1, 7, 13]].copy()
    # Deliberately shift only held-out targets; those errors must not fit away.
    test['energy_corrected'] += 2.*test['NUMBER_OF_ATOMS']
    fitter.test_df = test
    fitter.test_metrics = FitMetrics(
        test['w_energy'].to_numpy().reshape(-1, 1),
        np.concatenate(test['w_forces'].tolist()).reshape(-1, 1), .5, .5, fitter.ncoefs)
    before = fitter.last_source_energy_offsets.copy()
    result = fitter.compute_test_metric()
    pred = fitter.predict(test)
    energy = fitter.energy_offset_manager.apply_to_predictions(pred['energy_pred'], test, before)
    expected = np.sqrt(np.mean(((energy-test['energy_corrected'].to_numpy())
                                / test['NUMBER_OF_ATOMS'].to_numpy())**2))
    assert result['rmse_epa'] == pytest.approx(expected)
    assert fitter.last_source_energy_offsets == before
