"""Source offsets for the pinned, unmodified legacy TensorPotential fitter.

Offsets minimize the globally weighted per-atom energy loss at each parameter
vector. At this optimum the offset derivative vanishes, so TensorPotential's
ordinary parameter gradient with fixed adjusted targets is the exact gradient
of the minimized objective. Geometry, forces and regularization are unchanged.
"""

import time

import numpy as np
from tensorpotential.fit import FitTensorPotential


class SourceOffsetTensorPotential(FitTensorPotential):
    def __init__(self, *args, energy_offset_manager, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_offset_manager = energy_offset_manager
        self.last_source_energy_offsets = {}

    @staticmethod
    def _current_targets(dataframe):
        """Do not let cached tp_atoms silently override current dataframe labels."""
        result = dataframe.copy()
        records = []
        for _, row in result.iterrows():
            record = dict(row['tp_atoms'])
            record['_energy'] = np.array([[row['energy_corrected']]], dtype=float)
            record['_forces'] = np.asarray(row['forces'], dtype=float)
            records.append(record)
        result['tp_atoms'] = records
        return result

    def fit(self, df, test_df=None, **kwargs):
        optimizer = kwargs.get('optimizer', 'BFGS')
        if optimizer not in ('BFGS', 'L-BFGS-B', 'CG'):
            raise ValueError('Source offsets currently support BFGS, L-BFGS-B or CG')
        self.energy_offset_manager.validate_dataframe(df, 'fitting dataframe')
        self.energy_offset_manager._get_reference_value(df)
        if test_df is not None:
            self.energy_offset_manager.validate_dataframe(test_df, 'test dataframe')
            unseen = set(test_df[self.energy_offset_manager.column]) - set(
                df[self.energy_offset_manager.column])
            if unseen:
                raise ValueError(f'Test data contain source labels absent from training: {unseen}')
        return super().fit(self._current_targets(df),
                           test_df=None if test_df is None else self._current_targets(test_df),
                           **kwargs)

    def _evaluate_batch_energies(self, coefficients, batches):
        parallel = getattr(self, '_cpu_predict_energies', None)
        if parallel is not None:
            return parallel(coefficients, batches)
        self.tensorpot.potential.set_coefs(coefficients)
        evaluate = self.tensorpot.eager_evaluate if self.eager else self.tensorpot.evaluate
        return np.concatenate([evaluate(batch)[0].numpy().reshape(-1) for batch in batches])

    def fit_func(self, coefficients, batches):
        started = time.perf_counter()
        dataframe = self.data_df
        prediction = self._evaluate_batch_energies(coefficients, batches)
        offsets = self.energy_offset_manager.fit_offsets(dataframe, prediction)
        self.last_source_energy_offsets = offsets.copy()
        adjusted = dataframe.copy()
        adjusted['energy_corrected'] = self.energy_offset_manager.get_adjusted_target_energies(
            dataframe, offsets=offsets)
        adjusted['energy_corrected_per_atom'] = (
            adjusted['energy_corrected'] / adjusted['NUMBER_OF_ATOMS'])
        original_targets = [batch['energy'] for batch in batches]
        original_snapshot = self.metrics.to_FitMetricsDict

        def snapshot():
            self.eval_time = time.perf_counter() - started
            self.metrics.eval_time = self.eval_time
            result = original_snapshot()
            result['source_energy_offsets'] = self.energy_offset_manager.serialize_offsets(offsets)
            return result

        try:
            start = 0
            for batch in batches:
                end = start + len(batch['energy'])
                batch['energy'] = adjusted['energy_corrected'].iloc[start:end].to_numpy().reshape(-1, 1)
                start = end
            if start != len(dataframe):
                raise ValueError('Source-offset batches do not match the training dataframe')
            # Upstream computes its low-energy mask from this adjusted reference.
            # The CPU pool sends the new energy targets on every evaluation.
            self.data_df = adjusted
            self.metrics.to_FitMetricsDict = snapshot
            return super().fit_func(coefficients, batches)
        finally:
            self.data_df = dataframe
            self.metrics.to_FitMetricsDict = original_snapshot
            for batch, energy in zip(batches, original_targets):
                batch['energy'] = energy

    def update_last_metrics(self, coefficients):
        result = super().update_last_metrics(coefficients)
        # A rejected line-search trial must not leave its offsets on the model.
        self.last_source_energy_offsets = self.energy_offset_manager.restore_offsets(
            result['source_energy_offsets'])
        return result

    def predict(self, datadf=None):
        dataframe = self.data_df if datadf is None else datadf
        return super().predict(self._current_targets(dataframe))

    def compute_test_metric(self, updating_coefs=None):
        if self.test_df is None:
            return None
        original = self.test_df
        adjusted = original.copy()
        adjusted['energy_corrected'] = self.energy_offset_manager.get_adjusted_target_energies(
            original, offsets=self.last_source_energy_offsets)
        try:
            # Only training offsets are applied; test labels never fit offsets.
            self.test_df = self._current_targets(adjusted)
            result = super().compute_test_metric(updating_coefs)
            result['source_energy_offsets'] = self.energy_offset_manager.serialize_offsets(
                self.last_source_energy_offsets)
            return result
        finally:
            self.test_df = original
