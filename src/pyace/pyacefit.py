import logging
import numpy as np
import pandas as pd
import time

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from functools import partial
from typing import Union
from scipy.optimize import minimize

from pyace.basis import ACEBBasisSet, BBasisConfiguration
from pyace.evaluator import ACECTildeEvaluator, ACEBEvaluator
from pyace.calculator import ACECalculator
from pyace.paralleldataexecutor import ParallelDataExecutor, LOCAL_DATAFRAME_VARIALBE_NAME
from pyace.radial import *
from pyace.multispecies_basisextension import expand_trainable_parameters, compute_bbasisset_train_mask
from pyace.lossfuncspec import LossFunctionSpecification
from pyace.const import *
from pyace.metrics_aggregator import FitMetrics, MetricsAggregator
import __main__

required_structures_dataframe_columns = [ATOMIC_ENV_COL, ENERGY_CORRECTED_COL, FORCES_COL]


def batch_compute_energy_forces_function_wrapper(batch_indices, cbasis):
    _local_df = getattr(__main__, LOCAL_DATAFRAME_VARIALBE_NAME)
    batch_df = _local_df.loc[batch_indices]

    ace = ACECalculator()
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)
    ace.set_evaluator(evaluator)

    def pure_row_func(ae):
        ace.compute(ae)
        return ace.energy, ace.forces

    if isinstance(batch_df, pd.Series):
        return batch_df.map(pure_row_func)
    elif isinstance(batch_df, pd.DataFrame):
        return batch_df.apply(pure_row_func, axis=1)


def batch_compute_projections_function_wrapper(batch_indices, potential_params):
    _local_df = getattr(__main__, LOCAL_DATAFRAME_VARIALBE_NAME)
    batch_df = _local_df.loc[batch_indices]

    ace = ACECalculator()
    potential_params.is_sort_functions = False
    if isinstance(potential_params, BBasisConfiguration):
        bbasis = ACEBBasisSet(potential_params)
        evaluator = ACEBEvaluator()
        evaluator.set_basis(bbasis)
    elif isinstance(potential_params, ACECTildeBasisSet):
        evaluator = ACECTildeEvaluator()
        evaluator.set_basis(potential_params)
    elif isinstance(potential_params, ACEBBasisSet):
        evaluator = ACEBEvaluator()
        evaluator.set_basis(potential_params)
    else:
        raise ValueError("Unrecognized `potential_params` type: {}. Should be BBasisConfiguration, ACECTildeBasisSet "
                         "or ACEBBasisSet".format(type(potential_params)))

    ace.set_evaluator(evaluator)

    def pure_row_func(ae):
        ace.compute(ae, compute_projections=True)
        nat = ae.n_atoms_real
        # flatten projections
        pr = ace.projections
        return pr

    if isinstance(batch_df, pd.Series):
        return batch_df.map(pure_row_func)
    elif isinstance(batch_df, pd.DataFrame):
        return batch_df.apply(pure_row_func, axis=1)


class PyACEFit:
    """
    Create a class for fitting ACE potential

    :param basis (BBasisConfiguration, ACEBBasisSet) basis set specification
    :param loss_spec (LossFunctionSpecification)
    """

    def __init__(self, basis: Union[BBasisConfiguration] = None, structures_dataframe: pd.DataFrame = None,
                 loss_spec: LossFunctionSpecification = None, seed=None, executors_kw_args=None, display_step=10,
                 trainable_parameters=None):

        if basis is not None:
            if isinstance(basis, BBasisConfiguration):
                self.bbasis = ACEBBasisSet(basis)
            else:
                raise ValueError(
                    "`basis` argument should be either 'BBasisConfiguration' or 'ACEBBasisSet', but got " + type(basis))

            # blocks = basis.funcspecs_blocks

            # if trainable_parameters is None:
            #     log.info("`trainable_parameters_dict` is not provided, all blocks will be fitted")
            self.elements_name = self.bbasis.elements_name

            self.trainable_parameters_dict = expand_trainable_parameters(self.elements_name, trainable_parameters)
            self.elements_ind_map = {el: i for i, el in enumerate(self.elements_name)}

            self._init_trainable_params()
            self.cbasis = self.bbasis.to_ACECTildeBasisSet()
        else:
            self.bbasis = None
            self.cbasis = None
            self.trainable_params = None

        if loss_spec is None:
            loss_spec = LossFunctionSpecification()
        self.loss_spec = loss_spec
        self.seed = seed

        self._structures_dataframe = None

        self.eval_count = 0
        self.iter_num = 0

        self.best_loss = None
        self.best_params = None
        self.res_opt = None

        self.params_opt = None
        self.bbasis_opt = None
        self.cbasis_opt = None

        self.dataset_id_to_index = {}
        self.dataset_ids = []
        self.dataset_offset_params = np.array([], dtype=float)
        self._basis_param_count = 0
        self._dataset_offset_logged = False
        self._loss_offset_logged = False

        self.data_executor = None
        self.executors_kw_args = executors_kw_args or {}

        self.global_callback = None

        self.initial_loss = None
        self.last_loss = None
        self.last_epa_mae = None
        self.l1 = None
        self.l2 = None
        self.smooth_quad = None
        self._metrics = None
        self.eval_time = None
        self.display_step = display_step
        self.nfuncs = None
        if structures_dataframe is not None:
            self.structures_dataframe = structures_dataframe

        self.fit_metrics_data_dict = {}
        self.last_fit_metric_data = None
        self.last_test_metric_data = None
        self.fit_metric_callback = None

    def _init_trainable_params(self):
        self.trainable_params_mask = compute_bbasisset_train_mask(self.bbasis, self.trainable_parameters_dict)
        self._basis_param_count = int(np.sum(self.trainable_params_mask))
        self._refresh_trainable_params()

    @property
    def structures_dataframe(self):
        return self._structures_dataframe

    @property
    def metrics(self):
        if self._metrics is None:
            self.setup_metrics()
        return self._metrics

    @structures_dataframe.setter
    def structures_dataframe(self, value):
        log.info("structures_dataframe.setter: received value type=%s len=%s", type(value),
                 len(value) if value is not None else None)
        self._structures_dataframe = value  # .copy()
        # self.preprocess_dataframe(self._structures_dataframe)
        log.info("structures_dataframe.setter: calling _update_dataset_offset_info()")
        self._update_dataset_offset_info()
        log.info("structures_dataframe.setter: completed")

    def _refresh_trainable_params(self):
        log.info("_refresh_trainable_params: start")
        if self.bbasis is not None and self.trainable_params_mask is not None:
            basis_params = np.array(self.bbasis.all_coeffs)[self.trainable_params_mask]
        else:
            basis_params = np.array([], dtype=float)
        dataset_params = self.dataset_offset_params if self.dataset_offset_params is not None else np.array([], dtype=float)
        self.trainable_params = np.concatenate((basis_params, dataset_params))
        log.info("_refresh_trainable_params: basis_len=%d dataset_len=%d total_len=%d",
                 len(basis_params), len(dataset_params), len(self.trainable_params))

    def _update_dataset_offset_info(self):
        log.info("_update_dataset_offset_info: start")
        df = self._structures_dataframe
        if df is None or DATASET_ID_COL not in df.columns:
            log.info("_update_dataset_offset_info: dataframe None or missing `%s` column", DATASET_ID_COL)
            self.dataset_id_to_index = {}
            self.dataset_ids = []
            self.dataset_offset_params = np.array([], dtype=float)
            self._refresh_trainable_params()
            log.info("_update_dataset_offset_info early exit")
            return

        dataset_ids = list(pd.unique(df[DATASET_ID_COL]))
        log.info("_update_dataset_offset_info: dataset_ids=%s", dataset_ids)
        previous_offsets = {ds_id: self.dataset_offset_params[idx]
                            for ds_id, idx in self.dataset_id_to_index.items()
                            if idx < len(self.dataset_offset_params)}
        log.info("_update_dataset_offset_info: previous_offsets=%s", previous_offsets)

        energy_pa_series = None
        if E_CORRECTED_PER_ATOM_COLUMN in df.columns:
            energy_pa_series = df[E_CORRECTED_PER_ATOM_COLUMN]
        elif ENERGY_CORRECTED_COL in df.columns:
            if NUMBER_OF_ATOMS in df.columns:
                energy_pa_series = df[ENERGY_CORRECTED_COL] / df[NUMBER_OF_ATOMS]
            elif ASE_ATOMS in df.columns:
                energy_pa_series = df[ENERGY_CORRECTED_COL] / df[ASE_ATOMS].map(len)

        mean_map = {}
        if energy_pa_series is not None:
            for ds_id in dataset_ids:
                mask = df[DATASET_ID_COL] == ds_id
                if np.any(mask):
                    mean_map[ds_id] = float(energy_pa_series[mask].mean())
        log.info("_update_dataset_offset_info: mean_map=%s", mean_map)

        if mean_map:
            base_id = min(mean_map, key=lambda k: abs(mean_map[k]))
            base_mean = mean_map[base_id]
        else:
            base_id = dataset_ids[0] if dataset_ids else None
            base_mean = mean_map.get(base_id, 0.0) if base_id is not None else 0.0
        log.info("_update_dataset_offset_info: base_id=%s base_mean=%s", base_id, base_mean)

        new_offsets = np.zeros(len(dataset_ids), dtype=float)
        for idx, ds_id in enumerate(dataset_ids):
            if ds_id in previous_offsets:
                new_offsets[idx] = previous_offsets[ds_id]
            else:
                new_offsets[idx] = base_mean - mean_map.get(ds_id, 0.0)

        self.dataset_id_to_index = {ds_id: idx for idx, ds_id in enumerate(dataset_ids)}
        self.dataset_ids = dataset_ids
        self.dataset_offset_params = new_offsets
        log.info("Dataset offset setup: ids=%s, previous=%s, mean_map=%s, base_id=%s, base_mean=%s, offsets=%s",
                 dataset_ids, previous_offsets, mean_map, base_id, base_mean, new_offsets.tolist())
        if len(dataset_ids) > 1:
            offset_map = {ds_id: self.dataset_offset_params[idx] for idx, ds_id in enumerate(dataset_ids)}
            if mean_map:
                log.info("Detected multiple datasets with ids=%s (base id=%s)", dataset_ids, base_id)
                log.info("Per-atom energy means by dataset: %s", mean_map)
            else:
                log.info("Detected multiple datasets with ids=%s (base id=%s)", dataset_ids, base_id)
            log.info("Initial per-atom dataset offsets: %s", offset_map)
        self._dataset_offset_logged = False
        self._loss_offset_logged = False
        self._refresh_trainable_params()
        log.info("_update_dataset_offset_info: finished offsets now %s", self.dataset_offset_params.tolist())

    def preprocess_dataframe(self, structures_dataframe):
        log.info("preprocess_dataframe: start len=%s columns=%s",
                 len(structures_dataframe) if structures_dataframe is not None else None,
                 list(structures_dataframe.columns) if structures_dataframe is not None else None)
        # TODO: energies and forces weights are generated here, if columns not provided
        for col in required_structures_dataframe_columns:
            if col not in structures_dataframe.columns:
                raise ValueError("`structures_dataframe` doesn't contain column {}".format(col))

        if FORCES_COL in structures_dataframe.columns:
            structures_dataframe[FORCES_COL] = structures_dataframe[FORCES_COL].map(np.array)
        if FWEIGHTS_COL in structures_dataframe.columns:
            structures_dataframe[FWEIGHTS_COL] = structures_dataframe[FWEIGHTS_COL].map(np.array)

        # normalize_energy_forces_weights(structures_dataframe)
        log.info("preprocess_dataframe: completed")

    def _get_dataset_energy_shifts(self, dataframe):
        log.info("_get_dataset_energy_shifts: start len=%s", len(dataframe))
        if not len(self.dataset_offset_params) or DATASET_ID_COL not in dataframe.columns:
            log.info("_get_dataset_energy_shifts: no dataset offsets or missing %s column", DATASET_ID_COL)
            return np.zeros(len(dataframe))
        if NUMBER_OF_ATOMS in dataframe.columns:
            natoms = dataframe[NUMBER_OF_ATOMS].to_numpy()
        elif ASE_ATOMS in dataframe.columns:
            natoms = dataframe[ASE_ATOMS].map(len).to_numpy()
        else:
            raise ValueError(
                f"Either `{NUMBER_OF_ATOMS}` or `{ASE_ATOMS}` column is required to apply dataset energy offsets")
        dataset_ids = dataframe[DATASET_ID_COL].to_numpy()
        shifts = np.zeros(len(dataframe))
        for idx, ds_id in enumerate(dataset_ids):
            offset_idx = self.dataset_id_to_index.get(ds_id)
            if offset_idx is not None and offset_idx < len(self.dataset_offset_params):
                shifts[idx] = self.dataset_offset_params[offset_idx] * natoms[idx]
        if len(self.dataset_offset_params) and not self._dataset_offset_logged:
            per_atom_offsets = {ds_id: self.dataset_offset_params[self.dataset_id_to_index[ds_id]]
                                for ds_id in self.dataset_ids}
            log.info("Applying dataset per-atom offsets at prediction: %s", per_atom_offsets)
            self._dataset_offset_logged = True
        log.info("_get_dataset_energy_shifts: returning shifts=%s", shifts.tolist())
        return shifts

    def get_dataset_offsets(self):
        """Return dictionary mapping dataset ids to per-atom energy offsets."""
        return {ds_id: self.dataset_offset_params[idx]
                for idx, ds_id in enumerate(self.dataset_ids)}

    def update_bbasis(self, params):
        params = np.asarray(params, dtype=float)
        basis_param_count = self._basis_param_count
        if params.ndim != 1:
            params = params.flatten()
        if len(params) < basis_param_count:
            raise ValueError(
                "update_bbasis::number of provided parameters ({}) is less than required basis params ({})"
                .format(len(params), basis_param_count))

        basis_params = params[:basis_param_count]
        dataset_params = params[basis_param_count:]
        np_array = np.array(self.bbasis.all_coeffs)
        np_array[self.trainable_params_mask] = basis_params
        self.bbasis.all_coeffs = np_array
        self.dataset_offset_params = np.array(dataset_params, dtype=float)
        self._refresh_trainable_params()
        return self.bbasis

    def get_cbasis(self, params):
        self.bbasis = self.update_bbasis(params)
        self.cbasis = self.bbasis.to_ACECTildeBasisSet()
        return self.cbasis

    def loss(self, params=None, verbose=False):
        if params is None:
            params = self.trainable_params

        self.eval_count += 1
        log.info("loss: start eval=%s params_len=%s", self.eval_count, len(params))
        if not getattr(self, "_loss_offset_logged", False) and len(self.dataset_offset_params):
            log.info("Current dataset offsets before loss evaluation: %s", self.get_dataset_offsets())
            self._loss_offset_logged = True
        t0 = time.time()
        energy_forces_pred_df = self.predict_energy_forces(params, keep_parallel_dataexecutor=True)

        total_na = self.structures_dataframe["NUMBER_OF_ATOMS"].values
        dE = (energy_forces_pred_df[ENERGY_PRED_COL] - self.structures_dataframe[ENERGY_CORRECTED_COL]).values
        dE_per_atom = dE / total_na
        dF = (self.structures_dataframe[FORCES_COL] - energy_forces_pred_df[FORCES_PRED_COL]).values

        self.last_epa_mae = np.mean(np.abs(dE_per_atom))

        # de = dE #np.hstack(dE.tolist())
        # de_pa = dE_per_atom #np.hstack(dE_per_atom.tolist())
        df = np.vstack(dF)  # np.vstack([v.reshape(-1, 3) for v in dF.tolist()])
        self.metrics.compute_metrics(dE.reshape(-1, 1), dE_per_atom.reshape(-1, 1), df,
                                     total_na, dataframe=self.structures_dataframe)

        if self.loss_spec.kappa < 1:
            # dEsqr = dE ** 2
            dEsqr = dE_per_atom ** 2
            if EWEIGHTS_COL in self.structures_dataframe.columns:
                dEsqr = dEsqr * np.vstack(self.structures_dataframe[EWEIGHTS_COL]).reshape(-1)  # structure-wise
            e_loss = np.sum(dEsqr)
        else:
            e_loss = 0

        if self.loss_spec.kappa > 0:  # forces have contribution to loss function
            dFsqr = (self.structures_dataframe[FORCES_COL] - energy_forces_pred_df[FORCES_PRED_COL])
            dFsqr = dFsqr.map(lambda f: np.sum(f ** 2, axis=1))
            if FWEIGHTS_COL in self.structures_dataframe.columns:
                dFsqr = dFsqr * self.structures_dataframe[FWEIGHTS_COL]
            dFsqr = dFsqr.map(np.sum)

            f_loss = np.sum(dFsqr)
        else:  # forces have no contribution to loss function
            f_loss = 0

        basis_coeffs = np.array(self.bbasis.basis_coeffs)
        self.l1 = np.sum(np.abs(basis_coeffs))
        self.l2 = np.sum(basis_coeffs ** 2)

        loss_coeff = \
            self.loss_spec.L1_coeffs * self.l1 + self.loss_spec.L2_coeffs * self.l2

        loss_crad = 0
        if self.loss_spec.w0_rad > 0 or self.loss_spec.w1_rad > 0 or self.loss_spec.w2_rad > 0:
            smothness = RadialFunctionSmoothness(RadialFunctionsValues(self.bbasis))
            self.smooth_quad = smothness.smooth_quad
            loss_crad = self.loss_spec.w0_rad * self.smooth_quad[0] + \
                        self.loss_spec.w1_rad * self.smooth_quad[1] + \
                        self.loss_spec.w2_rad * self.smooth_quad[2]

        self.last_loss = (1 - self.loss_spec.kappa) * e_loss + self.loss_spec.kappa * f_loss + \
                         loss_coeff + loss_crad
        self.eval_time = time.time() - t0

        if self.best_loss is None or self.last_loss < self.best_loss:
            self.best_loss = self.last_loss
            self.best_params = params
        # collect all relevant info into FitMetrics
        self.metrics.regs = self.get_reg_components()
        self.metrics.reg_weights = self.get_reg_weights()
        # already computed above
        # self.metrics.compute_metrics(dE.reshape(-1, 1), dE_per_atom.reshape(-1, 1), df,
        #                              total_na, dataframe=self.structures_dataframe)
        self.metrics.loss = self.last_loss
        self.metrics.eval_time = self.eval_time
        # do a snapshot of  FitMetrics into FitMetricsData: loss, e_loss, f_loss, reg_loss, RMSE, MAE, MAX_E (E,F), timing
        curr_fit_metrics_data = self.metrics.to_FitMetricsDict()
        curr_fit_metrics_data["eval_count"] = self.eval_count
        curr_fit_metrics_data["iter_num"] = self.iter_num
        # store metrics_data into dict (x-> curr_fit_metrics_data)
        self.fit_metrics_data_dict[hash(params.tobytes())] = curr_fit_metrics_data
        self.last_fit_metric_data = curr_fit_metrics_data

        if verbose:
            print("Eval {}: loss={}".format(self.eval_count, self.last_loss) + " " * 40 + "\r", end="")

        return self.last_loss

    def predict_energy_forces(self, params=None,
                              structures_dataframe=None,
                              keep_parallel_dataexecutor=False):
        if params is not None:
            if isinstance(params, (list, tuple, np.ndarray)):
                self.cbasis = self.get_cbasis(params)
            elif isinstance(params, ACECTildeBasisSet):
                self.cbasis = params
            elif isinstance(params, ACEBBasisSet):
                self.bbasis = params
                self.cbasis = self.bbasis.to_ACECTildeBasisSet()
            else:
                raise ValueError(
                    "Type of parameters could be only np.array, list, tuple, ACECTildeBasisSet, ACEBBasisSet" +
                    "but got {}".format(type(params)))

        is_structures_dataframe_refreshed = False
        if structures_dataframe is not None:
            self.structures_dataframe = structures_dataframe
            is_structures_dataframe_refreshed = True

        par = partial(batch_compute_energy_forces_function_wrapper, cbasis=self.cbasis)

        self._initialize_executor(create_new=is_structures_dataframe_refreshed)
        energy_forces_pred = self.data_executor.map(wrapped_pure_func=par)
        if not keep_parallel_dataexecutor:
            self.data_executor.stop_executor()
        energy_forces_pred_df = pd.DataFrame({ENERGY_PRED_COL: energy_forces_pred.map(lambda d: d[0]),
                                              FORCES_PRED_COL: energy_forces_pred.map(lambda d: np.array(d[1]))},
                                             index=energy_forces_pred.index)
        if len(energy_forces_pred_df) and len(self.dataset_offset_params):
            df = self.structures_dataframe.loc[energy_forces_pred_df.index]
            energy_shifts = self._get_dataset_energy_shifts(df)
            energy_forces_pred_df[ENERGY_PRED_COL] = energy_forces_pred_df[ENERGY_PRED_COL].to_numpy() + energy_shifts

        return energy_forces_pred_df

    def predict(self, structures_dataframe=None):
        return self.predict_energy_forces(structures_dataframe=structures_dataframe)

    def predict_projections(self, params=None,
                            structures_dataframe=None,
                            keep_parallel_dataexecutor=False):
        is_structures_dataframe_refreshed = False
        if structures_dataframe is not None:
            self.structures_dataframe = structures_dataframe
            is_structures_dataframe_refreshed = True

        potential_params = None
        if params is not None:
            if isinstance(params, (list, tuple, np.ndarray)):
                self.cbasis = self.get_cbasis(params)
                potential_params = self.cbasis
            elif isinstance(params, ACECTildeBasisSet):
                self.cbasis = params
                potential_params = self.cbasis
            elif isinstance(params, ACEBBasisSet):
                self.bbasis = params
                potential_params = self.bbasis
            elif isinstance(params, BBasisConfiguration):
                potential_params = params
            else:
                raise ValueError(
                    "Type of parameters could be only np.array, list, tuple, ACECTildeBasisSet, ACEBBasisSet" +
                    "but got {}".format(type(params)))
        elif self.bbasis is not None:
            log.info("No 'params' provided to predict_projections, bbasis will be used")
            potential_params = self.bbasis
        else:
            raise ValueError(
                "Basis is not set. provide `params` argument or create PyACEFit with some predefined basis")

        par = partial(batch_compute_projections_function_wrapper, potential_params=potential_params)

        self._initialize_executor(create_new=is_structures_dataframe_refreshed)
        projections_pred_df = self.data_executor.map(wrapped_pure_func=par)
        if not keep_parallel_dataexecutor:
            self.data_executor.stop_executor()

        return projections_pred_df

    def get_reg_components(self):
        if self.smooth_quad is not None:
            return [self.l1, self.l2, self.smooth_quad[0], self.smooth_quad[1], self.smooth_quad[2]]
        else:
            return [self.l1, self.l2, 0., 0., 0.]

    def get_reg_weights(self):
        return [self.loss_spec.L1_coeffs, self.loss_spec.L2_coeffs,
                self.loss_spec.w0_rad, self.loss_spec.w1_rad, self.loss_spec.w2_rad]

    def get_fitting_data(self):
        return self.structures_dataframe

    def _callback(self, x, *args, **kwargs):
        self.iter_num += 1

        # call global callback
        self.metrics.record_time(self.eval_time)
        self.update_last_metrics(x)

        if self.fit_metric_callback is not None:
            # example of fit_metric_callback is:  print_extended_metrics
            self.fit_metric_callback(self.last_fit_metric_data)
        else:
            if self.iter_num % self.display_step == 0:
                MetricsAggregator.print_extended_metrics(self.last_fit_metric_data)
            else:
                MetricsAggregator.print_detailed_metrics(self.last_fit_metric_data)

        # clean self.fit_metrics_data_dict={} for next iteration
        # self.fit_metrics_data_dict = {}

        if self.global_callback is not None:
            self.global_callback(self.bbasis.to_BBasisConfiguration())

    def fit(self, structures_dataframe, method="Nelder-Mead",
            options=None, callback=None, verbose=True, fit_metric_callback=None):

        if options is None:
            options = {"maxiter": 100, "disp": True}

        if fit_metric_callback is not None:
            self.fit_metric_callback = fit_metric_callback

        if structures_dataframe is not None:
            self.preprocess_dataframe(structures_dataframe)
            self.structures_dataframe = structures_dataframe
        else:
            raise ValueError("structures_dataframe couldn't be None")

        self.global_callback = callback
        log.info("Data size:" + str(self.structures_dataframe.shape))
        # log.debug("self.structures_dataframe.columns = " + str(self.structures_dataframe.columns))
        log.info("Energy weights : " + str(EWEIGHTS_COL in self.structures_dataframe.columns))
        log.info("Forces weights : " + str(FWEIGHTS_COL in self.structures_dataframe.columns))
        self.eval_count = 0
        self.best_loss = None

        log.info('Number of trainable parameters: {0}'.format(len(self.trainable_params)))

        # self.setup_metrics() # no need, because .metrics is a property with auto initialization

        self._initialize_executor()
        if 'disp' not in options:
            options['disp'] = True
        if 'gtol' not in options:
            options['gtol'] = 1e-8
        log.info('Scipy minimize: method = {},  options = {}'.format(method, options))

        self.initial_loss = self.loss(self.trainable_params)
        if self.fit_metric_callback is not None:
            self.fit_metric_callback(self.last_fit_metric_data)
        else:
            MetricsAggregator.print_detailed_metrics(self.last_fit_metric_data, title='Initial state:')
            MetricsAggregator.print_extended_metrics(self.last_fit_metric_data, title='INIT STATS')
        self.fit_metrics_data_dict = {}
        res_opt = minimize(self.loss, x0=self.trainable_params, args=(verbose,), method=method, options=options,
                           callback=self._callback)

        self.res_opt = res_opt

        self.params_opt = res_opt.x
        self.bbasis_opt = self.update_bbasis(self.params_opt)
        self.cbasis_opt = self.bbasis.to_ACECTildeBasisSet()

        x = res_opt.x
        self.update_last_metrics(x)
        # TODO: call self.fit_metric_callback one more time?

    def update_last_metrics(self, x):
        x_hash = hash(x.tobytes())
        true_fit_metric_data = self.fit_metrics_data_dict[x_hash]
        true_fit_metric_data["iter_num"] = self.iter_num
        true_fit_metric_data["eval_count"] = self.eval_count
        # update self.metrics with this true metrics data
        self.metrics.from_FitMetricsDict(true_fit_metric_data)
        self.last_fit_metric_data = true_fit_metric_data
        return true_fit_metric_data

    def setup_metrics(self):
        w_e = np.hstack(self.structures_dataframe[EWEIGHTS_COL].tolist())
        w_f = np.hstack(self.structures_dataframe[FWEIGHTS_COL].tolist())
        self._metrics = FitMetrics(w_e.reshape(-1, 1), w_f.reshape(-1, 1), 1. - self.loss_spec.kappa,
                                   self.loss_spec.kappa, len(self.trainable_params))
        self._metrics.nfuncs = self.nfuncs

    def _initialize_executor(self, create_new=False):
        if self.data_executor is None or create_new:
            self.data_executor = ParallelDataExecutor(distributed_data=self.structures_dataframe[ATOMIC_ENV_COL],
                                                      **self.executors_kw_args)
