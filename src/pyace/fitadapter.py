import logging
import numpy as np
import pandas as pd
import warnings

from typing import Dict, Union, Callable

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from pyace.const import *
from pyace.basis import BBasisConfiguration, ACEBBasisSet

from pyace.multispecies_basisextension import compute_bbasisset_train_mask, expand_trainable_parameters
from pyace.lossfuncspec import LossFunctionSpecification


class BackendConfig:
    def __init__(self, backend_config_dict: Dict):
        self.backend_config_dict = backend_config_dict
        self.validate()

    @property
    def evaluator_name(self):
        return self.backend_config_dict[BACKEND_EVALUATOR_KW]

    def __getitem__(self, item):
        return self.backend_config_dict[item]

    def __setitem__(self, key, value):
        self.backend_config_dict[key] = value

    def validate(self):
        pass

    def get(self, item, default_value=None):
        return self.backend_config_dict.get(item, default_value)


class FitBackendAdapter:

    def __init__(self, backend_config: Union[Dict, BackendConfig], loss_spec: LossFunctionSpecification = None,
                 fit_config: Dict = None, callback: Callable = None, fit_metrics_callback: Callable = None,
                 test_metrics_callback: Callable = None):
        if isinstance(backend_config, dict):
            self.backend_config = BackendConfig(backend_config)
        else:
            self.backend_config = backend_config
        self.callback = callback
        self.loss_spec = loss_spec
        self.fit_config = fit_config
        self.res_opt = None
        self.fitter = None
        self.metrics = None
        self.fit_metrics_callback = fit_metrics_callback
        self.test_metrics_callback = test_metrics_callback
        self.tensorpot_dataset_offsets = None

    @property
    def evaluator_name(self):
        return self.backend_config.evaluator_name

    def _compute_dataset_offsets(self, dataframe: pd.DataFrame):
        """Return per-dataset energy-per-atom offsets relative to the dataset whose mean energy is closest to zero.

        Offsets are computed as (mean_energy_dataset - mean_energy_base) so that adding the offset (per atom)
        to the base prediction recovers the dataset energy."""
        if dataframe is None or DATASET_ID_COL not in dataframe.columns:
            return None, None, None

        if NUMBER_OF_ATOMS in dataframe.columns:
            natoms = dataframe[NUMBER_OF_ATOMS].to_numpy()
        elif ASE_ATOMS in dataframe.columns:
            natoms = dataframe[ASE_ATOMS].map(len).to_numpy()
            dataframe = dataframe.copy()
            dataframe[NUMBER_OF_ATOMS] = natoms
        else:
            raise ValueError(
                f"TensorFlow backend requires `{NUMBER_OF_ATOMS}` or `{ASE_ATOMS}` column when dataset ids are used")

        energy_per_atom = dataframe[ENERGY_CORRECTED_COL].to_numpy() / dataframe[NUMBER_OF_ATOMS].to_numpy()
        dataset_ids = dataframe[DATASET_ID_COL].to_numpy()

        mean_map = {}
        for ds_id in np.unique(dataset_ids):
            mask = dataset_ids == ds_id
            if np.any(mask):
                mean_map[ds_id] = float(energy_per_atom[mask].mean())

        if not mean_map:
            return None, None, None

        base_id = min(mean_map, key=lambda k: abs(mean_map[k]))
        base_mean = mean_map[base_id]
        offsets = {ds_id: mean_map[ds_id] - base_mean for ds_id in mean_map}
        offsets[base_id] = 0.0
        return offsets, base_id, mean_map

    def _apply_dataset_offsets(self, dataframe: pd.DataFrame, offsets: Dict, inverse: bool = False) -> pd.DataFrame:
        if not offsets or dataframe is None or DATASET_ID_COL not in dataframe.columns:
            return dataframe

        df = dataframe.copy()
        if NUMBER_OF_ATOMS in df.columns:
            natoms = df[NUMBER_OF_ATOMS].to_numpy()
        elif ASE_ATOMS in df.columns:
            natoms = df[ASE_ATOMS].map(len).to_numpy()
        else:
            raise ValueError(
                f"TensorFlow backend requires `{NUMBER_OF_ATOMS}` or `{ASE_ATOMS}` column when dataset ids are used")

        mapped_offsets = df[DATASET_ID_COL].map(lambda ds: offsets.get(ds, 0.0)).to_numpy()
        sign = -1 if inverse else 1
        delta = sign * mapped_offsets * natoms
        df[ENERGY_CORRECTED_COL] = df[ENERGY_CORRECTED_COL].to_numpy() + delta
        if E_CORRECTED_PER_ATOM_COLUMN in df.columns:
            df[E_CORRECTED_PER_ATOM_COLUMN] = df[ENERGY_CORRECTED_COL] / natoms
        if ENERGY in df.columns:
            df[ENERGY] = df[ENERGY].to_numpy() + delta
        if 'energy_per_atom' in df.columns:
            df['energy_per_atom'] = df[ENERGY_CORRECTED_COL] / natoms
        log.info("TensorFlow backend: %s offsets applied (%s) -> delta/atom range [%.3f, %.3f] eV",
                 "removing" if inverse else "adding",
                 "per atom" if not inverse else "re-centering",
                 float(np.min(mapped_offsets)), float(np.max(mapped_offsets)))
        return df

    def _log_energy_stats(self, dataframe: pd.DataFrame, label: str):
        if dataframe is None or ENERGY_CORRECTED_COL not in dataframe.columns:
            log.info("%s: dataframe missing `%s` column", label, ENERGY_CORRECTED_COL)
            return
        log.info("%s: columns=%s", label, list(dataframe.columns))
        if NUMBER_OF_ATOMS in dataframe.columns:
            natoms = dataframe[NUMBER_OF_ATOMS].to_numpy()
        elif ASE_ATOMS in dataframe.columns:
            natoms = dataframe[ASE_ATOMS].map(len).to_numpy()
        else:
            log.info("%s: unable to compute per-atom stats (missing `%s`/`%s`)", label, NUMBER_OF_ATOMS, ASE_ATOMS)
            return
        epa = dataframe[ENERGY_CORRECTED_COL].to_numpy() / natoms
        log.info("%s: energy/atom min=%.3f max=%.3f mean=%.3f std=%.3f", label, float(np.min(epa)),
                 float(np.max(epa)), float(np.mean(epa)), float(np.std(epa)))
        if DATASET_ID_COL in dataframe.columns:
            for ds_id, grp in dataframe.groupby(DATASET_ID_COL):
                if NUMBER_OF_ATOMS in grp.columns:
                    nat = grp[NUMBER_OF_ATOMS].to_numpy()
                else:
                    nat = grp[ASE_ATOMS].map(len).to_numpy()
                epa_ds = grp[ENERGY_CORRECTED_COL].to_numpy() / nat
                log.info("%s [ids=%s]: energy/atom min=%.3f max=%.3f mean=%.3f std=%.3f count=%d",
                         label, ds_id, float(np.min(epa_ds)), float(np.max(epa_ds)),
                         float(np.mean(epa_ds)), float(np.std(epa_ds)), len(epa_ds))

    def fit(self,
            bbasisconfig: BBasisConfiguration,
            dataframe: pd.DataFrame,
            loss_spec: LossFunctionSpecification = None,
            fit_config: Dict = None, callback: Callable = None,
            test_dataframe: pd.DataFrame = None
            ) -> BBasisConfiguration:
        if loss_spec is None:
            loss_spec = self.loss_spec
        else:
            self.loss_spec = loss_spec
        if fit_config is None:
            fit_config = self.fit_config

        if callback is not None:
            self.callback = callback

        trainable_parameters = fit_config.get("trainable_parameters", [])  # default value = [] -> ["ALL"]
        # convert fit_blocks to indices of the blocks to fit
        elements = ACEBBasisSet(bbasisconfig).elements_name
        trainable_parameters_dict = expand_trainable_parameters(elements=elements,
                                                                trainable_parameters=trainable_parameters)

        log.info("Trainable parameters: {}".format(trainable_parameters_dict))

        # save globally
        self.trainable_parameters_dict = trainable_parameters_dict
        self.bbasisconfig = bbasisconfig

        if self.backend_config.evaluator_name == TENSORPOT_EVAL:
            from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InternalError

            train_df_backend = dataframe
            test_df_backend = test_dataframe
            self.tensorpot_dataset_offsets = None
            if DATASET_ID_COL in dataframe.columns:
                self._log_energy_stats(dataframe, "TensorFlow backend: original train data")
                offsets, base_id, mean_map = self._compute_dataset_offsets(dataframe)
                if offsets:
                    log.info("TensorFlow backend: dataset offsets found base_id=%s mean_map=%s offsets=%s",
                             base_id, mean_map, offsets)
                    train_df_backend = self._apply_dataset_offsets(dataframe, offsets, inverse=True)
                    self._log_energy_stats(train_df_backend, "TensorFlow backend: adjusted train data")
                    if test_dataframe is not None:
                        test_df_backend = self._apply_dataset_offsets(test_dataframe, offsets, inverse=True)
                        self._log_energy_stats(test_df_backend, "TensorFlow backend: adjusted test data")
                    self.tensorpot_dataset_offsets = offsets
                else:
                    log.info("TensorFlow backend: unable to compute dataset offsets (mean_map empty)")
            else:
                self._log_energy_stats(dataframe, "TensorFlow backend: train data (no ids)")

            while True:
                try:
                    self.setup_tensorpot(bbasisconfig, train_df_backend, loss_spec, trainable_parameters_dict)
                    fit_res = self.run_tensorpot_fit(bbasisconfig, train_df_backend, loss_spec, fit_config,
                                                     trainable_parameters_dict,
                                                     test_dataframe=test_df_backend)
                    self.log_optimization_result()
                    return fit_res
                except (ResourceExhaustedError, InternalError) as e:
                    log.error("{} errors encountered".format(e))
                    if self.backend_config.get(BACKEND_BATCH_SIZE_REDUCTION_KW, True):
                        batch_size = self.backend_config.get(BACKEND_BATCH_SIZE_KW, 10)
                        batch_size_reduction_factor = self.backend_config.get(BACKEND_BATCH_SIZE_REDUCTION_FACTOR_KW,
                                                                              1.618)  # default - golden ratio
                        new_batch_size = int(batch_size / batch_size_reduction_factor)
                        log.info("Decrease batch size (by factor {}): {} -> {}".format(batch_size_reduction_factor,
                                                                                       batch_size, new_batch_size))
                        if new_batch_size < 1:
                            log.error("New batch size is too small, stopping")
                            raise RuntimeError("No further batch size reduction is possible, stopping")

                        self.backend_config[BACKEND_BATCH_SIZE_KW] = new_batch_size
                        # check the latest version of potential, update bbasisconfig to restart
                        try:
                            log.info("Attempt to get last version of potential")
                            bbasisconfig = self.fitter.tensorpot.potential.get_updated_config()
                            log.info("Last version of potential is extracted")
                        except Exception as e:
                            log.error("Can not get last version of potential: {}".format(e))
                    else:
                        log.error("Use `backend:batch_size_reduction` option for automatic batch size reduction")
                        raise RuntimeError("{} errors encountered. " +
                                           "Consider using `backend::{}=true` option for automatic batch size reduction".format(
                                               e,
                                               BACKEND_BATCH_SIZE_REDUCTION_KW))
                except Exception as e:
                    raise e

        elif self.backend_config.evaluator_name == PYACE_EVAL:
            self.setup_pyace(bbasisconfig, dataframe, loss_spec, trainable_parameters_dict)
            fit_res = self.run_pyace_fit(bbasisconfig, dataframe, loss_spec, fit_config, trainable_parameters_dict)
            self.log_optimization_result()
            return fit_res
        else:
            raise ValueError('{0} is not a valid evaluator'.format(self.backend_config.evaluator_name))

    def log_optimization_result(self, res_opt=None):
        if res_opt is None:
            res_opt = self.res_opt
        try:
            log.info(
                "Optimization result(success={success}, status={status}, message={message}, nfev={nfev}, njev={njev})".format(
                    success=res_opt.success,
                    status=res_opt.status,
                    message=res_opt.message,
                    nfev=res_opt.nfev, njev=res_opt.njev
                ))
        except Exception as e:
            log.error("Optimization result: not available: " + str(e))

    def get_evaluator_version_dict(self):
        try:
            if self.backend_config.evaluator_name == TENSORPOT_EVAL:
                import tensorpotential
                return {TENSORPOT_EVAL + "_version": tensorpotential.__version__}
            elif self.backend_config.evaluator_name == PYACE_EVAL:
                from pyace import __version__, get_ace_evaluator_version
                return {PYACE_EVAL + "_version": __version__, "ace_evaluator_version": get_ace_evaluator_version()}
            else:
                raise ValueError('{0} is not a valid evaluator'.format(self.backend_config.evaluator_name))
        except Exception as e:
            log.error(e)
        return {}

    def setup_tensorpot(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                        loss_spec: LossFunctionSpecification,
                        trainable_parameters_dict: Dict
                        ) -> BBasisConfiguration:
        from tensorpotential.potentials.ace import ACE
        from tensorpotential.tensorpot import TensorPotential
        from tensorpotential.fit import FitTensorPotential
        from tensorpotential.utils.utilities import batching_data, init_gpu_config
        from tensorpotential.constants import (LOSS_TYPE, LOSS_FORCE_FACTOR, LOSS_ENERGY_FACTOR, L1_REG,
                                               L2_REG, AUX_LOSS_FACTOR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            gpu_config = self.backend_config.get(BACKEND_GPU_CONFIG, None)
            init_gpu_config(gpu_config)
            batch_size = self.backend_config.get(BACKEND_BATCH_SIZE_KW, 10)
            log.info("Loss function specification: " + str(loss_spec))
            log.info("Batch size: {}".format(batch_size))
            batches = batching_data(dataframe, batch_size=batch_size)
            n_batches = len(batches)
            loss_force_factor = loss_spec.kappa
            has_smoothness = (np.array([loss_spec.w0_rad, loss_spec.w1_rad, loss_spec.w2_rad]) != 0).any()
            has_orthogonality = loss_spec.w_orth != 0
            aux_loss_factor = []
            if has_orthogonality:
                aux_loss_factor += [np.float64(loss_spec.w_orth) / n_batches]
            if has_smoothness:
                aux_loss_factor += [np.float64(loss_spec.w0_rad) / n_batches,
                                    np.float64(loss_spec.w1_rad) / n_batches,
                                    np.float64(loss_spec.w2_rad) / n_batches]
            loss_specs = {
                LOSS_TYPE: 'per-atom',
                LOSS_FORCE_FACTOR: loss_force_factor,
                LOSS_ENERGY_FACTOR: (1 - loss_force_factor),
                L1_REG: np.float64(loss_spec.L1_coeffs) / n_batches,
                L2_REG: np.float64(loss_spec.L2_coeffs) / n_batches
            }
            if aux_loss_factor:
                loss_specs[AUX_LOSS_FACTOR] = aux_loss_factor

            ace_potential = ACE(bbasisconfig, compute_orthogonality=has_orthogonality,
                                compute_smoothness=has_smoothness)
            tensorpotential = TensorPotential(ace_potential, loss_specs=loss_specs)

            display_step = self.backend_config.get('display_step', 20)
            self.fitter = FitTensorPotential(tensorpotential, display_step=display_step)
            # assign total_number_of_functions to fitter
            total_number_of_functions = bbasisconfig.total_number_of_functions
            self.fitter.nfuncs = total_number_of_functions

    def run_tensorpot_fit(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                          loss_spec: LossFunctionSpecification, fit_config: Dict,
                          trainable_parameters_dict: Dict,
                          test_dataframe: pd.DataFrame = None
                          ) -> BBasisConfiguration:

        with warnings.catch_warnings():
            # adapt call of self._callback(current_bbasisconfig) from FitTensorPotential.callback(coeffs)
            def adapted_callback(coeffs):
                new_config = self.fitter.tensorpot.potential.get_updated_config(updating_coefs=coeffs)
                self._callback(new_config)

            jacobian_factor = compute_bbasisset_train_mask(bbasisconfig, trainable_parameters_dict)
            if np.all(jacobian_factor):
                jacobian_factor = None  # default value - train all
            else:
                jacobian_factor = jacobian_factor.astype(float)

            train_df = dataframe
            test_df_local = test_dataframe
            if DATASET_ID_COL in train_df.columns:
                self._log_energy_stats(train_df, "TensorFlow backend: train data passed to fitter")
            else:
                self._log_energy_stats(train_df, "TensorFlow backend: train data passed to fitter (no ids)")

            batch_size = self.backend_config.get(BACKEND_BATCH_SIZE_KW, 10)
            fit_options = fit_config.get(FIT_OPTIONS_KW, None)
            self.fitter.fit(train_df, test_df=test_df_local, niter=fit_config[FIT_NITER_KW],
                            optimizer=fit_config[FIT_OPTIMIZER_KW],
                            batch_size=batch_size, jacobian_factor=jacobian_factor,
                            callback=adapted_callback,  # call adapted_callback to pass current_bbasisconfig
                            options=fit_options,
                            fit_metric_callback=self.fit_metrics_callback,
                            test_metric_callback=self.test_metrics_callback
                            )

            self.res_opt = self.fitter.res_opt
            new_config = self.fitter.tensorpot.potential.get_updated_config(updating_coefs=self.res_opt.x)
            return new_config

    def setup_pyace(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                    loss_spec: LossFunctionSpecification,
                    trainable_parameters_dict: Dict
                    ) -> BBasisConfiguration:
        from pyace.pyacefit import PyACEFit

        parallel_mode = self.backend_config.get(BACKEND_PARALLEL_MODE_KW) or "serial"
        batch_size = len(dataframe)

        log.info("Loss function specification: " + str(loss_spec))
        display_step = self.backend_config.get('display_step', 20)
        # TODO: consider loss_spec.w_orth
        self.fitter = PyACEFit(basis=bbasisconfig,
                               loss_spec=loss_spec,
                               executors_kw_args=dict(parallel_mode=parallel_mode,
                                                      batch_size=batch_size,
                                                      n_workers=self.backend_config.get(BACKEND_NWORKERS_KW, None)
                                                      ),
                               seed=42,
                               display_step=display_step, trainable_parameters=trainable_parameters_dict)

        # maxiter = fit_config.get(FIT_NITER_KW, 100)
        #
        # fit_options = fit_config.get(FIT_OPTIONS_KW, {})
        # options = {"maxiter": maxiter, "disp": True}
        # options.update(fit_options)

        # assign total_number_of_functions to fitter
        total_number_of_functions = bbasisconfig.total_number_of_functions
        self.fitter.nfuncs = total_number_of_functions

    def run_pyace_fit(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                      loss_spec: LossFunctionSpecification, fit_config: Dict,
                      trainable_parameters_dict: Dict,
                      test_dataframe: pd.DataFrame = None
                      ) -> BBasisConfiguration:
        maxiter = fit_config.get(FIT_NITER_KW, 100)
        fit_options = fit_config.get(FIT_OPTIONS_KW, {})
        options = {"maxiter": maxiter, "disp": True}
        options.update(fit_options)

        self.fitter.fit(structures_dataframe=dataframe, method=fit_config[FIT_OPTIMIZER_KW],
                        options=options,
                        callback=self._callback,
                        fit_metric_callback=self.fit_metrics_callback
                        )

        self.res_opt = self.fitter.res_opt
        new_bbasisconf = self.fitter.bbasis_opt.to_BBasisConfiguration()
        # bbasisconfig.set_all_coeffs(new_bbasisconf.get_all_coeffs())
        return new_bbasisconf

    def setup_backend_for_predict(self, bbasisconfig):
        if bbasisconfig is None:
            raise ValueError("`bbasisconfig` couldn't be None for FitAdapter.setup_backend_for_predict")
        log.info("Setting {} backend for predicting".format(self.backend_config.evaluator_name))
        if self.backend_config.evaluator_name == TENSORPOT_EVAL:
            from tensorpotential.potentials.ace import ACE
            from tensorpotential.tensorpot import TensorPotential
            from tensorpotential.fit import FitTensorPotential
            ace_pot = ACE(bbasisconfig)
            tp = TensorPotential(ace_pot)
            self.fitter = FitTensorPotential(tensorpot=tp, eager=True)
        elif self.backend_config.evaluator_name == PYACE_EVAL:
            from pyace import PyACEFit
            self.fitter = PyACEFit(bbasisconfig)
        else:
            raise ValueError('{0} is not a valid evaluator'.format(self.backend_config.evaluator_name))

    def predict(self, structures_dataframe=None, bbasisconfig=None):
        if self.fitter is None:
            self.setup_backend_for_predict(bbasisconfig)
        prediction = self.fitter.predict(structures_dataframe)
        if self.backend_config.evaluator_name == TENSORPOT_EVAL and self.tensorpot_dataset_offsets:
            if structures_dataframe is not None:
                source_df = structures_dataframe
            else:
                try:
                    source_df = self.fitter.get_fitting_data()
                except AttributeError:
                    source_df = None
            if source_df is not None and DATASET_ID_COL in source_df.columns:
                self._log_energy_stats(source_df, "TensorFlow backend: metrics dataframe before restoration")
                if NUMBER_OF_ATOMS in source_df.columns:
                    natoms = source_df[NUMBER_OF_ATOMS].to_numpy()
                elif ASE_ATOMS in source_df.columns:
                    natoms = source_df[ASE_ATOMS].map(len).to_numpy()
                else:
                    natoms = None
                if natoms is not None and ENERGY_PRED_COL in prediction.columns:
                    per_atom_offsets = source_df[DATASET_ID_COL].map(lambda ds: self.tensorpot_dataset_offsets.get(ds, 0.0)).to_numpy()
                    prediction[ENERGY_PRED_COL] = prediction[ENERGY_PRED_COL].to_numpy() + per_atom_offsets * natoms
                    if 'energy_pred_per_atom' in prediction.columns:
                        prediction['energy_pred_per_atom'] = prediction['energy_pred_per_atom'].to_numpy() + per_atom_offsets
                if ENERGY_PRED_COL in prediction.columns:
                    pred_df = pd.DataFrame({
                        ENERGY_CORRECTED_COL: prediction[ENERGY_PRED_COL].to_numpy(),
                        DATASET_ID_COL: source_df[DATASET_ID_COL].to_numpy(),
                        NUMBER_OF_ATOMS: natoms
                    })
                    self._log_energy_stats(pred_df,
                                           "TensorFlow backend: energy predictions after restoration")
        return prediction

    def compute_metrics(self, energy_col='energy_corrected',
                        nat_column='NUMBER_OF_ATOMS', force_col='forces'):
        results = {}
        prediction = self.predict()
        l1, l2, smth1, smth2, smth3 = self.fitter.get_reg_components()
        datadf = self.fitter.get_fitting_data()

        if self.backend_config.evaluator_name == TENSORPOT_EVAL and self.tensorpot_dataset_offsets and DATASET_ID_COL in datadf.columns:
            if NUMBER_OF_ATOMS in datadf.columns:
                natoms = datadf[NUMBER_OF_ATOMS].to_numpy()
            elif ASE_ATOMS in datadf.columns:
                natoms = datadf[ASE_ATOMS].map(len).to_numpy()
            else:
                natoms = None
            if natoms is not None:
                per_atom_offsets = datadf[DATASET_ID_COL].map(lambda ds: self.tensorpot_dataset_offsets.get(ds, 0.0)).to_numpy()
                datadf = datadf.copy()
                datadf[energy_col] = datadf[energy_col].to_numpy() + per_atom_offsets * natoms
                if ENERGY in datadf.columns:
                    datadf[ENERGY] = datadf[ENERGY].to_numpy() + per_atom_offsets * natoms
                if E_CORRECTED_PER_ATOM_COLUMN in datadf.columns:
                    datadf[E_CORRECTED_PER_ATOM_COLUMN] = datadf[energy_col] / natoms

        datadf[force_col] = datadf[force_col].apply(np.array)
        datadf['w_forces'] = datadf['w_forces'].apply(np.reshape, newshape=[-1, 1])
        de = prediction['energy_pred'] - datadf[energy_col]
        df = prediction['forces_pred'] - datadf[force_col]
        e_loss = float(np.sum(datadf['w_energy'] * de ** 2))
        f_loss = np.sum((datadf['w_forces'] * df ** 2).map(np.sum))
        if self.backend_config.evaluator_name == TENSORPOT_EVAL:
            nat_for_log = datadf[nat_column].to_numpy() if nat_column in datadf.columns else None
            if nat_for_log is None and ASE_ATOMS in datadf.columns:
                nat_for_log = datadf[ASE_ATOMS].map(len).to_numpy()
            de_values = de.to_numpy() if hasattr(de, "to_numpy") else np.asarray(de)
            if nat_for_log is not None and len(de_values) == len(nat_for_log):
                de_pa = de_values / nat_for_log
                sort_idx = np.argsort(np.abs(de_pa))
                worst_idx = sort_idx[-5:][::-1]
                for pos, idx in enumerate(worst_idx, start=1):
                    ds_id = datadf.iloc[idx][DATASET_ID_COL] if DATASET_ID_COL in datadf.columns else "n/a"
                    log.info("TensorFlow backend: worst energy residual #%d (global idx=%d, ids=%s) dE=%.6f eV, dE/atom=%.6f eV",
                             pos, idx, ds_id, float(de_values[idx]), float(de_pa[idx]))

        mae_pae = np.mean(np.abs(de / datadf[nat_column]))
        mae_e = np.mean(np.abs(de))
        mae_f = np.mean(np.abs(df).map(np.mean))
        rmse_pae = np.sqrt(np.mean(de ** 2 / datadf[nat_column]))
        rmse_e = np.sqrt(np.mean(de ** 2))
        rmse_f = np.sqrt(np.mean((df ** 2).map(np.mean)))

        results['mae_pae'] = mae_pae
        results['mae_e'] = mae_e
        results['mae_f'] = mae_f
        results['rmse_pae'] = rmse_pae
        results['rmse_e'] = rmse_e
        results['rmse_f'] = rmse_f

        results['e_loss'] = e_loss
        results['f_loss'] = f_loss
        results['l1'] = l1
        results['l2'] = l2
        results['radial_smooth'] = [smth1, smth2, smth3]

        return results

    def print_detailed_metrics(self, title='Iteration:'):
        if self.fitter is not None:
            self.fitter.print_detailed_metrics(title=title)

    def print_extended_metrics(self, title='Iteration:'):
        if self.fitter is not None:
            self.fitter.print_extended_metrics(title=title)

    def _callback(self, current_bbasisconfig: BBasisConfiguration):
        if self.callback is not None:
            self.callback(current_bbasisconfig)

    @property
    def last_loss(self):
        if self.backend_config.evaluator_name == TENSORPOT_EVAL:
            return self.fitter.loss_history[-1]
        elif self.backend_config.evaluator_name == PYACE_EVAL:
            return self.fitter.last_loss

    @property
    def last_fit_metric_data(self):
        if self.fitter is not None:
            last_fit_metric_data = self.fitter.last_fit_metric_data
            if last_fit_metric_data is None:
                last_fit_metric_data = {}
            return last_fit_metric_data

    @last_fit_metric_data.setter
    def last_fit_metric_data(self, value):
        if self.fitter is not None:
            self.fitter.last_fit_metric_data = value

    @property
    def last_test_metric_data(self):
        if self.fitter is not None:
            last_test_metric_data = self.fitter.last_test_metric_data
            if last_test_metric_data is None:
                last_test_metric_data = {}
            return last_test_metric_data

    @last_test_metric_data.setter
    def last_test_metric_data(self, value):
        if self.fitter is not None:
            self.fitter.last_test_metric_data = value
