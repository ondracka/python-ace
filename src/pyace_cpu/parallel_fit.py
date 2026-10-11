"""Opt-in persistent batch workers around upstream CPU fitting and callbacks.

No upstream files or input settings are rewritten. Pools live for one fit and
are closed on completion or failure. Global weights and regularization remain
those prepared by Pacemaker, with upstream metric/gradient aggregation.
"""

from functools import wraps
import inspect
import os
from pathlib import Path
import tempfile


def physical_cpus(count):
    """Select allowed physical cores, respecting scheduler CPU affinity."""
    selected, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((root / 'physical_package_id').read_text().strip(),
               (root / 'core_id').read_text().strip())
        if key not in seen:
            selected.append(cpu)
            seen.add(key)
        if len(selected) == count:
            return selected
    raise ValueError(f'Requested {count} physical cores, only {len(selected)} allowed')


def enable(workers, cores):
    from tensorpotential.fit import FitTensorPotential
    from parallel_batches import PersistentBatchPool
    if workers < 2 or cores % workers:
        raise ValueError('Use at least two workers, evenly dividing the core count')
    if getattr(FitTensorPotential.fit, '_wbc_parallel', False):
        raise RuntimeError('Parallel fitting is already enabled in this process')
    cpus = physical_cpus(cores)
    original_fit = FitTensorPotential.fit
    original_objective = FitTensorPotential.fit_func
    signature = inspect.signature(original_fit)

    @wraps(original_objective)
    def objective(self, coefficients, batches):
        context = getattr(self, '_wbc_parallel_context', None)
        if context is None:
            return original_objective(self, coefficients, batches)
        if context['pool'] is None:
            model = Path(context['directory']) / 'worker-model.yaml'
            self.tensorpot.potential.get_updated_config().save(str(model))
            context['pool'] = PersistentBatchPool(model, self, batches, workers,
                                                  cores, cpu_ids=cpus)
            print(f'CPU batch pool: {workers} workers on physical CPUs {cpus}, '
                  f'{len(batches)} globally weighted batches.', flush=True)
        return context['pool'].fit_func(self, coefficients, batches,
            aggregate=lambda x, b: original_objective(self, x, b))

    @wraps(original_fit)
    def fit(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments['optimizer'] not in ('BFGS', 'L-BFGS-B', 'CG'):
            raise ValueError('The CPU batch pool currently supports BFGS, L-BFGS-B or CG')
        if bound.arguments['test_df'] is not None or self.eager:
            raise ValueError('The CPU batch pool currently requires graph mode and no test split')
        if getattr(self, '_wbc_parallel_context', None) is not None:
            raise RuntimeError('Cannot nest a fit on the same TensorPotential object')
        with tempfile.TemporaryDirectory(prefix='wbc-cpu-fit-') as directory:
            context = dict(directory=directory, pool=None)
            self._wbc_parallel_context = context
            try:
                return original_fit(self, *args, **kwargs)
            finally:
                if context['pool'] is not None:
                    pool = context['pool']
                    self._wbc_parallel_last = dict(workers=workers, cpus=cpus,
                        worker_times=pool.worker_times, assignment=pool.assignment)
                    pool.close()
                del self._wbc_parallel_context

    fit._wbc_parallel = True
    FitTensorPotential.fit = fit
    FitTensorPotential.fit_func = objective
