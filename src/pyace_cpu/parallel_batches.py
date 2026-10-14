"""Persistent-process prototype for the unchanged TensorPotential batch objective.

Model and batches are sent once. Every call sends the complete coefficient vector.
Use the parent's already-scaled loss specifications, including regularization per
GLOBAL batch, and retain the original data weights. No worker normalizes a subset.
Used by the benchmark and an explicit CPU-fitting hook, never installed globally.
"""

import json
import multiprocessing as mp
import os
import resource
import time
import traceback


class _NumpyResult:
    """The upstream aggregator only requests .numpy() from cached results."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


def _worker(connection, cpus, model, loss_specs, potential_options, batches,
            progress_batches):
    try:
        os.sched_setaffinity(0, cpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(len(cpus))
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        import numpy as np
        import tensorflow as tf
        from pyace.basis import BBasisConfiguration
        from tensorpotential.potentials.ace import ACE
        from tensorpotential.tensorpot import TensorPotential
        if os.environ.get('WBC_CPU_EXPERIMENTS'):
            from cpu_experiments import apply
            apply(os.environ['WBC_CPU_EXPERIMENTS'])
        tf.config.threading.set_intra_op_parallelism_threads(len(cpus))
        tf.config.threading.set_inter_op_parallelism_threads(1)
        potential = ACE(BBasisConfiguration(model), **potential_options)
        tensorpot = TensorPotential(potential, loss_specs=loss_specs)
        set_coefs_once = os.environ.get('WBC_CPU_SET_COEFS_ONCE', '0') == '1'
        if os.environ.get('WBC_CPU_TENSOR_BATCHES', '0') == '1':
            # Replace only immutable raw inputs in this worker's private copy.
            # No model-dependent descriptors or geometry derivatives are cached.
            for _, batch in batches:
                for keys, dtype in ((tensorpot.list_of_float_placeholders, tf.float64),
                                    (tensorpot.list_of_int_placeholders, tf.int32)):
                    for key in keys:
                        batch[key] = tf.convert_to_tensor(batch[key], dtype=dtype, name=key)
        connection.send(('ready', os.getpid()))
        while True:
            request = connection.recv()
            if request is None:
                break
            coefficients, batch_limit = request[:2]
            diagnostic = request[2] if len(request) == 3 else None
            selected = batches[:batch_limit] if batch_limit else batches
            results = []
            batch_times = []
            if diagnostic and diagnostic.get('profile_dir'):
                tf.profiler.experimental.start(diagnostic['profile_dir'], options=
                    tf.profiler.experimental.ProfilerOptions(
                        host_tracer_level=2, python_tracer_level=0, device_tracer_level=0))
            usage_before = resource.getrusage(resource.RUSAGE_SELF) if diagnostic else None
            start = time.perf_counter()
            if set_coefs_once:
                # Every batch in this request uses exactly the same coefficients.
                # Reset for every request, including line-search evaluations.
                potential.set_coefs(coefficients)
            for completed, (index, batch) in enumerate(selected, 1):
                batch_start = time.perf_counter() if diagnostic else None
                result = (tensorpot.native_fit(batch) if set_coefs_once else
                          tensorpot.external_fit(coefficients, batch))
                results.append((index, tuple(np.asarray(item) for item in result),
                                np.asarray(tensorpot.reg_components)))
                if diagnostic:
                    batch_times.append(dict(index=index, seconds=time.perf_counter() - batch_start))
                if progress_batches and completed % progress_batches == 0:
                    print(json.dumps(dict(event='worker_batch_progress',
                        pid=os.getpid(), completed=completed, total=len(selected),
                        seconds=time.perf_counter() - start)), flush=True)
            seconds = time.perf_counter() - start
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            message = ('result', results, seconds, usage_after.ru_maxrss / 1024)
            if diagnostic:
                if diagnostic.get('profile_dir'):
                    tf.profiler.experimental.stop()
                message += (dict(batches=batch_times,
                    user_s=usage_after.ru_utime - usage_before.ru_utime,
                    system_s=usage_after.ru_stime - usage_before.ru_stime,
                    voluntary_switches=usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                    involuntary_switches=usage_after.ru_nivcsw - usage_before.ru_nivcsw,
                    minor_faults=usage_after.ru_minflt - usage_before.ru_minflt,
                    major_faults=usage_after.ru_majflt - usage_before.ru_majflt),)
            connection.send(message)
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class PersistentBatchPool:
    def __init__(self, model, fitter, batches, workers, cores, progress_batches=0,
                 cpu_ids=None, batch_order='largest-first'):
        if workers < 2 or workers > len(batches) or cores % workers:
            raise ValueError('Use >=2 workers, <= batch count, evenly dividing cores')
        cpu_ids = list(range(cores)) if cpu_ids is None else list(cpu_ids)
        if len(cpu_ids) != cores or len(set(cpu_ids)) != cores or not set(cpu_ids) <= os.sched_getaffinity(0):
            raise ValueError('Worker CPU IDs must be distinct allowed CPUs matching the core count')
        if batch_order not in ('largest-first', 'staggered', 'shuffled'):
            raise ValueError('Unknown worker-local batch order')
        self.batches = batches
        self.connections = []
        self.processes = []
        self.worker_times = []
        self.worker_warmup_s = []
        self.worker_peak_rss = []
        self.evaluation_phases = []
        self.last_worker_diagnostics = []
        self.numpy_results = os.environ.get('WBC_CPU_NUMPY_RESULTS', '0') == '1'
        self.ready_receive = os.environ.get('WBC_CPU_READY_RECEIVE', '0') == '1'
        self.closed = False
        self.batch_ids = [id(batch) for batch in batches]
        loss_specs = {key: value.numpy() if hasattr(value, 'numpy') else value
                      for key, value in fitter.tensorpot.loss_specs.items()}
        potential = fitter.tensorpot.potential
        options = dict(compute_orthogonality=potential.compute_orthogonality,
                       compute_smoothness=potential.compute_smoothness)
        groups = [[] for _ in range(workers)]
        loads = [0] * workers
        atom_cost = float(os.environ.get('WBC_CPU_BATCH_ATOM_COST', '0'))
        fixed_cost = float(os.environ.get('WBC_CPU_BATCH_FIXED_COST', '0'))
        if not 0 <= atom_cost < float('inf') or not 0 <= fixed_cost < float('inf'):
            raise ValueError('Batch cost estimates must be finite and nonnegative')
        self.cost_model = dict(atom_cost=atom_cost, fixed_cost=fixed_cost)
        costs = [len(batch['ind_i']) + atom_cost * len(batch['positions']) + fixed_cost
                 for batch in batches]
        # Balance whole, unchanged batches. Parent reduction keeps global order.
        for index in sorted(range(len(batches)), key=lambda i: costs[i], reverse=True):
            worker = min(range(workers), key=lambda i: loads[i])
            groups[worker].append((index, batches[index]))
            loads[worker] += costs[index]
        # Diagnostic scheduling variants preserve group membership and the
        # parent's original global summation order. Only execution order differs.
        for worker, group in enumerate(groups):
            if batch_order == 'staggered' and group:
                shift = (worker * len(group) * 5 // workers) % len(group)
                groups[worker] = group[shift:] + group[:shift]
            elif batch_order == 'shuffled':
                import random
                random.Random(913 + worker).shuffle(group)
        self.assignment = [[index for index, _ in group] for group in groups]
        context = mp.get_context('spawn')
        threads = cores // workers
        try:
            for worker, group in enumerate(groups):
                parent, child = context.Pipe()
                cpus = cpu_ids[worker * threads:(worker + 1) * threads]
                process = context.Process(target=_worker,
                    args=(child, cpus, str(model), loss_specs, options, group,
                          progress_batches), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.worker_pids = [self._receive(connection, 'ready')[1]
                                for connection in self.connections]
        except BaseException:
            self.close(abort=True)
            raise

    @staticmethod
    def _receive(connection, expected):
        if not connection.poll(900):
            raise TimeoutError('TensorPotential worker did not respond within 900 seconds')
        message = connection.recv()
        if message[0] == 'error':
            raise RuntimeError(message[1])
        if message[0] != expected:
            raise RuntimeError(f'Unexpected worker message: {message[0]}')
        return message

    def warmup(self, coefficients, batches_per_worker):
        """Warm each worker without another complete dataset evaluation."""
        if self.closed or batches_per_worker < 1:
            raise ValueError('Warmup requires an open pool and a positive batch limit')
        import numpy as np
        try:
            for connection in self.connections:
                connection.send((coefficients, batches_per_worker))
            loss, gradient = 0., np.zeros_like(coefficients)
            self.worker_warmup_s = []
            for connection in self.connections:
                _, entries, seconds, _ = self._receive(connection, 'result')
                self.worker_warmup_s.append(seconds)
                for _, values, _ in entries:
                    loss += float(values[0])
                    gradient += values[1]
            # Partial loss is only a warmup diagnostic, never a fit objective.
            return loss, gradient
        except BaseException:
            self.close(abort=True)
            raise

    def _result_connections(self):
        if not self.ready_receive:
            yield from enumerate(self.connections)
            return
        from multiprocessing.connection import wait
        pending = {connection: worker for worker, connection in enumerate(self.connections)}
        while pending:
            available = wait(pending, timeout=900)
            if not available:
                raise TimeoutError('TensorPotential workers did not respond within 900 seconds')
            for connection in available:
                yield pending.pop(connection), connection

    def fit_func(self, fitter, coefficients, batches, aggregate=None, diagnostic=None):
        """Reuse upstream loss/gradient/metric aggregation on worker results."""
        if self.closed or [id(batch) for batch in batches] != self.batch_ids:
            raise ValueError('The pool is closed or its immutable batch list changed')
        import tensorflow as tf
        start = time.perf_counter()
        try:
            for worker, connection in enumerate(self.connections):
                request = (coefficients, None)
                if diagnostic is not None:
                    options = {'enabled': True}
                    if worker in diagnostic.get('profile_workers', []):
                        from pathlib import Path
                        options['profile_dir'] = str(Path(diagnostic['directory']) / f'worker-{worker:02d}')
                    request += (options,)
                connection.send(request)
            sent = time.perf_counter()
            results = [None] * len(batches)
            times, memory = [None] * len(self.connections), [None] * len(self.connections)
            self.last_worker_diagnostics = []
            for worker, connection in self._result_connections():
                message = self._receive(connection, 'result')
                _, entries, seconds, rss = message[:4]
                if diagnostic is not None:
                    self.last_worker_diagnostics.append(dict(worker=worker,
                        seconds=seconds, received_after_s=time.perf_counter() - start,
                        peak_rss_MiB=rss, **message[4]))
                times[worker] = seconds
                memory[worker] = rss
                for index, values, regularization in entries:
                    results[index] = values, regularization
            received = time.perf_counter()
            self.worker_times.append(times)
            self.worker_peak_rss = memory
            iterator = iter(results)
            tensorpot = fitter.tensorpot
            original_external_fit = tensorpot.external_fit
            original_metrics_dict = fitter.metrics.to_FitMetricsDict

            def cached_external_fit(x, batch, eager=False):
                values, regularization = next(iterator)
                tensorpot.reg_components = tf.convert_to_tensor(regularization)
                if self.numpy_results:
                    return tuple(_NumpyResult(value) for value in values)
                return tuple(tf.convert_to_tensor(value) for value in values)

            def timed_metrics_dict():
                # Include workers in initial and subsequent callback timings.
                fitter.eval_time = time.perf_counter() - start
                fitter.metrics.eval_time = fitter.eval_time
                return original_metrics_dict()

            tensorpot.potential.set_coefs(coefficients)
            tensorpot.external_fit = cached_external_fit
            fitter.metrics.to_FitMetricsDict = timed_metrics_dict
            try:
                result = (aggregate or fitter.fit_func)(coefficients, batches)
            finally:
                tensorpot.external_fit = original_external_fit
                fitter.metrics.to_FitMetricsDict = original_metrics_dict
            # Include final aggregation overhead in retained metrics too.
            fitter.eval_time = time.perf_counter() - start
            fitter.metrics.eval_time = fitter.eval_time
            if fitter.last_fit_metric_data is not None:
                fitter.last_fit_metric_data['eval_time'] = fitter.eval_time
            self.evaluation_phases.append(dict(send_s=sent - start,
                compute_and_receive_s=received - sent,
                aggregate_s=time.perf_counter() - received,
                slowest_worker_s=max(times), total_s=time.perf_counter() - start,
                diagnostic=diagnostic is not None))
            return result
        except BaseException:
            self.close(abort=True)
            raise

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                if abort:
                    process.terminate()
                else:
                    try:
                        connection.send(None)
                    except (BrokenPipeError, EOFError, OSError):
                        pass
        for connection, process in zip(self.connections, self.processes):
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            connection.close()
