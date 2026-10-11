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
        connection.send(('ready', os.getpid()))
        while True:
            request = connection.recv()
            if request is None:
                break
            coefficients, batch_limit = request
            selected = batches[:batch_limit] if batch_limit else batches
            results = []
            start = time.perf_counter()
            for completed, (index, batch) in enumerate(selected, 1):
                result = tensorpot.external_fit(coefficients, batch)
                results.append((index, tuple(np.asarray(item) for item in result),
                                np.asarray(tensorpot.reg_components)))
                if progress_batches and completed % progress_batches == 0:
                    print(json.dumps(dict(event='worker_batch_progress',
                        pid=os.getpid(), completed=completed, total=len(selected),
                        seconds=time.perf_counter() - start)), flush=True)
            connection.send(('result', results, time.perf_counter() - start,
                             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class PersistentBatchPool:
    def __init__(self, model, fitter, batches, workers, cores, progress_batches=0,
                 cpu_ids=None):
        if workers < 2 or workers > len(batches) or cores % workers:
            raise ValueError('Use >=2 workers, <= batch count, evenly dividing cores')
        cpu_ids = list(range(cores)) if cpu_ids is None else list(cpu_ids)
        if len(cpu_ids) != cores or len(set(cpu_ids)) != cores or not set(cpu_ids) <= os.sched_getaffinity(0):
            raise ValueError('Worker CPU IDs must be distinct allowed CPUs matching the core count')
        self.batches = batches
        self.connections = []
        self.processes = []
        self.worker_times = []
        self.worker_warmup_s = []
        self.worker_peak_rss = []
        self.closed = False
        self.batch_ids = [id(batch) for batch in batches]
        loss_specs = {key: value.numpy() if hasattr(value, 'numpy') else value
                      for key, value in fitter.tensorpot.loss_specs.items()}
        potential = fitter.tensorpot.potential
        options = dict(compute_orthogonality=potential.compute_orthogonality,
                       compute_smoothness=potential.compute_smoothness)
        groups = [[] for _ in range(workers)]
        loads = [0] * workers
        # Balance whole, unchanged batches using neighbor-pair counts.
        for index in sorted(range(len(batches)),
                            key=lambda i: len(batches[i]['ind_i']), reverse=True):
            worker = min(range(workers), key=lambda i: loads[i])
            groups[worker].append((index, batches[index]))
            loads[worker] += len(batches[index]['ind_i'])
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

    def fit_func(self, fitter, coefficients, batches, aggregate=None):
        """Reuse upstream loss/gradient/metric aggregation on worker results."""
        if self.closed or [id(batch) for batch in batches] != self.batch_ids:
            raise ValueError('The pool is closed or its immutable batch list changed')
        import tensorflow as tf
        start = time.perf_counter()
        try:
            for connection in self.connections:
                connection.send((coefficients, None))
            results = [None] * len(batches)
            times, memory = [], []
            for connection in self.connections:
                _, entries, seconds, rss = self._receive(connection, 'result')
                times.append(seconds)
                memory.append(rss)
                for index, values, regularization in entries:
                    results[index] = values, regularization
            self.worker_times.append(times)
            self.worker_peak_rss = memory
            iterator = iter(results)
            tensorpot = fitter.tensorpot
            original_external_fit = tensorpot.external_fit
            original_metrics_dict = fitter.metrics.to_FitMetricsDict

            def cached_external_fit(x, batch, eager=False):
                values, regularization = next(iterator)
                tensorpot.reg_components = tf.convert_to_tensor(regularization)
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
