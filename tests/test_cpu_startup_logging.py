"""CPU startup descriptions reach the CLI log without changing ordinary fits."""
import importlib.util
import logging
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'cpu_runtime_under_test', ROOT/'src/pyace_cpu/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def cpu_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('TF_NUM_INTRAOP_THREADS', '1')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '12')
    monkeypatch.setenv('MKL_NUM_THREADS', '12')
    monkeypatch.setenv('WBC_CPU_DENSITY_BLOCK', '4')
    monkeypatch.setenv('PACEMAKER_CPU_CACHE_DIR', str(tmp_path/'kernels'))


@pytest.mark.parametrize('isa,compact,packed,description', [
    ('avx2', '1', '0', 'compact, separate real/imaginary'),
    ('avx2', '0', '0', 'full-node workspace, separate real/imaginary'),
    ('avx512', '1', '1', 'compact, packed-complex'),
    ('avx512', '1', '0', 'compact, separate real/imaginary'),
])
def test_kernel_variants(monkeypatch, cpu_environment, isa, compact, packed, description):
    monkeypatch.setenv('WBC_CPU_COMPACT_DENSITY', compact)
    monkeypatch.setenv('WBC_CPU_DENSITY_PACKED', packed)
    messages = '\n'.join(runtime.startup_messages(isa, workers=12, cores=12))
    assert f'CPU ISA: {isa}' in messages
    assert description in messages
    assert 'atom block=4, float64' in messages
    assert 'workers=12, physical cores=12, TensorFlow threads per worker=1' in messages
    assert 'OpenBLAS=12, MKL=12' in messages
    assert 'CPU native kernel cache:' in messages


def test_single_process_thread_override(monkeypatch, cpu_environment):
    monkeypatch.setenv('TF_NUM_INTRAOP_THREADS', '3')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '2')
    messages = '\n'.join(runtime.startup_messages('avx2', workers=1, cores=12))
    assert 'TensorFlow threads per worker=3' in messages
    assert 'OpenBLAS=2' in messages


@pytest.mark.parametrize('filename', ['log.txt', 'custom.log'])
@pytest.mark.parametrize('accelerated', [False, True])
def test_cli_startup_logs_after_file_opens(monkeypatch, tmp_path, cpu_environment,
                                         filename, accelerated):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('WBC_CPU_COMPACT_DENSITY', '1')
    monkeypatch.setenv('WBC_CPU_DENSITY_PACKED', '0')
    main = runpy.run_path(str(ROOT/'bin/pacemaker'))['main']
    logger = logging.getLogger()
    original_handlers = set(logger.handlers)
    arguments = ['missing.yaml']
    if filename != 'log.txt':
        arguments += ['--log', filename]
    options = {'startup_log': runtime.startup_messages('avx2', 12, 12)} if accelerated else {}
    try:
        with pytest.raises(FileNotFoundError, match='missing.yaml'):
            main(arguments, **options)
        text = (tmp_path/filename).read_text()
        assert 'Start pacemaker' in text
        if accelerated:
            assert 'CPU ISA: avx2' in text
            assert 'CPU density kernel: compact, separate real/imaginary' in text
            assert text.index('Start pacemaker') < text.index('CPU ISA: avx2')
        else:
            assert 'CPU ISA:' not in text
        if filename != 'log.txt':
            assert not (tmp_path/'log.txt').exists()
    finally:
        for handler in set(logger.handlers) - original_handlers:
            logger.removeHandler(handler)
            handler.close()
