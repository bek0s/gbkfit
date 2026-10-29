"""
Tests of the workers of clusters (see gbkfit.parallel): of Ray and Dask
in clusters of this machine, and of MPI in a job of this machine. They
evaluate batches, refit bootstraps and stop their sub-fits as the
workers of this machine do (see test_fitting).
"""

import os
import pathlib
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest

from gbkfit.driver.drivers.host import DriverHost
from gbkfit.fitting import (
    BootstrapBlocks, BootstrapOptions, FitControl, FitProblem, FitStopped,
    bootstrap, fitter_parser)
from gbkfit.params import ParamSpace
from gbkfit.parallel import Workers, WorkersDask, WorkersRay

from test_fitting import bounded_properties, mock_objective, run_until_stopped


def check_workers(workers: Workers) -> None:
    """
    Check that workers evaluate batches and refit a bootstrap as the
    objective alone, and stop their sub-fits.
    """
    objective = mock_objective(DriverHost())
    params = ParamSpace(
        objective.pdescs(), bounded_properties(),
        derived=dict(d='xpos + 2 * ypos'))
    xs = np.random.default_rng(2).uniform(
        [-1, -1, 30, 45, 0.8, 3], [1, 1, 50, 65, 1.2, 5], (5, 6))
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    start = fitter.fit(objective, params).best().elements
    options = BootstrapOptions(4, BootstrapBlocks([4, 4]))
    alone = bootstrap(objective, params, start, fitter, options)
    with workers.start(objective) as pool:
        np.testing.assert_allclose(
            FitProblem(objective, params, pool=pool).chi2_batch(xs),
            FitProblem(objective, params).chi2_batch(xs), rtol=1e-12)
        pooled = bootstrap(objective, params, start, fitter, options, pool)
        np.testing.assert_allclose(pooled.samples, alone.samples, rtol=1e-9)
        control = FitControl()
        threading.Timer(1.0, control.stop).start()
        with pytest.raises(FitStopped):
            pool.map_lazily(
                run_until_stopped, lambda i: i, 10, control=control)


def path_of_tests() -> str:
    """
    Return PYTHONPATH with the folder of the tests, whose modules the
    workers import (e.g. for run_until_stopped).
    """
    return os.pathsep.join(
        [str(pathlib.Path(__file__).parent), os.environ.get('PYTHONPATH', '')])


@pytest.fixture
def tests_importable(monkeypatch):
    # (for the processes of the clusters, which this process starts)
    monkeypatch.setenv('PYTHONPATH', path_of_tests())


def test_workers_of_ray(tests_importable):
    ray = pytest.importorskip('ray')
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        check_workers(WorkersRay(count=2))
    finally:
        ray.shutdown()


def test_workers_of_dask(tests_importable):
    distributed = pytest.importorskip('distributed')
    with distributed.LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True,
            dashboard_address=None) as cluster, \
            distributed.Client(cluster):
        check_workers(WorkersDask(count=2))


# The program of the job of the test of MPI
_MPI_PROGRAM = '''
from gbkfit.parallel import WorkersMPI
from test_parallel import check_workers
check_workers(WorkersMPI())
print('workers of MPI: ok')
'''


def test_workers_of_mpi(tmp_path):
    # (a job of 3 ranks: the program and 2 workers)
    pytest.importorskip('mpi4py')
    if shutil.which('mpiexec') is None:
        pytest.skip("needs mpiexec")
    program = tmp_path / 'program.py'
    program.write_text(_MPI_PROGRAM)
    env = os.environ | dict(PYTHONPATH=path_of_tests())
    result = subprocess.run(
        ['mpiexec', '-n', '3', '--oversubscribe', sys.executable, '-m',
         'gbkfit.parallel', str(program)],
        capture_output=True, text=True, env=env, timeout=300)
    assert 'workers of MPI: ok' in result.stdout, result.stderr[-3000:]
