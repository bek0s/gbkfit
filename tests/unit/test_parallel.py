"""
Tests of the workers of clusters (see gbkfit.parallel): of Ray and Dask
in clusters of this machine, and of MPI in a job of this machine. They
evaluate batches, refit bootstraps and stop their sub-fits as the
workers of this machine do (see test_fitting).
"""

import collections
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from gbkfit.control import StopControl, Stopped
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.fitting import (
    BootstrapBlocks, BootstrapOptions, FitProblem, fitter_parser,
    run_bootstrap)
from gbkfit.params import ParamSpace
from gbkfit.parallel import Workers, WorkersDask, WorkersRay
from gbkfit.parallel._detail import own_placement
from gbkfit.parallel.mpi import _serve_mpi

from test_fitting import bounded_properties, mock_objective, run_until_stopped


def check_workers(workers: Workers) -> None:
    """
    Check that workers evaluate batches and refit a bootstrap as the
    objective alone, and stop their sub-fits.
    """
    objective = mock_objective(DriverHost())
    params = ParamSpace.from_properties(
        objective.pdescs(), bounded_properties(),
        derived=dict(d='xpos + 2 * ypos'))
    xs = np.random.default_rng(2).uniform(
        [-1, -1, 30, 45, 0.8, 3], [1, 1, 50, 65, 1.2, 5], (5, 6))
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    start = fitter.fit(objective, params).best().elements
    options = BootstrapOptions(4, BootstrapBlocks([4, 4]))
    alone = run_bootstrap(objective, params, start, fitter, options)
    with workers.start(objective) as pool:
        np.testing.assert_allclose(
            FitProblem(objective, params, pool=pool).chi2_batch(xs),
            FitProblem(objective, params).chi2_batch(xs), rtol=1e-12)
        pooled = run_bootstrap(objective, params, start, fitter, options, pool)
        np.testing.assert_allclose(pooled.samples, alone.samples, rtol=1e-9)
        control = StopControl()
        threading.Timer(1.0, control.stop).start()
        with pytest.raises(Stopped):
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


def test_workers_of_dask_of_an_address_close_their_client(tests_importable):
    # The client that they made for the address used to stay open
    distributed = pytest.importorskip('distributed')
    with distributed.LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True,
            dashboard_address=None) as cluster:
        check_workers(WorkersDask(count=2, address=cluster.scheduler_address))
        with pytest.raises(ValueError):
            distributed.get_client()


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


def test_a_wrong_command_of_the_launcher_ends_its_job():
    # The other ranks used to wait for the first, which had exited
    pytest.importorskip('mpi4py')
    if shutil.which('mpiexec') is None:
        pytest.skip("needs mpiexec")
    result = subprocess.run(
        ['mpiexec', '-n', '2', '--oversubscribe', sys.executable, '-m',
         'gbkfit.parallel', '-m'],
        capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert 'usage: mpiexec' in result.stderr


class FakeComm:
    """
    The communicator of a rank of workers of MPI, with the messages of the
    first rank given at once; its replies are pickled, as MPI sends them.
    """

    def __init__(self, messages):
        self.messages = collections.deque(messages)
        self.replies = []

    def recv(self, source, tag):
        return self.messages.popleft()

    def iprobe(self, source, tag):
        return bool(self.messages)

    def send(self, message, dest, tag):
        self.replies.append(pickle.loads(pickle.dumps(message)))


def start_message(key, objective):
    """The message that starts a copy of an objective on a rank."""
    return (
        'start', key, objective.likelihoods(), objective.solves(),
        own_placement(objective), None)


def stop_on_control(copy, item, control):
    """A function that runs until its control stops it."""
    while not control.stopped():
        time.sleep(0.01)
    return 'stopped'


def test_a_rank_whose_copy_failed_answers_the_calls_of_its_pool():
    # It used to exit, and the first rank waited for its reply forever
    comm = FakeComm([
        ('start', 0, None, None, ((), ()), None),
        ('run', 0, stop_on_control, [1]),
        ('run_controlled', 0, stop_on_control, 1),
        ('close', 0), ('exit',)])
    _serve_mpi(comm)
    assert [ok for ok, _ in comm.replies] == [False, False]


def test_a_rank_keeps_the_messages_that_come_as_a_function_runs():
    # A message of another pool that came before the stop used to be lost
    objective = mock_objective(DriverHost())
    comm = FakeComm([
        start_message(0, objective), start_message(1, objective),
        ('run_controlled', 0, stop_on_control, None),
        ('run', 1, lambda copy, item: item + 1, [41]), ('stop',),
        ('close', 0), ('close', 1), ('exit',)])
    _serve_mpi(comm)
    assert comm.replies == [(True, 'stopped'), (True, [42])]


def test_a_rank_sends_an_error_for_a_result_it_cannot_send():
    objective = mock_objective(DriverHost())
    comm = FakeComm([
        start_message(0, objective),
        ('run', 0, lambda copy, item: lambda: item, [1]),
        ('close', 0), ('exit',)])
    _serve_mpi(comm)
    [(ok, error)] = comm.replies
    assert not ok and 'could not be sent' in str(error)


def test_workers_of_mpi_are_made_outside_their_job():
    # (the configuration of a cluster fit is evaluated anywhere; the job
    # is needed on first use: audit 2026-10-31 #28)
    from gbkfit.parallel import WorkersMPI
    pool = WorkersMPI().start(mock_objective(DriverHost()))
    with pytest.raises(RuntimeError, match="launcher of GBKFIT"):
        pool.size()
