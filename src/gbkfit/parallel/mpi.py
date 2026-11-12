"""
Workers in the ranks of an MPI job, and the launcher of programs with
them (see main).
"""

import collections
import concurrent.futures
import itertools
import runpy
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any

import threadpoolctl
from annotated_types import Ge

from gbkfit.config import check_options
from gbkfit.control import StopControl
from gbkfit.driver import Driver
from gbkfit.objective import Objective
from gbkfit.observation import Likelihoods
from gbkfit.params import Solve
from ._detail import (
    Drivers, Item, PolledFlag, Result, Worker, dump_placements,
    objective_on, own_placement, placement_drivers)
from .pool import Placement, Workers


__all__ = [
    'WorkersMPI'
]


class WorkersMPI(Workers):
    """
    Workers in the processes (ranks) of an MPI job, one on each rank but
    the first. The first rank runs the program, started by the launcher
    of GBKFIT (see main), e.g. with 4 workers:

        mpiexec -n 5 python -m gbkfit.parallel script.py [arguments]
        mpiexec -n 5 python -m gbkfit.parallel -m module [arguments]

    The ranks of the workers serve every pool of the program, each with
    its copy of the objective, which goes to it pickled (see Objective).
    Like processes (see WorkersProcesses), they evaluate at the same time
    the parts of the models that run in Python and models on the host,
    and run stoppable functions at once (e.g. sub-fits), on the machines
    of the job.

    Parameters
    ----------
    drivers : Sequence of Placement, optional
        The placement of the copy of each worker (one per rank but the
        first); by default, the drivers of the objective.
    threads : int, optional
        The threads of the host of each worker (of the kernels of the host
        driver, and of BLAS); by default, those of its rank.

    Raises
    ------
    ValueError
        If drivers gives no placements.
    """

    requires = {'mpi4py': "pip install 'gbkfit[mpi]'"}

    # Configurations name the drivers of the placements, see Workers
    config_options = {'drivers': Sequence[str | Driver] | None}

    @staticmethod
    def type() -> str:
        return 'mpi'

    @staticmethod
    def type_title() -> str:
        return 'MPI job'

    def __init__(
            self,
            drivers: Sequence[Placement] | None = None,
            threads: Annotated[int, Ge(1)] | None = None
    ):
        check_options(WorkersMPI.__init__, locals())
        if drivers is not None and not drivers:
            raise ValueError("drivers must give at least one placement")
        self._drivers = None if drivers is None else tuple(drivers)
        self._threads = threads

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self._drivers is not None:
            info['drivers'] = dump_placements(self._drivers)
        if self._threads is not None:
            info['threads'] = self._threads
        return info

    def check(self, objective: Objective) -> None:
        # The number of workers is that of the job, known only under its
        # launcher, so that the configuration of a run on a cluster loads
        # anywhere, e.g. to be evaluated on a login node
        for placement in self._drivers or ():
            placement_drivers(objective, placement)

    def placements(self, objective: Objective) -> list[Drivers]:
        nworkers = _mpi_size() - 1
        if self._drivers is None:
            return [own_placement(objective)] * nworkers
        if len(self._drivers) != nworkers:
            raise ValueError(
                f"drivers must give a placement for each of the {nworkers} "
                f"workers (the ranks of the job but the first); it gives "
                f"{len(self._drivers)}")
        return [placement_drivers(objective, p) for p in self._drivers]

    def launch(
            self,
            objective: Objective,
            placements: Sequence[Drivers]
    ) -> list[Worker]:
        from mpi4py import MPI
        key = next(_mpi_pools)
        return [
            _MPIWorker(
                MPI.COMM_WORLD, rank, key, objective.likelihoods(),
                objective.solves(), placement, self._threads)
            for rank, placement in enumerate(placements, 1)]


# Whether the program runs under the launcher of GBKFIT (see main), and
# the keys of the pools of its workers of MPI
_mpi_launched = False
_mpi_pools = itertools.count()

# The tags of the messages of MPI: of the first rank to a worker, and of a
# worker to the first rank. As a function runs on a worker, the first rank
# sends it stops, and the other pools of the program may send it messages
# of their own (see _serve_mpi)
_MPI_TO_WORKER = 1
_MPI_TO_FIRST = 2

# The lock of the calls of MPI of the first rank, from several threads
_mpi_lock = threading.Lock()


def _mpi_size() -> int:
    """
    Return the number of ranks of the MPI job, or raise RuntimeError unless
    the launcher of GBKFIT started the program, and the job has ranks of
    workers.
    """
    if not _mpi_launched:
        raise RuntimeError(
            "workers of MPI need the program to run under the launcher of "
            "GBKFIT: mpiexec -n N python -m gbkfit.parallel <program>")
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    if size < 2:
        raise RuntimeError(
            "the MPI job has no ranks for workers (its first runs the "
            "program): start it with mpiexec -n N, N of at least 2")
    return size


class _MPIWorker(Worker):
    """
    A worker on a rank of an MPI job (see WorkersMPI): the first rank sends
    it messages, and a thread of its own waits for its results.
    """

    def __init__(
            self,
            comm: Any,
            rank: int,
            key: int,
            likelihoods: Likelihoods,
            solves: Mapping[str, Solve],
            placement: Drivers,
            threads: int | None
    ):
        self._comm = comm
        self._rank = rank
        self._key = key
        self._executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix='gbkfit-mpi')
        self._send(('start', key, likelihoods, solves, placement, threads))

    def submit(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            self._call, ('run', self._key, function, items))

    def submit_controlled(
            self,
            function: Callable[[Objective, Item, StopControl], Result],
            item: Item
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            self._call, ('run_controlled', self._key, function, item))

    def interrupt(self) -> None:
        self._send(('stop',))

    def clear(self) -> None:
        # A function on a rank has a flag of its own, see _serve_mpi
        pass

    def close(self) -> None:
        self._executor.shutdown()
        self._send(('close', self._key))

    def _send(self, message: tuple) -> None:
        """Send a message to the rank."""
        with _mpi_lock:
            self._comm.send(message, dest=self._rank, tag=_MPI_TO_WORKER)

    def _call(self, message: tuple) -> Any:
        """
        Send a message of a function to the rank, and return its result
        (or raise its exception).
        """
        self._send(message)
        with _mpi_lock:
            request = self._comm.irecv(source=self._rank, tag=_MPI_TO_FIRST)
        while True:
            with _mpi_lock:
                done, reply = request.test()
            if done:
                break
            time.sleep(0.001)
        ok, value = reply
        if not ok:
            raise value
        return value


def _serve_mpi(comm: Any) -> None:
    """
    Serve the workers of the pools of the program on a rank of an MPI job
    (see WorkersMPI), until the first rank says to exit.

    Every call gets a reply, so that the first rank never waits for a
    rank that failed: a copy that cannot be made fails the calls of its
    pool with its error.
    """
    copies: dict[int, Objective | Exception] = {}
    # The messages that came as a function ran (see _mpi_stop_sent)
    pending: collections.deque[tuple] = collections.deque()
    while True:
        kind, *args = pending.popleft() if pending \
            else comm.recv(source=0, tag=_MPI_TO_WORKER)
        if kind == 'exit':
            return
        if kind == 'start':
            key, likelihoods, solves, placement, threads = args
            try:
                copies[key] = objective_on(likelihoods, solves, placement)
            except Exception as e:
                copies[key] = e
            if threads is not None:
                threadpoolctl.threadpool_limits(threads)
        elif kind == 'close':
            copy = copies.pop(args[0])
            if isinstance(copy, Objective):
                copy.close()
        elif kind in ('run', 'run_controlled'):
            key, function, payload = args
            try:
                copy = copies[key]
                if isinstance(copy, Exception):
                    raise copy
                if kind == 'run':
                    value = [function(copy, item) for item in payload]
                else:
                    control = StopControl(PolledFlag(
                        lambda: _mpi_stop_sent(comm, pending)))
                    value = function(copy, payload, control)
                reply = (True, value)
            except Exception as e:
                reply = (False, e)
            _mpi_reply(comm, reply)
        # Else a stop that came after its function ended, which is dropped


def _mpi_reply(comm: Any, reply: tuple[bool, Any]) -> None:
    """
    Send the reply of a call to the first rank: a value or an error that
    cannot be pickled goes as a RuntimeError that says what it was.
    """
    try:
        comm.send(reply, dest=0, tag=_MPI_TO_FIRST)
    except Exception as e:
        ok, value = reply
        what = f"the result of type {type(value).__name__}" if ok \
            else f"the error {type(value).__name__}: {value}"
        comm.send(
            (False, RuntimeError(f"{what} could not be sent: {e}")),
            dest=0, tag=_MPI_TO_FIRST)


def _mpi_stop_sent(comm: Any, pending: collections.deque[tuple]) -> bool:
    """
    Return whether the first rank sent a rank a stop (and take it), as a
    function runs there; the other messages that came (e.g. of another
    pool) are kept for after it, in order.
    """
    while comm.iprobe(source=0, tag=_MPI_TO_WORKER):
        message = comm.recv(source=0, tag=_MPI_TO_WORKER)
        if message[0] == 'stop':
            return True
        pending.append(message)
    return False


def main(argv: Sequence[str]) -> None:
    """
    Run a program under MPI with workers of MPI (see WorkersMPI): on the
    first rank, the Python file or the module (-m) of argv, with the
    arguments after it; on the others, the workers.
    """
    from mpi4py import MPI
    global _mpi_launched
    _mpi_launched = True
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        _serve_mpi(comm)
        return
    # The other ranks exit however the program ends, a wrong command too
    try:
        if not argv or (argv[0] == '-m' and len(argv) < 2):
            raise SystemExit(
                "usage: mpiexec -n N python -m gbkfit.parallel "
                "(file | -m module) [arguments]")
        if argv[0] == '-m':
            sys.argv = list(argv[1:])
            runpy.run_module(argv[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = list(argv)
            runpy.run_path(argv[0], run_name='__main__')
    finally:
        for rank in range(1, comm.Get_size()):
            comm.send(('exit',), dest=rank, tag=_MPI_TO_WORKER)
