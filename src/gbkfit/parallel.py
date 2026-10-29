"""
Workers that evaluate an objective at the same time, each on a copy of
it (see Workers): e.g. the batches of points of a fit (see Fitter.fit),
and its sub-fits (e.g. the refits of a bootstrap; see run_subfits), in
threads of this process (e.g. one per GPU), in processes of this
machine, or on the machines of a cluster (MPI, Ray, Dask; their modules
are optional, see requires).

Run as a module, it is the launcher of programs with workers of MPI (see
WorkersMPI): mpiexec -n N python -m gbkfit.parallel (file | -m module)
[arguments].
"""

import abc
import concurrent.futures
import itertools
import math
import multiprocessing
import os
import runpy
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Self, TypeAlias, TypeVar

import threadpoolctl
from annotated_types import Ge, Gt

from gbkfit.driver import Driver, driver_parser
from gbkfit.fitting.control import FitControl, FitStopped, StopFlag
from gbkfit.objective import Objective
from gbkfit.observation import GroupPlan, ObservationGroup
from gbkfit.params import Solve
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, check_options


__all__ = [
    'Placement',
    'WorkerPool',
    'Workers',
    'WorkersDask',
    'WorkersMPI',
    'WorkersProcesses',
    'WorkersRay',
    'WorkersThreads',
    'workers_parser'
]


_Item = TypeVar('_Item')
_Result = TypeVar('_Result')

# How often the feeders of the workers look whether to stop them, as they
# wait for their results (s; see WorkerPool.map_lazily)
_POLL_INTERVAL = 0.2

# Where a worker evaluates its copy of an objective: a driver for all its
# work, or one with other drivers of some of its galaxies and
# observations, by label (e.g. (gpu0, {'hi': gpu1}): a copy split over two
# GPUs, for models too big for one; see ObservationGroup.plan)
Placement: TypeAlias = Driver | tuple[Driver, Mapping[str, Driver]]

# The drivers of a copy: of each galaxy and of each observation (see
# ObservationGroup.placement)
_Drivers: TypeAlias = tuple[tuple[Driver, ...], tuple[Driver, ...]]


class Workers(parseutils.TypedSerializable, abc.ABC):
    """
    A kind of workers that evaluate an objective at the same time, each on
    a copy of it (see start and WorkerPool).

    In configurations (the section 'workers'), the placements of the
    workers (the option drivers of the kinds that have it) are drivers:
    names of those of the configuration (its section 'drivers'), or
    drivers.
    """

    def start(self, objective: Objective) -> 'WorkerPool':
        """
        Return workers of copies of an objective (see WorkerPool), which
        start on their first use.

        Parameters
        ----------
        objective : Objective
            The objective.

        Returns
        -------
        WorkerPool
            The workers.

        Raises
        ------
        ConfigError
            If a placement names unknown galaxies or observations (see
            ObservationGroup.placement).
        """
        return WorkerPool(objective, self)

    @abc.abstractmethod
    def placements(self, objective: Objective) -> list[_Drivers]:
        """
        Return the drivers of the copy of each worker of an objective.

        Raises
        ------
        ConfigError
            If a placement names unknown galaxies or observations.
        """

    @abc.abstractmethod
    def launch(
            self,
            objective: Objective,
            placements: Sequence[_Drivers]
    ) -> list['_Worker']:
        """Start a worker of a copy of an objective on each placement."""


class _Worker(abc.ABC):
    """
    A worker, which runs functions of its copy of an objective, and stops
    them when it is interrupted (see submit_controlled).
    """

    @abc.abstractmethod
    def submit(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> concurrent.futures.Future:
        """
        Return the future of function(copy, item) for each item, in order.
        """

    @abc.abstractmethod
    def submit_controlled(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            item: _Item
    ) -> concurrent.futures.Future:
        """
        Return the future of function(copy, item, control), with the
        control of its flag (see interrupt).
        """

    @abc.abstractmethod
    def interrupt(self) -> None:
        """
        Set the flag of the control of the function it runs: stop it (see
        submit_controlled).
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear the flag."""

    @abc.abstractmethod
    def close(self) -> None:
        """Stop the worker."""


class _LocalWorker(_Worker, abc.ABC):
    """A worker of this machine, whose flag is a shared event."""

    def __init__(self, flag: StopFlag):
        self._flag = flag

    def interrupt(self) -> None:
        self._flag.set()

    def clear(self) -> None:
        self._flag.clear()


class WorkerPool:
    """
    Workers of copies of an objective (see Workers.start), which evaluate
    functions of it at the same time (see map). Item i of a map goes to
    worker i mod N (of N), so that a fit gives the same results each time
    it runs (to the rounding of the sums that its kernels make in
    parallel), also with workers on different devices.

    The placements are checked when it is made; the workers start on first
    use (e.g. the first batch of a fit), so that workers that are not used
    (e.g. those of the configuration of a fit that is only evaluated or
    checked) take no memory.

    Parameters
    ----------
    objective : Objective
        The objective.
    workers : Workers
        The kind of the workers.

    Raises
    ------
    ConfigError
        If a placement names unknown galaxies or observations (see
        ObservationGroup.placement).
    """

    def __init__(self, objective: Objective, workers: Workers):
        self._objective = objective
        self._workers = workers
        self._placements = workers.placements(objective)
        self._started: list[_Worker] | None = None

    def objective(self) -> Objective:
        """Return the objective that the workers have copies of."""
        return self._objective

    def size(self) -> int:
        """Return the number of workers."""
        return len(self._placements)

    def map(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> list[_Result]:
        """
        Return function(copy, item) for each item, in order: item i on
        worker i mod N, the workers at once. It returns once every worker
        has finished, also when one raises an exception (then it raises
        that of the first worker that raised one). In workers of other
        processes, the function and the items are pickled.
        """
        workers = self._start()
        n = len(workers)
        futures = [
            workers[k].submit(function, items[k::n])
            for k in range(min(n, len(items)))]
        concurrent.futures.wait(futures)
        chunks = [future.result() for future in futures]
        return [chunks[i % n][i // n] for i in range(len(items))]

    def map_lazily(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            make: Callable[[int], _Item],
            count: int,
            on_result: Callable[[int, _Result], None] | None = None,
            control: FitControl | None = None
    ) -> list[_Result]:
        """
        Return function(copy, make(i), control_k) for i < count, in order:
        item i is made when worker k = i mod N takes it, after item i - N,
        so that at most N items exist at once (e.g. the data of the refits
        of a bootstrap), and its result goes to on_result, if given, as it
        comes. Workers of threads run one item at a time (see
        WorkersThreads). make and on_result are called in threads of this process,
        one per worker. It returns once every worker has finished, also
        when one raises an exception (then it raises that of the first
        worker that raised one).

        The function takes the control of its worker (control_k; see
        FitControl), which stops it (e.g. a sub-fit; see run_subfits) once
        the given control is stopped: then no item is made any more, and
        it raises FitStopped once the workers have finished (the results
        that came went to on_result).
        """
        workers = self._start()
        n = len(workers)
        for worker in workers:
            worker.clear()

        def stopped() -> bool:
            return control is not None and control.stopped()

        def run(k: int) -> list[_Result]:
            results = []
            for i in range(k, count, n):
                if stopped():
                    raise FitStopped("the work was asked to stop")
                future = workers[k].submit_controlled(function, make(i))
                while not concurrent.futures.wait(
                        [future], timeout=_POLL_INTERVAL).done:
                    if stopped():
                        workers[k].interrupt()
                result = future.result()
                if on_result is not None:
                    on_result(i, result)
                results.append(result)
            return results
        with concurrent.futures.ThreadPoolExecutor(
                max(1, n), thread_name_prefix='gbkfit-feeder') as feeders:
            futures = [feeders.submit(run, k) for k in range(min(n, count))]
            concurrent.futures.wait(futures)
            chunks = [future.result() for future in futures]
        return [chunks[i % n][i // n] for i in range(count)]

    def _start(self) -> list['_Worker']:
        """Return the workers, and start them on the first call."""
        if self._started is None:
            self._started = self._workers.launch(
                self._objective, self._placements)
        return self._started

    def close(self) -> None:
        """Stop the workers, if started."""
        for worker in self._started or ():
            worker.close()
        self._started = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class WorkersThreads(Workers):
    """
    Workers in threads of this process, one on each of the given
    placements (e.g. one per GPU). The worker of the first placement of
    the objective has the objective itself; the others have copies of it
    that share its data, with memory of their own (see GroupPlan).

    The threads overlap while they wait for their devices (GPUs); the host
    driver gains nothing from them (its kernels use every core), nor do
    the parts of the models that run in Python (see WorkersProcesses).

    Its sub-fits (see WorkerPool.map_lazily; e.g. the refits of a
    bootstrap) run one after the other: fitters and their libraries keep
    state of their process (e.g. CMA-ES its random numbers, of numpy), so
    that two at once in one process would not give the results of each
    alone. Workers of processes run them at once.

    Parameters
    ----------
    drivers : Sequence of Placement
        The placement of the copy of each worker; a placement can be
        listed more than once (e.g. two copies on one GPU).

    Raises
    ------
    ConfigError
        If there are no placements.
    """

    # (see Workers)
    config_options = {'drivers': Sequence[str | Driver]}

    @staticmethod
    def type() -> str:
        return 'threads'

    def __init__(self, drivers: Sequence[Placement]):
        check_options(WorkersThreads.__init__, locals())
        if not drivers:
            raise ConfigError("drivers must give at least one placement")
        self._drivers = tuple(drivers)

    def dump(self) -> dict[str, Any]:
        return dict(drivers=_dump_placements(self._drivers))

    def placements(self, objective: Objective) -> list[_Drivers]:
        return [_placement(objective, p) for p in self._drivers]

    def launch(
            self,
            objective: Objective,
            placements: Sequence[_Drivers]
    ) -> list[_Worker]:
        own = _own_placement(objective)
        # (one sub-fit at a time, see the class)
        lock = threading.Lock()
        workers = []
        for placement in placements:
            if placement == own and objective not in (
                    w.copy for w in workers):
                workers.append(_ThreadWorker(objective, False, lock))
            else:
                workers.append(_ThreadWorker(Objective(
                    GroupPlan(objective.group(), *placement),
                    objective.solves()), True, lock))
        return workers


class _ThreadWorker(_LocalWorker):
    """
    A worker in a thread of its own, of a copy of an objective (or of the
    objective itself, which it does not own: it does not close it), whose
    controlled functions hold the lock that the workers of its pool share
    (see WorkersThreads).
    """

    def __init__(self, copy: Objective, owned: bool, lock: threading.Lock):
        super().__init__(threading.Event())
        self.copy = copy
        self._owned = owned
        self._lock = lock
        self._control = FitControl(self._flag)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix='gbkfit-worker')

    def submit(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            lambda: [function(self.copy, item) for item in items])

    def submit_controlled(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            item: _Item
    ) -> concurrent.futures.Future:
        def run() -> _Result:
            with self._lock:
                return function(self.copy, item, self._control)
        return self._executor.submit(run)

    def close(self) -> None:
        self._executor.shutdown()
        if self._owned:
            self.copy.close()


class WorkersProcesses(Workers):
    """
    Workers in processes of this machine: a number of them on the drivers
    of the objective, or one on each of the given placements (e.g. one per
    GPU). Each process imports GBKFIT (spawned, see multiprocessing) and
    makes its copy of the objective, which goes to it pickled (see
    Objective): its data, once.

    Unlike threads (see WorkersThreads), processes evaluate at the same
    time the parts of the models that run in Python, and models on the
    host: then each has its share of the cores, as the kernels of the host
    driver use every core they are given.

    Parameters
    ----------
    count : int, optional
        The number of workers on the drivers of the objective.
    drivers : Sequence of Placement, optional
        The placement of the copy of each worker, in place of count; a
        placement can be listed more than once (e.g. two copies on one
        GPU).
    threads : int, optional
        The threads of the host of each worker (of the kernels of the host
        driver, and of BLAS); by default, the cores of the machine shared
        out among the workers.

    Raises
    ------
    ConfigError
        If neither or both of count and drivers are given, or there are no
        placements.
    """

    # (see Workers)
    config_options = {'drivers': Sequence[str | Driver] | None}

    @staticmethod
    def type() -> str:
        return 'processes'

    def __init__(
            self,
            count: Annotated[int, Ge(1)] | None = None,
            drivers: Sequence[Placement] | None = None,
            threads: Annotated[int, Ge(1)] | None = None
    ):
        check_options(WorkersProcesses.__init__, locals())
        if (count is None) == (drivers is None):
            raise ConfigError("give count or drivers (and not both)")
        if drivers is not None and not drivers:
            raise ConfigError("drivers must give at least one placement")
        self._count = count
        self._drivers = None if drivers is None else tuple(drivers)
        self._threads = threads

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(count=self._count) \
            if self._drivers is None \
            else dict(drivers=_dump_placements(self._drivers))
        if self._threads is not None:
            info['threads'] = self._threads
        return info

    def placements(self, objective: Objective) -> list[_Drivers]:
        if self._drivers is None:
            return [_own_placement(objective)] * self._count
        return [_placement(objective, p) for p in self._drivers]

    def launch(
            self,
            objective: Objective,
            placements: Sequence[_Drivers]
    ) -> list[_Worker]:
        threads = self._threads or max(1, _cores() // len(placements))
        return [
            _ProcessWorker(
                objective.group(), objective.solves(), placement, threads)
            for placement in placements]


class _ProcessWorker(_LocalWorker):
    """A worker in a process of its own, of a copy of an objective."""

    def __init__(
            self,
            group: ObservationGroup,
            solves: Mapping[str, Solve],
            placement: _Drivers,
            threads: int
    ):
        context = multiprocessing.get_context('spawn')
        super().__init__(context.Event())
        self._executor = concurrent.futures.ProcessPoolExecutor(
            1, mp_context=context, initializer=_start_process,
            initargs=(group, solves, placement, threads, self._flag))

    def submit(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(_run_in_process, function, items)

    def submit_controlled(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            item: _Item
    ) -> concurrent.futures.Future:
        return self._executor.submit(_run_controlled_in_process, function, item)

    def close(self) -> None:
        self._executor.shutdown()


# The copy of the objective of a worker process, and the control of its
# flag (see _start_process)
_process_copy: Objective | None = None
_process_control: FitControl | None = None


def _start_process(
        group: ObservationGroup,
        solves: Mapping[str, Solve],
        placement: _Drivers,
        threads: int,
        flag: StopFlag
) -> None:
    """
    Make the copy of the objective of a worker process and the control of
    its flag, and give its host its threads (those of the libraries loaded
    by then, as the host driver loads its own).
    """
    global _process_copy, _process_control
    _process_copy = Objective(GroupPlan(group, *placement), solves)
    _process_control = FitControl(flag)
    threadpoolctl.threadpool_limits(threads)


def _run_in_process(
        function: Callable[[Objective, _Item], _Result],
        items: Sequence[_Item]
) -> list[_Result]:
    """Return function(copy, item) for each item, in a worker process."""
    return [function(_process_copy, item) for item in items]


def _run_controlled_in_process(
        function: Callable[[Objective, _Item, FitControl], _Result],
        item: _Item
) -> _Result:
    """Return function(copy, item, control), in a worker process."""
    return function(_process_copy, item, _process_control)


class WorkersMPI(Workers):
    """
    Workers in the processes (ranks) of an MPI job: one on each rank but
    the first, which runs the program, started by the launcher of GBKFIT
    (see _main), e.g. with 4 workers:

        mpiexec -n 5 python -m gbkfit.parallel script.py [arguments]
        mpiexec -n 5 python -m gbkfit.parallel -m module [arguments]

    The ranks of the workers serve every pool of the program, each with
    its copy of the objective, which goes to it pickled (see Objective).
    Like processes (see WorkersProcesses), they evaluate at the same time
    the parts of the models that run in Python and models on the host,
    and run sub-fits at once, on the machines of the job.

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
    ConfigError
        If drivers gives no placements.
    """

    requires = {'mpi4py': "pip install 'gbkfit[mpi]'"}

    # (see Workers)
    config_options = {'drivers': Sequence[str | Driver] | None}

    @staticmethod
    def type() -> str:
        return 'mpi'

    def __init__(
            self,
            drivers: Sequence[Placement] | None = None,
            threads: Annotated[int, Ge(1)] | None = None
    ):
        check_options(WorkersMPI.__init__, locals())
        if drivers is not None and not drivers:
            raise ConfigError("drivers must give at least one placement")
        self._drivers = None if drivers is None else tuple(drivers)
        self._threads = threads

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self._drivers is not None:
            info['drivers'] = _dump_placements(self._drivers)
        if self._threads is not None:
            info['threads'] = self._threads
        return info

    def placements(self, objective: Objective) -> list[_Drivers]:
        nworkers = _mpi_size() - 1
        if self._drivers is None:
            return [_own_placement(objective)] * nworkers
        if len(self._drivers) != nworkers:
            raise ConfigError(
                f"drivers must give a placement for each of the {nworkers} "
                f"workers (the ranks of the job but the first); it gives "
                f"{len(self._drivers)}")
        return [_placement(objective, p) for p in self._drivers]

    def launch(
            self,
            objective: Objective,
            placements: Sequence[_Drivers]
    ) -> list[_Worker]:
        from mpi4py import MPI
        key = next(_mpi_pools)
        return [
            _MPIWorker(
                MPI.COMM_WORLD, rank, key, objective.group(),
                objective.solves(), placement, self._threads)
            for rank, placement in enumerate(placements, 1)]


# Whether the program runs under the launcher of GBKFIT (see _main), and
# the keys of the pools of its workers of MPI
_mpi_launched = False
_mpi_pools = itertools.count()

# The tags of the messages of MPI: of the first rank to a worker, and of a
# worker to the first rank. The first rank sends a worker one message at a
# time (to stop the function it runs, the second), in order
_MPI_TO_WORKER = 1
_MPI_TO_FIRST = 2

# The lock of the calls of MPI of the first rank, from several threads
_mpi_lock = threading.Lock()


def _mpi_size() -> int:
    """
    Return the number of ranks of the MPI job, or raise ConfigError unless
    the launcher of GBKFIT started the program, and the job has ranks of
    workers.
    """
    if not _mpi_launched:
        raise ConfigError(
            "workers of MPI need the program to run under the launcher of "
            "GBKFIT: mpiexec -n N python -m gbkfit.parallel <program>")
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    if size < 2:
        raise ConfigError(
            "the MPI job has no ranks for workers (its first runs the "
            "program): start it with mpiexec -n N, N of at least 2")
    return size


class _MPIWorker(_Worker):
    """
    A worker on a rank of an MPI job (see WorkersMPI): the first rank sends
    it messages, and a thread of its own waits for its results.
    """

    def __init__(
            self,
            comm: Any,
            rank: int,
            key: int,
            group: ObservationGroup,
            solves: Mapping[str, Solve],
            placement: _Drivers,
            threads: int | None
    ):
        self._comm = comm
        self._rank = rank
        self._key = key
        self._executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix='gbkfit-mpi')
        self._send(('start', key, group, solves, placement, threads))

    def submit(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            self._call, ('run', self._key, function, items))

    def submit_controlled(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            item: _Item
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            self._call, ('run_controlled', self._key, function, item))

    def interrupt(self) -> None:
        self._send(('stop',))

    def clear(self) -> None:
        # (a function on a rank has a flag of its own, see _serve_mpi)
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
    """
    copies: dict[int, Objective] = {}
    while True:
        kind, *args = comm.recv(source=0, tag=_MPI_TO_WORKER)
        if kind == 'exit':
            return
        if kind == 'start':
            key, group, solves, placement, threads = args
            copies[key] = Objective(GroupPlan(group, *placement), solves)
            if threads is not None:
                threadpoolctl.threadpool_limits(threads)
        elif kind == 'close':
            copies.pop(args[0]).close()
        elif kind in ('run', 'run_controlled'):
            key, function, payload = args
            try:
                if kind == 'run':
                    value = [function(copies[key], item) for item in payload]
                else:
                    control = FitControl(_PolledFlag(
                        lambda: _mpi_stop_sent(comm)))
                    value = function(copies[key], payload, control)
                reply = (True, value)
            except Exception as e:
                reply = (False, e)
            comm.send(reply, dest=0, tag=_MPI_TO_FIRST)
        # (else a stop that came after the function ended)


def _mpi_stop_sent(comm: Any) -> bool:
    """
    Return whether the first rank sent a rank a stop (and take it), as a
    function runs there: it sends nothing else then.
    """
    if not comm.iprobe(source=0, tag=_MPI_TO_WORKER):
        return False
    kind, *_ = comm.recv(source=0, tag=_MPI_TO_WORKER)
    return kind == 'stop'


def _main(argv: Sequence[str]) -> None:
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
    if not argv or (argv[0] == '-m' and len(argv) < 2):
        raise SystemExit(
            "usage: mpiexec -n N python -m gbkfit.parallel "
            "(file | -m module) [arguments]")
    try:
        if argv[0] == '-m':
            sys.argv = list(argv[1:])
            runpy.run_module(argv[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = list(argv)
            runpy.run_path(argv[0], run_name='__main__')
    finally:
        for rank in range(1, comm.Get_size()):
            comm.send(('exit',), dest=rank, tag=_MPI_TO_WORKER)


class _PolledFlag:
    """
    A stop flag kept elsewhere (e.g. by the first rank of an MPI job, or
    by the scheduler of a Dask cluster), asked at most every
    _POLL_INTERVAL seconds (see StopFlag); once set, it stays set.
    """

    def __init__(self, ask: Callable[[], bool]):
        self._ask = ask
        self._set = False
        self._asked = -math.inf

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        now = time.monotonic()
        if not self._set and now - self._asked >= _POLL_INTERVAL:
            self._asked = now
            self._set = self._ask()
        return self._set


class _ActorCopy:
    """
    The copy of an objective of a worker that is an actor of Ray or Dask
    (see WorkersRay and WorkersDask), which runs functions of it: those
    with a control, with a flag of its own (Ray), or the given one (Dask).
    """

    def __init__(
            self,
            group: ObservationGroup,
            solves: Mapping[str, Solve],
            placement: _Drivers,
            threads: int | None
    ):
        self._copy = Objective(GroupPlan(group, *placement), solves)
        self._flag = threading.Event()
        if threads is not None:
            threadpoolctl.threadpool_limits(threads)

    def run(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> list[_Result]:
        return [function(self._copy, item) for item in items]

    def run_controlled(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            item: _Item,
            flag: StopFlag | None = None
    ) -> _Result:
        control = FitControl(self._flag if flag is None else flag)
        return function(self._copy, item, control)

    def interrupt(self) -> None:
        self._flag.set()

    def clear(self) -> None:
        self._flag.clear()


class WorkersRay(Workers):
    """
    Workers in actors of a Ray cluster: a number of them, each on CPUs and
    GPUs of the cluster, with its copy of the objective, which goes to it
    pickled (see Objective). Ray gives an actor its GPUs: the CUDA drivers
    of its copy of device 0 are on the first. Like processes (see
    WorkersProcesses), they evaluate at the same time the parts of the
    models that run in Python, and run sub-fits at once, on the machines
    of the cluster.

    The cluster is that of Ray in this process, if it has one (see
    ray.init), or else that of the address.

    Parameters
    ----------
    count : int
        The number of workers.
    address : str, optional
        The address of the cluster (see ray.init); by default, that of the
        environment variable RAY_ADDRESS, or else a cluster of this
        machine, started for them.
    cpus : float, optional
        The CPUs of each worker.
    gpus : float, optional
        The GPUs of each worker (none by default).
    threads : int, optional
        The threads of the host of each worker (of the kernels of the host
        driver, and of BLAS); by default, its CPUs (at least 1).
    """

    requires = {'ray': "pip install 'gbkfit[ray]'"}

    @staticmethod
    def type() -> str:
        return 'ray'

    def __init__(
            self,
            count: Annotated[int, Ge(1)],
            address: str | None = None,
            cpus: Annotated[float, Gt(0)] = 1.0,
            gpus: Annotated[float, Ge(0)] = 0.0,
            threads: Annotated[int, Ge(1)] | None = None
    ):
        check_options(WorkersRay.__init__, locals())
        self._count = count
        self._address = address
        self._cpus = cpus
        self._gpus = gpus
        self._threads = threads

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(
            count=self._count, cpus=self._cpus, gpus=self._gpus)
        if self._address is not None:
            info['address'] = self._address
        if self._threads is not None:
            info['threads'] = self._threads
        return info

    def placements(self, objective: Objective) -> list[_Drivers]:
        return [_own_placement(objective)] * self._count

    def launch(
            self,
            objective: Objective,
            placements: Sequence[_Drivers]
    ) -> list[_Worker]:
        import ray
        if not ray.is_initialized():
            ray.init(address=self._address)
        # (two at once: a function, and its interruption)
        actor = ray.remote(_ActorCopy).options(
            num_cpus=self._cpus, num_gpus=self._gpus, max_concurrency=2)
        threads = self._threads or max(1, math.floor(self._cpus))
        return [
            _RayWorker(actor.remote(
                objective.group(), objective.solves(), placement, threads))
            for placement in placements]


class _RayWorker(_Worker):
    """A worker that is an actor of Ray (see WorkersRay)."""

    def __init__(self, actor: Any):
        self._actor = actor

    def submit(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> concurrent.futures.Future:
        return self._actor.run.remote(function, items).future()

    def submit_controlled(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            item: _Item
    ) -> concurrent.futures.Future:
        return self._actor.run_controlled.remote(function, item).future()

    def interrupt(self) -> None:
        self._actor.interrupt.remote()

    def clear(self) -> None:
        import ray
        ray.get(self._actor.clear.remote())

    def close(self) -> None:
        import ray
        ray.kill(self._actor)


class WorkersDask(Workers):
    """
    Workers in actors of a Dask cluster: a number of them, each on a worker
    of the cluster (of its own, while there are enough), with its copy of
    the objective, which goes to it pickled (see Objective). Like
    processes (see WorkersProcesses), they evaluate at the same time the
    parts of the models that run in Python, and run sub-fits at once, on
    the machines of the cluster.

    The cluster is that of the client of this process (see
    distributed.get_client), or else that of the address of its scheduler.

    Parameters
    ----------
    count : int
        The number of workers.
    address : str, optional
        The address of the scheduler of the cluster (e.g.
        'tcp://10.0.0.1:8786'); by default, that of the client of this
        process.
    threads : int, optional
        The threads of the host of each worker (of the kernels of the host
        driver, and of BLAS); by default, those of its Dask worker.
    """

    requires = {'distributed': "pip install 'gbkfit[dask]'"}

    @staticmethod
    def type() -> str:
        return 'dask'

    def __init__(
            self,
            count: Annotated[int, Ge(1)],
            address: str | None = None,
            threads: Annotated[int, Ge(1)] | None = None
    ):
        check_options(WorkersDask.__init__, locals())
        self._count = count
        self._address = address
        self._threads = threads

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(count=self._count)
        if self._address is not None:
            info['address'] = self._address
        if self._threads is not None:
            info['threads'] = self._threads
        return info

    def placements(self, objective: Objective) -> list[_Drivers]:
        return [_own_placement(objective)] * self._count

    def launch(
            self,
            objective: Objective,
            placements: Sequence[_Drivers]
    ) -> list[_Worker]:
        import distributed
        client = distributed.get_client() if self._address is None \
            else distributed.Client(self._address)
        hosts = list(client.scheduler_info()['workers'])
        return [
            _DaskWorker(client.submit(
                _ActorCopy, objective.group(), objective.solves(),
                placement, self._threads, actor=True, pure=False,
                workers=[hosts[k % len(hosts)]]).result())
            for k, placement in enumerate(placements)]


class _DaskWorker(_Worker):
    """
    A worker that is an actor of Dask (see WorkersDask): a thread of its
    own waits for its results. The flag of each function it runs is an
    event of the cluster of its own (see distributed.Event).
    """

    def __init__(self, actor: Any):
        self._actor = actor
        self._event: str | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix='gbkfit-dask')

    def submit(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            lambda: self._actor.run(function, items).result())

    def submit_controlled(
            self,
            function: Callable[[Objective, _Item, FitControl], _Result],
            item: _Item
    ) -> concurrent.futures.Future:
        self._event = f'gbkfit-stop-{uuid.uuid4().hex}'
        flag = _DaskFlag(self._event)
        return self._executor.submit(
            lambda: self._actor.run_controlled(function, item, flag).result())

    def interrupt(self) -> None:
        if self._event is not None:
            import distributed
            distributed.Event(self._event).set()

    def clear(self) -> None:
        # (each function has a flag of its own)
        pass

    def close(self) -> None:
        self._executor.shutdown()


class _DaskFlag(_PolledFlag):
    """The flag of a function of a worker of Dask: an event of its cluster."""

    def __init__(self, name: str):
        super().__init__(self._ask_cluster)
        self._name = name

    def _ask_cluster(self) -> bool:
        import distributed
        return distributed.Event(self._name).is_set()

    def __reduce__(self) -> tuple[type, tuple[str]]:
        return _DaskFlag, (self._name,)


def _cores() -> int:
    """
    Return the cores that this process may run on (e.g. those of a job of
    a cluster), or those of the machine where that is not known.
    """
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _own_placement(objective: Objective) -> _Drivers:
    """Return the drivers of an objective."""
    plan = objective.group_plan()
    return plan.galaxy_drivers(), plan.observation_drivers()


def _placement(objective: Objective, placement: Placement) -> _Drivers:
    """
    Return the drivers of a copy of an objective on a placement, or raise
    ConfigError if it names unknown galaxies or observations.
    """
    driver, drivers = (placement, None) if isinstance(placement, Driver) \
        else placement
    return objective.group().placement(driver, drivers)


def _dump_placements(placements: Sequence[Placement]) -> list[Any]:
    """
    Return the configurations of placements: those of their drivers (see
    Workers). Raise ConfigError for placements of drivers by label, which
    configurations do not have.
    """
    if any(not isinstance(p, Driver) for p in placements):
        raise ConfigError(
            "workers on placements of drivers by label cannot be dumped: "
            "configurations give one driver of each worker")
    return [driver_parser.dump(p) for p in placements]


workers_parser = parseutils.TypedParser(Workers)
workers_parser.register([
    WorkersDask, WorkersMPI, WorkersProcesses, WorkersRay, WorkersThreads])


if __name__ == '__main__':
    # (in the module itself, not this copy of it: see _main)
    import gbkfit.parallel
    gbkfit.parallel._main(sys.argv[1:])
