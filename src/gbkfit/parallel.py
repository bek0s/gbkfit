"""
Workers that evaluate an objective at the same time, each on a copy of
it (see Workers): e.g. the batches of points of a fit (see Fitter.fit),
in threads of this process (e.g. one per GPU) or in processes of this
machine.
"""

import abc
import concurrent.futures
import multiprocessing
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Self, TypeAlias, TypeVar

import threadpoolctl
from annotated_types import Ge

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
    'WorkersProcesses',
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
    workers are drivers: names of those of the configuration (its section
    'drivers'), or drivers.
    """

    # (names of drivers of the configuration, which its load turns into the
    # drivers, or drivers)
    config_options = {'drivers': Sequence[str | Driver] | None}

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
    them when its flag is set (see submit_controlled).
    """

    def __init__(self, flag: StopFlag):
        self._flag = flag

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

    def interrupt(self) -> None:
        """Set its flag: stop the function it runs (see submit_controlled)."""
        self._flag.set()

    def clear(self) -> None:
        """Clear its flag."""
        self._flag.clear()

    @abc.abstractmethod
    def close(self) -> None:
        """Stop the worker."""


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


class _ThreadWorker(_Worker):
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


class _ProcessWorker(_Worker):
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
workers_parser.register([WorkersProcesses, WorkersThreads])
