"""
The kinds of workers (see Workers), and their pools (see WorkerPool),
which evaluate functions of copies of an objective at the same time.
"""

import abc
import concurrent.futures
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Self, TypeAlias

from gbkfit.config import TypedParser, TypedSerializable
from gbkfit.control import StopControl, Stopped
from gbkfit.driver import Driver
from gbkfit.objective import Objective
from ._detail import POLL_INTERVAL, Drivers, Item, Result, Worker


__all__ = [
    'Placement',
    'WorkerPool',
    'Workers',
    'workers_parser'
]


# Where a worker evaluates its copy of an objective: a driver for all its
# work, or one with other drivers of some of its galaxies and
# observations, by label (e.g. (gpu0, {'hi': gpu1}): a copy split over two
# GPUs, for models too big for one; see ObservationGroup.plan)
Placement: TypeAlias = Driver | tuple[Driver, Mapping[str, Driver]]


class Workers(TypedSerializable, abc.ABC):
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
        ValueError
            If a placement names unknown galaxies or observations (see
            ObservationGroup.placement).
        """
        return WorkerPool(objective, self)

    def check(self, objective: Objective) -> None:
        """
        Check the placements of the workers of an objective, as far as
        that needs no running job: by default, by making them (see
        placements).

        Raises
        ------
        ValueError
            If a placement names unknown galaxies or observations.
        """
        self.placements(objective)

    @abc.abstractmethod
    def placements(self, objective: Objective) -> list[Drivers]:
        """
        Return the drivers of the copy of each worker of an objective.

        Raises
        ------
        ValueError
            If a placement names unknown galaxies or observations.
        RuntimeError
            If the job of the workers is not running (see WorkersMPI).
        """

    @abc.abstractmethod
    def launch(
            self,
            objective: Objective,
            placements: Sequence[Drivers]
    ) -> list['Worker']:
        """Start a worker of a copy of an objective on each placement."""


class WorkerPool:
    """
    Workers of copies of an objective (see Workers.start), which evaluate
    functions of it at the same time (see map). Item i of a map goes to
    worker i mod N (of N), so that a run gives the same results each time
    (to the rounding of the sums that kernels make in parallel), also with
    workers on different devices.

    The placements are checked when it is made, as far as that needs no
    running job (see Workers.check); they are made, and the workers start,
    on first use (e.g. the first batch of a fit), so that workers that are
    not used (e.g. those of a configuration that is only evaluated or
    checked, also outside its MPI job) take no memory.

    Parameters
    ----------
    objective : Objective
        The objective.
    workers : Workers
        The kind of the workers.

    Raises
    ------
    ValueError
        If a placement names unknown galaxies or observations (see
        ObservationGroup.placement).
    """

    def __init__(self, objective: Objective, workers: Workers):
        workers.check(objective)
        self._objective = objective
        self._workers = workers
        self._placements: list[Drivers] | None = None
        self._started: list[Worker] | None = None

    def objective(self) -> Objective:
        """Return the objective that the workers have copies of."""
        return self._objective

    def size(self) -> int:
        """Return the number of workers."""
        return len(self._placed())

    def map(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> list[Result]:
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
            function: Callable[[Objective, Item, StopControl], Result],
            make: Callable[[int], Item],
            count: int,
            on_result: Callable[[int, Result], None] | None = None,
            control: StopControl | None = None
    ) -> list[Result]:
        """
        Return function(copy, make(i), control_k) for i < count, in order:
        item i is made when worker k = i mod N takes it, after item i - N,
        so that at most N items exist at once (e.g. the data of the refits
        of a bootstrap), and its result goes to on_result, if given, as it
        comes. Workers of threads run one item at a time (see
        WorkersThreads). make and on_result are called in threads of this
        process, one per worker. It returns once every worker has
        finished, also when one raises an exception (then it raises that
        of the first worker that raised one).

        The function takes the control of its worker (control_k; see
        StopControl), which stops it (e.g. a sub-fit; see run_subfits) once
        the given control is stopped: then no item is made any more, and
        it raises Stopped once the workers have finished (the results
        that came went to on_result).
        """
        workers = self._start()
        n = len(workers)
        for worker in workers:
            worker.clear()

        def stopped() -> bool:
            return control is not None and control.stopped()

        # Set by the first feeder that raises, e.g. of a bug in an item:
        # the others take no new items, so that the error comes at once,
        # not after the rest of the items
        failed = threading.Event()

        def run(k: int) -> list[Result]:
            results = []
            try:
                for i in range(k, count, n):
                    if failed.is_set():
                        break
                    if stopped():
                        raise Stopped("the work was asked to stop")
                    future = workers[k].submit_controlled(function, make(i))
                    while not concurrent.futures.wait(
                            [future], timeout=POLL_INTERVAL).done:
                        if stopped():
                            workers[k].interrupt()
                    result = future.result()
                    if on_result is not None:
                        on_result(i, result)
                    results.append(result)
            except BaseException:
                failed.set()
                raise
            return results
        with concurrent.futures.ThreadPoolExecutor(
                max(1, n), thread_name_prefix='gbkfit-feeder') as feeders:
            futures = [feeders.submit(run, k) for k in range(min(n, count))]
            concurrent.futures.wait(futures)
            chunks = [future.result() for future in futures]
        return [chunks[i % n][i // n] for i in range(count)]

    def _placed(self) -> list[Drivers]:
        """Return the placements, made on the first call."""
        if self._placements is None:
            self._placements = self._workers.placements(self._objective)
        return self._placements

    def _start(self) -> list['Worker']:
        """Return the workers, and start them on the first call."""
        if self._started is None:
            self._started = self._workers.launch(
                self._objective, self._placed())
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


workers_parser = TypedParser(Workers, family='workers')
