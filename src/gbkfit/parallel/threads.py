"""
Workers in threads of this process.
"""

import concurrent.futures
import threading
from collections.abc import Callable, Sequence
from typing import Any

from gbkfit.config import check_options
from gbkfit.control import StopControl
from gbkfit.driver import Driver
from gbkfit.objective import Objective
from gbkfit.observation import GroupPlan
from ._detail import (
    Drivers, Item, LocalWorker, Result, Worker, dump_placements,
    own_placement, placement_drivers)
from .pool import Placement, Workers


__all__ = [
    'WorkersThreads'
]


class WorkersThreads(Workers):
    """
    Workers in threads of this process, one on each of the given
    placements (e.g. one per GPU). The worker of the first placement of
    the objective has the objective itself; the others have copies of it
    that share its data, with memory of their own (see GroupPlan).

    The threads overlap while they wait for their devices: one copy per
    GPU (copies on one GPU do not overlap: each evaluation waits for its
    device to finish); the host driver gains nothing from them (its
    kernels use every core), nor do the parts of the models that run in
    Python (see WorkersProcesses).

    Its stoppable functions (see WorkerPool.map_lazily; e.g. the refits of
    a bootstrap) run one after the other: the libraries they use may keep
    state of their process (e.g. the random numbers of numpy, which CMA-ES
    draws), so that two at once in one process would not give the results
    of each alone. Workers of processes run them at once.

    Parameters
    ----------
    drivers : Sequence of Placement
        The placement of the copy of each worker (e.g. one per GPU).

    Raises
    ------
    ValueError
        If there are no placements.
    """

    # Configurations name the drivers of the placements, see Workers
    config_options = {'drivers': Sequence[str | Driver]}

    @staticmethod
    def type() -> str:
        return 'threads'

    @staticmethod
    def type_title() -> str:
        return 'Threads'

    def __init__(self, drivers: Sequence[Placement]):
        check_options(WorkersThreads.__init__, locals())
        if not drivers:
            raise ValueError("drivers must give at least one placement")
        self._drivers = tuple(drivers)

    def dump(self) -> dict[str, Any]:
        return dict(drivers=dump_placements(self._drivers))

    def placements(self, objective: Objective) -> list[Drivers]:
        return [placement_drivers(objective, p) for p in self._drivers]

    def launch(
            self,
            objective: Objective,
            placements: Sequence[Drivers]
    ) -> list[Worker]:
        own = own_placement(objective)
        # One stoppable function at a time, see the class
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


class _ThreadWorker(LocalWorker):
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
        self._control = StopControl(self._flag)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix='gbkfit-worker')

    def submit(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            lambda: [function(self.copy, item) for item in items])

    def submit_controlled(
            self,
            function: Callable[[Objective, Item, StopControl], Result],
            item: Item
    ) -> concurrent.futures.Future:
        def run() -> Result:
            with self._lock:
                return function(self.copy, item, self._control)
        return self._executor.submit(run)

    def close(self) -> None:
        self._executor.shutdown()
        if self._owned:
            self.copy.close()
