"""
Workers in processes of this machine.
"""

import concurrent.futures
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any

import threadpoolctl
from annotated_types import Ge

from gbkfit.config import check_options
from gbkfit.control import StopControl, StopFlag
from gbkfit.driver import Driver
from gbkfit.objective import Objective
from gbkfit.observation import GroupPlan, ObservationGroup
from gbkfit.params import Solve
from ._detail import (
    Drivers, Item, LocalWorker, Result, Worker, cores, dump_placements,
    own_placement, placement_drivers)
from .pool import Placement, Workers


__all__ = [
    'WorkersProcesses'
]


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
    ValueError
        If neither or both of count and drivers are given, or there are no
        placements.
    """

    # Configurations name the drivers of the placements, see Workers
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
            raise ValueError("give count or drivers (and not both)")
        if drivers is not None and not drivers:
            raise ValueError("drivers must give at least one placement")
        self._count = count
        self._drivers = None if drivers is None else tuple(drivers)
        self._threads = threads

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(count=self._count) \
            if self._drivers is None \
            else dict(drivers=dump_placements(self._drivers))
        if self._threads is not None:
            info['threads'] = self._threads
        return info

    def placements(self, objective: Objective) -> list[Drivers]:
        if self._drivers is None:
            return [own_placement(objective)] * self._count
        return [placement_drivers(objective, p) for p in self._drivers]

    def launch(
            self,
            objective: Objective,
            placements: Sequence[Drivers]
    ) -> list[Worker]:
        threads = self._threads or max(1, cores() // len(placements))
        return [
            _ProcessWorker(
                objective.group(), objective.solves(), placement, threads)
            for placement in placements]


class _ProcessWorker(LocalWorker):
    """A worker in a process of its own, of a copy of an objective."""

    def __init__(
            self,
            group: ObservationGroup,
            solves: Mapping[str, Solve],
            placement: Drivers,
            threads: int
    ):
        context = multiprocessing.get_context('spawn')
        super().__init__(context.Event())
        self._executor = concurrent.futures.ProcessPoolExecutor(
            1, mp_context=context, initializer=_start_process,
            initargs=(group, solves, placement, threads, self._flag))

    def submit(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(_run_in_process, function, items)

    def submit_controlled(
            self,
            function: Callable[[Objective, Item, StopControl], Result],
            item: Item
    ) -> concurrent.futures.Future:
        return self._executor.submit(_run_controlled_in_process, function, item)

    def close(self) -> None:
        self._executor.shutdown()


# The copy of the objective of a worker process, and the control of its
# flag (see _start_process)
_process_copy: Objective | None = None
_process_control: StopControl | None = None


def _start_process(
        group: ObservationGroup,
        solves: Mapping[str, Solve],
        placement: Drivers,
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
    _process_control = StopControl(flag)
    threadpoolctl.threadpool_limits(threads)


def _run_in_process(
        function: Callable[[Objective, Item], Result],
        items: Sequence[Item]
) -> list[Result]:
    """Return function(copy, item) for each item, in a worker process."""
    return [function(_process_copy, item) for item in items]


def _run_controlled_in_process(
        function: Callable[[Objective, Item, StopControl], Result],
        item: Item
) -> Result:
    """Return function(copy, item, control), in a worker process."""
    return function(_process_copy, item, _process_control)
