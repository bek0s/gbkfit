"""
The parts of the kinds of workers that they share: workers, their flags
of stops, the copies of actors of clusters, and placements.
"""

import abc
import concurrent.futures
import math
import os
import threading
import time
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias, TypeVar

import threadpoolctl

from gbkfit.control import StopControl, StopFlag
from gbkfit.driver import Driver, driver_parser
from gbkfit.objective import Objective
from gbkfit.observation import GroupPlan, Likelihoods
from gbkfit.params import Solve

if typing.TYPE_CHECKING:
    from .pool import Placement


Item = TypeVar('Item')
Result = TypeVar('Result')

# How often the feeders of the workers look whether to stop them, as they
# wait for their results (s; see WorkerPool.map_lazily)
POLL_INTERVAL = 0.2


# The drivers of a copy: of each galaxy and of each observation (see
# ObservationGroup.drivers)
Drivers: TypeAlias = tuple[tuple[Driver, ...], tuple[Driver, ...]]


class Worker(abc.ABC):
    """
    A worker, which runs functions of its copy of an objective, and stops
    them when it is interrupted (see submit_controlled).
    """

    @abc.abstractmethod
    def submit(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> concurrent.futures.Future:
        """
        Return the future of function(copy, item) for each item, in order.
        """

    @abc.abstractmethod
    def submit_controlled(
            self,
            function: Callable[[Objective, Item, StopControl], Result],
            item: Item
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


class LocalWorker(Worker, abc.ABC):
    """A worker of this machine, whose flag is a shared event."""

    def __init__(self, flag: StopFlag):
        self._flag = flag

    def interrupt(self) -> None:
        self._flag.set()

    def clear(self) -> None:
        self._flag.clear()


class PolledFlag:
    """
    A stop flag kept elsewhere (e.g. by the first rank of an MPI job, or
    by the scheduler of a Dask cluster), asked at most every
    POLL_INTERVAL seconds (see StopFlag); once set, it stays set.
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
        if not self._set and now - self._asked >= POLL_INTERVAL:
            self._asked = now
            self._set = self._ask()
        return self._set


class ActorCopy:
    """
    The copy of an objective of a worker that is an actor of Ray or Dask
    (see WorkersRay and WorkersDask), which runs functions of it: those
    with a control, with a flag of its own (Ray), or the given one (Dask).
    """

    def __init__(
            self,
            likelihoods: Likelihoods,
            solves: Mapping[str, Solve],
            placement: Drivers,
            threads: int | None
    ):
        self._copy = objective_on(likelihoods, solves, placement)
        self._flag = threading.Event()
        if threads is not None:
            threadpoolctl.threadpool_limits(threads)

    def run(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> list[Result]:
        return [function(self._copy, item) for item in items]

    def run_controlled(
            self,
            function: Callable[[Objective, Item, StopControl], Result],
            item: Item,
            flag: StopFlag | None = None
    ) -> Result:
        control = StopControl(self._flag if flag is None else flag)
        return function(self._copy, item, control)

    def interrupt(self) -> None:
        self._flag.set()

    def clear(self) -> None:
        self._flag.clear()


def cores() -> int:
    """
    Return the cores that this process may run on (e.g. those of a job of
    a cluster), or those of the machine where that is not known.
    """
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def objective_on(
        likelihoods: Likelihoods,
        solves: Mapping[str, Solve],
        placement: Drivers
) -> Objective:
    """
    Return a copy of an objective on the drivers of a placement, of its
    likelihoods (see Objective.likelihoods), whose group it plans, and its
    solves: what a worker makes of an objective (sent to it pickled, as
    the group and the likelihoods of its observations, which share it).
    """
    return Objective(
        GroupPlan(likelihoods.group(), *placement), likelihoods, solves)


def own_placement(objective: Objective) -> Drivers:
    """Return the drivers of an objective."""
    plan = objective.group_plan()
    return plan.galaxy_drivers(), plan.observation_drivers()


def placement_drivers(objective: Objective, placement: Placement) -> Drivers:
    """
    Return the drivers of a copy of an objective on a placement, or raise
    ValueError if it names unknown galaxies or observations.
    """
    driver, by_label = (placement, None) if isinstance(placement, Driver) \
        else placement
    return objective.group().drivers(driver, by_label)


def dump_placements(placements: Sequence[Placement]) -> list[Any]:
    """
    Return the configurations of placements: those of their drivers (see
    Workers). Raise TypeError for placements of drivers by label, which
    configurations do not have.
    """
    if any(not isinstance(p, Driver) for p in placements):
        raise TypeError(
            "workers on placements of drivers by label cannot be dumped: "
            "configurations give one driver of each worker")
    return [driver_parser.dump(p) for p in placements]
