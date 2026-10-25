"""
Evaluating several points of a fit at once, on copies of its objective
on several drivers (e.g. one per GPU; see EvaluationPool).
"""

import concurrent.futures
from collections.abc import Callable, Mapping, Sequence
from typing import Self, TypeVar

from gbkfit.driver import Driver
from gbkfit.objective import Objective


__all__ = [
    'EvaluationPool'
]


_Item = TypeVar('_Item')
_Result = TypeVar('_Result')

# Where a copy of an objective evaluates: a driver for all its work, or
# one with other drivers of some galaxies and observations, by label (see
# ObservationGroup.plan)
_Placement = Driver | tuple[Driver, Mapping[str, Driver]]


class EvaluationPool:
    """
    Copies of an objective, one on each of the given placements (see
    Objective.with_driver), that evaluate the batches of points of a fit
    at once, each in a thread of its own (see Fitter.fit). Point i of a
    batch goes to copy i mod N (of N copies), so that a fit gives the same
    results each time it runs, also with drivers of different devices.

    A placement is a driver for all the work of a copy, or a driver and
    other drivers of some of its galaxies and observations, by label
    (e.g. (gpu0, {'hi': gpu1}): a copy split over two GPUs, for models too
    big for one; see ObservationGroup.plan). The first placement of the
    objective itself takes the objective; the others take copies. A
    placement can be listed more than once (e.g. two copies on one GPU).

    The threads overlap while they wait for their devices (GPUs); the
    host driver gains nothing from them (its kernels use every core).

    The placements are checked when the pool is made; the copies and
    their threads are made on first use (see copies; e.g. by the first
    batch of a fit), so that a pool that is not used (e.g. that of the
    configuration of a fit that is only evaluated or checked) takes no
    memory.

    Parameters
    ----------
    objective : Objective
        The objective.
    placements : Sequence of Driver or tuple of Driver and Mapping
        The placement of each copy.

    Raises
    ------
    ValueError
        If there are no placements.
    ConfigError
        If a placement names unknown galaxies or observations (see
        ObservationGroup.placement).
    """

    def __init__(self, objective: Objective, placements: Sequence[_Placement]):
        if not placements:
            raise ValueError("an evaluation pool needs at least one placement")
        group_plan = objective.group_plan()
        own = (group_plan.galaxy_drivers(), group_plan.observation_drivers())
        # (each placement, and whether it is that of the objective)
        self._placements = []
        for placement in placements:
            driver, drivers = (placement, None) \
                if isinstance(placement, Driver) else placement
            self._placements.append((
                driver, drivers,
                objective.group().placement(driver, drivers) == own))
        self._objective = objective
        self._copies: tuple[Objective, ...] | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def objective(self) -> Objective:
        """Return the objective that the copies are of."""
        return self._objective

    def copies(self) -> tuple[Objective, ...]:
        """
        Return the copies of the objective, one per placement (the first
        of its own placement is the objective itself), and make them, and
        their threads, on the first call.
        """
        if self._copies is None:
            copies = []
            for driver, drivers, own in self._placements:
                if own and self._objective not in copies:
                    copies.append(self._objective)
                else:
                    copies.append(self._objective.with_driver(driver, drivers))
            self._copies = tuple(copies)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                len(copies), thread_name_prefix='gbkfit-evaluation')
        return self._copies

    def map(
            self,
            function: Callable[[Objective, _Item], _Result],
            items: Sequence[_Item]
    ) -> list[_Result]:
        """
        Return function(copy, item) for each item, in order: item i on
        copy i mod N, the copies at once. It returns once every copy has
        finished, also when one raises an exception (then it raises that
        of the first copy that raised one).
        """
        copies = self.copies()
        n = len(copies)

        def run(k: int) -> list[_Result]:
            copy = copies[k]
            return [function(copy, item) for item in items[k::n]]

        futures = [
            self._executor.submit(run, k) for k in range(min(n, len(items)))]
        concurrent.futures.wait(futures)
        chunks = [future.result() for future in futures]
        return [chunks[i % n][i // n] for i in range(len(items))]

    def close(self) -> None:
        """Stop the threads, if made."""
        if self._executor is not None:
            self._executor.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

