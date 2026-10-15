"""
Evaluating several points of a fit at once, on copies of its objective
on several drivers (e.g. one per GPU; see EvaluationPool).
"""

import concurrent.futures
from collections.abc import Callable, Sequence
from typing import Self, TypeVar

from gbkfit.driver import Driver
from gbkfit.objective import Objective


__all__ = [
    'EvaluationPool'
]


_Item = TypeVar('_Item')
_Result = TypeVar('_Result')


class EvaluationPool:
    """
    Copies of an objective, one on each of the given drivers (see
    Objective.with_driver), that evaluate the batches of points of a fit
    at once, each in a thread of its own (see Fitter.fit). Point i of a
    batch goes to copy i mod N (of N copies), so that a fit gives the same
    results each time it runs, also with drivers of different devices.

    The first driver on which the objective does all its work (its
    observations and the gravity of its galaxies with mass) takes the
    objective itself; the others take copies. A driver can be listed more
    than once (e.g. two copies on one GPU).

    The threads overlap while they wait for their devices (GPUs); the
    host driver gains nothing from them (its kernels use every core).

    Parameters
    ----------
    objective : Objective
        The objective.
    drivers : Sequence of Driver
        The driver of each copy.

    Raises
    ------
    ValueError
        If there are no drivers.
    """

    def __init__(self, objective: Objective, drivers: Sequence[Driver]):
        if not drivers:
            raise ValueError("an evaluation pool needs at least one driver")
        copies = []
        for driver in drivers:
            if objective not in copies and _all_on(objective, driver):
                copies.append(objective)
            else:
                copies.append(objective.with_driver(driver))
        self._objective = objective
        self._copies = tuple(copies)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            len(copies), thread_name_prefix='gbkfit-evaluation')

    def objective(self) -> Objective:
        """Return the objective that the copies are of."""
        return self._objective

    def copies(self) -> tuple[Objective, ...]:
        """Return the copies of the objective, one per driver."""
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
        n = len(self._copies)

        def run(k: int) -> list[_Result]:
            copy = self._copies[k]
            return [function(copy, item) for item in items[k::n]]

        futures = [
            self._executor.submit(run, k) for k in range(min(n, len(items)))]
        concurrent.futures.wait(futures)
        chunks = [future.result() for future in futures]
        return [chunks[i % n][i // n] for i in range(len(items))]

    def close(self) -> None:
        """Stop the threads."""
        self._executor.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _all_on(objective: Objective, driver: Driver) -> bool:
    """
    Return whether an objective does all its work on a driver: its
    observations, and the gravity of its galaxies with mass.
    """
    group = objective.group()
    return all(obs.driver() is driver for obs in group.observations()) \
        and all(galaxy.driver() is driver
                for galaxy in group.galaxies() if galaxy.mass())
