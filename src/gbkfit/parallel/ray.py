"""
Workers in actors of a Ray cluster.
"""

import concurrent.futures
import math
from collections.abc import Callable, Sequence
from typing import Annotated, Any

from annotated_types import Ge, Gt

from gbkfit.config import check_options
from gbkfit.control import StopControl
from gbkfit.objective import Objective
from ._detail import ActorCopy, Drivers, Item, Result, Worker, own_placement
from .pool import Workers


__all__ = [
    'WorkersRay'
]


class WorkersRay(Workers):
    """
    Workers in actors of a Ray cluster, each on CPUs and GPUs of the
    cluster with its copy of the objective. There are a number of them;
    the objective goes to each pickled (see Objective). Ray gives an actor
    its GPUs: the CUDA drivers of its copy of device 0 are on the first.
    Like processes (see WorkersProcesses), they evaluate at the same time
    the parts of the models that run in Python, and run stoppable
    functions at once (e.g. sub-fits), on the machines of the cluster.

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

    @staticmethod
    def type_title() -> str:
        return 'Ray cluster'

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

    def placements(self, objective: Objective) -> list[Drivers]:
        return [own_placement(objective)] * self._count

    def launch(
            self,
            objective: Objective,
            placements: Sequence[Drivers]
    ) -> list[Worker]:
        import ray
        if not ray.is_initialized():
            ray.init(address=self._address)
        # Two calls at once: a function, and its interruption
        actor = ray.remote(ActorCopy).options(
            num_cpus=self._cpus, num_gpus=self._gpus, max_concurrency=2)
        threads = self._threads or max(1, math.floor(self._cpus))
        return [
            _RayWorker(actor.remote(
                objective.group(), objective.solves(), placement, threads))
            for placement in placements]


class _RayWorker(Worker):
    """A worker that is an actor of Ray (see WorkersRay)."""

    def __init__(self, actor: Any):
        self._actor = actor

    def submit(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> concurrent.futures.Future:
        return self._actor.run.remote(function, items).future()

    def submit_controlled(
            self,
            function: Callable[[Objective, Item, StopControl], Result],
            item: Item
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
