"""
Workers in actors of a Dask cluster.
"""

import concurrent.futures
import uuid
from collections.abc import Callable, Sequence
from typing import Annotated, Any

from annotated_types import Ge

from gbkfit.config import check_options
from gbkfit.control import StopControl
from gbkfit.objective import Objective
from ._detail import (
    ActorCopy, Drivers, Item, PolledFlag, Result, Worker, own_placement)
from .pool import Workers


__all__ = [
    'WorkersDask'
]


class WorkersDask(Workers):
    """
    Workers in actors of a Dask cluster: a number of them, each on a worker
    of the cluster (of its own, while there are enough), with its copy of
    the objective, which goes to it pickled (see Objective). Like
    processes (see WorkersProcesses), they evaluate at the same time the
    parts of the models that run in Python, and run stoppable functions
    at once (e.g. sub-fits), on the machines of the cluster.

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

    def placements(self, objective: Objective) -> list[Drivers]:
        return [own_placement(objective)] * self._count

    def launch(
            self,
            objective: Objective,
            placements: Sequence[Drivers]
    ) -> list[Worker]:
        import distributed
        # A client of the address is the pool's, which closes it; it is
        # the default client meanwhile, through which actors call theirs
        own = self._address is not None
        client = distributed.Client(self._address) if own \
            else distributed.get_client()
        hosts = list(client.scheduler_info()['workers'])
        return [
            _DaskWorker(client.submit(
                ActorCopy, objective.group(), objective.solves(),
                placement, self._threads, actor=True, pure=False,
                workers=[hosts[k % len(hosts)]]).result(),
                client, own)
            for k, placement in enumerate(placements)]


class _DaskWorker(Worker):
    """
    A worker that is an actor of Dask (see WorkersDask): a thread of its
    own waits for its results. The flag of each function it runs is an
    event of the cluster of its own (see distributed.Event), which it sets
    through its client. A client it owns is closed with it: the workers of
    a pool share it, and closing a closed client does nothing.
    """

    def __init__(self, actor: Any, client: Any, owned: bool):
        self._actor = actor
        self._client = client
        self._owned = owned
        self._event: str | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix='gbkfit-dask')

    def submit(
            self,
            function: Callable[[Objective, Item], Result],
            items: Sequence[Item]
    ) -> concurrent.futures.Future:
        return self._executor.submit(
            lambda: self._actor.run(function, items).result())

    def submit_controlled(
            self,
            function: Callable[[Objective, Item, StopControl], Result],
            item: Item
    ) -> concurrent.futures.Future:
        self._event = f'gbkfit-stop-{uuid.uuid4().hex}'
        flag = _DaskFlag(self._event)
        return self._executor.submit(
            lambda: self._actor.run_controlled(function, item, flag).result())

    def interrupt(self) -> None:
        if self._event is not None:
            import distributed
            distributed.Event(self._event, client=self._client).set()

    def clear(self) -> None:
        # Each function has a flag of its own
        pass

    def close(self) -> None:
        self._executor.shutdown()
        if self._owned:
            self._client.close()


class _DaskFlag(PolledFlag):
    """The flag of a function of a worker of Dask: an event of its cluster."""

    def __init__(self, name: str):
        super().__init__(self._ask_cluster)
        self._name = name

    def _ask_cluster(self) -> bool:
        import distributed
        return distributed.Event(self._name).is_set()

    def __reduce__(self) -> tuple[type, tuple[str]]:
        return _DaskFlag, (self._name,)
