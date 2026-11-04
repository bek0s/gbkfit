"""
Stopping work as it runs: the caller holds a StopControl, and the work
checks it as it goes and raises Stopped at its next step after it is
asked to stop. The workers of gbkfit.parallel stop the same way, with a
flag that their processes share.
"""

import threading
from typing import Protocol


__all__ = [
    'StopControl',
    'StopFlag',
    'Stopped'
]


class StopFlag(Protocol):
    """
    A flag of work that is asked to stop: e.g. a threading.Event, or a
    multiprocessing Event, which processes share (see StopControl).
    """

    def set(self) -> None: ...

    def clear(self) -> None: ...

    def is_set(self) -> bool: ...


class StopControl:
    """
    The control of running work: stop asks it to stop at its next step,
    and the work ends with what it has. With its own flag, stop can be
    called from another thread or a signal handler; a flag of
    multiprocessing takes a lock that the interrupted thread may hold.

    Parameters
    ----------
    flag : StopFlag, optional
        Its flag: by default its own (a threading.Event); a flag that
        processes share stops the work in them (e.g. that of workers, see
        WorkerPool.map_lazily).
    """

    def __init__(self, flag: StopFlag | None = None):
        self._stop = threading.Event() if flag is None else flag

    def stop(self) -> None:
        """Ask the work to stop."""
        self._stop.set()

    def stopped(self) -> bool:
        """Return whether the work was asked to stop."""
        return self._stop.is_set()


class Stopped(Exception):
    """
    Raised by work that was asked to stop (see StopControl). The code
    that runs it lets it through, or catches it to keep what the work
    has done so far.
    """
