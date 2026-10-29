"""
Stopping fits as they run: the caller of Fitter.fit holds a FitControl,
and the fit problem raises FitStopped at the next model evaluation after
it is asked to stop.
"""

import threading
from typing import Protocol


__all__ = [
    'FitControl',
    'FitStopped',
    'StopFlag'
]


class StopFlag(Protocol):
    """
    A flag of a fit that is asked to stop: e.g. a threading.Event, or a
    multiprocessing Event, which processes share (see FitControl).
    """

    def set(self) -> None: ...

    def clear(self) -> None: ...

    def is_set(self) -> bool: ...


class FitControl:
    """
    The control of a running fit: stop asks it to stop at its next model
    evaluation, and the fit ends with what it has (see Fitter.fit). It can
    be called from another thread or a signal handler.

    Parameters
    ----------
    flag : StopFlag, optional
        Its flag: by default its own (a threading.Event); a flag that
        processes share stops fits in them (e.g. the sub-fits of workers,
        see WorkerPool.map_lazily).
    """

    def __init__(self, flag: StopFlag | None = None):
        self._stop = threading.Event() if flag is None else flag

    def stop(self) -> None:
        """Ask the fit to stop."""
        self._stop.set()

    def stopped(self) -> bool:
        """Return whether the fit was asked to stop."""
        return self._stop.is_set()


class FitStopped(Exception):
    """
    Raised by the fit problem when its fit was asked to stop (see
    FitControl). Fitters let it through, or catch it to return what they
    have (e.g. the chain of an MCMC sampler so far).
    """
