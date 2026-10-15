"""
Stopping fits as they run: the caller of Fitter.fit holds a FitControl,
and the fit problem raises FitStopped at the next model evaluation after
it is asked to stop.
"""

import threading


__all__ = [
    'FitControl',
    'FitStopped'
]


class FitControl:
    """
    The control of a running fit: stop asks it to stop at its next model
    evaluation, and the fit ends with what it has (see Fitter.fit). It can
    be called from another thread or a signal handler.
    """

    def __init__(self):
        self._stop = threading.Event()

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
