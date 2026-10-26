"""
The threads of the libraries that GBKFIT calls.
"""

import contextlib
import functools
import threading
from collections.abc import Iterator
from typing import Protocol

import threadpoolctl


__all__ = [
    'blas_limit',
    'single_threaded_blas'
]


@functools.cache
def _blas_libraries() -> threadpoolctl.ThreadpoolController:
    """
    Return the BLAS libraries loaded (e.g. the OpenBLAS of numpy and that
    of scipy), found once: scanning the loaded libraries takes about 2 ms.
    """
    return threadpoolctl.ThreadpoolController().select(user_api='blas')


class _Limiter(Protocol):
    """A limit of threads of threadpoolctl, which restores their own."""

    def restore_original_limits(self) -> None: ...


class _BlasLimits:
    """
    The contexts of blas_limit, one for all threads: the BLAS libraries
    use the fewest threads of the contexts that threads are in, and their
    own limits again once no thread is in one.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # The limits of the contexts entered (one per entry), the limit
        # applied, and the limiter that restores the own limits
        self._limits: list[int] = []
        self._applied: int | None = None
        self._limiter: _Limiter | None = None

    @contextlib.contextmanager
    def limit(self, threads: int) -> Iterator[None]:
        """Return a context of BLAS on at most the given threads."""
        with self._lock:
            self._limits.append(threads)
            self._apply()
        try:
            yield
        finally:
            with self._lock:
                self._limits.remove(threads)
                self._apply()

    def _apply(self) -> None:
        """Apply the fewest threads of the contexts entered, if any."""
        fewest = min(self._limits, default=None)
        if fewest == self._applied:
            return
        if fewest is None:
            assert self._limiter is not None
            self._limiter.restore_original_limits()
            self._limiter = None
        elif self._limiter is None:
            self._limiter = _blas_libraries().limit(limits=fewest)
        else:
            _blas_libraries().limit(limits=fewest)
        self._applied = fewest


_BLAS_LIMITS = _BlasLimits()


def blas_limit(threads: int) -> contextlib.AbstractContextManager:
    """
    Return a context in which the BLAS libraries use at most the given
    threads.

    The limit applies to the whole process while any thread is in the
    context (evaluations can run in several threads at once), and to the
    libraries loaded when it is first used (gbkfit loads numpy and scipy
    at import); of contexts that threads are in at once, the fewest
    threads. Threads that spin from a call made before it keep spinning
    (see single_threaded_blas).

    Parameters
    ----------
    threads : int
        The most threads.

    Returns
    -------
    contextlib.AbstractContextManager
        The context; nested contexts, and contexts in several threads at
        once, are allowed.
    """
    return _BLAS_LIMITS.limit(threads)


def single_threaded_blas() -> contextlib.AbstractContextManager:
    """
    Return a context in which the BLAS libraries use one thread (see
    blas_limit).

    GBKFIT evaluates models with BLAS calls on small arrays (e.g. the
    gravity of disks) between the OpenMP kernels of its host driver. The
    threads of a multithreaded BLAS gain nothing there, and OpenBLAS keeps
    them spinning for about 0.1 s after each call, on the cores that the
    kernels need, which stalls the kernels. Their evaluations run in this
    context.

    Returns
    -------
    contextlib.AbstractContextManager
        The context; nested contexts, and contexts in several threads at
        once, are allowed.
    """
    return blas_limit(1)
