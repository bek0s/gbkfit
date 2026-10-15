"""
The threads of the libraries that GBKFIT calls.
"""

import contextlib
import functools
import threading

import threadpoolctl


__all__ = [
    'single_threaded_blas'
]


@functools.cache
def _blas_libraries() -> threadpoolctl.ThreadpoolController:
    """
    Return the BLAS libraries loaded (e.g. the OpenBLAS of numpy and that
    of scipy), found once: scanning the loaded libraries takes about 2 ms.
    """
    return threadpoolctl.ThreadpoolController().select(user_api='blas')


class _SingleThreadedBlas:
    """
    The context of single_threaded_blas, one for all threads: the first
    to enter it limits the BLAS libraries to one thread, and the last to
    leave it restores their limits.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._users = 0
        self._limit = contextlib.ExitStack()

    def __enter__(self) -> None:
        with self._lock:
            if self._users == 0:
                self._limit.enter_context(_blas_libraries().limit(limits=1))
            self._users += 1

    def __exit__(self, *exc_info: object) -> None:
        with self._lock:
            self._users -= 1
            if self._users == 0:
                self._limit.close()


_SINGLE_THREADED_BLAS = _SingleThreadedBlas()


def single_threaded_blas() -> contextlib.AbstractContextManager:
    """
    Return a context in which the BLAS libraries use one thread.

    GBKFIT evaluates models with BLAS calls on small arrays (e.g. the
    gravity of disks) between the OpenMP kernels of its host driver. The
    threads of a multithreaded BLAS gain nothing there, and OpenBLAS keeps
    them spinning for about 0.1 s after each call, on the cores that the
    kernels need, which stalls the kernels. Their evaluations run in this
    context. The limit applies to the whole process while any thread is
    in the context (evaluations can run in several threads at once), and
    to the libraries loaded when it is first used (gbkfit loads numpy and
    scipy at import). Threads that spin from a call made before it keep
    spinning.

    Returns
    -------
    contextlib.AbstractContextManager
        The context; nested contexts, and contexts in several threads at
        once, are allowed.
    """
    return _SINGLE_THREADED_BLAS
