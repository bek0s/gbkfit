"""
The threads of the libraries that GBKFIT calls.
"""

import contextlib
import functools

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


def single_threaded_blas() -> contextlib.AbstractContextManager:
    """
    Return a context in which the BLAS libraries use one thread.

    GBKFIT evaluates models with BLAS calls on small arrays (e.g. the
    gravity of disks) between the OpenMP kernels of its host driver. The
    threads of a multithreaded BLAS gain nothing there, and OpenBLAS keeps
    them spinning for about 0.1 s after each call, on the cores that the
    kernels need, which stalls the kernels. Their evaluations run in this
    context. The limit applies to the whole process while it lasts, and to
    the libraries loaded when it is first used (gbkfit loads numpy and
    scipy at import). Threads that spin from a call made before it keep
    spinning.

    Returns
    -------
    contextlib.AbstractContextManager
        The context; nested contexts are allowed.
    """
    return _blas_libraries().limit(limits=1)
