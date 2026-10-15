import threading

import threadpoolctl

from gbkfit.utils import threadutils


def blas_threads() -> list[int]:
    """The number of threads of each BLAS library loaded."""
    return [
        info['num_threads'] for info in threadpoolctl.threadpool_info()
        if info['user_api'] == 'blas']


def test_blas_runs_on_one_thread_in_the_context():
    before = blas_threads()
    with threadutils.single_threaded_blas():
        assert blas_threads() == [1] * len(before)
        with threadutils.single_threaded_blas():
            assert blas_threads() == [1] * len(before)
        assert blas_threads() == [1] * len(before)
    assert blas_threads() == before


def test_the_limit_lasts_while_any_thread_is_in_the_context():
    before = blas_threads()
    entered, leave = threading.Event(), threading.Event()

    def other_thread():
        with threadutils.single_threaded_blas():
            entered.set()
            leave.wait()

    thread = threading.Thread(target=other_thread)
    thread.start()
    entered.wait()
    with threadutils.single_threaded_blas():
        # (the other thread leaves first)
        leave.set()
        thread.join()
        assert blas_threads() == [1] * len(before)
    assert blas_threads() == before
