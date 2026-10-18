from typing import Any

import numpy as np

from gbkfit.driver.base import DeviceArray, DriverNUFFT, NUFFT

try:
    import finufft
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the non-uniform FFTs of the host driver need FINUFFT: "
        "pip install 'gbkfit[finufft]'") from e


__all__ = [
    'DriverNUFFTHost'
]


class DriverNUFFTHost(DriverNUFFT):
    """
    The non-uniform FFTs of the host driver: FINUFFT, in its threads
    (OpenMP; OMP_NUM_THREADS of them).
    """

    def _type3(
            self,
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...],
            sign: int,
            eps: float
    ) -> NUFFT:
        plan = finufft.Plan(
            3, len(points), eps=eps, isign=sign, dtype=np.complex128)
        plan.setpts(**_points(points, frequencies))
        return _NUFFTHost(plan)

    def _type1(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float
    ) -> NUFFT:
        plan = finufft.Plan(
            1, shape, eps=eps, isign=sign, dtype=np.complex128)
        plan.setpts(**_points(points))
        return _NUFFTHost(plan)

    def _type2(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        plan = finufft.Plan(
            2, shape, n_trans=ntransforms, eps=eps, isign=sign,
            dtype=np.complex128)
        plan.setpts(**_points(points))
        return _NUFFTHost(plan)


class _NUFFTHost(NUFFT):
    """A NUFFT of a FINUFFT plan."""

    def __init__(self, plan: Any):
        self._plan = plan

    def __call__(self, data: DeviceArray, out: DeviceArray) -> None:
        self._plan.execute(data, out=out)


def _points(
        points: tuple[np.ndarray, ...],
        frequencies: tuple[np.ndarray, ...] = ()
) -> dict[str, np.ndarray]:
    """
    Return the points and the frequencies of a NUFFT as the arguments of
    the setpts of FINUFFT: their columns x, y, z and s, t, u.
    """
    return dict(zip('xyz', points)) | dict(zip('stu', frequencies))
