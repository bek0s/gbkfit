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
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        return _NUFFTHost(
            3, len(points), points, frequencies, sign, eps, ntransforms)

    def _type1(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        return _NUFFTHost(1, shape, points, (), sign, eps, ntransforms)

    def _type2(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        return _NUFFTHost(2, shape, points, (), sign, eps, ntransforms)


class _NUFFTHost(NUFFT):
    """
    A NUFFT of a FINUFFT plan of a type, of the number of dimensions of
    type 3 or the shape of the arrays of types 1 and 2, and of columns of
    points and frequencies (see DriverNUFFT._type3).
    """

    def __init__(
            self,
            nufft_type: int,
            modes_or_dimensions: int | tuple[int, ...],
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ):
        super().__init__(nufft_type, len(points))
        self._plan = finufft.Plan(
            nufft_type, modes_or_dimensions, n_trans=ntransforms, eps=eps,
            isign=sign, dtype=np.complex128)
        self._set_points(points, frequencies)

    def __call__(self, data: DeviceArray, out: DeviceArray) -> None:
        self._plan.execute(data, out=out)

    def _set_points(
            self,
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...]
    ) -> None:
        self._plan.setpts(**_points(points, frequencies))


def _points(
        points: tuple[np.ndarray, ...],
        frequencies: tuple[np.ndarray, ...]
) -> dict[str, Any]:
    """
    Return the points and the frequencies of a NUFFT as the arguments of
    the setpts of FINUFFT: their columns x, y, z and s, t, u.
    """
    return dict(zip('xyz', points)) | dict(zip('stu', frequencies))
