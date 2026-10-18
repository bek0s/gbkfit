from typing import Any

import cupy as cp
import numpy as np

from gbkfit.driver.base import DeviceArray, DriverNUFFT, NUFFT

try:
    import cufinufft
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the non-uniform FFTs of the cuda driver need cuFINUFFT: "
        "pip install 'gbkfit[cufinufft]'") from e


__all__ = [
    'DriverNUFFTCuda'
]


class DriverNUFFTCuda(DriverNUFFT):
    """
    The non-uniform FFTs of the cuda driver: cuFINUFFT, on its GPU (in the
    default stream, as all the work of the driver). The points and the
    frequencies are uploaded when a NUFFT is made.
    """

    def __init__(self, device: cp.cuda.Device):
        self._device = device

    def _type3(
            self,
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...],
            sign: int,
            eps: float
    ) -> NUFFT:
        return self._nufft(
            3, len(points), _points(points, frequencies), sign, eps)

    def _type1(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float
    ) -> NUFFT:
        return self._nufft(1, shape, _points(points), sign, eps)

    def _type2(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        return self._nufft(2, shape, _points(points), sign, eps, ntransforms)

    def _nufft(
            self,
            nufft_type: int,
            modes_or_dimensions: int | tuple[int, ...],
            points: dict[str, np.ndarray],
            sign: int,
            eps: float,
            ntransforms: int = 1
    ) -> NUFFT:
        """
        Return a NUFFT of a cuFINUFFT plan of a type, of the arguments of
        its setpts (see _points), on the host, and of a number of
        transforms.
        """
        with self._device:
            plan = cufinufft.Plan(
                nufft_type, modes_or_dimensions, n_trans=ntransforms, eps=eps,
                isign=sign, dtype=np.complex128,
                gpu_device_id=self._device.id)
            plan.setpts(**{
                name: cp.asarray(values) for name, values in points.items()})
        return _NUFFTCuda(plan, self._device)


class _NUFFTCuda(NUFFT):
    """A NUFFT of a cuFINUFFT plan, on a GPU."""

    def __init__(self, plan: Any, device: cp.cuda.Device):
        self._plan = plan
        self._device = device

    def __call__(self, data: DeviceArray, out: DeviceArray) -> None:
        with self._device:
            self._plan.execute(data, out=out)


def _points(
        points: tuple[np.ndarray, ...],
        frequencies: tuple[np.ndarray, ...] = ()
) -> dict[str, np.ndarray]:
    """
    Return the points and the frequencies of a NUFFT as the arguments of
    the setpts of cuFINUFFT: their columns x, y, z and s, t, u.
    """
    return dict(zip('xyz', points)) | dict(zip('stu', frequencies))
