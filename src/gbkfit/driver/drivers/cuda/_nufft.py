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
    frequencies are uploaded when a NUFFT is made or moved.
    """

    def __init__(self, device: cp.cuda.Device):
        self._device = device

    def _type3(
            self,
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        return _NUFFTCuda(
            3, len(points), points, frequencies, sign, eps, ntransforms,
            self._device)

    def _type1(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        return _NUFFTCuda(
            1, shape, points, (), sign, eps, ntransforms, self._device)

    def _type2(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        return _NUFFTCuda(
            2, shape, points, (), sign, eps, ntransforms, self._device)


class _NUFFTCuda(NUFFT):
    """
    A NUFFT of a cuFINUFFT plan on a GPU (see _NUFFTHost of the host
    driver, of the same arguments).
    """

    def __init__(
            self,
            nufft_type: int,
            modes_or_dimensions: int | tuple[int, ...],
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...],
            sign: int,
            eps: float,
            ntransforms: int,
            device: cp.cuda.Device
    ):
        super().__init__(nufft_type, len(points))
        self._device = device
        with device:
            self._plan = cufinufft.Plan(
                nufft_type, modes_or_dimensions, n_trans=ntransforms,
                eps=eps, isign=sign, dtype=np.complex128,
                gpu_device_id=device.id)
        self._set_points(points, frequencies)

    def __call__(self, data: DeviceArray, out: DeviceArray) -> None:
        with self._device:
            self._plan.execute(data, out=out)

    def _set_points(
            self,
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...]
    ) -> None:
        with self._device:
            self._plan.setpts(**{
                name: cp.asarray(values)
                for name, values in _points(points, frequencies).items()})


def _points(
        points: tuple[np.ndarray, ...],
        frequencies: tuple[np.ndarray, ...]
) -> dict[str, Any]:
    """
    Return the points and the frequencies of a NUFFT as the arguments of
    the setpts of cuFINUFFT: their columns x, y, z and s, t, u.
    """
    return dict(zip('xyz', points)) | dict(zip('stu', frequencies))
