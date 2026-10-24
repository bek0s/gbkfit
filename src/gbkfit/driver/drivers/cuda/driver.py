import types
from typing import Any

import cupy as cp
import cupyx as cpx
import cupyx.scipy.ndimage
import numpy as np
import numpy.typing as npt

import gbkfit.driver.native._cuda as native_module
from gbkfit.driver.base import (
    DeviceArray, Driver, DriverFFT, DriverNUFFT, Shape, check_array_size)
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'DriverCuda'
]


class DriverCuda(Driver):
    """
    One CUDA GPU: CuPy arrays, and the CUDA kernels of the native cuda
    module. All its work (its memory, transfers and FFTs, and the kernels
    on its arrays) runs on its GPU, whatever the current CuPy device of
    the calling thread; drivers of several GPUs can work in several
    threads at once. The host side of a transfer is in pinned memory.

    Parameters
    ----------
    device : int, optional
        The number of the GPU (as CUDA numbers them, e.g. 0 and 1 for two;
        see CUDA_VISIBLE_DEVICES).

    Raises
    ------
    ConfigError
        If the machine has no GPU of that number.
    """

    @staticmethod
    def type() -> str:
        return 'cuda'

    def __init__(self, device: int = 0):
        ndevices = cp.cuda.runtime.getDeviceCount()
        if not 0 <= device < ndevices:
            raise ConfigError(
                f"device must be the number of a GPU of this machine, from 0 "
                f"to {ndevices - 1}; it is {device}")
        self._device = cp.cuda.Device(device)

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(device=self.device())

    def device(self) -> int:
        """Return the number of the GPU."""
        return self._device.id

    def mem_alloc_s(
            self,
            shape: Shape,
            dtype: npt.DTypeLike
    ) -> tuple[np.ndarray, DeviceArray]:
        check_array_size(shape)
        with self._device:
            return cpx.empty_pinned(shape, dtype), cp.empty(shape, dtype)

    def mem_alloc_h(self, shape: Shape, dtype: npt.DTypeLike) -> np.ndarray:
        with self._device:
            return cpx.empty_pinned(shape, dtype)

    def mem_alloc_d(self, shape: Shape, dtype: npt.DTypeLike) -> DeviceArray:
        check_array_size(shape)
        with self._device:
            return cp.empty(shape, dtype)

    def mem_copy_h2d(
            self,
            h_src: np.ndarray,
            d_dst: DeviceArray | None = None
    ) -> DeviceArray:
        if d_dst is None:
            d_dst = self.mem_alloc_d(h_src.shape, h_src.dtype)
        if h_src is not d_dst:
            with self._device:
                d_dst.set(h_src)
        return d_dst

    def mem_copy_d2h(
            self,
            d_src: DeviceArray,
            h_dst: np.ndarray | None = None
    ) -> np.ndarray:
        if h_dst is None:
            h_dst = self.mem_alloc_h(d_src.shape, d_src.dtype)
        if d_src is not h_dst:
            with self._device:
                d_src.get(out=h_dst)
        return h_dst

    def mem_fill(self, x: DeviceArray, value: float) -> None:
        with self._device:
            x.fill(value)

    def mem_copy_d2d(self, d_src: DeviceArray, d_dst: DeviceArray) -> None:
        with self._device:
            cp.copyto(d_dst, d_src, casting='same_kind')

    def math_add(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        with self._device:
            return cp.add(x1, x2, out=out)

    def math_mul(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        with self._device:
            return cp.multiply(x1, x2, out=out)

    def math_median(
            self,
            x: DeviceArray,
            size: tuple[int, ...],
            out: DeviceArray
    ) -> None:
        with self._device:
            cupyx.scipy.ndimage.median_filter(
                x, size=size, mode='constant', cval=0.0, output=out)

    def math_matmul(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        with self._device:
            return cp.matmul(x1, x2, out=out)

    def mem_gather(
            self,
            x: DeviceArray,
            indices: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        with self._device:
            return cp.take(x, indices, axis=-1, out=out)

    def mem_scatter(
            self,
            x: DeviceArray,
            indices: DeviceArray,
            out: DeviceArray
    ) -> None:
        with self._device:
            out[..., indices] = x

    def native_module(self) -> types.ModuleType:
        return native_module

    def fft(self, dtype: npt.DTypeLike) -> DriverFFT:
        from ._fft import DriverFFTCuda
        return DriverFFTCuda(dtype, self._device)

    def nufft(self) -> DriverNUFFT:
        from ._nufft import DriverNUFFTCuda
        return DriverNUFFTCuda(self._device)
