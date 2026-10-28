import types

import numpy as np
import numpy.typing as npt
import scipy.ndimage

from gbkfit.driver.base import (
    DeviceArray, Driver, DriverFFT, DriverNUFFT, Shape, check_array_size)


__all__ = [
    'DriverHost'
]


class DriverHost(Driver):
    """
    The host CPU: numpy arrays, and the C++ kernels (OpenMP) of the native
    host module. The host and device sides of a transfer are one memory.
    """

    requires = {
        'gbkfit.driver.native._host':
            "a build of gbkfit with its host module (CMake option "
            "GBKFIT_HOST, on by default)"}

    @staticmethod
    def type() -> str:
        return 'host'

    def mem_alloc_s(
            self,
            shape: Shape,
            dtype: npt.DTypeLike
    ) -> tuple[np.ndarray, DeviceArray]:
        check_array_size(shape)
        data = np.empty(shape, dtype)
        return data, data

    def mem_alloc_h(self, shape: Shape, dtype: npt.DTypeLike) -> np.ndarray:
        return np.empty(shape, dtype)

    def mem_alloc_d(self, shape: Shape, dtype: npt.DTypeLike) -> DeviceArray:
        check_array_size(shape)
        return np.empty(shape, dtype)

    def mem_copy_h2d(
            self,
            h_src: np.ndarray,
            d_dst: DeviceArray | None = None
    ) -> DeviceArray:
        if d_dst is None:
            d_dst = self.mem_alloc_d(h_src.shape, h_src.dtype)
        if h_src is not d_dst:
            d_dst[:] = h_src
        return d_dst

    def mem_copy_d2h(
            self,
            d_src: DeviceArray,
            h_dst: np.ndarray | None = None
    ) -> np.ndarray:
        if h_dst is None:
            h_dst = self.mem_alloc_h(d_src.shape, d_src.dtype)
        if d_src is not h_dst:
            h_dst[:] = d_src
        return h_dst

    def mem_fill(self, x: DeviceArray, value: float) -> None:
        x.fill(value)

    def mem_copy_d2d(self, d_src: DeviceArray, d_dst: DeviceArray) -> None:
        np.copyto(d_dst, d_src, casting='same_kind')

    def math_add(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        return np.add(x1, x2, out=out)

    def math_sub(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        return np.subtract(x1, x2, out=out)

    def math_mul(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        return np.multiply(x1, x2, out=out)

    def math_median(
            self,
            x: DeviceArray,
            size: tuple[int, ...],
            out: DeviceArray
    ) -> None:
        scipy.ndimage.median_filter(
            x, size=size, mode='constant', cval=0.0, output=out)

    def math_matmul(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        return np.matmul(x1, x2, out=out)

    def mem_gather(
            self,
            x: DeviceArray,
            indices: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        return np.take(x, indices, axis=-1, out=out)

    def mem_scatter(
            self,
            x: DeviceArray,
            indices: DeviceArray,
            out: DeviceArray
    ) -> None:
        out[..., indices] = x

    def native_module(self) -> types.ModuleType:
        import gbkfit.driver.native._host as native_module
        return native_module

    def fft(self, dtype: npt.DTypeLike) -> DriverFFT:
        from ._fft import DriverFFTHost
        return DriverFFTHost(self.native_module(), dtype)

    def nufft(self) -> DriverNUFFT:
        from ._nufft import DriverNUFFTHost
        return DriverNUFFTHost()
