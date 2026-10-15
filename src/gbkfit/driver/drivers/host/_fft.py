
import types
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.driver.base import (
    _NATIVE_CLASS_SUFFIXES, DeviceArray, DriverFFT, FFTConvolution)


__all__ = [
    'DriverFFTHost'
]


class DriverFFTHost(DriverFFT):
    """
    The FFT of the host driver: pocketfft, in the native module (its
    FFTConvolution classes).
    """

    def __init__(self, native_module: types.ModuleType, dtype: npt.DTypeLike):
        super().__init__(32)
        self._native_module = native_module
        self._dtype = np.dtype(dtype)

    def dtype(self) -> np.dtype:
        return self._dtype

    def _convolution(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...],
            dtype: np.dtype
    ) -> FFTConvolution:
        # The classes are named by the types of the cubes and, if more
        # precise, of the transforms (e.g. FFTConvolutionf32f64)
        name = 'FFTConvolution' + _NATIVE_CLASS_SUFFIXES[dtype]
        if dtype != self._dtype:
            name += _NATIVE_CLASS_SUFFIXES[self._dtype]
        native = getattr(self._native_module, name)
        return _FFTConvolutionHost(native(kernel, shape, axes))


class _FFTConvolutionHost(FFTConvolution):
    """A convolution of the native module."""

    def __init__(self, native: Any):
        self._native = native

    def __call__(self, cube: DeviceArray) -> None:
        self._native(cube)
