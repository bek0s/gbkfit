
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.driver.base import DeviceArray, DriverFFT, FFTConvolution


__all__ = [
    'DriverFFTHost'
]


class DriverFFTHost(DriverFFT):
    """
    The FFT of the host driver: pocketfft, in the native module (its
    FFTConvolution classes).
    """

    def __init__(self, native_convolution: type, dtype: npt.DTypeLike):
        super().__init__(32)
        self._native_convolution = native_convolution
        self._dtype = np.dtype(dtype)

    def dtype(self) -> np.dtype:
        return self._dtype

    def _convolution(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...]
    ) -> FFTConvolution:
        return _FFTConvolutionHost(
            self._native_convolution(kernel, shape, axes))


class _FFTConvolutionHost(FFTConvolution):
    """A convolution of the native module."""

    def __init__(self, native: Any):
        self._native = native

    def __call__(self, cube: DeviceArray) -> None:
        self._native(cube)
