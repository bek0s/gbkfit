
from typing import Any

import cupy as cp
import numpy as np
import numpy.typing as npt
from cupy.cuda import cufft
from cupyx.scipy import fft as cupyx_fft

from gbkfit.driver.base import DeviceArray, DriverFFT, FFTConvolution


__all__ = [
    'DriverFFTCuda'
]


class DriverFFTCuda(DriverFFT):
    """The FFT of the cuda driver: cuFFT, through CuPy."""

    def __init__(self, dtype: npt.DTypeLike):
        super().__init__(32)
        self._dtype = np.dtype(dtype)

    def dtype(self) -> np.dtype:
        return self._dtype

    def _convolution(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...]
    ) -> FFTConvolution:
        return _FFTConvolutionCuda(kernel, shape, axes)


class _FFTConvolutionCuda(FFTConvolution):
    """
    A convolution by cuFFT: one cuFFT plan for each direction, and a
    buffer for the spectra of the cubes, through which the kernel is also
    uploaded (so it never has device memory of its own). After it is made,
    no device memory is allocated.
    """

    def __init__(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...]
    ):
        complex_dtype = np.result_type(kernel.dtype, np.complex64)
        self._spectrum = cp.empty(_complex_shape(shape), complex_dtype)
        cube = _real_view(self._spectrum, shape, kernel.dtype)
        self._r2c = cupyx_fft.get_fft_plan(cube, axes=axes, value_type='R2C')
        self._c2r = cupyx_fft.get_fft_plan(
            self._spectrum, shape=tuple(shape[axis] for axis in axes),
            axes=axes, value_type='C2R')
        # The spectrum of the kernel, scaled by the normalisation of the
        # inverse transform
        kernel_d = _real_view(self._spectrum, kernel.shape, kernel.dtype)
        kernel_d.set(kernel)
        plan = self._r2c if kernel.shape == shape else \
            cupyx_fft.get_fft_plan(kernel_d, axes=axes, value_type='R2C')
        self._kernel_spectrum = cp.empty(
            _complex_shape(kernel.shape), complex_dtype)
        plan.fft(kernel_d, self._kernel_spectrum, cufft.CUFFT_FORWARD)
        size = np.prod([shape[axis] for axis in axes])
        self._kernel_spectrum *= kernel.dtype.type(1 / size)

    def __call__(self, cube: DeviceArray) -> None:
        # (the kernel is broadcast if it has one image for all channels;
        # cuFFT overwrites the input of a c2r transform)
        self._r2c.fft(cube, self._spectrum, cufft.CUFFT_FORWARD)
        self._spectrum *= self._kernel_spectrum
        self._c2r.fft(self._spectrum, cube, cufft.CUFFT_INVERSE)


def _complex_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the shape of the spectrum of a real array."""
    return tuple(shape[:-1]) + (shape[-1] // 2 + 1,)


def _real_view(
        spectrum: Any,
        shape: tuple[int, ...],
        dtype: np.dtype
) -> Any:
    """
    Return a contiguous real array of a shape in the memory of a
    spectrum (of at least as many bytes).
    """
    size = int(np.prod(shape))
    return spectrum.view(dtype).ravel()[:size].reshape(shape)
