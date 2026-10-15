
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
    """The FFT of the cuda driver: cuFFT, through CuPy, on its GPU."""

    # (measured 2026-10-15 with float64 FFTs: 6-35% faster than the 3D
    # FFT, and less memory; on the host, the 3D FFT is faster)
    separable = True

    def __init__(self, dtype: npt.DTypeLike, device: cp.cuda.Device):
        super().__init__(32)
        self._dtype = np.dtype(dtype)
        self._device = device

    def dtype(self) -> np.dtype:
        return self._dtype

    def _convolution(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...],
            dtype: np.dtype
    ) -> FFTConvolution:
        return _FFTConvolutionCuda(kernel, shape, axes, dtype, self._device)


class _FFTConvolutionCuda(FFTConvolution):
    """
    A convolution by cuFFT on a GPU: one cuFFT plan for each direction,
    and a buffer for the spectra of the cubes, through which the kernel is
    also uploaded (so it never has device memory of its own). After it is
    made, no device memory is allocated. Cubes less precise than the
    transforms are copied into the buffer, as real rows padded to the
    length of its complex rows, transformed there in place, and copied
    back.
    """

    def __init__(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...],
            dtype: np.dtype,
            device: cp.cuda.Device
    ):
        self._device = device
        fft_dtype = kernel.dtype
        complex_dtype = np.result_type(fft_dtype, np.complex64)
        with device:
            self._spectrum = cp.empty(_complex_shape(shape), complex_dtype)
            # The spectrum of the kernel, scaled by the normalisation of the
            # inverse transform
            kernel_d = _real_view(self._spectrum, kernel.shape, fft_dtype)
            kernel_d.set(kernel)
            self._kernel_spectrum = cp.empty(
                _complex_shape(kernel.shape), complex_dtype)
            cupyx_fft.get_fft_plan(kernel_d, axes=axes, value_type='R2C').fft(
                kernel_d, self._kernel_spectrum, cufft.CUFFT_FORWARD)
            size = np.prod([shape[axis] for axis in axes])
            self._kernel_spectrum *= fft_dtype.type(1 / size)
            self._padded = None
            if dtype == fft_dtype:
                self._r2c = cupyx_fft.get_fft_plan(
                    _real_view(self._spectrum, shape, fft_dtype), axes=axes,
                    value_type='R2C')
                self._c2r = cupyx_fft.get_fft_plan(
                    self._spectrum, shape=tuple(shape[axis] for axis in axes),
                    axes=axes, value_type='C2R')
            else:
                self._padded = self._spectrum.view(fft_dtype)[..., :shape[2]]
                self._r2c, self._c2r = _in_place_plans(shape, axes, fft_dtype)

    def __call__(self, cube: DeviceArray) -> None:
        # (the kernel is broadcast if it has one image for all channels;
        # cuFFT overwrites the input of a c2r transform)
        spectrum = self._spectrum
        with self._device:
            if self._padded is None:
                self._r2c.fft(cube, spectrum, cufft.CUFFT_FORWARD)
                spectrum *= self._kernel_spectrum
                self._c2r.fft(spectrum, cube, cufft.CUFFT_INVERSE)
            else:
                self._padded[...] = cube
                self._r2c.fft(spectrum, spectrum, cufft.CUFFT_FORWARD)
                spectrum *= self._kernel_spectrum
                self._c2r.fft(spectrum, spectrum, cufft.CUFFT_INVERSE)
                cube[...] = self._padded


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


def _in_place_plans(
        shape: tuple[int, int, int],
        axes: tuple[int, ...],
        dtype: np.dtype
) -> tuple[cufft.PlanNd, cufft.PlanNd]:
    """
    Return the cuFFT plans of the transforms (r2c and c2r) along the axes
    of real arrays of a shape, in place in their spectra: their rows are
    padded to the length of the complex rows (x // 2 + 1 values).
    """
    # The transformed shape, and how many times (each channel, or once)
    transformed = [shape[axis] for axis in axes]
    batch = shape[0] if axes == (1, 2) else 1
    columns = shape[2] // 2 + 1
    real = transformed[:-1] + [2 * columns]
    complex_ = transformed[:-1] + [columns]
    real_size, complex_size = int(np.prod(real)), int(np.prod(complex_))
    r2c_type, c2r_type = (cufft.CUFFT_D2Z, cufft.CUFFT_Z2D) \
        if dtype == np.float64 else (cufft.CUFFT_R2C, cufft.CUFFT_C2R)
    last = len(transformed) - 1
    r2c = cufft.PlanNd(
        transformed, real, 1, real_size, complex_, 1, complex_size,
        r2c_type, batch, 'C', last, columns)
    c2r = cufft.PlanNd(
        transformed, complex_, 1, complex_size, real, 1, real_size,
        c2r_type, batch, 'C', last, shape[2])
    return r2c, c2r
