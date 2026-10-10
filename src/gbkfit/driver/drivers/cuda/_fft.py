
import cupy as cp
import numpy as np
from cupy.cuda import cufft
from cupyx.scipy import fft as cupyx_fft

from gbkfit.driver.fft import DriverFFT


__all__ = [
    'DriverFFTCuda'
]


class DriverFFTCuda(DriverFFT):
    """
    The FFT of the cuda driver: cuFFT, through CuPy.

    Like the host FFT, it caches one cuFFT plan per array shape
    and transform direction, and the buffers used by fft_convolve_cached()
    (including the spectrum of the second array, e.g., the PSF/LSF). After
    the first call, no device memory is allocated.
    """

    def __init__(self, dtype):
        super().__init__(32)
        self._dtype = np.dtype(dtype)
        self._complex_dtype = np.result_type(self._dtype, np.complex64)
        self._plans = {}
        self._data_spectra = {}
        self._kernel_spectra = {}

    def __deepcopy__(self, memodict):
        return self.__class__(self.dtype())

    def dtype(self):
        return self._dtype

    def fft_r2c(self, data_r, data_c, axes=(0, 1, 2)):
        key = (data_r.shape, axes, 'r2c')
        if key not in self._plans:
            self._plans[key] = cupyx_fft.get_fft_plan(
                data_r, axes=axes, value_type='R2C')
        self._plans[key].fft(data_r, data_c, cufft.CUFFT_FORWARD)

    def fft_c2r(self, data_c, data_r, axes=(0, 1, 2)):
        # Note: cuFFT overwrites the input of a c2r transform
        key = (data_r.shape, axes, 'c2r')
        if key not in self._plans:
            # (shape is that of the transformed axes of the output)
            self._plans[key] = cupyx_fft.get_fft_plan(
                data_c, shape=tuple(data_r.shape[axis] for axis in axes),
                axes=axes, value_type='C2R')
        self._plans[key].fft(data_c, data_r, cufft.CUFFT_INVERSE)

    def fft_convolve_cached(self, data1_r, data2_r):
        shape = data1_r.shape
        data1_key = (shape, data1_r.data.ptr)
        data2_key = (shape, data2_r.data.ptr)
        if data2_key not in self._kernel_spectra:
            # Store the spectrum already scaled by the normalisation factor
            spectrum = self._alloc_spectrum(shape)
            self.fft_r2c(data2_r, spectrum)
            spectrum *= self._dtype.type(1 / data1_r.size)
            self._kernel_spectra[data2_key] = spectrum
        if data1_key not in self._data_spectra:
            self._data_spectra[data1_key] = self._alloc_spectrum(shape)
        data1_c = self._data_spectra[data1_key]
        self.fft_r2c(data1_r, data1_c)
        data1_c *= self._kernel_spectra[data2_key]
        self.fft_c2r(data1_c, data1_r)

    def fft_convolve_xy_cached(self, data1_r, data2_r):
        # The spectra of the images (2D transforms along y and x), in their
        # own caches; the kernel is broadcast if it has one image
        shape = data1_r.shape
        axes = (1, 2)
        data1_key = (shape, data1_r.data.ptr, 'xy')
        data2_key = (data2_r.shape, data2_r.data.ptr, 'xy')
        if data2_key not in self._kernel_spectra:
            spectrum = self._alloc_spectrum(data2_r.shape)
            self.fft_r2c(data2_r, spectrum, axes)
            spectrum *= self._dtype.type(1 / (shape[1] * shape[2]))
            self._kernel_spectra[data2_key] = spectrum
        if data1_key not in self._data_spectra:
            self._data_spectra[data1_key] = self._alloc_spectrum(shape)
        data1_c = self._data_spectra[data1_key]
        self.fft_r2c(data1_r, data1_c, axes)
        data1_c *= self._kernel_spectra[data2_key]
        self.fft_c2r(data1_c, data1_r, axes)

    def _alloc_spectrum(self, shape):
        return cp.empty(self.fft_complex_shape(shape), self._complex_dtype)
