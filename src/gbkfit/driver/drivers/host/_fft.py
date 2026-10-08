
import numpy as np

from gbkfit.driver.fft import DriverFFT


__all__ = [
    'DriverFFTHost'
]


class DriverFFTHost(DriverFFT):
    """
    The FFT of the host driver: pocketfft, in the native module.

    Like the cuda FFT, it caches the spectra used by fft_convolve_cached()
    (including the one of the second array, e.g., the PSF/LSF).
    """

    def __init__(self, native_fft, dtype):
        super().__init__(32)
        self._fft = native_fft
        self._dtype = np.dtype(dtype)

    def dtype(self):
        return self._dtype

    def fft_r2c(self, data_r, data_c):
        self._fft.fft_r2c(data_r, data_c)

    def fft_c2r(self, data_c, data_r):
        self._fft.fft_c2r(data_c, data_r)

    def fft_convolve_cached(self, data1_r, data2_r):
        self._fft.fft_convolve_cached(data1_r, data2_r)
