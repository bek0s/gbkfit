
import abc

import numpy as np

import gbkfit.math


__all__ = [
    'DriverFFT'
]


class DriverFFT(abc.ABC):
    """
    The FFT of a driver, for the FFT-based convolution of DCube: the
    transforms, and the shapes of the arrays they need.
    """

    @staticmethod
    def fft_complex_shape(shape):
        return tuple(shape[:-1]) + (shape[-1] // 2 + 1,)

    @staticmethod
    def fft_convolution_shift(data):
        axis = tuple(range(data.ndim))
        return np.roll(data, np.array(data.shape) // 2 + 1, axis=axis)

    def __init__(self, fft_shape_threshold):
        self._fft_shape_threshold = fft_shape_threshold

    def fft_optimal_shape(self, shape):
        threshold = self._fft_shape_threshold
        new_shape = []
        for dim in shape:
            dim_po2 = gbkfit.math.roundu_po2(dim)
            dim_mul = gbkfit.math.roundu_multiple(dim, threshold)
            dim_new = dim_po2 if dim_po2 <= threshold else dim_mul
            new_shape.append(dim_new)
        return np.array(new_shape)

    def fft_convolution_shape(self, data1_shape, data2_shape):
        shape = np.array(data1_shape) + np.array(data2_shape) - 1
        shape = self.fft_optimal_shape(shape)
        margin = np.array(data2_shape) // 2
        return tuple(shape), tuple(margin)

    @abc.abstractmethod
    def fft_r2c(self, data_r, data_c):
        pass

    @abc.abstractmethod
    def fft_c2r(self, data_c, data_r):
        pass

    @abc.abstractmethod
    def fft_convolve_cached(self, data1_r, data2_r):
        pass
