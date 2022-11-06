
import numpy as np

import gbkfit.driver.native.libgbkfit_cuda as native_module
from gbkfit.driver.modules import DriverBackends
from .._detail.native import *


__all__ = [
    'DriverBackendsCuda'
]


class NativeMemoryCuda(NativeMemory):

    @staticmethod
    def ptr(x):
        return x.__cuda_array_interface__['data'][0] if x is not None else 0

    @staticmethod
    def size(x):
        return x.size if x is not None else 0

    @staticmethod
    def shape(x):
        return x.__cuda_array_interface__['shape'] if x is not None else None

    @staticmethod
    def dtype(x):
        return x.__cuda_array_interface__['typestr'] if x is not None else None


class DriverBackendsCuda(DriverBackends):

    def fft(self, dtype):
        return DriverBackendFFTCuda(dtype)

    def dmodel(self, dtype):
        return DriverBackendDModelCuda(dtype)

    def gmodel(self, dtype):
        return DriverBackendGModelCuda(dtype)

    def objective(self, dtype):
        raise NotImplementedError()


class DriverBackendFFTCuda(DriverBackendFFTNative):

    def __init__(self, dtype):
        super().__init__(dtype, NativeMemoryCuda, {
            np.float32: native_module.FFTf32
        })


class DriverBackendDModelCuda(DriverBackendDModelNative):

    def __init__(self, dtype):
        super().__init__(dtype, NativeMemoryCuda, {
            np.float32: native_module.DModelf32
        })


class DriverBackendGModelCuda(DriverBackendGModelNative):

    def __init__(self, dtype):
        super().__init__(dtype, NativeMemoryCuda, {
            np.float32: native_module.GModelf32
        })
