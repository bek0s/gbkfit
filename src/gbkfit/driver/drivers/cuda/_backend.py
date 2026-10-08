
import numpy as np

import gbkfit.driver.native._cuda as native_module
from gbkfit.driver.backend import DriverBackends
from ._fft import DriverBackendFFTCuda
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
        return native_class('DModel', dtype, {
            np.dtype(np.float32): native_module.DModelf32})()

    def gmodel(self, dtype):
        return DriverBackendGModelCuda(dtype)

    def objective(self, dtype):
        return native_class('Objective', dtype, {
            np.dtype(np.float32): native_module.Objectivef32})()


class DriverBackendGModelCuda(DriverBackendGModelNative):

    def __init__(self, dtype):
        super().__init__(dtype, NativeMemoryCuda, {
            np.dtype(np.float32): native_module.GModelf32
        })

    def __deepcopy__(self, memodict):
        return self.__class__(self.dtype())
