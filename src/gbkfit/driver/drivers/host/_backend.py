
import numpy as np

import gbkfit.driver.native.libgbkfit_host as native_module
from gbkfit.driver.modules import DriverBackends
from .._detail.native import *


__all__ = ['DriverBackendsHost']


class NativeMemoryHost(NativeMemory):

    @staticmethod
    def ptr(x):
        return x.__array_interface__['data'][0] if x is not None else 0

    @staticmethod
    def size(x):
        return x.size if x is not None else 0

    @staticmethod
    def shape(x):
        return x.__array_interface__['shape'] if x is not None else None

    @staticmethod
    def dtype(x):
        return x.__array_interface__['typestr'] if x is not None else None


class DriverBackendsHost(DriverBackends):

    def fft(self, dtype):
        return DriverBackendFFTHost(dtype)

    def dmodel(self, dtype):
        return DriverBackendDModelHost(dtype)

    def gmodel(self, dtype):
        return DriverBackendGModelHost(dtype)

    def objective(self, dtype):
        raise NotImplementedError()


class DriverBackendFFTHost(DriverBackendFFTNative):

    def __init__(self, dtype):
        super().__init__(dtype, NativeMemoryHost, {
            np.float32: native_module.FFTf32
        })


class DriverBackendDModelHost(DriverBackendDModelNative):

    def __init__(self, dtype):
        super().__init__(dtype, NativeMemoryHost, {
            np.float32: native_module.DModelf32
        })


class DriverBackendGModelHost(DriverBackendGModelNative):

    def __init__(self, dtype):
        super().__init__(dtype, NativeMemoryHost, {
            np.float32: native_module.GModelf32
        })
