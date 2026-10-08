
import abc

import numpy as np

from gbkfit.utils import parseutils


# The suffix of the native classes for each supported dtype
_NATIVE_CLASS_SUFFIXES = {np.dtype(np.float32): 'f32'}


class Driver(parseutils.TypedSerializable, abc.ABC):

    @classmethod
    def load(cls, info, *args, **kwargs):
        return cls()

    @abc.abstractmethod
    def mem_alloc_s(self, shape, dtype):
        pass

    @abc.abstractmethod
    def mem_alloc_h(self, shape, dtype):
        pass

    @abc.abstractmethod
    def mem_alloc_d(self, shape, dtype):
        pass

    @abc.abstractmethod
    def mem_copy_h2d(self, h_src, d_dst=None):
        pass

    @abc.abstractmethod
    def mem_copy_d2h(self, d_src, h_dst=None):
        pass

    @abc.abstractmethod
    def mem_fill(self, x, value):
        pass

    @abc.abstractmethod
    def math_abs(self, x, out=None):
        pass

    @abc.abstractmethod
    def math_sum(self, x, out=None):
        pass

    @abc.abstractmethod
    def math_add(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def math_sub(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def math_mul(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def math_div(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def math_pow(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def native_module(self):
        """The native (C++/cuda) module of the driver."""
        pass

    def native_class(self, name, dtype):
        """
        The class of the native module with the given name, for the given
        dtype (e.g., 'GModel' for float32 is the class GModelf32).
        """
        dtype = np.dtype(dtype)
        if dtype not in _NATIVE_CLASS_SUFFIXES:
            supported = [dt.name for dt in _NATIVE_CLASS_SUFFIXES]
            raise RuntimeError(
                f"the {self.type()} driver does not support dtype "
                f"{dtype.name}; supported dtypes: {supported}")
        return getattr(
            self.native_module(), name + _NATIVE_CLASS_SUFFIXES[dtype])

    @abc.abstractmethod
    def fft(self, dtype):
        """The FFT of the driver (a DriverFFT) for the given dtype."""
        pass


driver_parser = parseutils.TypedParser(Driver)
