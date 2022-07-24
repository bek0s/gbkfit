
import cupy as cp
import numpy as np

from gbkfit.driver.core import Driver


__all__ = ['DriverCuda']


class DriverCuda(Driver):

    @staticmethod
    def type():
        return 'cuda'

    def mem_alloc_s(self, shape, dtype):
        h_data = np.empty(shape, dtype)
        d_data = cp.empty(shape, dtype)
        return h_data, d_data

    def mem_alloc_h(self, shape, dtype):
        return np.empty(shape, dtype)

    def mem_alloc_d(self, shape, dtype):
        return cp.empty(shape, dtype)

    def mem_copy_h2d(self, h_src, d_dst=None):
        if d_dst is None:
            d_dst = self.mem_alloc_d(h_src.shape, h_src.dtype)
        if h_src is not d_dst:
            d_dst.set(h_src)
        return d_dst

    def mem_copy_d2h(self, d_src, h_dst=None):
        if h_dst is None:
            h_dst = self.mem_alloc_h(d_src.shape, d_src.dtype)
        if d_src is not h_dst:
            h_dst[:] = d_src.get()
        return h_dst

    def mem_fill(self, x, value):
        x.fill(value)

    def math_abs(self, x, out=None):
        return cp.abs(x, out=out)

    def math_sum(self, x, out=None):
        return cp.nansum(x, out=out, keepdims=True)

    def math_add(self, x1, x2, out=None):
        return cp.add(x1, x2, out=out)

    def math_sub(self, x1, x2, out=None):
        return cp.subtract(x1, x2, out=out)

    def math_mul(self, x1, x2, out=None):
        return cp.multiply(x1, x2, out=out)

    def math_div(self, x1, x2, out=None):
        return cp.divide(x1, x2, out=out)

    def math_pow(self, x1, x2, out=None):
        return cp.power(x1, x2, out=out)

    def backends(self):
        from ._backend import DriverBackendsCuda
        return DriverBackendsCuda()
