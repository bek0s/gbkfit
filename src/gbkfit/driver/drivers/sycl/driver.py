import dpctl
import dpctl.tensor as dpt
import dpnp as np

from gbkfit.driver.core import Driver


__all__ = [
    'DriverSycl'
]


class DriverSycl(Driver):

    @staticmethod
    def type():
        return 'sycl'

    def mem_alloc_s(self, shape, dtype):
        # Shared memory (accessible from both host and device)
        h_data = dpt.empty(shape, dtype, usm_type='shared')
        d_data = h_data
        return h_data, d_data

    def mem_alloc_h(self, shape, dtype):
        return np.empty(shape, dtype)

    def mem_alloc_d(self, shape, dtype):
        return dpt.empty(shape, dtype, usm_type='device')

    def mem_copy_h2d(self, h_src, d_dst=None):
        if d_dst is None:
            d_dst = self.mem_alloc_d(h_src.shape, h_src.dtype)
        if h_src is not d_dst:
            d_dst[:] = dpt.asarray(h_src, copy=True)
        return d_dst

    def mem_copy_d2h(self, d_src, h_dst=None):
        if h_dst is None:
            h_dst = self.mem_alloc_h(d_src.shape, d_src.dtype)
        if d_src is not h_dst:
            dpt.copy(d_src, h_dst)
        return h_dst

    def mem_fill(self, x, value):
        x[:] = value

    def math_abs(self, x, out=None):
        return dpt.abs(x, out=out)

    def math_sum(self, x, out=None):
        return dpt.sum(x, out=out, keepdims=True)

    def math_add(self, x1, x2, out=None):
        return dpt.add(x1, x2, out=out)

    def math_sub(self, x1, x2, out=None):
        return dpt.subtract(x1, x2, out=out)

    def math_mul(self, x1, x2, out=None):
        return dpt.multiply(x1, x2, out=out)

    def math_div(self, x1, x2, out=None):
        return dpt.divide(x1, x2, out=out)

    def math_pow(self, x1, x2, out=None):
        return dpt.power(x1, x2, out=out)

    def backends(self):
        from ._backend import DriverBackendsSycl
        return DriverBackendsSycl()
