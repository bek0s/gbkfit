
import abc

from gbkfit.utils import parseutils


class DriverNativeModules(abc.ABC):

    @abc.abstractmethod
    def fft(self, dtype):
        pass

    @abc.abstractmethod
    def dmodel(self, dtype):
        pass

    @abc.abstractmethod
    def gmodel(self, dtype):
        pass

    @abc.abstractmethod
    def objective(self, dtype):
        pass


class Driver(parseutils.TypedSerializable, abc.ABC):

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return dict(type=self.type())

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
    def backends(self):
        pass


driver_parser = parseutils.TypedParser(Driver)
