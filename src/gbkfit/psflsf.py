
import abc

import gbkfit.math
from gbkfit.utils import parseutils


class LSF(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self, **kwargs):
        pass

    def size(self, step, offset=0):
        size = self._size_impl(step)
        return int(gbkfit.math.roundu_odd(size + offset))

    def asarray(self, step, size=None, offset=0):
        if size is None:
            size = self.size(step, offset)
        if gbkfit.math.is_even(size + offset):
            raise RuntimeError(f"size {size} + offset {offset} must be odd")
        return self._asarray_impl(step, size, offset)

    @abc.abstractmethod
    def _size_impl(self, step):
        pass

    @abc.abstractmethod
    def _asarray_impl(self, step, size, offset):
        pass


class PSF(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self, **kwargs):
        pass

    def size(self, step, offset=(0, 0)):
        size = self._size_impl(step)
        return (int(gbkfit.math.roundu_odd(size[0] + offset[0])),
                int(gbkfit.math.roundu_odd(size[1] + offset[1])))

    def asarray(self, step, size=None, offset=(0, 0)):
        if size is None:
            size = self.size(step, offset)
        if any(gbkfit.math.is_even(s + o) for s, o in zip(size, offset)):
            raise RuntimeError(f"size {size} + offset {offset} must be odd")
        return self._asarray_impl(step, size, offset)

    @abc.abstractmethod
    def _size_impl(self, step):
        pass

    @abc.abstractmethod
    def _asarray_impl(self, step, size, offset):
        pass


lsf_parser = parseutils.SimpleParser(LSF)
psf_parser = parseutils.SimpleParser(PSF)
