
import abc

import numpy as np

from gbkfit.utils import parseutils


class DModel(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @staticmethod
    @abc.abstractmethod
    def is_compatible(gmodel):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    def __init__(self):
        self._driver = None
        self._gmodel = None

    def ndim(self):
        return len(self.size())

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def cval(self):
        pass

    @abc.abstractmethod
    def onames(self):
        pass

    def submodel(self, min_, max_):
        if self.ndim() != len(min_) != len(max_):
            raise RuntimeError()
        min_ = np.asarray(min_)
        max_ = np.asarray(max_)
        size = np.asarray(self.size())
        step = np.asarray(self.step())
        cval = np.asarray(self.cval())
        cpix = (size - 1) / 2
        new_size = (max_ - min_ + 1)
        new_cpix = (min_ + max_) / 2
        new_cval = cval + (new_cpix - cpix) * step
        return self._submodel_impl(tuple(new_size), tuple(new_cval))

    def prepare(self, driver, gmodel):
        if not self.is_compatible(gmodel):
            raise RuntimeError(
                f"DModel of type '{self.type()}' is not compatible with "
                f"GModel of type '{gmodel.type()}'")
        self._driver = driver
        self._gmodel = gmodel
        self._prepare_impl()

    def evaluate(self, driver, gmodel, params, out_extra=None):
        if self._driver is not driver or self._gmodel is not gmodel:
            self.prepare(driver, gmodel)

        out_dextra = None if out_extra is None else {}
        out_gextra = None if out_extra is None else {}

        out = self._evaluate_impl(params, out_dextra, out_gextra)

        if out_dextra is not None:
            out_extra.update({f'dmodel_{k}': v for k, v in out_dextra.items()})
        if out_gextra is not None:
            out_extra.update({f'gmodel_{k}': v for k, v in out_gextra.items()})

        return out

    @abc.abstractmethod
    def _submodel_impl(self, size, cval):
        pass

    @abc.abstractmethod
    def _prepare_impl(self):
        pass

    @abc.abstractmethod
    def _evaluate_impl(self, params, out_dextra, out_gextra):
        pass


parser = parseutils.TypedParser(DModel)