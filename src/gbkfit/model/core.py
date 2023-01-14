
import abc

import numpy as np

from gbkfit.utils import parseutils


__all__ = [
    'DModel',
    'GModel',
    'GModelImage',
    'GModelSCube',
    'dmodel_parser',
    'gmodel_parser'
]


class DModel(parseutils.TypedParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def is_compatible(gmodel):
        pass

    def __init__(self):
        self._driver = None
        self._gmodel = None

    def ndim(self):
        return len(self.size())

    def npix(self):
        return int(np.prod(self.size()))

    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero(self):
        pass

    def _prepare(self, driver, gmodel):
        if not self.is_compatible(gmodel):
            dmodel_desc = parseutils.make_typed_desc(self.__class__, 'dmodel')
            gmodel_desc = parseutils.make_typed_desc(gmodel.__class__, 'gmodel')
            raise RuntimeError(
                f"{dmodel_desc} is not compatible with {gmodel_desc}")
        self._driver = driver
        self._gmodel = gmodel
        self._prepare_impl(gmodel)

    def evaluate(self, driver, gmodel, params, out_extra=None):
        if self._driver is not driver or self._gmodel is not gmodel:
            self._prepare(driver, gmodel)
        out_dmodel_extra = None if out_extra is None else {}
        out_gmodel_extra = None if out_extra is None else {}
        out = self._evaluate_impl(params, out_dmodel_extra, out_gmodel_extra)
        if out_dmodel_extra:
            out_extra.update(
                {f'dmodel_{k}': v for k, v in out_dmodel_extra.items()})
        if out_gmodel_extra:
            out_extra.update(
                {f'gmodel_{k}': v for k, v in out_gmodel_extra.items()})
        return out

    @abc.abstractmethod
    def _prepare_impl(self, gmodel):
        pass

    @abc.abstractmethod
    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        pass


class GModel(parseutils.TypedParserSupport, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def is_weighted(self):
        pass


class GModelImage(GModel, abc.ABC):

    @abc.abstractmethod
    def evaluate_image(
            self, driver, params, image, weights, size, step, zero, rota, dtype,
            out_extra):
        pass


class GModelSCube(GModel, abc.ABC):

    @abc.abstractmethod
    def evaluate_scube(
            self, driver, params, scube, weights, size, step, zero, rota, dtype,
            out_extra):
        pass


dmodel_parser = parseutils.TypedParser(DModel)
gmodel_parser = parseutils.TypedParser(GModel)
