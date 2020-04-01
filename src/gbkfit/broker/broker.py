
import abc

from gbkfit.utils import parseutils


class Broker(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
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
        self._dmodel = None
        self._gmodel = None

    def _prepare(self, driver, dmodel, gmodel):
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._impl_prepare(driver, dmodel, gmodel)

    def evaluate(
            self, driver, dmodel, gmodel, params,
            out_dextra=None, out_gextra=None):
        if (self._driver is not driver
                or self._dmodel is not dmodel
                or self._gmodel is not gmodel):
            self._prepare(driver, dmodel, gmodel)
        return self._impl_evaluate(
            driver, dmodel, gmodel, params, out_dextra, out_gextra)

    @abc.abstractmethod
    def _impl_prepare(self, driver, dmodel, gmodel):
        pass

    @abc.abstractmethod
    def _impl_evaluate(self, driver, dmodel, gmodel, params, dextra, gextra):
        pass


parser = parseutils.TypedParser(Broker)
