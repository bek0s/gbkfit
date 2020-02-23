
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

    def prepare(self, driver, dmodel, gmodel):
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._prepare_impl()

    def evaluate(self, driver, dmodel, gmodel, params, dextra, gextra):
        if (self._driver is not driver
                or self._dmodel is not dmodel
                or self._gmodel is not gmodel):
            self.prepare(driver, dmodel, gmodel)
        self._evaluate_impl(params, dextra, gextra)

    @abc.abstractmethod
    def output(self):
        pass

    @abc.abstractmethod
    def _prepare_impl(self):
        pass

    @abc.abstractmethod
    def _evaluate_impl(self, params, dextra, gextra):
        pass


parser = parseutils.TypedParser(Broker)
