
import abc

from gbkfit.utils import parseutils


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

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero(self):
        pass

    @abc.abstractmethod
    def onames(self):
        pass

    def prepare(self, driver, gmodel):
        if not self.is_compatible(gmodel):
            raise RuntimeError(
                f"{make_dmodel_desc(self.__class__)} "
                f"is not compatible with "
                f"{make_gmodel_desc(gmodel.__class__)}")
        self._driver = driver
        self._gmodel = gmodel
        self._prepare_impl()

    def evaluate(self, driver, gmodel, params, out_extra=None):
        if self._driver is not driver or self._gmodel is not gmodel:
            self.prepare(driver, gmodel)
        out_dextra = None if out_extra is None else {}
        out_gextra = None if out_extra is None else {}
        out = self._evaluate_impl(params, out_dextra, out_gextra)
        if out_dextra:
            out_extra.update({f'dmodel_{k}': v for k, v in out_dextra.items()})
        if out_gextra:
            out_extra.update({f'gmodel_{k}': v for k, v in out_gextra.items()})
        return out

    @abc.abstractmethod
    def _prepare_impl(self):
        pass

    @abc.abstractmethod
    def _evaluate_impl(self, params, out_dextra, out_gextra):
        pass


class GModel(parseutils.TypedParserSupport, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass


class GModelImage(GModel, abc.ABC):

    @abc.abstractmethod
    def evaluate_image(
            self, driver, params, image, size, step, zero, rota, dtype,
            out_extra):
        pass


class GModelSCube(GModel, abc.ABC):

    @abc.abstractmethod
    def evaluate_scube(
            self, driver, params, scube, size, step, zero, rota, dtype,
            out_extra):
        pass


dmodel_parser = parseutils.TypedParser(DModel)
gmodel_parser = parseutils.TypedParser(GModel)
