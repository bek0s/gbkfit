
import abc

from gbkfit.utils import parseutils


class GModel(abc.ABC):

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

    @abc.abstractmethod
    def params(self):
        pass


class GModelImageSupport(abc.ABC):

    @abc.abstractmethod
    def evaluate_image(
            self, driver, params, image, dtype, size, step, zero, out_extra):
        pass


class GModelSCubeSupport(abc.ABC):

    @abc.abstractmethod
    def evaluate_scube(
            self, driver, params, scube, dtype, size, step, zero, out_extra):
        pass


parser = parseutils.SimpleParser(GModel)
