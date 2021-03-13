
import abc

from gbkfit.utils import parseutils


class DensityComponent2D(parseutils.TypedParserSupport, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params, image,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        pass


class DensityComponent3D(parseutils.TypedParserSupport, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params, image, rcube,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        pass


class SpectralComponent2D(parseutils.TypedParserSupport, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params, scube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        pass


class SpectralComponent3D(parseutils.TypedParserSupport, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params, scube, rcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        pass
