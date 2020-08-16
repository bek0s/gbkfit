
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


component_d2d_parser = parseutils.TypedParser(DensityComponent2D)
component_d3d_parser = parseutils.TypedParser(DensityComponent3D)
component_s2d_parser = parseutils.TypedParser(SpectralComponent2D)
component_s3d_parser = parseutils.TypedParser(SpectralComponent3D)


def make_component_desc(cls):
    return f'{cls.type()} gmodel component (class={cls.__qualname__})'
