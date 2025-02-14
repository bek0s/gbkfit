
import abc

from gbkfit.utils import parseutils


__all__ = [
    'BrightnessComponent2D',
    'BrightnessComponent3D',
    'SpectralComponent2D',
    'SpectralComponent3D',
    'OpacityComponent3D'
]


class BrightnessComponent2D(parseutils.TypedSerializable, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params,
            image_d, image_w, bdata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        pass


class BrightnessComponent3D(parseutils.TypedSerializable, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params, odata,
            image, wdata, rdata, ordata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        pass


class SpectralComponent2D(parseutils.TypedSerializable, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params,
            scube_d, scube_w, rdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        pass


class SpectralComponent3D(parseutils.TypedSerializable, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params, odata,
            scube_d, scube_w, rdata, ordata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        pass


class OpacityComponent3D(parseutils.TypedSerializable, abc.ABC):

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, driver, params,
            odata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        pass
