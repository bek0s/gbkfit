
from gbkfit.model.core import GModelImage
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import DensityComponent2D
from .density_smdisk_2d import DensitySMDisk2D


__all__ = ['GModelIntensity2D']


_dcmp_parser = parseutils.TypedParser(DensityComponent2D, [
    DensitySMDisk2D])


class GModelIntensity2D(GModelImage):

    @staticmethod
    def type():
        return 'intensity_2d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=_dcmp_parser.load(info['components']))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            components=_dcmp_parser.dump(self._components))

    def __init__(self, components):
        self._components = iterutils.tuplify(components)
        self._size = [None, None]
        self._step = [None, None]
        self._zero = [None, None]
        self._wcube = None
        self._dtype = None
        self._driver = None
        self._backend = None
        (self._params,
         self._mappings) = _detail.make_gmodel_2d_params(
            self._components)

    def params(self):
        return self._params

    def is_weighted(self):
        return True

    def _prepare(self, driver, wdata, size, step, zero, dtype):
        self._driver = driver
        self._size = size
        self._step = step
        self._zero = zero
        self._wcube = None
        self._dtype = dtype
        # If weighting is requested, store it in a 2d spatial array.
        if wdata is not None:
            self._wcube = driver.mem_alloc_d(self._size[::-1], dtype)
        # Create backend
        self._backend = driver.backends().gmodel(dtype)

    def evaluate_image(
            self, driver, params, image, wdata, size, step, zero, rota, dtype,
            out_extra):

        if (self._driver is not driver
                or self._size != size
                or self._dtype != dtype):
            self._prepare(driver, wdata, size, step, zero, dtype)

        # Convenience variables
        spat_size = self._size
        spat_step = self._step
        spat_zero = self._zero
        spat_rota = rota
        wcube = self._wcube
        components = self._components
        mappings = self._mappings
        backend = self._backend

        # Evaluate components
        _detail.evaluate_components_d2d(
            components, driver, params, mappings, image, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra, '')

        # Evaluate the provided spectral weight cube using
        # the spatial weight cube evaluated above
        if wcube is not None:
            backend.wcube_evaluate(
                tuple(spat_size) + (1,), 1, wcube, wdata)
