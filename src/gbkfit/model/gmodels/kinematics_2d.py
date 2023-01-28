
from gbkfit.model.core import GModelSCube
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import SpectralComponent2D
from .spectral_smdisk_2d import SpectralSMDisk2D


__all__ = [
    'GModelKinematics2D'
]


_scmp_parser = parseutils.TypedParser(SpectralComponent2D, [
    SpectralSMDisk2D])


class GModelKinematics2D(GModelSCube):

    @staticmethod
    def type():
        return 'kinematics_2d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=_scmp_parser.load(info['components']))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            components=_scmp_parser.dump(self._components))

    def __init__(self, components):
        self._components = iterutils.tuplify(components)
        self._size = [None, None]
        self._step = [None, None]
        self._zero = [None, None]
        self._wdata = None
        self._dtype = None
        self._driver = None
        self._backend = None
        (self._params,
         self._mappings) = _detail.make_gmodel_2d_params(
            self._components)

    def params(self):
        return self._params

    def is_weighted(self):
        return _detail.is_gmodel_weighted(self._components)

    def _prepare(self, driver, scube_w, size, step, zero, dtype):
        self._driver = driver
        self._size = size
        self._step = step
        self._zero = zero
        self._wdata = None
        self._dtype = dtype
        # If weighting is requested, store it in a 2d spatial array.
        if scube_w is not None:
            self._wdata = driver.mem_alloc_d(self._size[::-1], dtype)
        # Create backend
        self._backend = driver.backends().gmodel(dtype)

    def evaluate_scube(
            self, driver, params,
            scube_d, scube_w, size, step, zero, rota, dtype,
            out_extra):

        if (self._driver is not driver
                or self._size != size[:2]
                or self._dtype != dtype):
            self._prepare(driver, scube_w, size[:2], step[:2], zero[:2], dtype)

        spat_size = self._size
        spat_step = self._step
        spat_zero = self._zero
        spat_rota = rota
        spec_size = size[2]
        spec_step = step[2]
        spec_zero = zero[2]
        wdata = self._wdata
        components = self._components
        mappings = self._mappings
        backend = self._backend

        bdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(spat_size[::-1], dtype)
            driver.mem_fill(bdata, 0)

        # Evaluate components
        _detail.evaluate_components_s2d(
            components, driver, params, mappings,
            scube_d, scube_w, bdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra, '')

        # Evaluate the provided spectral weight cube using
        # the 2d spatial weight data evaluated above
        if scube_w is not None:
            backend.wcube_evaluate(
                tuple(spat_size) + (1,), spec_size, wdata, scube_w)

        if out_extra is not None:
            if wdata is not None:
                out_extra['total_wdata'] = driver.mem_copy_d2h(wdata)
            if bdata is not None:
                out_extra['total_bdata'] = driver.mem_copy_d2h(bdata)
