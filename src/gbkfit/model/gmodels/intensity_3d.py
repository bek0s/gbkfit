
from gbkfit.model.core import GModelImage
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import DensityComponent3D
from .density_mcdisk_3d import DensityMCDisk3D
from .density_smdisk_3d import DensitySMDisk3D


__all__ = ['GModelIntensity3D']


_dcmp_parser = parseutils.TypedParser(DensityComponent3D, [
    DensityMCDisk3D, DensitySMDisk3D])


class GModelIntensity3D(GModelImage):

    @staticmethod
    def type():
        return 'intensity_3d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=_dcmp_parser.load(opts['components']),
            tcomponents=_dcmp_parser.load(opts.get('tcomponents')))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            size_z=self._size[2],
            step_z=self._step[2],
            zero_z=self._zero[2],
            tauto=self._tauto,
            components=_dcmp_parser.dump(self._components),
            tcomponents=_dcmp_parser.dump(self._tcomponents))

    def __init__(
            self, components, size_z=None, step_z=None, zero_z=None,
            tauto=False, tcomponents=None):
        if tcomponents is None:
            tcomponents = ()
        self._tauto = tauto
        self._components = iterutils.tuplify(components)
        self._tcomponents = iterutils.tuplify(tcomponents)
        self._size = [None, None, size_z]
        self._step = [None, None, step_z]
        self._zero = [None, None, zero_z]
        self._wcube = None
        self._tcube = None
        self._dtype = None
        self._driver = None
        self._backend = None
        (self._params,
         self._mappings,
         self._tmappings) = _detail.make_gmodel_3d_params(
            self._components, self._tcomponents, self._tauto)

    def params(self):
        return self._params

    def is_weighted(self):
        return True

    def _prepare(self, driver, wdata, size, step, zero, dtype):
        self._driver = driver
        self._size[:2] = size
        self._step[:2] = step
        self._zero[:2] = zero
        self._wcube = None
        self._tcube = None
        self._dtype = dtype
        # Calculate spatial size, step, zero for all three dimensions
        # For x and y we use the supplied x and y spatial dimensions.
        # For z things are not that easy. We need size, step, and zero
        # that can enclose the entire galaxy along the z axis. This is
        # hard to calculate. Instead, we pick the size and step of the
        # longest of the x and y axes, and we place zero in the middle.
        # Alternatively, the size, step, and zero of the z spatial axis
        # can be provided in the constructor and hence by the user.
        if self._size[2] is None:
            self._size[2] = size[int(size[0] < size[1])]
        if self._step[2] is None:
            self._step[2] = step[int(size[0] < size[1])]
        if self._zero[2] is None:
            self._zero[2] = -(self._size[2] / 2 - 0.5) * self._step[2]
        # If transparency is enabled, we need a 3d spatial cube to
        # store it. The dimensions of the cube were calculated above
        if self._tauto or self._tcomponents:
            self._tcube = driver.mem_alloc_d(self._size[::-1], dtype)
        # If weighting is requested, store it in a 3d spatial array.
        # Its dimensions were calculated above
        if wdata is not None:
            self._wcube = driver.mem_alloc_d(self._size[::-1], dtype)
        # Create backend
        self._backend = driver.backends().gmodel(dtype)

    def evaluate_image(
            self, driver, params, image, wdata, size, step, zero, rota, dtype,
            out_extra):

        if (self._driver is not driver
                or self._size[:2] != size
                or self._step[:2] != step
                or self._zero[:2] != zero
                or self._dtype != dtype):
            self._prepare(driver, wdata, size, step, zero, dtype)

        # Convenience variables
        spat_size = self._size
        spat_step = self._step
        spat_zero = self._zero
        spat_rota = rota
        tcube = self._tcube
        wcube = self._wcube
        tauto = self._tauto
        components = self._components
        tcomponents = self._tcomponents
        mappings = self._mappings
        tmappings = self._tmappings
        backend = self._backend

        # Evaluate transparency components
        if tcomponents:
            _detail.evaluate_components_d3d(
                tcomponents, driver, params, tmappings, None, None, tcube,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, out_extra, 'transparency_manual')

        # Evaluate the density of normal components as transparency
        if tauto:
            _detail.evaluate_components_d3d(
                components, driver, params, mappings, None, None, tcube,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, out_extra, 'transparency_auto')

        # Evaluate normal components
        if components:
            _detail.evaluate_components_d3d(
                components, driver, params, mappings, image, wcube, tcube,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, out_extra, '')

        # Evaluate the provided spectral weight cube using
        # the spatial weight cube evaluated above
        if wcube is not None:
            backend.make_wcube(spat_size, 1, wcube, wdata)

        # Save total transparency to extras
        if out_extra is not None:
            if tcube is not None:
                out_extra['transparency_total'] = \
                    driver.mem_copy_d2h(tcube).copy()
