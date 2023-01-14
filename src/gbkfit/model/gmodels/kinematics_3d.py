
from gbkfit.model.core import GModelSCube
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import OpacityComponent3D, SpectralComponent3D
from .opacity_mcdisk_3d import OpacityMCDisk3D
from .opacity_smdisk_3d import OpacitySMDisk3D
from .spectral_mcdisk_3d import SpectralMCDisk3D
from .spectral_smdisk_3d import SpectralSMDisk3D


__all__ = [
    'GModelKinematics3D'
]


_ocmp_parser = parseutils.TypedParser(OpacityComponent3D, [
    OpacityMCDisk3D,
    OpacitySMDisk3D])

_scmp_parser = parseutils.TypedParser(SpectralComponent3D, [
    SpectralMCDisk3D,
    SpectralSMDisk3D])


class GModelKinematics3D(GModelSCube):

    @staticmethod
    def type():
        return 'kinematics_3d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=_scmp_parser.load(info['components']),
            opacity_components=_ocmp_parser.load(
                info.get('opacity_components')))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            size_z=self._size[2],
            step_z=self._step[2],
            zero_z=self._zero[2],
            components=_scmp_parser.dump(self._components),
            opacity_components=_ocmp_parser.dump(self._ocomponents))

    def __init__(
            self, components, opacity_components=None,
            size_z=None, step_z=None, zero_z=None):
        if opacity_components is None:
            opacity_components = ()
        self._components = iterutils.tuplify(components)
        self._ocomponents = iterutils.tuplify(opacity_components)
        self._size = [None, None, size_z]
        self._step = [None, None, step_z]
        self._zero = [None, None, zero_z]
        self._wdata = None
        self._odata = None
        self._dtype = None
        self._driver = None
        self._backend = None
        (self._params,
         self._mappings,
         self._omappings) = _detail.make_gmodel_3d_params(
            self._components, self._ocomponents)

    def params(self):
        return self._params

    def is_weighted(self):
        return _detail.is_gmodel_weighted(self._components)

    def _prepare(self, driver, scube_w, size, step, zero, dtype):
        self._driver = driver
        self._size[:2] = size
        self._step[:2] = step
        self._zero[:2] = zero
        self._wdata = None
        self._odata = None
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
        # If weighting is requested, store it in a 3d spatial array.
        # Its dimensions were calculated above
        if scube_w is not None:
            self._wdata = driver.mem_alloc_d(self._size[::-1], dtype)
        # If opacity is enabled, store it in a 3d spatial array.
        # The dimensions of the cube were calculated above
        if self._ocomponents:
            self._odata = driver.mem_alloc_d(self._size[::-1], dtype)
        # Create backend
        self._backend = driver.backends().gmodel(dtype)

    def evaluate_scube(
            self, driver, params,
            scube_d, scube_w, size, step, zero, rota, dtype,
            out_extra):

        if (self._driver is not driver
                or self._size[:2] != size[:2]
                or self._step[:2] != step[:2]
                or self._zero[:2] != zero[:2]
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
        odata = self._odata
        components = self._components
        ocomponents = self._ocomponents
        mappings = self._mappings
        omappings = self._omappings
        backend = self._backend

        bdata = None
        obdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(spat_size[::-1], dtype)
            driver.mem_fill(bdata, 0)
            obdata = driver.mem_alloc_d(spat_size[::-1], dtype)
            driver.mem_fill(obdata, 0)

        # Evaluate opacity components
        if ocomponents:
            _detail.evaluate_components_o3d(
                ocomponents, driver, params, omappings,
                odata,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, out_extra, 'opacity_')

        # Evaluate normal components
        if components:
            _detail.evaluate_components_s3d(
                components, driver, params, mappings,
                odata,
                scube_d, wdata, bdata, obdata,
                spat_size, spat_step, spat_zero, spat_rota,
                spec_size, spec_step, spec_zero,
                dtype, out_extra, '')

        # Evaluate the provided spectral weight cube using
        # the spatial weight cube evaluated above
        if scube_w is not None:
            backend.wcube_evaluate(spat_size, spec_size, wdata, scube_w)

        if out_extra is not None:
            if wdata is not None:
                out_extra['wdata'] = driver.mem_copy_d2h(wdata)
            if odata is not None:
                out_extra['odata'] = driver.mem_copy_d2h(odata)
            if bdata is not None:
                out_extra['bdata'] = driver.mem_copy_d2h(bdata)
            if obdata is not None:
                out_extra['obdata'] = driver.mem_copy_d2h(obdata)
