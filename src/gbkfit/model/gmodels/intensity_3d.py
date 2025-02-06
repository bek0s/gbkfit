
from collections.abc import Sequence

from gbkfit.model.core import GModelImage
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import BrightnessComponent3D, OpacityComponent3D
from .brightness_mcdisk_3d import BrightnessMCDisk3D
from .brightness_smdisk_3d import BrightnessSMDisk3D
from .opacity_mcdisk_3d import OpacityMCDisk3D
from .opacity_smdisk_3d import OpacitySMDisk3D


__all__ = [
    'GModelIntensity3D'
]


_bcmp_parser = parseutils.TypedParser(BrightnessComponent3D, [
    BrightnessMCDisk3D,
    BrightnessSMDisk3D])

_ocmp_parser = parseutils.TypedParser(OpacityComponent3D, [
    OpacityMCDisk3D,
    OpacitySMDisk3D])


class GModelIntensity3D(GModelImage):

    @staticmethod
    def type():
        return 'intensity_3d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        parseutils.load_option_and_update_info(
            _bcmp_parser, info, 'components', required=True)
        parseutils.load_option_and_update_info(
            _ocmp_parser, info, 'opacity_components', required=False)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            size_z=self._size[2],
            step_z=self._step[2],
            zero_z=self._zero[2],
            components=_bcmp_parser.dump(self._components),
            opacity_components=_ocmp_parser.dump(self._ocomponents))

    def __init__(
            self,
            components:
            BrightnessComponent3D | Sequence[BrightnessComponent3D],
            opacity_components:
            OpacityComponent3D | Sequence[OpacityComponent3D] | None = None,
            size_z: int = None,
            step_z: int | float | None = None,
            zero_z: int | float | None = None
    ):
        if not components:
            raise RuntimeError("at least one component must be configured")
        if not opacity_components:
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

    def _prepare(self, driver, image_w, size, step, zero, dtype):
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
        # can be provided in the constructor.
        if self._size[2] is None:
            self._size[2] = size[int(size[0] < size[1])]
        if self._step[2] is None:
            self._step[2] = step[int(size[0] < size[1])]
        if self._zero[2] is None:
            self._zero[2] = -(self._size[2] / 2 - 0.5) * self._step[2]
        # If weighting is requested, store it in a 3d spatial array.
        # Its dimensions were calculated above
        if image_w is not None:
            self._wdata = driver.mem_alloc_d(self._size[::-1], dtype)
        # If opacity is enabled, store it in a 3d spatial array.
        # The dimensions of the cube were calculated above
        if self._ocomponents:
            self._odata = driver.mem_alloc_d(self._size[::-1], dtype)
        # Create backend
        self._backend = driver.backends().gmodel(dtype)

    def evaluate_image(
            self, driver, params,
            image_d, image_w, size, step, zero, rota, dtype,
            out_extra):

        if (self._driver is not driver
                or self._size[:2] != size
                or self._step[:2] != step
                or self._zero[:2] != zero
                or self._dtype != dtype):
            self._prepare(driver, image_w, size, step, zero, dtype)

        spat_size = self._size
        spat_step = self._step
        spat_zero = self._zero
        spat_rota = rota
        spec_size = 1
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
            _detail.evaluate_components_b3d(
                components, driver, params, mappings,
                odata,
                image_d, wdata, bdata, obdata,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, out_extra, '')

        # Evaluate the provided weight image using
        # the 3d spatial weight data evaluated above
        if image_w is not None:
            backend.wcube_evaluate(spat_size, spec_size, wdata, image_w)

        if out_extra is not None:
            if wdata is not None:
                out_extra['total_wdata'] = driver.mem_copy_d2h(wdata)
            if odata is not None:
                out_extra['total_odata'] = driver.mem_copy_d2h(odata)
            if bdata is not None:
                out_extra['total_bdata'] = driver.mem_copy_d2h(bdata)
            if obdata is not None:
                out_extra['total_obdata'] = driver.mem_copy_d2h(obdata)
