from collections.abc import Sequence

from gbkfit.model.core import GModelImage
from gbkfit.utils import parseutils
from ._component_set import ComponentSet3D, ComponentSetGModelPlan
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
            _bcmp_parser, info, 'components', required=True, allow_none=False)
        parseutils.load_option_and_update_info(
            _ocmp_parser, info, 'opacity_components')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        component_set = self._component_set
        return dict(
            type=self.type(),
            size_z=component_set.size_z(),
            step_z=component_set.step_z(),
            zero_z=component_set.zero_z(),
            components=_bcmp_parser.dump(component_set.components()),
            opacity_components=_ocmp_parser.dump(
                component_set.opacity_components()))

    def __init__(
            self,
            components:
            BrightnessComponent3D | Sequence[BrightnessComponent3D],
            opacity_components:
            OpacityComponent3D | Sequence[OpacityComponent3D] | None = None,
            size_z: int | None = None,
            step_z: int | float | None = None,
            zero_z: int | float | None = None
    ):
        self._component_set = ComponentSet3D(
            components, opacity_components, size_z, step_z, zero_z)

    def pdescs(self):
        return self._component_set.pdescs()

    def has_weights(self):
        return self._component_set.has_weights()

    def constants(self):
        return self._component_set.constants()

    def plan(self, driver, grid, has_weights, dtype):
        # An image has a spectral axis of size 1
        return ComponentSetGModelPlan(
            self._component_set.plan(
                driver, grid, (1, 0, 0), has_weights, dtype),
            'image')
