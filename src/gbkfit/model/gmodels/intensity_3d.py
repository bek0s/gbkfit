from collections.abc import Sequence

from gbkfit.model.base import GModelImage, Selection
from gbkfit.utils import parseutils
from ._component_set import (
    IMAGE_SPECTRAL_AXIS, ComponentSet3D, ComponentSetGModelPlan)
from .base import BrightnessComponent3D, OpacityComponent3D
from .brightness_mcdisk_3d import BrightnessMCDisk3D
from .brightness_point_3d import BrightnessPoint3D
from .brightness_smdisk_3d import BrightnessSMDisk3D
from .opacity_mcdisk_3d import OpacityMCDisk3D
from .opacity_smdisk_3d import OpacitySMDisk3D


__all__ = [
    'GModelIntensity3D'
]


_bcmp_parser = parseutils.TypedParser(BrightnessComponent3D, [
    BrightnessPoint3D,
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
        name = dict(name=self.name()) if self.name() is not None else {}
        return dict(
            type=self.type(),
            **name,
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
            zero_z: int | float | None = None,
            name: str | None = None
    ):
        super().__init__(name)
        self._component_set = ComponentSet3D(
            components, opacity_components, size_z, step_z, zero_z)

    def pdescs(self):
        return self._component_set.pdescs()

    def has_weights(self):
        return self._component_set.has_weights()

    def constants(self):
        return self._component_set.constants()

    def plan(self, driver, grid, has_weights, dtype, selection=Selection()):
        return ComponentSetGModelPlan(
            self._component_set.plan(
                driver, grid, IMAGE_SPECTRAL_AXIS, has_weights, dtype,
                selection),
            'image')
