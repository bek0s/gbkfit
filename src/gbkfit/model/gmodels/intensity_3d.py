
from collections.abc import Sequence

from gbkfit.model.core import GModelImage
from gbkfit.utils import parseutils
from ._gmodel import ComponentGModel
from .core import BrightnessComponent3D, OpacityComponent3D
from .brightness_mcdisk_3d import BrightnessMCDisk3D
from .brightness_smdisk_3d import BrightnessSMDisk3D
from .opacity_mcdisk_3d import OpacityMCDisk3D
from .opacity_smdisk_3d import OpacitySMDisk3D


__all__ = [
    'GModelIntensity3D'
]


class GModelIntensity3D(ComponentGModel, GModelImage):

    _cmp_parser = parseutils.TypedParser(BrightnessComponent3D, [
        BrightnessMCDisk3D,
        BrightnessSMDisk3D])

    _ocmp_parser = parseutils.TypedParser(OpacityComponent3D, [
        OpacityMCDisk3D,
        OpacitySMDisk3D])

    @staticmethod
    def type():
        return 'intensity_3d'

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
        super().__init__(
            components, opacity_components, size_z, step_z, zero_z)

    def evaluate_image(
            self, driver, params, image, weights, size, step, zero, rota,
            dtype, out_extra):
        self._evaluate(
            driver, params, dict(image=image), weights,
            size, step, zero, rota, dtype, out_extra)
