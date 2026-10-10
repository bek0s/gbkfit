from collections.abc import Sequence

from gbkfit.model.base import GModelImage, Selection
from gbkfit.utils import parseutils
from ._component_set import (
    IMAGE_SPECTRAL_AXIS, ComponentSet2D, ComponentSetGModelPlan)
from .base import BrightnessComponent2D
from .brightness_point_2d import BrightnessPoint2D
from .brightness_smdisk_2d import BrightnessSMDisk2D


__all__ = [
    'GModelIntensity2D'
]


_bcmp_parser = parseutils.TypedParser(BrightnessComponent2D, [
    BrightnessPoint2D,
    BrightnessSMDisk2D])


class GModelIntensity2D(GModelImage):

    @staticmethod
    def type():
        return 'intensity_2d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        parseutils.load_option_and_update_info(
            _bcmp_parser, info, 'components', required=True)
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self):
        name = dict(name=self.name()) if self.name() is not None else {}
        return dict(
            type=self.type(),
            **name,
            components=_bcmp_parser.dump(self._component_set.components()))

    def __init__(
            self,
            components: BrightnessComponent2D | Sequence[BrightnessComponent2D],
            name: str | None = None
    ):
        super().__init__(name)
        self._component_set = ComponentSet2D(components)

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
