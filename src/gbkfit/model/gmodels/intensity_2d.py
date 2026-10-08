from collections.abc import Sequence

from gbkfit.model.core import GModelImage
from gbkfit.utils import parseutils
from ._component_set import ComponentSet2D
from .core import BrightnessComponent2D
from .brightness_smdisk_2d import BrightnessSMDisk2D


__all__ = [
    'GModelIntensity2D'
]


_bcmp_parser = parseutils.TypedParser(BrightnessComponent2D, [
    BrightnessSMDisk2D])


class GModelIntensity2D(GModelImage):

    @staticmethod
    def type():
        return 'intensity_2d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        parseutils.load_option_and_update_info(
            _bcmp_parser, info, 'components', required=True, allow_none=False)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            components=_bcmp_parser.dump(self._component_set.components()))

    def __init__(
            self,
            components: BrightnessComponent2D | Sequence[BrightnessComponent2D]
    ):
        self._component_set = ComponentSet2D(components)

    def pdescs(self):
        return self._component_set.pdescs()

    def has_weights(self):
        return self._component_set.has_weights()

    def constants(self):
        return self._component_set.constants()

    def evaluate_image(
            self, driver, params, image, weights, size, step, zero, rota,
            dtype, out_extra):
        # An image has a spectral axis of size 1
        self._component_set.evaluate(
            driver, params, dict(image=image), (1, 0, 0), weights,
            size, step, zero, rota, dtype, out_extra)
