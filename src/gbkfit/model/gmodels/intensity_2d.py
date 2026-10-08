
from collections.abc import Sequence

from gbkfit.model.core import GModelImage
from gbkfit.utils import parseutils
from ._gmodel import GModel2D
from .core import BrightnessComponent2D
from .brightness_smdisk_2d import BrightnessSMDisk2D


__all__ = [
    'GModelIntensity2D'
]


class GModelIntensity2D(GModel2D, GModelImage):

    _cmp_parser = parseutils.TypedParser(BrightnessComponent2D, [
        BrightnessSMDisk2D])

    @staticmethod
    def type():
        return 'intensity_2d'

    def __init__(
            self,
            components: BrightnessComponent2D | Sequence[BrightnessComponent2D]
    ):
        super().__init__(components)

    def evaluate_image(
            self, driver, params, image, weights, size, step, zero, rota,
            dtype, out_extra):
        # An image has a spectral axis of size 1
        self._evaluate(
            driver, params, dict(image=image), (1, 0, 0), weights,
            size, step, zero, rota, dtype, out_extra)
