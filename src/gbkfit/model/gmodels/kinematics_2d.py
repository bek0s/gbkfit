from collections.abc import Sequence

from gbkfit.model.core import GModelSCube
from gbkfit.utils import parseutils
from ._component_set import ComponentSet2D
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
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        parseutils.load_option_and_update_info(
            _scmp_parser, info, 'components', required=True, allow_none=False)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            components=_scmp_parser.dump(self._component_set.components()))

    def __init__(
            self,
            components: SpectralComponent2D | Sequence[SpectralComponent2D]
    ):
        self._component_set = ComponentSet2D(components)

    def pdescs(self):
        return self._component_set.pdescs()

    def has_weights(self):
        return self._component_set.has_weights()

    def constants(self):
        return self._component_set.constants()

    def evaluate_scube(
            self, driver, params, scube, weights, size, step, zero, rota,
            dtype, out_extra):
        self._component_set.evaluate(
            driver, params, dict(scube=scube), (size[2], step[2], zero[2]),
            weights, size, step, zero, rota, dtype, out_extra)
