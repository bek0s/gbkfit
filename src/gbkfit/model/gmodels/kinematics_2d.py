from collections.abc import Sequence

from gbkfit.model.core import GModelSCube, Selection
from gbkfit.utils import parseutils
from ._component_set import ComponentSet2D, ComponentSetGModelPlan
from .core import SpectralComponent2D
from .spectral_point_2d import SpectralPoint2D
from .spectral_smdisk_2d import SpectralSMDisk2D


__all__ = [
    'GModelKinematics2D'
]


_scmp_parser = parseutils.TypedParser(SpectralComponent2D, [
    SpectralPoint2D,
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
        name = dict(name=self.name()) if self.name() is not None else {}
        return dict(
            type=self.type(),
            **name,
            components=_scmp_parser.dump(self._component_set.components()))

    def __init__(
            self,
            components: SpectralComponent2D | Sequence[SpectralComponent2D],
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
                driver, grid.spatial(), grid.spectral(), has_weights, dtype,
                selection),
            'scube')
