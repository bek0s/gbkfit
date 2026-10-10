from collections.abc import Sequence

from gbkfit.model.base import GModelSCube, Selection
from gbkfit.utils import parseutils
from ._component_set import ComponentSet2D, ComponentSetGModelPlan
from .base import SpectralComponent2D
from .mass import MassModel, mass_model_parser
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
        parseutils.load_option_and_update_info(
            _scmp_parser, info, 'components', required=True)
        parseutils.load_option_and_update_info(
            mass_model_parser, info, 'mass_model')
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self):
        component_set = self._component_set
        name = dict(name=self.name()) if self.name() is not None else {}
        mass_model = component_set.mass_model()
        mass = dict(mass_model=mass_model_parser.dump(mass_model)) \
            if mass_model is not None else {}
        return dict(
            type=self.type(),
            **name,
            components=_scmp_parser.dump(component_set.components()),
            **mass)

    def __init__(
            self,
            components: SpectralComponent2D | Sequence[SpectralComponent2D],
            mass_model: MassModel | None = None,
            name: str | None = None
    ):
        """
        mass_model is the mass of the galaxy, whose circular velocity the
        'mass' velocity traits of the components take (see MassModel).
        """
        super().__init__(name)
        self._component_set = ComponentSet2D(components, mass_model)

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
