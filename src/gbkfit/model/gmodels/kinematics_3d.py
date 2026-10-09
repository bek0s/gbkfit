from collections.abc import Sequence

from gbkfit.model.base import GModelSCube, Selection
from gbkfit.utils import parseutils
from ._component_set import ComponentSet3D, ComponentSetGModelPlan
from .base import OpacityComponent3D, SpectralComponent3D
from .mass import MassModel, mass_model_parser
from .opacity_mcdisk_3d import OpacityMCDisk3D
from .opacity_smdisk_3d import OpacitySMDisk3D
from .spectral_mcdisk_3d import SpectralMCDisk3D
from .spectral_point_3d import SpectralPoint3D
from .spectral_smdisk_3d import SpectralSMDisk3D


__all__ = [
    'GModelKinematics3D'
]


_scmp_parser = parseutils.TypedParser(SpectralComponent3D, [
    SpectralPoint3D,
    SpectralMCDisk3D,
    SpectralSMDisk3D])

_ocmp_parser = parseutils.TypedParser(OpacityComponent3D, [
    OpacityMCDisk3D,
    OpacitySMDisk3D])


class GModelKinematics3D(GModelSCube):

    @staticmethod
    def type():
        return 'kinematics_3d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        parseutils.load_option_and_update_info(
            _scmp_parser, info, 'components', required=True, allow_none=False)
        parseutils.load_option_and_update_info(
            _ocmp_parser, info, 'opacity_components')
        parseutils.load_option_and_update_info(
            mass_model_parser, info, 'mass_model')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
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
            size_z=component_set.size_z(),
            step_z=component_set.step_z(),
            zero_z=component_set.zero_z(),
            components=_scmp_parser.dump(component_set.components()),
            opacity_components=_ocmp_parser.dump(
                component_set.opacity_components()),
            **mass)

    def __init__(
            self,
            components:
            SpectralComponent3D | Sequence[SpectralComponent3D],
            opacity_components:
            OpacityComponent3D | Sequence[OpacityComponent3D] | None = None,
            size_z: int | None = None,
            step_z: int | float | None = None,
            zero_z: int | float | None = None,
            mass_model: MassModel | None = None,
            name: str | None = None
    ):
        """
        mass_model is the mass of the galaxy, whose circular velocity the
        'mass' velocity traits of the components take (see MassModel).
        """
        super().__init__(name)
        self._component_set = ComponentSet3D(
            components, opacity_components, size_z, step_z, zero_z,
            mass_model)

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
