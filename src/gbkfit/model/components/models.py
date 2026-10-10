from collections.abc import Sequence

from gbkfit.utils import parseutils
from ..base import ModelImage, ModelSCube, Selection
from ..mass import MassModel, mass_model_parser
from ._component_set import (
    IMAGE_SPECTRAL_AXIS, ComponentSet2D, ComponentSet3D,
    ComponentSetModelPlan)
from .base import (
    BrightnessComponent2D, BrightnessComponent3D, OpacityComponent3D,
    SpectralComponent2D, SpectralComponent3D)
from .disks import (
    BrightnessMCDisk3D, BrightnessSMDisk2D, BrightnessSMDisk3D,
    OpacityMCDisk3D, OpacitySMDisk3D, SpectralMCDisk3D, SpectralSMDisk2D,
    SpectralSMDisk3D)
from .points import (
    BrightnessPoint2D, BrightnessPoint3D, SpectralPoint2D, SpectralPoint3D)


__all__ = [
    'ModelIntensity2D',
    'ModelIntensity3D',
    'ModelKinematics2D',
    'ModelKinematics3D'
]


# The components that each model accepts
_bcmp2d_parser = parseutils.TypedParser(BrightnessComponent2D, [
    BrightnessPoint2D,
    BrightnessSMDisk2D])

_bcmp3d_parser = parseutils.TypedParser(BrightnessComponent3D, [
    BrightnessPoint3D,
    BrightnessMCDisk3D,
    BrightnessSMDisk3D])

_scmp2d_parser = parseutils.TypedParser(SpectralComponent2D, [
    SpectralPoint2D,
    SpectralSMDisk2D])

_scmp3d_parser = parseutils.TypedParser(SpectralComponent3D, [
    SpectralPoint3D,
    SpectralMCDisk3D,
    SpectralSMDisk3D])

_ocmp_parser = parseutils.TypedParser(OpacityComponent3D, [
    OpacityMCDisk3D,
    OpacitySMDisk3D])


class ModelIntensity2D(ModelImage):

    @staticmethod
    def type():
        return 'intensity_2d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        parseutils.load_option_and_update_info(
            _bcmp2d_parser, info, 'components', required=True)
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self):
        name = dict(name=self.name()) if self.name() is not None else {}
        return dict(
            type=self.type(),
            **name,
            components=_bcmp2d_parser.dump(self._component_set.components()))

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
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid, IMAGE_SPECTRAL_AXIS, has_weights, dtype,
                selection),
            'image')


class ModelIntensity3D(ModelImage):

    @staticmethod
    def type():
        return 'intensity_3d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        parseutils.load_option_and_update_info(
            _bcmp3d_parser, info, 'components', required=True)
        parseutils.load_option_and_update_info(
            _ocmp_parser, info, 'opacity_components')
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
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
            components=_bcmp3d_parser.dump(component_set.components()),
            opacity_components=_ocmp_parser.dump(
                component_set.opacity_components()))

    def __init__(
            self,
            components:
            BrightnessComponent3D | Sequence[BrightnessComponent3D],
            opacity_components:
            OpacityComponent3D | Sequence[OpacityComponent3D] | None = None,
            size_z: int | None = None,
            step_z: float | None = None,
            zero_z: float | None = None,
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
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid, IMAGE_SPECTRAL_AXIS, has_weights, dtype,
                selection),
            'image')


class ModelKinematics2D(ModelSCube):

    @staticmethod
    def type():
        return 'kinematics_2d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        parseutils.load_option_and_update_info(
            _scmp2d_parser, info, 'components', required=True)
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
            components=_scmp2d_parser.dump(component_set.components()),
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
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid.spatial(), grid.spectral(), has_weights, dtype,
                selection),
            'scube')


class ModelKinematics3D(ModelSCube):

    @staticmethod
    def type():
        return 'kinematics_3d'

    @classmethod
    def load(cls, info, *args, **kwargs):
        parseutils.load_option_and_update_info(
            _scmp3d_parser, info, 'components', required=True)
        parseutils.load_option_and_update_info(
            _ocmp_parser, info, 'opacity_components')
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
            size_z=component_set.size_z(),
            step_z=component_set.step_z(),
            zero_z=component_set.zero_z(),
            components=_scmp3d_parser.dump(component_set.components()),
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
            step_z: float | None = None,
            zero_z: float | None = None,
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
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid.spatial(), grid.spectral(), has_weights, dtype,
                selection),
            'scube')
