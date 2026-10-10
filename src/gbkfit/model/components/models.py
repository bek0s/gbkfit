from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils
from ..base import ModelImage, ModelSCube, Selection
from ..mass import MassModel, mass_model_parser
from . import _detail
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
from .geometries import geometry_parser
from .points import (
    BrightnessPoint2D, BrightnessPoint3D, SpectralPoint2D, SpectralPoint3D)


__all__ = [
    'ModelIntensity2D',
    'ModelIntensity3D',
    'ModelKinematics2D',
    'ModelKinematics3D'
]


def _dump_geometries(
        component_set: ComponentSet2D | ComponentSet3D
) -> dict[str, Any]:
    """
    Return the geometries option of a model: none if its components have
    none.
    """
    geometries = component_set.geometries()
    return dict(geometries=geometry_parser.dump(geometries)) \
        if geometries else {}


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
    """
    A thin galaxy of brightness: its components evaluated into images.

    Parameters
    ----------
    components : BrightnessComponent2D or Sequence of them
        Its components: smooth disks (smdisk) and points (point).
    name : str, optional
        Its name (see Model).

    Raises
    ------
    ConfigError
        If there are no components, or the names of the components,
        their geometries and their parameters conflict.
    """

    @staticmethod
    def type() -> str:
        return 'intensity_2d'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        _detail.load_geometries(info, ('components',))
        parseutils.load_option_and_update_info(
            _bcmp2d_parser, info, 'components', required=True)
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        name = dict(name=self.name()) if self.name() is not None else {}
        return dict(
            type=self.type(),
            **name,
            **_dump_geometries(self._component_set),
            components=_bcmp2d_parser.dump(self._component_set.components()))

    def __init__(
            self,
            components:
            BrightnessComponent2D | Sequence[BrightnessComponent2D],
            name: str | None = None
    ):
        super().__init__(name)
        self._component_set = ComponentSet2D(components)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._component_set.pdescs()

    def has_weights(self) -> bool:
        return self._component_set.has_weights()

    def constants(self) -> dict[str, Any]:
        return self._component_set.constants()

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection = Selection()
    ) -> ComponentSetModelPlan:
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid, IMAGE_SPECTRAL_AXIS, has_weights, dtype,
                selection),
            'image')


class ModelIntensity3D(ModelImage):
    """
    A thick galaxy of brightness: its components evaluated in 3D and
    projected into images, through its opacity.

    Parameters
    ----------
    components : BrightnessComponent3D or Sequence of them
        Its components: smooth and cloud disks (smdisk, mcdisk) and
        points (point).
    opacity_components : OpacityComponent3D or Sequence of them, optional
        Its opacity components: smooth and cloud disks of dust, which
        absorb the light of its disks (not that of its points).
    size_z, step_z, zero_z : optional
        The axis along the line of sight (z): its number of voxels, their
        size and the position of the first; by default, those of the
        longer of the x and y axes, centred on 0.
    name : str, optional
        Its name (see Model).

    Raises
    ------
    ConfigError
        If there are no components, or the names of the components,
        their geometries and their parameters conflict.
    """

    @staticmethod
    def type() -> str:
        return 'intensity_3d'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        _detail.load_geometries(
            info, ('components', 'opacity_components'))
        parseutils.load_option_and_update_info(
            _bcmp3d_parser, info, 'components', required=True)
        parseutils.load_option_and_update_info(
            _ocmp_parser, info, 'opacity_components')
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        component_set = self._component_set
        name = dict(name=self.name()) if self.name() is not None else {}
        return dict(
            type=self.type(),
            **name,
            **_dump_geometries(self._component_set),
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

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._component_set.pdescs()

    def has_weights(self) -> bool:
        return self._component_set.has_weights()

    def constants(self) -> dict[str, Any]:
        return self._component_set.constants()

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection = Selection()
    ) -> ComponentSetModelPlan:
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid, IMAGE_SPECTRAL_AXIS, has_weights, dtype,
                selection),
            'image')


class ModelKinematics2D(ModelSCube):
    """
    A thin galaxy of emission lines: its components evaluated into
    spectral cubes.

    Parameters
    ----------
    components : SpectralComponent2D or Sequence of them
        Its components: smooth disks (smdisk) and points (point).
    mass_model : MassModel, optional
        The mass of the galaxy, whose circular velocity the 'mass'
        velocity traits of its components take.
    name : str, optional
        Its name (see Model).

    Raises
    ------
    ConfigError
        If there are no components, or the names of the components,
        their geometries and their parameters conflict.
    """

    @staticmethod
    def type() -> str:
        return 'kinematics_2d'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        _detail.load_geometries(info, ('components',))
        parseutils.load_option_and_update_info(
            _scmp2d_parser, info, 'components', required=True)
        parseutils.load_option_and_update_info(
            mass_model_parser, info, 'mass_model')
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        component_set = self._component_set
        name = dict(name=self.name()) if self.name() is not None else {}
        mass_model = component_set.mass_model()
        mass = dict(mass_model=mass_model_parser.dump(mass_model)) \
            if mass_model is not None else {}
        return dict(
            type=self.type(),
            **name,
            **_dump_geometries(self._component_set),
            components=_scmp2d_parser.dump(component_set.components()),
            **mass)

    def __init__(
            self,
            components: SpectralComponent2D | Sequence[SpectralComponent2D],
            mass_model: MassModel | None = None,
            name: str | None = None
    ):
        super().__init__(name)
        self._component_set = ComponentSet2D(components, mass_model)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._component_set.pdescs()

    def has_weights(self) -> bool:
        return self._component_set.has_weights()

    def constants(self) -> dict[str, Any]:
        return self._component_set.constants()

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection = Selection()
    ) -> ComponentSetModelPlan:
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid.spatial(), grid.spectral(), has_weights, dtype,
                selection),
            'scube')


class ModelKinematics3D(ModelSCube):
    """
    A thick galaxy of emission lines: its components evaluated in 3D
    and projected into spectral cubes, through its opacity.

    Parameters
    ----------
    components : SpectralComponent3D or Sequence of them
        Its components: smooth and cloud disks (smdisk, mcdisk) and
        points (point).
    opacity_components : OpacityComponent3D or Sequence of them, optional
        Its opacity components: smooth and cloud disks of dust, which
        absorb the light of its disks (not that of its points).
    size_z, step_z, zero_z : optional
        The axis along the line of sight (z): its number of voxels, their
        size and the position of the first; by default, those of the
        longer of the x and y axes, centred on 0.
    mass_model : MassModel, optional
        The mass of the galaxy, whose circular velocity the 'mass'
        velocity traits of its components take.
    name : str, optional
        Its name (see Model).

    Raises
    ------
    ConfigError
        If there are no components, or the names of the components,
        their geometries and their parameters conflict.
    """

    @staticmethod
    def type() -> str:
        return 'kinematics_3d'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        _detail.load_geometries(
            info, ('components', 'opacity_components'))
        parseutils.load_option_and_update_info(
            _scmp3d_parser, info, 'components', required=True)
        parseutils.load_option_and_update_info(
            _ocmp_parser, info, 'opacity_components')
        parseutils.load_option_and_update_info(
            mass_model_parser, info, 'mass_model')
        opts = parseutils.parse_options_for_callable(info, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        component_set = self._component_set
        name = dict(name=self.name()) if self.name() is not None else {}
        mass_model = component_set.mass_model()
        mass = dict(mass_model=mass_model_parser.dump(mass_model)) \
            if mass_model is not None else {}
        return dict(
            type=self.type(),
            **name,
            **_dump_geometries(self._component_set),
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
        super().__init__(name)
        self._component_set = ComponentSet3D(
            components, opacity_components, size_z, step_z, zero_z,
            mass_model)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._component_set.pdescs()

    def has_weights(self) -> bool:
        return self._component_set.has_weights()

    def constants(self) -> dict[str, Any]:
        return self._component_set.constants()

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection = Selection()
    ) -> ComponentSetModelPlan:
        return ComponentSetModelPlan(
            self._component_set.plan(
                driver, grid.spatial(), grid.spectral(), has_weights, dtype,
                selection),
            'scube')
