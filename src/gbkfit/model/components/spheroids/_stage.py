"""
The part of a spheroid in the stage of its galaxy (SpheroidStage).
"""

from typing import TYPE_CHECKING

from gbkfit.params import ParamValues
from ...extras import Extras
from ...mass import SpheroidDensity
from .._stage import GalaxyMass
from ..base import ComponentStage, ComponentState

if TYPE_CHECKING:
    from .spheroid import Spheroid


__all__ = [
    'SpheroidStage'
]


class SpheroidStage(ComponentStage):
    """
    The part of a spheroid in the stage of its galaxy (see GalaxyStage):
    its density, which the mass of the galaxy follows (see
    mass.MassLightSpheroid), of the given mass surface density of one unit
    of its light (Msun/kpc^2; see _units.mass_per_unit), if its emission
    has a unit; and its state, that of the motion of its kinematics, if
    any (see SpheroidMotion.state; e.g. the tables of its Jeans
    kinematics), of its groundwork (see SpheroidMotion.groundwork). Its
    light reads the parameters of its profile and its shape (see
    Spheroid.light_params), its state those of its motion that it
    declares (see SpheroidMotion.state_params).
    """

    def __init__(self, spheroid: 'Spheroid', mass_per_unit: float | None):
        self._spheroid = spheroid
        self._mass_per_unit = mass_per_unit

    def light_params(self) -> tuple[str, ...]:
        return self._spheroid.light_params()

    def values(self, own: ParamValues) -> ParamValues:
        return own

    def density(
            self, values: ParamValues, kpc_per_arcsec: float
    ) -> SpheroidDensity:
        spheroid = self._spheroid
        return SpheroidDensity(
            spheroid.profile().mass_terms(
                values, spheroid.shape(), kpc_per_arcsec),
            self._mass_per_unit)

    def state_params(self) -> tuple[str, ...]:
        motion = self._spheroid.motion()
        return () if motion is None else motion.state_params()

    def groundwork(
            self, values: ParamValues, mass: GalaxyMass | None
    ) -> object:
        spheroid = self._spheroid
        motion = spheroid.motion()
        if motion is None:
            return None
        return motion.groundwork(spheroid.light(values), mass)

    def state(
            self,
            values: ParamValues,
            groundwork: object,
            mass: GalaxyMass | None,
            out_extra: Extras | None
    ) -> ComponentState | None:
        motion = self._spheroid.motion()
        if motion is None:
            return None
        return motion.state(values, groundwork)
