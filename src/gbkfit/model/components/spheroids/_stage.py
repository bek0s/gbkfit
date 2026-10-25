"""
The part of a spheroid in the stage of its galaxy (SpheroidStage).
"""

from typing import TYPE_CHECKING

from gbkfit.params import ParamValues
from ...extras import Extras
from ...mass import SpheroidDensity
from .._stage import GalaxyMass
from ..base import ComponentStage

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
    has a unit.
    """

    def __init__(self, spheroid: 'Spheroid', mass_per_unit: float | None):
        self._spheroid = spheroid
        self._mass_per_unit = mass_per_unit

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

    def given(
            self,
            values: ParamValues,
            mass: GalaxyMass | None,
            out_extra: Extras | None
    ) -> ParamValues:
        return {}
