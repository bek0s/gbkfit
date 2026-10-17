"""
Spherical spheroids (e.g. a bulge, or a dark matter halo), of a profile
(see profiles and halos).
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from .base import MassContext, MassSpherical
from .halos import (
    ProfileBurkert, ProfileEinasto, ProfileNFW, ProfilePseudoIsothermal,
    ProfileZhao)
from .profiles import ProfileSersic, SpheroidProfile


__all__ = [
    'MassSpheroid',
    'spheroid_profile_parser'
]


spheroid_profile_parser = parseutils.TypedParser(SpheroidProfile, [
    ProfileSersic,
    ProfileNFW,
    ProfileBurkert,
    ProfilePseudoIsothermal,
    ProfileEinasto,
    ProfileZhao])


class MassSpheroid(MassSpherical):
    """
    A spherical spheroid of a profile: its density, enclosed mass and
    potential are those of the profile, and its parameters too.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    profile : SpheroidProfile
        Its profile.

    Raises
    ------
    ConfigError
        If the name is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'spheroid'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'MassSpheroid':
        parseutils.load_option_and_update_info(
            spheroid_profile_parser, info, 'profile', required=True)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            profile=spheroid_profile_parser.dump(self._profile))

    def __init__(self, name: str, profile: SpheroidProfile):
        super().__init__(name)
        self._profile = profile

    def profile(self) -> SpheroidProfile:
        """Return its profile."""
        return self._profile

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._profile.pdescs()

    def density_spherical(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._profile.density(r, params, context)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._profile.enclosed_mass(r, params, context)

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._profile.potential(np.hypot(R, z), params, context)
