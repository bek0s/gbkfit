"""
Spheroids (e.g. a bulge, or a dark matter halo): a profile (see profiles
and halos) of a shape (see shapes), a sphere or a flattened spheroid.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from .base import MassComponent, MassContext
from .halos import (
    ProfileBurkert, ProfileDekelZhao, ProfileEinasto, ProfileNFW,
    ProfilePseudoIsothermal, ProfileZhao)
from .profiles import ProfileSersic, SpheroidProfile
from .shapes import ShapeSphere, SpheroidShape, spheroid_shape_parser


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
    ProfileZhao,
    ProfileDekelZhao])


class MassSpheroid(MassComponent):
    """
    A spheroid of a profile and a shape: its density, circular velocity,
    forces and potential are those of the profile laid out in the shape
    (see shapes), and its parameters those of both.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    profile : SpheroidProfile
        Its profile.
    shape : SpheroidShape, optional
        Its shape; by default, a sphere.

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
        parseutils.load_option_and_update_info(
            spheroid_shape_parser, info, 'shape')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            profile=spheroid_profile_parser.dump(self._profile),
            shape=spheroid_shape_parser.dump(self._shape))

    def __init__(
            self,
            name: str,
            profile: SpheroidProfile,
            shape: SpheroidShape | None = None
    ):
        super().__init__(name)
        self._profile = profile
        self._shape = ShapeSphere() if shape is None else shape

    def profile(self) -> SpheroidProfile:
        """Return its profile."""
        return self._profile

    def shape(self) -> SpheroidShape:
        """Return its shape."""
        return self._shape

    def spherical(self) -> bool:
        return self._shape.spherical()

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._profile.pdescs() | self._shape.pdescs()

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._shape.vcirc2(R, self._profile, params, context)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._shape.density(R, z, self._profile, params, context)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._shape.forces(R, z, self._profile, params, context)

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._shape.potential(R, z, self._profile, params, context)
