"""
The radial profiles of a spheroid (see ComponentProfiles), along the
distance from its axis: its surface density seen face-on; and of its
motion, if it gives them, those of its traits at its midplane (see
TraitSpheroid) or of its Jeans tables there (see JeansSpheroid), within
its rmax only, where it has stars. The orbits of a spheroid give its
views, not its profiles.
"""

import math
from typing import TYPE_CHECKING

import numpy as np

from gbkfit.params import ParamValues
from .._profiles import (
    ANGLES, MOTION_FAMILIES, ComponentProfiles, TraitsPlan, trait_dispersion,
    trait_speed)
from ..base import ComponentState
from ._jeans import JeansTables, midplane
from ._motion import TraitsMotion

if TYPE_CHECKING:
    from .spheroid import Spheroid


__all__ = [
    'JeansSpheroidProfiles',
    'SpheroidProfiles',
    'TraitSpheroidProfiles',
    'spheroid_families'
]


# The directions of the dispersion of a spheroid of Jeans kinematics, the
# rows of its tables at its midplane (see _jeans.midplane)
_JEANS_DISPERSIONS = dict(rad=1, tan=2, ver=3)


def spheroid_families(
        spheroid: 'Spheroid', moves: bool
) -> dict[str, tuple[float, float]]:
    """
    Return the families of the profiles of a spheroid (see
    Component.profile_families), whether it gives those of its motion or
    not: those of its motion, to its rmax, beyond which it has no stars;
    its density; and its orientation if it has one (a sphere without
    kinematics has none).
    """
    families = dict.fromkeys(MOTION_FAMILIES, (0.0, spheroid.rmax())) \
        if moves else {}
    families['density'] = (0.0, math.inf)
    if all(angle in spheroid.pdescs() for angle in ANGLES):
        families['orientation'] = (0.0, math.inf)
    return families


class SpheroidProfiles(ComponentProfiles):
    """
    The profiles of a spheroid without those of its motion: its surface
    density seen face-on (see Spheroid.surface_density) and its
    orientation.
    """

    def __init__(self, spheroid: 'Spheroid'):
        self._spheroid = spheroid

    def density(self, params: ParamValues, R: np.ndarray) -> np.ndarray:
        # (seen face-on: neither inclined nor turned)
        face_on = dict(params) | dict(posa=0.0, incl=0.0)
        return self._spheroid.surface_density(face_on, np.zeros_like(R), R)

    def _where_lit(self, R: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Return values of its motion at the radii R, NaN beyond its rmax,
        where it has no stars to move (see spheroid_families).
        """
        return np.where(R <= self._spheroid.rmax(), values, np.nan)


class TraitSpheroidProfiles(SpheroidProfiles):
    """
    The profiles of a spheroid whose stars move as traits give (see
    TraitSpheroid): those of its traits at its midplane, from their plan.
    """

    def __init__(
            self, spheroid: 'Spheroid', motion: TraitsMotion,
            plan: TraitsPlan
    ):
        super().__init__(spheroid)
        self._motion = motion
        self._plan = plan

    def speed(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str
    ) -> np.ndarray:
        return self._where_lit(R, trait_speed(
            self._motion, self._plan, params, R, direction))

    def dispersion(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str | None = None
    ) -> np.ndarray:
        return self._where_lit(R, trait_dispersion(
            self._motion, self._plan, params, R, direction))


class JeansSpheroidProfiles(SpheroidProfiles):
    """
    The profiles of a spheroid of Jeans kinematics (see JeansSpheroid):
    those of its Jeans tables at its midplane, its state (see
    _jeans.midplane: its stars move around its axis only).
    """

    def speed(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str
    ) -> np.ndarray:
        speeds = midplane(_jeans_tables(state), R)[0] if direction == 'tan' \
            else np.zeros(len(R))
        return self._where_lit(R, speeds)

    def dispersion(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str | None = None
    ) -> np.ndarray:
        if direction is None:
            raise ValueError(
                "the dispersion of a spheroid of Jeans kinematics is not "
                "isotropic: give its direction, 'rad', 'tan' or 'ver'")
        if direction not in _JEANS_DISPERSIONS:
            raise ValueError(
                f"direction must be 'rad', 'tan' or 'ver'; it is "
                f"{direction!r}")
        return self._where_lit(R, midplane(_jeans_tables(state), R)[
            _JEANS_DISPERSIONS[direction]])


def _jeans_tables(state: ComponentState | None) -> JeansTables:
    """
    Return the Jeans tables of a spheroid, its state.

    Raises
    ------
    ValueError
        If the state is not its tables.
    """
    if not isinstance(state, JeansTables):
        raise ValueError(
            "the profiles of the motion of a spheroid of Jeans kinematics "
            "need its tables (see JeansTables)")
    return state
