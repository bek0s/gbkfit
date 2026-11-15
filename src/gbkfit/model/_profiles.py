"""
The radial profiles of the components of a galaxy, as its functions give
them (see GalaxyFunctions): of a disk, a cone or a spheroid, its rotation,
its speeds along each direction, its dispersion, its density and its
orientation along the radius, those it has (see ComponentProfiles).
"""

import abc
import math
import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from .components._profiles import (
    NodalTraited, trait_density, trait_dispersion, trait_speed)
from .components.base import Component, ComponentState
from .components.cones.cone import Cone
from .components.disks._profiles import (
    dispersion, rotation, speed, surface_density)
from .components.disks.disk import Disk
from .components.spheroids._jeans import JeansTables, midplane
from .components.spheroids._kinematics import TraitsMotion
from .components.spheroids.spheroid import (
    JeansSpheroid, Spheroid, SpheroidPlan)

if TYPE_CHECKING:
    from .components.disks._stage import DiskState


__all__ = [
    'PROFILE_FAMILIES',
    'ComponentProfiles',
    'checked_radii',
    'profile_families',
    'profile_radius_range',
    'profiles_of',
    'spheroid_families'
]


# The families of the profiles of a component (see ComponentProfiles):
# those of its motion, and the others
PROFILE_FAMILIES = (
    'rotation', 'speed', 'dispersion', 'density', 'orientation')
_MOTION_FAMILIES = ('rotation', 'speed', 'dispersion')

# The angles of the orientation of a component (see ComponentProfiles)
_ANGLES = ('posa', 'incl')

# The directions of the dispersion of a spheroid of Jeans kinematics, the
# rows of its tables at its midplane (see _jeans.midplane)
_JEANS_DISPERSIONS = dict(rad=1, tan=2, ver=3)


class ComponentProfiles(abc.ABC):
    """
    The radial profiles of a component, at the values of its parameters
    (by its own names) and with what its galaxy gives it (its state), at
    the radii R (arcsec, from its centre, in the frame of its traits):

    - rotation(R): its rotation at its midplane (km/s): its speed around
      its axis
    - speed(R, direction): its speed along a direction at its midplane
      (km/s; see traits.VPT_DIRECTIONS)
    - dispersion(R, direction=None): its velocity dispersion at its
      midplane (km/s), along a direction ('rad', 'tan' or 'ver') if it is
      not isotropic
    - density(R): its density seen face-on, in the unit of its density
    - orientation(R, which): its position angle or its inclination
      (which: 'posa' or 'incl'; degrees)

    Those of its families (see families) only.
    """

    def __init__(self, component: Component):
        self._component = component

    def families(self) -> tuple[str, ...]:
        """Return the families of its profiles (see PROFILE_FAMILIES)."""
        return profile_families(self._component)

    def rotation(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray
    ) -> np.ndarray:
        return self.speed(params, state, R, 'tan')

    def speed(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str
    ) -> np.ndarray:
        raise NotImplementedError

    def dispersion(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def density(self, params: ParamValues, R: np.ndarray) -> np.ndarray:
        pass

    def orientation(
            self, params: ParamValues, R: np.ndarray, which: str
    ) -> np.ndarray:
        """(of one value of each angle, its value at every radius)"""
        return np.full(np.shape(R), float(params[_checked_angle(which)]))


class _DiskProfiles(ComponentProfiles):
    """
    The profiles of a disk with velocity or dispersion traits, as its
    galaxy moves it (see disks._profiles: with the circular velocity of
    the mass, the pressure support and the vertical equilibrium that its
    state gives).
    """

    def __init__(self, disk: Disk, driver: Driver):
        super().__init__(disk)
        self._disk = disk.density_disk()
        self._plan = self._disk.plan(driver, 0, np.float64)

    def rotation(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray
    ) -> np.ndarray:
        return rotation(
            self._disk, self._plan, params, R,
            typing.cast('DiskState | None', state))

    def speed(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str
    ) -> np.ndarray:
        return speed(
            self._disk, self._plan, params, R, direction,
            typing.cast('DiskState | None', state))

    def dispersion(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str | None = None
    ) -> np.ndarray:
        state = typing.cast('DiskState | None', state)
        return dispersion(
            self._disk, self._plan, params, R,
            None if state is None else state.equilibrium, direction)

    def density(self, params: ParamValues, R: np.ndarray) -> np.ndarray:
        return surface_density(self._disk, self._plan, params, R)

    def orientation(
            self, params: ParamValues, R: np.ndarray, which: str
    ) -> np.ndarray:
        return self._disk.orientation(params, R, _checked_angle(which))


class _ConeProfiles(ComponentProfiles):
    """
    The profiles of a cone with velocity or dispersion traits, along the
    distance from its apex, at the mid-surface of its walls: those of its
    traits (see components._profiles); its density per unit area of the
    mid-surface.
    """

    def __init__(self, cone: Cone, driver: Driver):
        super().__init__(cone)
        self._cone = cone.nodal_cone()
        self._plan = self._cone.plan(driver, 0, np.float64)

    def speed(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str
    ) -> np.ndarray:
        return trait_speed(self._cone, self._plan, params, R, direction)

    def dispersion(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str | None = None
    ) -> np.ndarray:
        return trait_dispersion(
            self._cone, self._plan, params, R, direction)

    def density(self, params: ParamValues, R: np.ndarray) -> np.ndarray:
        return trait_density(self._cone, self._plan, params, R)


class _SpheroidProfiles(ComponentProfiles):
    """
    The profiles of a spheroid along the distance from its axis: its
    surface density seen face-on (see Spheroid.surface_density); and of
    its kinematics, those of their traits at its midplane (see
    components._profiles), or of its Jeans tables there (see
    _jeans.midplane: its stars move around its axis only), if it has
    either (the orbits of a spheroid give its views, not its profiles);
    those of its motion within its rmax only, where it has stars.
    """

    def __init__(self, spheroid: Spheroid, driver: Driver):
        super().__init__(spheroid)
        self._spheroid = spheroid
        motion = spheroid.motion()
        self._traits = motion if isinstance(motion, TraitsMotion) else None
        self._plan = SpheroidPlan(spheroid, driver, 0, np.float64) \
            if self._traits is not None else None
        self._jeans = isinstance(spheroid, JeansSpheroid)

    def speed(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str
    ) -> np.ndarray:
        if self._traits is not None:
            speeds = trait_speed(
                self._traits, self._plan, params, R, direction)
        elif direction == 'tan':
            speeds = midplane(_jeans_tables(state), R)[0]
        else:
            speeds = np.zeros(len(R))
        return self._where_lit(R, speeds)

    def dispersion(
            self, params: ParamValues, state: ComponentState | None,
            R: np.ndarray, direction: str | None = None
    ) -> np.ndarray:
        if self._traits is not None:
            return self._where_lit(R, trait_dispersion(
                self._traits, self._plan, params, R, direction))
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

    def _where_lit(self, R: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Return values of its motion at the radii R, NaN beyond its rmax,
        where it has no stars to move (see profile_radius_range).
        """
        return np.where(R <= self._spheroid.rmax(), values, np.nan)

    def density(self, params: ParamValues, R: np.ndarray) -> np.ndarray:
        # (seen face-on: neither inclined nor turned)
        face_on = dict(params) | dict(posa=0.0, incl=0.0)
        return self._spheroid.surface_density(face_on, np.zeros_like(R), R)


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


def _checked_angle(which: str) -> str:
    """
    Return an angle of the orientation of a component ('posa' or 'incl').

    Raises
    ------
    ValueError
        If it is not one.
    """
    if which not in _ANGLES:
        raise ValueError(f"which must be 'posa' or 'incl'; it is {which!r}")
    return which


def spheroid_families(spheroid: Spheroid) -> tuple[str, ...]:
    """
    Return the families of the profiles of a spheroid (see
    _SpheroidProfiles): its density; those of its motion if its
    kinematics are of traits (with velocity or dispersion traits) or of
    the Jeans equations; and its orientation if it has one (a sphere
    without kinematics has none).
    """
    motion = spheroid.motion()
    moves = isinstance(spheroid, JeansSpheroid) or (
        isinstance(motion, TraitsMotion)
        and bool(motion.traits('vpt') or motion.traits('dpt')))
    turns = all(angle in spheroid.pdescs() for angle in _ANGLES)
    return (*(_MOTION_FAMILIES if moves else ()), 'density',
            *(('orientation',) if turns else ()))


def profile_families(component: Component) -> tuple[str, ...]:
    """
    Return the families of the profiles of a component (see
    ComponentProfiles), without planning them: of a disk or a cone, its
    density and its orientation, and those of its motion if it has
    velocity or dispersion traits; of a spheroid, see spheroid_families;
    none of the others.
    """
    if isinstance(component, Spheroid):
        return spheroid_families(component)
    if isinstance(component, Disk):
        traited: NodalTraited = component.density_disk()
    elif isinstance(component, Cone):
        traited = component.nodal_cone()
    else:
        return ()
    moves = bool(traited.traits('vpt') or traited.traits('dpt'))
    return (*(_MOTION_FAMILIES if moves else ()), 'density', 'orientation')


def profile_radius_range(
        component: Component, family: str
) -> tuple[float, float]:
    """
    Return the range of the radii (arcsec) of the profiles of a family of
    a component (see ComponentProfiles): of the motion of a spheroid, to
    its rmax, beyond which it has no stars; of the others, from its centre
    on (a disk or a cone does not end at its last node).
    """
    if isinstance(component, Spheroid) and family in _MOTION_FAMILIES:
        return 0.0, component.rmax()
    return 0.0, math.inf


def profiles_of(
        component: Component, driver: Driver
) -> ComponentProfiles | None:
    """
    Return the profiles of a component (see profile_families), planned on
    the given driver in float64; None if it has none.
    """
    if not profile_families(component):
        return None
    if isinstance(component, Disk):
        return _DiskProfiles(component, driver)
    if isinstance(component, Cone):
        return _ConeProfiles(component, driver)
    return _SpheroidProfiles(typing.cast(Spheroid, component), driver)


def checked_radii(name: str, R: npt.ArrayLike) -> np.ndarray:
    """
    Return the radii R (arcsec) of the profile function of the given name,
    as an array of floats.

    Raises
    ------
    ValueError
        If a radius is negative.
    """
    radii = np.asarray(R, dtype=float)
    if np.any(radii < 0):
        raise ValueError(
            f"{name}: the radii must not be negative; they are {radii}")
    return radii
