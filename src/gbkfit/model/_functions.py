"""
The functions of a galaxy (GalaxyFunctions) that the derived values of
its parameters can use (see ParamSpace): the circular velocity, the
enclosed mass and the density slope of its mass, and the rotation, the
speeds, the dispersion and the density of its components, at any
radius.
"""

import functools
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
import numpy.typing as npt

from gbkfit.cosmology import Cosmology
from gbkfit.driver import Driver
from gbkfit.grid import doppler
from gbkfit.params import ParamValues
from ._galaxy_stage import GalaxyStage, GalaxyState
from ._profiles import (
    ComponentProfiles, checked_radii, profile_families, profile_radius_range,
    profiles_of)
from .components.base import GalaxyMass
from .components.base import ComponentState

if TYPE_CHECKING:
    from .galaxy import Galaxy


__all__ = [
    'GalaxyFunctions'
]


# The step in ln r of the differences of the density slopes (see
# _density_slope): their error, about 1e-6, and that of masses of
# quadratures (to about 1e-10) amplified by 1 / step^2
_SLOPE_STEP = 1e-3

# The functions of radii by family (see GalaxyFunctions.family): the units
# of their values and of their radii, and whether the family has one for
# each mass component or disk ('vcirc_halo')
_RADIAL_FAMILIES = dict(
    vcirc=(u.km / u.s, u.kpc, True),
    vnewton=(u.km / u.s, u.kpc, False),
    mass=(u.Msun, u.kpc, True),
    slope=(u.dimensionless_unscaled, u.kpc, True),
    rotation=(u.km / u.s, u.arcsec, True),
    speed=(u.km / u.s, u.arcsec, True),
    dispersion=(u.km / u.s, u.arcsec, True),
    density=(u.dimensionless_unscaled, u.arcsec, True),
    orientation=(u.deg, u.arcsec, True))

# The families of functions of the mass of the galaxy
_MASS_FAMILIES = ('vcirc', 'vnewton', 'mass', 'slope')


class GalaxyFunctions:
    """
    The functions of a galaxy that the derived values of its parameters
    can use (see ParamSpace), by name, at the values of its parameters
    (see bind). With mass:

    - vcirc(R): the circular velocity of its mass (km/s) at the radii R
      (kpc, in the plane of its disks), in its gravity (see Gravity),
      signed as in the rotation tables of its disks: negative where its
      square is (see DiskStage.state)
    - vnewton(R): that of its mass in Newtonian gravity (vcirc(R), with
      Newtonian gravity), signed
    - vcirc_{name}(R): that of its mass component of the name in
      Newtonian gravity, signed (negative where it pulls outwards)
    - mass(r): its mass (Msun) within the spheres of the radii r (kpc)
      about its centre, in any gravity (see MassModel.enclosed_mass), if
      every mass component gives it
    - mass_{name}(r): that of its mass component of the name, if it gives
      it (a user component without a function of it does not; see
      MassUser)
    - slope(r): the logarithmic slope d ln rho / d ln r of the spherically
      averaged density of its mass, rho(r) = (dM / dr) / (4 pi r^2), at
      the radii r > 0 (kpc), from its enclosed mass M (see mass): of a
      spherical mass (e.g. a halo), that of its density; NaN where it has
      none (e.g. about a point mass alone)
    - slope_{name}(r): that of its mass component of the name, if it
      gives its enclosed mass
    - kpc_per_arcsec: the length of an arcsec in it (kpc), if it has a
      distance (with or without mass)
    - redshift_at(v): its redshift at the systemic velocities v (km/s;
      e.g. that of a component, vsys), (1 + z) D(v) - 1, of its redshift
      z and the relativistic Doppler factor D (see LinePlacement)

    For each of its components with profiles (see ComponentProfiles),
    named after it ('rotation_gas'; 'rotation' for one without a name), at
    the radii R (arcsec, from its centre on: of a disk, on the disk; of a
    cone, from its apex, on the mid-surface of its walls; of a spheroid,
    from its axis, at its midplane):

    - rotation_{component}(R): its rotation (km/s); of a disk, as its
      gravity moves it: the azimuthal mean of its tangential velocity
      traits (with the circular velocity of the mass, and its pressure
      support, that its gravity gives them; see rotation)
    - speed_{component}(R, direction): its speed along a direction (km/s):
      the azimuthal mean of its velocity traits along it ('rad', 'ver',
      'away', 'out' or 'los', see traits); along 'tan', its rotation (see
      speed)
    - dispersion_{component}(R): its velocity dispersion (km/s): the
      azimuthal mean of its dispersion traits (of a disk, with the
      dispersion of its vertical equilibrium in quadrature; see
      dispersion); dispersion_{component}(R, direction), along a
      direction, 'rad', 'tan' or 'ver' (see traits), as one whose
      dispersion is not isotropic needs
    - density_{component}(R): its density, in the unit of its density
      (e.g. its light): of a disk, its surface density seen face-on (the
      azimuthal mean of its density traits where its selection traits
      keep it; see surface_density); of a cone, the same per unit area of
      the mid-surface of its walls; of a spheroid, its surface density
      seen face-on
    - orientation_{component}(R, which): its position angle or its
      inclination (which: 'posa' or 'incl'; degrees): of a disk that
      tilts, as it is evaluated (see NodalDisk.orientation); else its one
      value

    A disk or a cone has its density and its orientation, and those of
    its motion if it has velocity or dispersion traits (of them). A
    spheroid has its density, its orientation unless it is a sphere
    without kinematics, and those of its motion if its kinematics are of
    traits (of them) or of the Jeans equations (of its tables: it
    rotates, and has no other speed; see JeansKinematics); its orbits
    give its views, not its profiles.

    The functions take numbers or arrays of radii, and return arrays of
    their shape. They are of the mass of the galaxy and of what it gives
    its components (see GalaxyState): of a state that they are given, or
    of their own stage of the galaxy (see bind). They have their own stage
    and plans of the components, on the given driver and in float64, apart
    from those of the evaluations of the galaxy (which may run at the same
    time, e.g. in workers of threads; see WorkersThreads).

    Parameters
    ----------
    galaxy : Galaxy
        The galaxy.
    driver : Driver
        The driver where the components sample their profiles.
    sky_origin : tuple of float, optional
        The RA and Dec (degrees) of the origin of the frame of the
        parameters of the galaxy, which its image traits need (see
        GalaxyStage).
    cosmology : Cosmology, optional
        The cosmology of the galaxy (see GalaxyStage).
    """

    def __init__(
            self,
            galaxy: 'Galaxy',
            driver: Driver,
            sky_origin: tuple[float, float] | None = None,
            cosmology: Cosmology | None = None
    ):
        components = galaxy.components()
        self._galaxy = galaxy
        self._redshift = galaxy.redshift()
        self._stage = GalaxyStage(galaxy, driver, sky_origin, cosmology)
        self._suffixes = _profile_suffixes(galaxy)
        # (the components of the suffixes have profiles)
        self._profiles = {
            i: typing.cast(
                ComponentProfiles, profiles_of(components[i], driver))
            for i in self._suffixes}
        self._names = function_names(galaxy)

    @staticmethod
    def family(name: str) -> str | None:
        """
        Return the family of the function of radii of a name (see the
        class): the name without the name of its mass component or disk
        (e.g. 'vcirc' of 'vcirc_halo'), or None if no function of radii
        has the name (e.g. 'kpc_per_arcsec').
        """
        family, _, own = name.partition('_')
        if family in _RADIAL_FAMILIES \
                and (not own or _RADIAL_FAMILIES[family][2]):
            return family
        return None

    @staticmethod
    def of_mass(name: str) -> bool:
        """
        Check whether the function of radii of a name is of the mass of
        the galaxy (its circular velocity, enclosed mass or density slope,
        not the rotation or the dispersion of a disk).
        """
        return GalaxyFunctions.family(name) in _MASS_FAMILIES

    @staticmethod
    def radial_units(
            name: str
    ) -> tuple[u.UnitBase, u.UnitBase] | None:
        """
        Return the units of the values and of the radii of the function of
        radii of a name (see family; e.g. km/s and kpc of vcirc), or None
        if no function of radii has the name.
        """
        family = GalaxyFunctions.family(name)
        return None if family is None else _RADIAL_FAMILIES[family][:2]

    def names(self) -> tuple[str, ...]:
        """Return the names of its functions (see the class)."""
        return self._names


    def bind(
            self,
            params: ParamValues,
            state: GalaxyState | None = None
    ) -> dict[str, Any]:
        """
        Return its functions (see the class) at the values of the
        parameters of the galaxy (see Galaxy.pdescs), by name.

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the galaxy.
        state : GalaxyState, optional
            The galaxy at the values, if evaluated (e.g. by its observation
            group): the functions take its mass and what it gives the
            components. By default, its own stage evaluates the mass here,
            and what the galaxy gives a disk once, when a function of the
            disk first needs it (see GalaxyStage.state).

        Returns
        -------
        dict
            The functions, by name.
        """
        if state is None:
            mass = self._stage.mass(params)
            state_of = functools.cache(functools.partial(
                self._stage.state, params=params, mass=mass))
        else:
            mass = state.mass

            def state_of(index: int) -> ComponentState | None:
                return state.states.get(index)
        functions: dict[str, Any] = {}
        if mass is not None:
            functions['vcirc'] = functools.partial(
                _vcirc, 'vcirc', mass.field.vcirc2)
            functions['vnewton'] = functools.partial(
                _vcirc, 'vnewton', functools.partial(
                    mass.model.vcirc2, params=mass.params,
                    context=mass.context))
            for cmp in mass.model.components():
                functions[f'vcirc_{cmp.name()}'] = functools.partial(
                    _vcirc, f'vcirc_{cmp.name()}', functools.partial(
                        _component_vcirc2, mass, cmp.name()))
            masses = {}
            if mass.model.gives_enclosed_mass():
                masses[''] = functools.partial(
                    mass.model.enclosed_mass, params=mass.params,
                    context=mass.context)
            for cmp in mass.model.components():
                if cmp.gives_enclosed_mass():
                    masses[f'_{cmp.name()}'] = functools.partial(
                        _component_enclosed_mass, mass, cmp.name())
            for suffix, enclosed in masses.items():
                functions[f'mass{suffix}'] = functools.partial(
                    _enclosed_mass, f'mass{suffix}', enclosed)
            for suffix, enclosed in masses.items():
                functions[f'slope{suffix}'] = functools.partial(
                    _density_slope, f'slope{suffix}', enclosed)
        if (kpc_per_arcsec := self._stage.kpc_per_arcsec(params)) is not None:
            functions['kpc_per_arcsec'] = kpc_per_arcsec
        functions['redshift_at'] = functools.partial(
            _redshift_at, self._redshift)
        methods = dict(
            rotation=self._rotation, speed=self._speed,
            dispersion=self._dispersion, density=self._density,
            orientation=self._orientation)
        for i, suffix in self._suffixes.items():
            for family in self._profiles[i].families():
                functions[f'{family}{suffix}'] = functools.partial(
                    methods[family], i, params, state_of)
        return functions

    def _rotation(
            self,
            index: int,
            params: ParamValues,
            state_of: Callable[[int], ComponentState | None],
            R: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the rotation (km/s) of the component of the given index at
        the radii R (arcsec), at the values of the parameters of the galaxy
        and with what its galaxy gives it (its state, by the index of its
        component; see ComponentProfiles).
        """
        radii = self._radii(index, R, 'rotation')
        return self._profiles[index].rotation(
            self._own_params(index, params), state_of(index), radii.ravel()
        ).reshape(radii.shape)

    def _speed(
            self,
            index: int,
            params: ParamValues,
            state_of: Callable[[int], ComponentState | None],
            R: npt.ArrayLike,
            direction: str
    ) -> np.ndarray:
        """
        Return the speed (km/s) along a direction of the component of the
        given index at the radii R (arcsec), as _rotation.
        """
        radii = self._radii(index, R, 'speed')
        return self._profiles[index].speed(
            self._own_params(index, params), state_of(index), radii.ravel(),
            direction).reshape(radii.shape)

    def _dispersion(
            self,
            index: int,
            params: ParamValues,
            state_of: Callable[[int], ComponentState | None],
            R: npt.ArrayLike,
            direction: str | None = None
    ) -> np.ndarray:
        """
        Return the dispersion (km/s) of the component of the given index at
        the radii R (arcsec), along the direction if given, as _rotation.
        """
        radii = self._radii(index, R, 'dispersion')
        return self._profiles[index].dispersion(
            self._own_params(index, params), state_of(index), radii.ravel(),
            direction).reshape(radii.shape)

    def _density(
            self,
            index: int,
            params: ParamValues,
            state_of: Callable[[int], ComponentState | None],
            R: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the density of the component of the given index seen
        face-on at the radii R (arcsec), at the values of the parameters of
        the galaxy (it takes nothing from its galaxy).
        """
        radii = self._radii(index, R, 'density')
        return self._profiles[index].density(
            self._own_params(index, params), radii.ravel()
        ).reshape(radii.shape)

    def _orientation(
            self,
            index: int,
            params: ParamValues,
            state_of: Callable[[int], ComponentState | None],
            R: npt.ArrayLike,
            which: str
    ) -> np.ndarray:
        """
        Return the position angle or the inclination (which: 'posa' or
        'incl'; degrees) of the component of the given index at the radii R
        (arcsec), at the values of the parameters of the galaxy.
        """
        radii = self._radii(index, R, 'orientation')
        return self._profiles[index].orientation(
            self._own_params(index, params), radii.ravel(), which
        ).reshape(radii.shape)

    def _own_params(self, index: int, params: ParamValues) -> ParamValues:
        """
        Return the values of the parameters of the component of the given
        index (by its own names).
        """
        component = self._galaxy.components()[index]
        mapping = self._galaxy.component_param_names(index)
        return {p: params[mapping[p]] for p in component.pdescs()}

    def _radii(self, index: int, R: npt.ArrayLike, what: str) -> np.ndarray:
        """
        Return the radii R (arcsec) of a function of a profile of the
        component of the given index, as an array of floats (see
        checked_radii).
        """
        return checked_radii(f'{what}{self._suffixes[index]}', R)


def function_names(galaxy: 'Galaxy') -> tuple[str, ...]:
    """
    Return the names of the functions of a galaxy (see GalaxyFunctions),
    without planning them.
    """
    mass_model = galaxy.mass_model()
    names = []
    if mass_model is not None:
        names += ['vcirc', 'vnewton'] + [
            f'vcirc_{cmp.name()}' for cmp in mass_model.components()]
        if mass_model.gives_enclosed_mass():
            names += ['mass']
        names += [
            f'mass_{cmp.name()}' for cmp in mass_model.components()
            if cmp.gives_enclosed_mass()]
        if mass_model.gives_enclosed_mass():
            names += ['slope']
        names += [
            f'slope_{cmp.name()}' for cmp in mass_model.components()
            if cmp.gives_enclosed_mass()]
    if galaxy.distance() is not None:
        names += ['kpc_per_arcsec']
    names += ['redshift_at']
    for index, suffix in _profile_suffixes(galaxy).items():
        names += [
            f'{family}{suffix}'
            for family in profile_families(galaxy.components()[index])]
    return tuple(names)


def function_radius_range(
        galaxy: 'Galaxy', name: str
) -> tuple[float, float] | None:
    """
    Return the range of the radii (arcsec) of the function of a profile of
    a component of the name (e.g. rotation_gas) of a galaxy, without
    planning it (see profile_radius_range); None for its other functions,
    of any radius.
    """
    for index, suffix in _profile_suffixes(galaxy).items():
        component = galaxy.components()[index]
        for family in profile_families(component):
            if name == f'{family}{suffix}':
                return profile_radius_range(component, family)
    return None


def _redshift_at(redshift: float, v: npt.ArrayLike) -> np.ndarray:
    """
    Return the redshift of a galaxy of the given redshift at the systemic
    velocities v (km/s; see GalaxyFunctions).
    """
    return (1 + redshift) * doppler(np.asarray(v, dtype=float)) - 1


def _vcirc(
        function: str,
        vcirc2: Callable[[np.ndarray], np.ndarray],
        R: npt.ArrayLike
) -> np.ndarray:
    """
    Return a circular velocity (km/s, signed) of the function of the given
    name, from its square (vcirc2, of radii in kpc), at the radii R (kpc).

    Raises
    ------
    ValueError
        If a radius is negative or not finite.
    """
    radii = np.asarray(R, dtype=float)
    if not np.all(np.isfinite(radii) & (radii >= 0)):
        raise ValueError(
            f"{function}: the radii must be finite and at least 0; they are "
            f"{radii}")
    v2 = vcirc2(radii.ravel())
    return (np.sign(v2) * np.sqrt(np.abs(v2))).reshape(radii.shape)


def _component_vcirc2(
        mass: GalaxyMass, name: str, R: np.ndarray
) -> np.ndarray:
    """
    Return the Newtonian square of the circular velocity ((km/s)^2) of the
    mass component of the given name of the mass of a galaxy, at the
    radii R (kpc).
    """
    return mass.model.vcirc2_components(R, mass.params, mass.context)[name]


def _enclosed_mass(
        function: str,
        masses: Callable[[np.ndarray], np.ndarray],
        r: npt.ArrayLike
) -> np.ndarray:
    """
    Return an enclosed mass (Msun) of the function of the given name (of
    radii in kpc) at the radii r (kpc).

    Raises
    ------
    ValueError
        If a radius is negative or not finite.
    """
    radii = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(radii) & (radii >= 0)):
        raise ValueError(
            f"{function}: the radii must be finite and at least 0; they are "
            f"{radii}")
    return np.reshape(masses(radii.ravel()), radii.shape)


def _component_enclosed_mass(
        mass: GalaxyMass, name: str, r: np.ndarray
) -> np.ndarray:
    """
    Return the mass (Msun) of the mass component of the given name of the
    mass of a galaxy within the spheres of the radii r (kpc), in its
    context (e.g. a contracted halo, with the baryons it responds to).
    """
    return mass.model.enclosed_mass_components(
        r, mass.params, mass.context)[name]


def _density_slope(
        function: str,
        masses: Callable[[np.ndarray], np.ndarray],
        r: npt.ArrayLike
) -> np.ndarray:
    """
    Return the logarithmic slope of a spherically averaged density (see
    GalaxyFunctions), of the enclosed masses of the function of the given
    name (of radii in kpc), at the radii r (kpc): with L(s) = ln M(e^s),
    rho = M L' / (4 pi r^3), so d ln rho / d ln r = L' + L'' / L' - 3, by
    central differences of the step _SLOPE_STEP in s; NaN where L' is not
    positive (no density).

    Raises
    ------
    ValueError
        If a radius is not positive or not finite.
    """
    radii = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(radii) & (radii > 0)):
        raise ValueError(
            f"{function}: the radii must be finite and positive; they are "
            f"{radii}")
    s, h = np.log(radii.ravel()), _SLOPE_STEP
    with np.errstate(divide='ignore', invalid='ignore'):
        lo, mid, hi = (np.log(masses(np.exp(s + d))) for d in (-h, 0, h))
        first = (hi - lo) / (2 * h)
        second = (hi - 2 * mid + lo) / h ** 2
        slope = np.where(first > 0, first + second / first - 3, np.nan)
    return slope.reshape(radii.shape)


def _profile_suffixes(galaxy: 'Galaxy') -> dict[int, str]:
    """
    Return the components of a galaxy with profiles (see
    profile_families), by their index: the suffixes of their functions
    (see GalaxyFunctions).
    """
    return {
        i: '' if cmp.name() is None else f'_{cmp.name()}'
        for i, cmp in enumerate(galaxy.components())
        if profile_families(cmp)}
