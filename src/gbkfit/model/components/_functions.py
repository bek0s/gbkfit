"""
The functions of a galaxy (GalaxyFunctions) that the derived values of
its parameters can use (see ParamSpace): the circular velocity of its
mass, and the rotation and dispersion of its disks, at any radius.
"""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from .disks._disk import EQUILIBRIUM_DISPERSION
from .disks._profiles import dispersion, rotation
from ._gravity import GalaxyMass, GravityPlan

if TYPE_CHECKING:
    from ._component_set import ComponentSet


__all__ = [
    'GalaxyFunctions'
]


class GalaxyFunctions:
    """
    The functions of a galaxy that the derived values of its parameters
    can use (see ParamSpace), by name, at the values of its parameters
    (see bind). With mass:

    - vcirc(R): the circular velocity of its mass (km/s) at the radii R
      (kpc, in the plane of its disks), signed as in the rotation tables
      of its gravity: negative where its square is (see
      GravityPlan.evaluate)
    - vcirc_{name}(R): that of its mass component of the name, signed
      (negative where it pulls outwards)
    - kpc_per_arcsec: the length of an arcsec in it (kpc)

    For each of its disks with velocity or dispersion traits, named
    after its component ('rotation_gas'; 'rotation' for one without a
    name), at the radii R (arcsec, on the disk, from its first radial
    node to its last):

    - rotation_{component}(R): its rotation at its midplane (km/s), as
      its gravity moves it: the azimuthal mean of its tangential velocity
      traits (with the circular velocity of the mass, and its pressure
      support, that its gravity gives them; see rotation)
    - dispersion_{component}(R): its velocity dispersion at its midplane
      (km/s): the azimuthal mean of its dispersion traits, with the
      dispersion of its vertical equilibrium in quadrature (see
      dispersion)

    The functions take numbers or arrays of radii, and return arrays of
    their shape. They have their own plans of the gravity and of the
    disks, on the given driver and in float64, apart from those of the
    evaluations of the galaxy (which may run at the same time, e.g. in
    an EvaluationPool).

    Parameters
    ----------
    component_set : ComponentSet
        The components of the galaxy, its distance and its mass.
    driver : Driver
        The driver where the disks sample their profiles.
    redshift : float
        The redshift of the galaxy (for its distance, see Distance).
    """

    def __init__(
            self,
            component_set: 'ComponentSet',
            driver: Driver,
            redshift: float
    ):
        components = component_set.components()
        mass_model = component_set.mass_model()
        self._component_set = component_set
        self._gravity = GravityPlan(component_set, driver, redshift)
        # The disks with velocity or dispersion traits, by the index of
        # their component, and the suffixes of their functions
        self._suffixes = {
            i: '' if cmp.name() is None else f'_{cmp.name()}'
            for i, cmp in enumerate(components)
            if (disk := cmp.density_disk()) is not None
            and (disk.traits('vpt') or disk.traits('dpt'))}
        self._disk_plans = {
            i: components[i].density_disk().plan(driver, 0, np.float64)
            for i in self._suffixes}
        names = []
        if mass_model is not None:
            names += ['vcirc'] + [
                f'vcirc_{cmp.name()}' for cmp in mass_model.components()]
            names += ['kpc_per_arcsec']
        for suffix in self._suffixes.values():
            names += [f'rotation{suffix}', f'dispersion{suffix}']
        self._names = tuple(names)

    def names(self) -> tuple[str, ...]:
        """Return the names of its functions (see the class)."""
        return self._names

    def bind(self, params: ParamValues) -> dict[str, Any]:
        """
        Return its functions (see the class) at the values of the
        parameters of the galaxy (see Galaxy.pdescs), by name. Its
        gravity is evaluated once, when a function of a disk that takes
        it first needs it (see GravityPlan.evaluate).
        """
        mass = self._gravity.mass(params)
        given = functools.cache(functools.partial(
            self._gravity.evaluate, params))
        functions: dict[str, Any] = {}
        if mass is not None:
            functions['vcirc'] = functools.partial(_vcirc, mass, None)
            for cmp in mass.model.components():
                functions[f'vcirc_{cmp.name()}'] = functools.partial(
                    _vcirc, mass, cmp.name())
            functions['kpc_per_arcsec'] = mass.kpc_per_arcsec
        for i, suffix in self._suffixes.items():
            functions[f'rotation{suffix}'] = functools.partial(
                self._rotation, i, params, given)
            functions[f'dispersion{suffix}'] = functools.partial(
                self._dispersion, i, params, given)
        return functions

    def _rotation(
            self,
            index: int,
            params: ParamValues,
            given: Callable[[], dict[int, ParamValues]],
            R: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the rotation (km/s) of the disk of the component of the
        given index at the radii R (arcsec), at the values of the
        parameters of the galaxy and with what its gravity gives it.
        """
        disk = self._component_set.components()[index].density_disk()
        radii = self._radii(index, R, 'rotation')
        values = self._disk_params(index, params, given)
        return rotation(
            disk, self._disk_plans[index], values,
            radii.ravel()).reshape(radii.shape)

    def _dispersion(
            self,
            index: int,
            params: ParamValues,
            given: Callable[[], dict[int, ParamValues]],
            R: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the dispersion (km/s) of the disk of the component of the
        given index at the radii R (arcsec), at the values of the
        parameters of the galaxy and with what its gravity gives it.
        """
        disk = self._component_set.components()[index].density_disk()
        radii = self._radii(index, R, 'dispersion')
        values = self._disk_params(index, params, given)
        return dispersion(
            disk, self._disk_plans[index], values, radii.ravel(),
            values.get(EQUILIBRIUM_DISPERSION)).reshape(radii.shape)

    def _disk_params(
            self,
            index: int,
            params: ParamValues,
            given: Callable[[], dict[int, ParamValues]]
    ) -> ParamValues:
        """
        Return the values of the parameters of the component of the given
        index (by its own names), with those that its gravity gives it (if
        any; see GravityPlan.evaluate).
        """
        component = self._component_set.components()[index]
        mapping = self._component_set.mappings()[index]
        values = {p: params[mapping[p]] for p in component.pdescs()}
        disk = component.density_disk()
        if disk.circular_velocity_params() or disk.equilibrium() is not None:
            values |= given()[index]
        return values

    def _radii(self, index: int, R: npt.ArrayLike, what: str) -> np.ndarray:
        """
        Return the radii R (arcsec) of a function of the disk of the
        component of the given index, as an array of floats.

        Raises
        ------
        ValueError
            If a radius is outside the disk: below its first radial node,
            or above its last.
        """
        component = self._component_set.components()[index]
        rnodes = component.density_disk().rnodes()
        radii = np.asarray(R, dtype=float)
        if not np.all((rnodes[0] <= radii) & (radii <= rnodes[-1])):
            name = f'{what}{self._suffixes[index]}'
            raise ValueError(
                f"{name}: the radii must be on the disk, from {rnodes[0]:g} "
                f"to {rnodes[-1]:g} arcsec; they are {radii}")
        return radii


def _vcirc(mass: GalaxyMass, name: str | None, R: npt.ArrayLike) -> np.ndarray:
    """
    Return the circular velocity (km/s, signed) of the mass of a galaxy,
    or of its mass component of the given name, at the radii R (kpc).

    Raises
    ------
    ValueError
        If a radius is negative or not finite.
    """
    radii = np.asarray(R, dtype=float)
    if not np.all(np.isfinite(radii) & (radii >= 0)):
        function = 'vcirc' if name is None else f'vcirc_{name}'
        raise ValueError(
            f"{function}: the radii must be finite and at least 0; they are "
            f"{radii}")
    parts = mass.model.vcirc2_components(
        radii.ravel(), mass.params, mass.context)
    vcirc2 = sum(parts.values()) if name is None else parts[name]
    return (np.sign(vcirc2) * np.sqrt(np.abs(vcirc2))).reshape(radii.shape)
