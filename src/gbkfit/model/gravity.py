"""
The gravity of a galaxy: how its mass (see MassModel) pulls, in
Newtonian gravity (GravityNewton, by default) or in MOND (GravityMOND).
Each gravity gives the field of the mass in one evaluation (see
GravityField), whose circular velocity in the plane moves the
components; what else a field gives (see Gravity), the features that
need it check when the galaxy is made.
"""

import abc
from collections.abc import Callable
from typing import Any

import astropy.units
import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .mass import MassContext, MassModel


__all__ = [
    'A0_UNIT',
    'Gravity',
    'GravityField',
    'GravityMOND',
    'GravityNewton',
    'MOND_INTERPOLATIONS',
    'gravity_parser'
]


# The unit of the acceleration a0 of MOND, 1e-10 m/s^2, in (km/s)^2/kpc
A0_UNIT = (1e-10 * astropy.units.Unit('m / s2')).to_value('km2 / (s2 kpc)')

# The interpolation functions nu(y) of MOND, of y = g_N / a0 (see
# GravityMOND), by name: infinite at y = 0, 1 at y = inf
_MOND_NU: dict[str, Callable[[np.ndarray], np.ndarray]] = dict(
    simple=lambda y: 0.5 + np.sqrt(0.25 + 1 / y),
    standard=lambda y: np.sqrt(0.5 + np.sqrt(0.25 + 1 / y ** 2)),
    rar=lambda y: 1 / -np.expm1(-np.sqrt(y)))

# The names of the interpolation functions of MOND
MOND_INTERPOLATIONS = tuple(_MOND_NU)


class GravityField(abc.ABC):
    """
    The gravity of the mass of a galaxy in one evaluation, in kpc (see
    Gravity.field).
    """

    @abc.abstractmethod
    def vcirc2(self, R: npt.ArrayLike) -> np.ndarray:
        """
        Return the square of the circular velocity ((km/s)^2) in the plane
        at the radii R (kpc), signed: negative where the mass pulls
        outwards.
        """

    @abc.abstractmethod
    def forces(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the forces per unit mass ((km/s)^2/kpc), radial and
        vertical, at the points (R, z) (kpc), broadcast together, if its
        gravity gives them (see Gravity.check_forces).
        """

    @abc.abstractmethod
    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the vertical force per unit mass ((km/s)^2/kpc) at the
        points (R, z) (kpc), if its gravity gives it (see
        Gravity.check_forces).
        """


class Gravity(parseutils.TypedSerializable, abc.ABC):
    """
    How the mass of a galaxy pulls: its parameters, and the field of the
    mass in each evaluation (see GravityField). Every field gives the
    circular velocity in the plane; the features that need more check
    when the galaxy is made that its gravity gives it:

    - the forces anywhere (off the plane too), which the jeans
      equilibrium takes (see check_forces and parts.Kinematics)
    - the vertical structure of self-gravitating isothermal sheets,
      which the self_gravitating equilibrium and pressure support take
      (see check_self_gravitating_sheets, parts.Kinematics and
      traits.VPTraitMass)
    - the shares of the components in the squared circular velocity,
      which the dark fraction of a spheroid takes (see check_shares and
      mass.DarkFraction)
    """

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name; none here."""
        return {}

    @abc.abstractmethod
    def field(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ) -> GravityField:
        """
        Return the field of a mass model, at the values of the parameters
        of its galaxy (those of the gravity among them), in its context.
        """

    @abc.abstractmethod
    def check_forces(self, mass_model: MassModel) -> None:
        """
        Raise ConfigError, saying why, unless its fields give the forces
        of the mass model anywhere (off the plane too).
        """

    @abc.abstractmethod
    def check_self_gravitating_sheets(self) -> None:
        """
        Raise ConfigError, saying why, unless the vertical structure of a
        self-gravitating isothermal sheet in it is that of Newtonian
        gravity (h ~ sigma^2 / (G Sigma)).
        """

    @abc.abstractmethod
    def check_shares(self) -> None:
        """
        Raise ConfigError, saying why, unless the squared circular
        velocities of the components add up to that of the mass.
        """


class GravityNewton(Gravity):
    """
    Newtonian gravity, by default: the fields of the components of the
    mass add up, and so do their squared circular velocities.
    """

    @staticmethod
    def type() -> str:
        return 'newton'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def field(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ) -> GravityField:
        return _NewtonField(mass_model, params, context)

    def check_forces(self, mass_model: MassModel) -> None:
        _check_components_give_forces(mass_model)

    def check_self_gravitating_sheets(self) -> None:
        pass

    def check_shares(self) -> None:
        pass


class _NewtonField(GravityField):
    """The field of a mass model in Newtonian gravity (see GravityNewton)."""

    def __init__(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ):
        self._mass_model = mass_model
        self._params = params
        self._context = context

    def vcirc2(self, R: npt.ArrayLike) -> np.ndarray:
        return self._mass_model.vcirc2(R, self._params, self._context)

    def forces(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._mass_model.forces(R, z, self._params, self._context)

    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        return self._mass_model.vertical_force(
            R, z, self._params, self._context)


class GravityMOND(Gravity):
    """
    MOND (Milgrom 1983) in its algebraic form: the acceleration in the
    plane is nu(|g_N| / a0) g_N, of the Newtonian acceleration g_N =
    v_N^2 / R of all the mass (see GravityNewton) and the acceleration
    a0, so that its squared circular velocity is v^2 = nu(|g_N| / a0)
    v_N^2, signed as v_N^2. Its parameter: 'gravity_a0', a0 in units of
    1e-10 m/s^2 (a0 = 0 is Newtonian gravity). Its interpolation
    function nu, of y = |g_N| / a0:

    - 'simple': nu(y) = 1/2 + sqrt(1/4 + 1/y)
    - 'standard': nu(y) = sqrt(1/2 + sqrt(1/4 + 1/y^2)) (that of GIPSY
      ROTMOD)
    - 'rar': nu(y) = 1 / (1 - exp(-sqrt(y))) (McGaugh, Lelli & Schombert
      2016, the radial acceleration relation)

    Its forces anywhere follow the same rule, nu(|g_N| / a0) g_N, of the
    Newtonian acceleration g_N = (F_R, F_z) of all the mass (in the
    plane, those of v^2). The algebraic form is exact for spherical mass
    and for a Kuzmin disk, and an approximation for other disks, which
    rotation-curve fits take: very good for exponential disks in their
    plane, worse for disks with holes or sharp edges (Brada & Milgrom
    1995); off the plane, its field is not that of a potential (nu g_N
    curls). It does not give the vertical structure of self-gravitating
    sheets, or the shares of the components (the squared circular
    velocities of MOND do not add up); and there is no external field
    effect. The Newtonian circular velocities of the components remain
    in the rotation tables and the functions of the galaxy (see
    GravityPlan.evaluate and GalaxyFunctions).

    Parameters
    ----------
    interpolation : {'simple', 'standard', 'rar'}
        The interpolation function.

    Raises
    ------
    ConfigError
        If the interpolation function is not one of them.
    """

    @staticmethod
    def type() -> str:
        return 'mond'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), interpolation=self._interpolation)

    def __init__(self, interpolation: str):
        if interpolation not in MOND_INTERPOLATIONS:
            raise ConfigError(
                f"interpolation must be one of {list(MOND_INTERPOLATIONS)}; "
                f"it is {interpolation!r}")
        self._interpolation = interpolation

    def interpolation(self) -> str:
        """Return the name of its interpolation function."""
        return self._interpolation

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(gravity_a0=ParamScalarDesc(
            desc="the acceleration a0 of MOND (1e-10 m/s^2)", minimum=0))

    def field(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ) -> GravityField:
        return _MONDField(
            _NewtonField(mass_model, params, context),
            _MOND_NU[self._interpolation],
            float(params['gravity_a0']) * A0_UNIT)

    def check_forces(self, mass_model: MassModel) -> None:
        # (of the Newtonian forces of all the mass)
        _check_components_give_forces(mass_model)

    def check_self_gravitating_sheets(self) -> None:
        raise ConfigError(
            "the self-gravitating sheets of MOND are not those of Newtonian "
            "gravity")

    def check_shares(self) -> None:
        raise ConfigError(
            "the squared circular velocities of the components do not add "
            "up in MOND")


class _MONDField(GravityField):
    """
    The field of a mass model in MOND (see GravityMOND): its Newtonian
    field, its interpolation function nu and a0 ((km/s)^2/kpc).
    """

    def __init__(
            self,
            newton: GravityField,
            nu: Callable[[np.ndarray], np.ndarray],
            a0: float
    ):
        self._newton = newton
        self._nu = nu
        self._a0 = a0

    def vcirc2(self, R: npt.ArrayLike) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        v2 = self._newton.vcirc2(R)
        # (y = |g_N| / a0 is infinite at the centre and with a0 = 0, where
        # nu is 1; where g_N = 0, nu v_N^2 is 0)
        scale = R * self._a0
        y = np.divide(
            np.abs(v2), scale, out=np.full_like(v2, np.inf), where=scale > 0)
        with np.errstate(divide='ignore'):
            nu = self._nu(y)
        return np.where(v2 == 0, 0.0, nu * v2)

    def forces(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        force_R, force_z = self._newton.forces(R, z)
        # (y = |g_N| / a0 is infinite with a0 = 0, where nu is 1; where g_N
        # = 0, nu g_N is 0)
        size = np.hypot(force_R, force_z)
        y = size / self._a0 if self._a0 > 0 else np.full_like(size, np.inf)
        with np.errstate(divide='ignore'):
            nu = np.where(size == 0, 0.0, self._nu(y))
        return nu * force_R, nu * force_z

    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        return self.forces(R, z)[1]


def _check_components_give_forces(mass_model: MassModel) -> None:
    """
    Raise ConfigError unless every component of the mass model gives its
    forces off its plane (see MassComponent.gives_forces).
    """
    if forceless := [
            component.name() for component in mass_model.components()
            if not component.gives_forces()]:
        raise ConfigError(
            f"the mass components {forceless} do not give their forces off "
            f"their plane (user components give them by a function of the "
            f"forces, see MassUser)")


gravity_parser = parseutils.TypedParser(Gravity, [GravityNewton, GravityMOND])
