"""
The gravity of a galaxy: how its mass (see MassModel) pulls, in
Newtonian gravity (GravityNewton, by default), in MOND (GravityMOND), in
the f(R) gravity of Hu & Sawicki (GravityHuSawicki) or in Palatini f(R)
gravity (GravityPalatini).
Each gravity gives the field of the mass in one evaluation (see
GravityField), whose circular velocity in the plane moves the
components; what else a field gives (see Gravity), the features that
need it check when the galaxy is made.
"""

import abc
import enum
from collections.abc import Callable, Collection, Sequence
from typing import Annotated, Any, Literal

import astropy.constants
import astropy.units
import numpy as np
import numpy.typing as npt
from annotated_types import Gt

from gbkfit.params import ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, check_options
from ._scalar_field import FieldGradient, solve_field
from .mass import MassContext, MassModel
from .mass.units import G


__all__ = [
    'A0_UNIT',
    'Capability',
    'Gravity',
    'GravityField',
    'GravityHuSawicki',
    'GravityMOND',
    'GravityNewton',
    'GravityPalatini',
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

# The square of the speed of light ((km/s)^2)
_C2 = astropy.constants.c.to_value('km/s') ** 2

# The central differences of the gradient of the density of Palatini
# f(R): their steps, of this fraction of R and of |z|, and at least this
# much (kpc)
_PALATINI_STEP = 1e-4
_PALATINI_MIN_STEP = 1e-9

# The outer radius of the field of f(R) (kpc): this many of its Compton
# wavelengths (beyond which it is in equilibrium with the density), and
# at least this much
_FR_RMAX_COMPTON = 10.0
_FR_RMAX_MIN = 1000.0


class Capability(enum.Enum):
    """
    What a galaxy gives the components that need it (see Component.needs):
    its mass, and what its gravity gives of it (see Gravity.gives).
    """

    # The mass (e.g. the circular velocity in its gravity)
    MASS = 'mass'
    # The forces of the mass anywhere (off the plane too)
    FORCES = 'forces'
    # Forces that are the gradient of a potential, in which orbits keep
    # their energy
    POTENTIAL = 'potential'
    # The vertical structure of self-gravitating isothermal sheets of
    # Newtonian gravity (h ~ sigma^2 / (G Sigma))
    SHEETS = 'sheets'
    # The shares of the components in the squared circular velocity (they
    # add up to that of the mass)
    SHARES = 'shares'


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
        gravity gives them (see Gravity.gives).
        """

    @abc.abstractmethod
    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the vertical force per unit mass ((km/s)^2/kpc) at the
        points (R, z) (kpc), if its gravity gives it (see Gravity.gives).
        """


class Gravity(parseutils.TypedSerializable, abc.ABC):
    """
    How the mass of a galaxy pulls: its parameters, and the field of the
    mass in each evaluation (see GravityField). Every field gives the
    circular velocity in the plane; what features need beyond it, a
    gravity gives or not (see gives), which the galaxy checks when it is
    made (see Component.needs):

    - the forces anywhere (off the plane too; Capability.FORCES), which
      the jeans equilibrium and the rotation at each height take (see
      parts.JeansEquilibrium and traits.VHTraitMass)
    - forces that are the gradient of a potential (POTENTIAL), in which
      orbits keep their energy, as the orbits of Schwarzschild models take
    - the vertical structure of self-gravitating isothermal sheets
      (SHEETS), which the self_gravitating equilibrium and pressure
      support take (see parts.SelfGravitatingEquilibrium and
      traits.VPTraitMass)
    - the shares of the components in the squared circular velocity
      (SHARES), which the dark fraction of a spheroid takes (see
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

    def check_mass_model(
            self, mass_model: MassModel, thick: Collection[str]
    ) -> None:
        """
        Raise ConfigError, saying why, unless it takes the mass model of
        its galaxy, whose thick components have the given names (e.g. the
        names of mass components it refers to, or what it takes of them,
        see MassComponent.gives_density); here, it does.
        """

    @abc.abstractmethod
    def gives(self, mass_model: MassModel) -> dict[Capability, str | None]:
        """
        Return what it gives of a mass model to the components of its
        galaxy that need it (see Capability; all but the mass, which is
        that of the galaxy): of each, None if it gives it, or else why
        not.

        Parameters
        ----------
        mass_model : MassModel
            The mass model of its galaxy.

        Returns
        -------
        dict
            None, or why not, by capability.
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

    def gives(self, mass_model: MassModel) -> dict[Capability, str | None]:
        # (the potential of the mass, wherever it gives its forces)
        forces = _forces_of_components(mass_model)
        return {
            Capability.FORCES: forces, Capability.POTENTIAL: forces,
            Capability.SHEETS: None, Capability.SHARES: None}


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
    DiskStage.state and GalaxyFunctions).

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

    def __init__(self, interpolation: Literal[MOND_INTERPOLATIONS]):
        check_options(GravityMOND.__init__, locals())
        self._interpolation = interpolation

    def interpolation(self) -> str:
        """Return the name of its interpolation function."""
        return self._interpolation

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(gravity_a0=ParamScalarDesc(
            desc="the acceleration a0 of MOND", min=0,
            unit='1e-10 m / s2'))

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

    def gives(self, mass_model: MassModel) -> dict[Capability, str | None]:
        # (the forces of the Newtonian forces of all the mass)
        return {
            Capability.FORCES: _forces_of_components(mass_model),
            Capability.POTENTIAL:
                "the forces of MOND here (the Newtonian forces times the "
                "interpolation function) are not the gradient of a "
                "potential off the plane: orbits do not keep their energy "
                "in them",
            Capability.SHEETS:
                "the self-gravitating sheets of MOND are not those of "
                "Newtonian gravity",
            Capability.SHARES:
                "the squared circular velocities of the components do not "
                "add up in MOND"}


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


class GravityHuSawicki(Gravity):
    """
    The f(R) gravity of Hu & Sawicki (2007), with its chameleon screening:
    the Newtonian field of all the mass, plus the fifth force (c^2 / 2)
    grad(f_R) of the scalar field f_R, of the quasi-static equation

        laplacian(f_R) = (R(f_R) - Rbar) / 3 - 8 pi G rho / (3 c^2)

    with R(f_R) = Rbar (fbar_R / f_R)^(1/(n+1)) (Hu & Sawicki 2007, of
    high curvature), Rbar and fbar_R the Ricci scalar and the field of the
    background at the epoch of the galaxy, fbar_R = f_R0 (Rbar_0 /
    Rbar)^(n+1) of the field today f_R0, and rho the density of the
    galaxy (an overdensity in a universe of mean density, so far from it
    f_R is in equilibrium with its density, and fbar_R where there is
    none). Where the field is unscreened, the fifth force is a third of
    the Newtonian one within the Compton wavelength lambda_C (some Mpc for
    |f_R0| = 1e-6, at the epoch 0); where the density is high enough, the
    field is screened and the gravity Newtonian. Its parameter:
    'gravity_log_fr0', log10 |f_R0|.

    The field is solved in each evaluation, on a grid of the meridional
    plane to beyond lambda_C, as SPAM (Naik et al. 2019) does: some 0.1 to
    1 s (more where the screening is partial). It is sourced by the mass
    of the mass components (in each cell of the grid; thin disks in the
    cells next to the plane), but of those screened (e.g. the stars, as
    SPAM can: each star screens itself; tables and user components, which
    do not give where their mass is, must be), and but within 10 pc of the
    centre (e.g. a point mass, compact, screens itself). With the
    environment, its parameter 'gravity_log_phi_ext' is log10 of the depth
    of the Newtonian potential of the environment of the galaxy, -Phi_ext
    / c^2 (e.g. of the screening maps of Desmond et al. 2018), which also
    sources the field as SPAM models it: a sphere of uniform density and
    of radius lambda_C about the galaxy, of potential Phi_ext at its
    centre (which only sources the field: its Newtonian force is left
    out). Its forces anywhere are those of the Newtonian forces of all the
    mass, plus the fifth force; it does not give the vertical structure
    of self-gravitating sheets, or the shares of the components (the
    fifth force is not a sum over them). The Newtonian circular
    velocities of the components remain in the rotation tables and the
    functions of the galaxy (see DiskStage.state and
    GalaxyFunctions).

    Parameters
    ----------
    n : float, optional
        The index n of Hu & Sawicki (1 by default, as SPAM).
    screened : sequence of str, optional
        The names of the mass components that do not source the field.
    environment : bool, optional
        Whether the environment of the galaxy sources the field.

    Raises
    ------
    ConfigError
        If n is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'hu_sawicki'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), n=self._n, screened=list(self._screened),
            environment=self._environment)

    def __init__(
            self,
            n: Annotated[float, Gt(0)] = 1,
            screened: Sequence[str] = (),
            environment: bool = False
    ):
        check_options(GravityHuSawicki.__init__, locals())
        self._n = float(n)
        self._screened = tuple(screened)
        self._environment = bool(environment)

    def n(self) -> float:
        """Return the index n of Hu & Sawicki."""
        return self._n

    def screened(self) -> tuple[str, ...]:
        """Return the names of the mass components that do not source it."""
        return self._screened

    def environment(self) -> bool:
        """Return whether the environment of the galaxy sources it."""
        return self._environment

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        pdescs = dict(gravity_log_fr0=ParamScalarDesc(
            desc="log10 of |f_R0|, the scalar field of the background "
                 "today", unit='dex'))
        if self._environment:
            pdescs['gravity_log_phi_ext'] = ParamScalarDesc(
                desc="log10 of -Phi_ext / c^2, the depth of the Newtonian "
                     "potential of the environment", unit='dex')
        return pdescs

    def field(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ) -> GravityField:
        if context.ricci_scalar is None or context.ricci_scalar_today is None:
            raise ValueError(
                "f(R) gravity takes the Ricci scalar of the background "
                "universe, which the context of the mass does not give")
        n, ricci = self._n, context.ricci_scalar
        background = 10 ** float(params['gravity_log_fr0']) * (
            context.ricci_scalar_today / ricci) ** (n + 1)
        a = ricci / (3 * background)
        b = 8 * np.pi * G / (3 * _C2 * background)
        compton = np.sqrt((n + 1) / a)
        # (of the uniform sphere of the environment: Phi(0) = -2 pi G rho
        # lambda_C^2)
        environment = 0.0
        if self._environment:
            environment = (10 ** float(params['gravity_log_phi_ext']) * _C2
                           / (2 * np.pi * G * compton ** 2))
        sources = [
            component.name() for component in mass_model.components()
            if component.name() not in self._screened]

        def masses(r_edges: np.ndarray, b_edges: np.ndarray) -> np.ndarray:
            try:
                galaxy = mass_model.cell_masses(
                    r_edges, b_edges, params, context, sources)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"the field of f(R) gravity is sourced by the mass of "
                    f"the mass components that are not screened: {e}") from e
            return galaxy + 4 * np.pi / 3 * environment * np.outer(
                np.diff(np.minimum(r_edges, compton) ** 3),
                np.diff(np.sin(b_edges)))

        rmax = max(_FR_RMAX_COMPTON * compton, _FR_RMAX_MIN)
        grid, w, w_out = solve_field(rmax, masses, a, b, n)
        return _HuSawickiField(
            _NewtonField(mass_model, params, context),
            grid.gradient(w, w_out), _C2 * background / 2)

    def check_mass_model(
            self, mass_model: MassModel, thick: Collection[str]
    ) -> None:
        names = [component.name() for component in mass_model.components()]
        if unknown := [name for name in self._screened if name not in names]:
            raise ConfigError(
                f"the screened mass components {unknown} are not in the "
                f"mass model; its components: {names}")
        if massless := [
                component.name() for component in mass_model.components()
                if component.name() not in self._screened
                and not component.gives_cell_masses(thick)]:
            raise ConfigError(
                f"the field of f(R) gravity is sourced by the mass of the "
                f"mass components that are not screened; the mass "
                f"components {massless} do not give where their mass is "
                f"(e.g. tables or user components): screen them")

    def gives(self, mass_model: MassModel) -> dict[Capability, str | None]:
        # (the forces of the Newtonian forces of all the mass)
        return {
            Capability.FORCES: _forces_of_components(mass_model),
            Capability.POTENTIAL:
                "the forces of f(R) gravity here are not given as the "
                "gradient of a potential off the plane",
            Capability.SHEETS:
                "the self-gravitating sheets of f(R) gravity are not those "
                "of Newtonian gravity",
            Capability.SHARES:
                "the fifth force of f(R) gravity is not a sum over the "
                "components"}


class _HuSawickiField(GravityField):
    """
    The field of a mass model in f(R) gravity (see GravityHuSawicki): its
    Newtonian field, plus the fifth force -scale grad(w) of the field w =
    f_R / fbar_R (scale = c^2 |fbar_R| / 2, (km/s)^2), of the gradient of
    w (kpc^-1) at any points (R, z).
    """

    def __init__(
            self, newton: GravityField, gradient: FieldGradient, scale: float
    ):
        self._newton = newton
        self._gradient = gradient
        self._scale = scale

    def vcirc2(self, R: npt.ArrayLike) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        grad_R, _ = self._gradient(R, np.zeros_like(R))
        return self._newton.vcirc2(R) + self._scale * R * grad_R

    def forces(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        force_R, force_z = self._newton.forces(R, z)
        grad_R, grad_z = self._gradient(R, z)
        return (force_R - self._scale * grad_R,
                force_z - self._scale * grad_z)

    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        return self.forces(R, z)[1]


class GravityPalatini(Gravity):
    """
    Palatini f(R) gravity in its weak-field limit, of any f(R) = R + alpha
    R^2 + O(R^3) (Toniato et al. 2020; Hernandez-Arboleda et al. 2023, eq.
    16): the Poisson equation of the potential of the motion is
    laplacian(Phi) = 4 pi G (rho + 2 alpha laplacian(rho)), so Phi = Phi_N
    + 8 pi G alpha rho, of the density rho of all the mass. Its correction
    is local: where there is mass (in vacuum, it is Newtonian gravity).
    Its squared circular velocity is v^2 = v_N^2 + 8 pi G alpha R
    drho/dR, at z = 0 (their eq. 32), and its forces are the Newtonian
    ones minus 8 pi G alpha grad(rho). Its parameter: 'gravity_alpha',
    alpha (kpc^2; disks are stable for alpha < 0, which pulls inwards
    where the density falls outwards).

    The gradient of the density is numerical: central differences of
    steps of 1e-4 of R and of |z| (relative truncation errors of about
    (1e-4 R / L)^2 / 6, for a density that changes on scales L). Every
    mass component must give its density (see check_mass_model): thin
    disks do not (the correction in their plane depends on Sigma(R) Z(0)
    of a vertical law Z, infinite for a thin disk), nor do tables, user
    components and halos that respond to their baryons; point masses
    have none away from the centre. It does not give the vertical
    structure of self-gravitating sheets, or the shares of the
    components (those of the dark fraction are Newtonian). The Newtonian
    circular velocities of the components remain in the rotation tables
    and the functions of the galaxy (see DiskStage.state and
    GalaxyFunctions).
    """

    @staticmethod
    def type() -> str:
        return 'palatini'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(gravity_alpha=ParamScalarDesc(
            desc="alpha of f(R) = R + alpha R^2", unit='kpc2'))

    def field(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ) -> GravityField:

        def density(R: np.ndarray, z: np.ndarray) -> np.ndarray:
            try:
                return mass_model.density(R, z, params, context)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"Palatini f(R) gravity takes the density of the mass "
                    f"components: {e}") from e

        return _PalatiniField(
            _NewtonField(mass_model, params, context), density,
            8 * np.pi * G * float(params['gravity_alpha']))

    def check_mass_model(
            self, mass_model: MassModel, thick: Collection[str]
    ) -> None:
        if densityless := [
                component.name() for component in mass_model.components()
                if not component.gives_density(thick)]:
            raise ConfigError(
                f"Palatini f(R) gravity takes the density of every mass "
                f"component; the mass components {densityless} do not give "
                f"it (e.g. thin disks, tables, user components or halos "
                f"that respond to their baryons)")

    def gives(self, mass_model: MassModel) -> dict[Capability, str | None]:
        # (the forces of the Newtonian forces of all the mass)
        return {
            Capability.FORCES: _forces_of_components(mass_model),
            Capability.POTENTIAL:
                "the forces of Palatini f(R) gravity here are not given as "
                "the gradient of a potential off the plane",
            Capability.SHEETS:
                "the self-gravitating sheets of Palatini f(R) gravity are "
                "not those of Newtonian gravity",
            Capability.SHARES:
                "the shares of the components that the dark fraction takes "
                "are Newtonian, not those of Palatini f(R) gravity"}


class _PalatiniField(GravityField):
    """
    The field of a mass model in Palatini f(R) gravity (see
    GravityPalatini): its Newtonian field, plus -scale grad(rho) of its
    density rho, a function of (R, z) (scale = 8 pi G alpha).
    """

    def __init__(
            self,
            newton: GravityField,
            density: Callable[[np.ndarray, np.ndarray], np.ndarray],
            scale: float
    ):
        self._newton = newton
        self._density = density
        self._scale = scale

    def vcirc2(self, R: npt.ArrayLike) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        grad_R, _ = self._gradient(R, np.zeros_like(R))
        return self._newton.vcirc2(R) + self._scale * R * grad_R

    def forces(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        force_R, force_z = self._newton.forces(R, z)
        grad_R, grad_z = self._gradient(R, z)
        return (force_R - self._scale * grad_R,
                force_z - self._scale * grad_z)

    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        return self.forces(R, z)[1]

    def _gradient(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the gradient of the density, (drho/dR, drho/dz), at the
        points (R, z), by central differences (none without a correction).
        """
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        if self._scale == 0:
            return np.zeros_like(R), np.zeros_like(R)
        step_R = _PALATINI_STEP * np.maximum(R, _PALATINI_MIN_STEP)
        step_z = _PALATINI_STEP * np.maximum(np.abs(z), _PALATINI_MIN_STEP)
        grad_R = (self._density(R + step_R, z)
                  - self._density(R - step_R, z)) / (2 * step_R)
        grad_z = (self._density(R, z + step_z)
                  - self._density(R, z - step_z)) / (2 * step_z)
        return grad_R, grad_z


def _forces_of_components(mass_model: MassModel) -> str | None:
    """
    Return None if every component of the mass model gives its forces off
    its plane (see MassComponent.gives_forces), or else why not.
    """
    if forceless := [
            component.name() for component in mass_model.components()
            if not component.gives_forces()]:
        return (
            f"the mass components {forceless} do not give their forces off "
            f"their plane (user components give them by a function of the "
            f"forces, see MassUser)")
    return None


gravity_parser = parseutils.TypedParser(
    Gravity, [GravityNewton, GravityMOND, GravityHuSawicki, GravityPalatini])
