"""
The responses of spherical halos to the gravity of their baryons (see
MassSpheroid): the adiabatic contractions of Blumenthal et al. (1986) and
Gnedin et al. (2004), that of Cautun et al. (2020), and the isothermal
core of self-interacting dark matter.
"""

import abc
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import astropy.units
import numpy as np
import scipy.integrate
import scipy.interpolate

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, check_options
from .base import MassContext
from .halos import ProfileHalo
from .profiles import SpheroidProfile
from .units import G


__all__ = [
    'HaloResponse',
    'ResponseBlumenthal86',
    'ResponseCautun20',
    'ResponseGnedin04',
    'ResponseSIDM',
    'SIDMCore',
    'halo_response_parser'
]


# The orbit-averaged radii of the contraction of Gnedin et al. (2004):
# r_bar = A0 r_D (r / r_D)^w
_GNEDIN04_A0 = 0.85
_GNEDIN04_W = 0.8

# The bisections of the initial radius of a contracted shell, in ln r
# over [ln r, ln r + _CONTRACTION_RANGE] (to the precision of floats)
_CONTRACTION_RANGE = 50.0
_CONTRACTION_BISECTIONS = 80

# A squared circular velocity of the radius (kpc)
_VCirc2 = Callable[[np.ndarray], np.ndarray]

# A mass (Msun) within spheres of the radii (kpc)
_Mass = Callable[[np.ndarray], np.ndarray]

# The units of the scattering rate of SIDM: the cross-section (cm^2/g) in
# kpc^2/Msun, and a velocity (km/s) times a time (Gyr) in kpc
_SIGMA_UNIT = (1 * astropy.units.Unit('cm2 / g')).to_value('kpc2 / solMass')
_TIME_UNIT = (1 * astropy.units.Unit('km Gyr / s')).to_value('kpc')

# The radii (kpc) at which the halo before its SIDM core is taken: its
# mass, density and velocity dispersion (of the Jeans equation)
_SIDM_RADII = np.geomspace(1e-4, 1e4, 801)

# The isothermal core of SIDM: integrated by Runge-Kutta in ln r, of
# these steps, from this fraction of r1 (uniform within); the density at
# its centre searched from that of the halo at r1 to this many times it,
# and its velocity dispersion over this range of that of the halo at r1
# (Jiang et al. 2023), on a grid of these many values of each, then
# refined by Newton iterations to the tolerance (in ln; at most this
# many)
_SIDM_STEP = 0.05
_SIDM_START = 1e-4
_SIDM_DENSITIES = (1e6, 48)
_SIDM_DISPERSIONS = (0.5, 2.0, 24)
_SIDM_ITERATIONS = 50
_SIDM_TOLERANCE = 1e-10


class HaloResponse(parseutils.TypedSerializable, abc.ABC):
    """
    How a spherical halo responds to the gravity of its baryons: its
    squared circular velocity after, of those of the halo before and of
    the baryons, and its parameters (of the spheroid of the halo, with
    those of its profile and shape).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name; none here."""
        return {}

    def check_profile(self, profile: SpheroidProfile) -> None:
        """
        Raise ConfigError, saying why, unless it takes the profile of its
        halo; here, it does.
        """

    def check_baryons(self, baryons: Sequence[str]) -> None:
        """
        Raise ConfigError, saying why, unless it takes the baryons its
        halo responds to (by their names); here, it needs some.
        """
        if not baryons:
            raise ConfigError(
                f"the response '{self.type()}' needs the baryons that the "
                f"spheroid responds to (baryons)")

    @abc.abstractmethod
    def vcirc2(
            self,
            R: np.ndarray,
            halo_vcirc2: _VCirc2,
            baryons_mass: _Mass,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the squared circular velocity ((km/s)^2) of the halo after
        its response at the radii R (kpc; 1-D), of that of the halo before
        and the mass of the baryons within spheres (functions of the
        radius), the profile of the halo, the values of the parameters of
        its spheroid and its context (see MassComponent.vcirc2).
        """


class ResponseBlumenthal86(HaloResponse):
    """
    The adiabatic contraction of Blumenthal et al. (1986): the dark matter
    within r' ends within r, with r' M(r') = r (M(r') + M_b(r)) (Burkert
    et al. 2010, Eq. 19), of the masses within spheres of the halo before
    (M = v^2 r / G) and of the baryons: v^2(r) = G M(r') / r = v^2(r') r'
    / r. (Burkert
    et al. 2010, Eq. 18, which DysmalPy's documentation quotes, writes
    v^2(r') for this term, smaller by r / r': it does not keep the mass
    that Eq. 19 moves within r.)
    """

    @staticmethod
    def type() -> str:
        return 'blumenthal86'

    def vcirc2(
            self,
            R: np.ndarray,
            halo_vcirc2: _VCirc2,
            baryons_mass: _Mass,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return _contracted(R, halo_vcirc2, baryons_mass)


class ResponseGnedin04(HaloResponse):
    """
    The adiabatic contraction of Gnedin et al. (2004): that of Blumenthal
    et al. (see ResponseBlumenthal86) with the orbit-averaged radii r_bar
    = 0.85 r_D (r / r_D)^0.8 in the masses, of the radius r_D of the halo
    (of a parameterisation of a mass or a velocity).
    """

    @staticmethod
    def type() -> str:
        return 'gnedin04'

    def check_profile(self, profile: SpheroidProfile) -> None:
        if not (isinstance(profile, ProfileHalo)
                and profile.parameterisation() != 'density_radius'):
            raise ConfigError(
                "the contraction 'gnedin04' takes the radius r_D of a halo: "
                "it needs a halo of a parameterisation of a mass or a "
                "velocity")

    def vcirc2(
            self,
            R: np.ndarray,
            halo_vcirc2: _VCirc2,
            baryons_mass: _Mass,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return _contracted(
            R, halo_vcirc2, baryons_mass, _GNEDIN04_A0, _GNEDIN04_W,
            profile.halo_radius(params, context))


class ResponseCautun20(HaloResponse):
    """
    The contraction of Cautun et al. (2020, Eq. 11), as they fit their
    simulations: the mass of the halo within r times 0.45 + 0.38 (eta +
    1.16)^0.53, eta = M_b (1 - f_b) / (f_b M) of the masses within the
    sphere of r of the halo before (of dark matter only, M = v^2 r / G)
    and of the baryons, f_b the cosmic baryon fraction of the cosmology of
    the galaxy (about 1 at eta = 1: baryons that follow the dark matter).
    """

    @staticmethod
    def type() -> str:
        return 'cautun20'

    def vcirc2(
            self,
            R: np.ndarray,
            halo_vcirc2: _VCirc2,
            baryons_mass: _Mass,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        fraction = context.baryon_fraction
        if fraction is None:
            raise ValueError(
                "the contraction 'cautun20' takes the cosmic baryon "
                "fraction, which the cosmology of the galaxy lacks "
                "(Omega_b)")
        # (G M / r of both)
        halo = halo_vcirc2(R)
        with np.errstate(divide='ignore', invalid='ignore'):
            baryons = G * baryons_mass(R) / R
            eta = baryons * (1 - fraction) / (fraction * halo)
            out = halo * (0.45 + 0.38 * (eta + 1.16) ** 0.53)
        out[R == 0] = 0
        return out


class SIDMCore(NamedTuple):
    """
    The isothermal core of a halo of SIDM (see ResponseSIDM): its radius
    r1 (kpc), its central density rho0 (Msun/kpc^3) and velocity
    dispersion v0 (km/s), and its mass (Msun) within radii below r1 (a
    function of the radii).
    """
    r1: float
    rho0: float
    v0: float
    mass: Callable[[np.ndarray], np.ndarray]


class ResponseSIDM(HaloResponse):
    """
    The isothermal core of self-interacting dark matter (SIDM; Kaplinghat
    et al. 2014, 2016; Jiang et al. 2023). Within the radius r1 where a
    particle of the halo has scattered once in the age of the halo,

        (4 / sqrt(pi)) rho(r1) v(r1) sigma_m t_age = 1

    (Jiang et al. 2023, Eq. 1; rho and v the density and the 1D velocity
    dispersion of the halo before, isotropic, of the Jeans equation in the
    potential of the halo and of the baryons), the dark matter is
    isothermal: rho = rho0 exp(-(Phi - Phi(0)) / v0^2) in the potential Phi
    of itself and of the baryons (Eq. 4-5), of rho0 and v0 such that its
    density and its mass match those of the halo before at r1 (the
    solution of the lower density: Elbert et al. 2018; Jiang et al. 2023,
    Sect. 2.3). Outside r1, it is the halo before. The baryons are taken
    spherical, of their mass within spheres (as the contractions take
    them; Jiang et al. 2023 take a Hernquist sphere). Without scatterings
    within the age, the halo is that before. (Jiang et al. 2023 take v
    from a Dekel-Zhao profile fitted to the halo before, of dark matter
    alone, for speed.)

    Its parameters: 'sigma_m', the cross-section per unit mass (cm^2/g),
    and 't_age', the age of the halo (Gyr). The core is integrated by
    Runge-Kutta in ln r (steps of 0.05, from 1e-4 r1, to some 1e-6) and
    matched by Newton iterations (to 1e-10, from a grid search of the
    lowest density that matches); some 0.1 s.

    Parameters
    ----------
    contraction : HaloResponse, optional
        The contraction of the halo before by the baryons (Jiang et al.
        2023: 'gnedin04'); none by default (Kaplinghat et al. 2016).

    Raises
    ------
    ConfigError
        If the contraction is that of SIDM.
    """

    @staticmethod
    def type() -> str:
        return 'sidm'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'ResponseSIDM':
        parseutils.load_option_and_update_info(
            halo_response_parser, info, 'contraction')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = super().dump()
        if self._contraction is not None:
            info['contraction'] = halo_response_parser.dump(
                self._contraction)
        return info

    def __init__(self, contraction: HaloResponse | None = None):
        check_options(ResponseSIDM.__init__, locals())
        if isinstance(contraction, ResponseSIDM):
            raise ConfigError(
                "the contraction of the halo before its SIDM core is not "
                "SIDM")
        self._contraction = contraction

    def contraction(self) -> HaloResponse | None:
        """Return the contraction of the halo before, if any."""
        return self._contraction

    def pdescs(self) -> dict[str, ParamDesc]:
        pdescs = {} if self._contraction is None \
            else self._contraction.pdescs()
        return pdescs | dict(
            sigma_m=ParamScalarDesc(
                desc="the cross-section of SIDM per unit mass",
                min=0, unit='cm2 / g'),
            t_age=ParamScalarDesc(
                desc="the age of the halo", min=0, unit='Gyr'))

    def check_profile(self, profile: SpheroidProfile) -> None:
        if self._contraction is not None:
            self._contraction.check_profile(profile)

    def check_baryons(self, baryons: Sequence[str]) -> None:
        # (of dark matter alone without baryons)
        if self._contraction is not None:
            self._contraction.check_baryons(baryons)

    def vcirc2(
            self,
            R: np.ndarray,
            halo_vcirc2: _VCirc2,
            baryons_mass: _Mass,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        before = self._before(
            halo_vcirc2, baryons_mass, profile, params, context)
        core = self.core(
            halo_vcirc2, baryons_mass, profile, params, context)
        out = before(R)
        if core is not None:
            inside = (R > 0) & (R < core.r1)
            out[inside] = G * core.mass(R[inside]) / R[inside]
        return out

    def core(
            self,
            halo_vcirc2: _VCirc2,
            baryons_mass: _Mass,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> SIDMCore | None:
        """
        Return the isothermal core of the halo (see vcirc2 for the
        arguments); None without scatterings within the age of the halo.

        Raises
        ------
        ValueError
            If the halo scatters at every radius (to 1e4 kpc), or no core
            matches it at r1.
        """
        rate = (4 / np.sqrt(np.pi) * params['sigma_m'] * _SIGMA_UNIT
                * params['t_age'] * _TIME_UNIT)
        return _sidm_core(
            self._before(halo_vcirc2, baryons_mass, profile, params,
                         context),
            baryons_mass, rate)

    def _before(
            self,
            halo_vcirc2: _VCirc2,
            baryons_mass: _Mass,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> _VCirc2:
        """
        Return the squared circular velocity of the halo before its core:
        contracted by the baryons if it is.
        """
        if self._contraction is None:
            return halo_vcirc2
        contraction = self._contraction

        def before(radii: np.ndarray) -> np.ndarray:
            return contraction.vcirc2(
                radii, halo_vcirc2, baryons_mass, profile, params, context)

        return before


def _sidm_core(
        before: _VCirc2,
        baryons_mass: _Mass,
        rate: float
) -> SIDMCore | None:
    """
    Return the isothermal core of SIDM of a halo (see ResponseSIDM), of
    the squared circular velocity of the halo before, the mass of the
    baryons within spheres and rate = (4 / sqrt(pi)) sigma_m t_age
    (kpc^3/Msun / (km/s)); None
    without scatterings within the age.

    Raises
    ------
    ValueError
        If the halo scatters at every radius of the grid, or no core
        matches the halo at r1.
    """
    radii = _SIDM_RADII
    lnr = np.log(radii)
    mass = scipy.interpolate.CubicSpline(lnr, before(radii) * radii / G)
    baryons = baryons_mass(radii)
    density = mass(lnr, 1) / (4 * np.pi * radii ** 3)
    # (rho v^2 = int_r^inf rho G (M + M_b) / r'^2 dr', in ln r')
    integrand = density * G * (mass(lnr) + baryons) / radii
    pressure = scipy.integrate.cumulative_trapezoid(
        integrand[::-1], -lnr[::-1], initial=0)[::-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        dispersion = np.sqrt(pressure / density)
    scatterings = rate * density * dispersion
    # (r1: the outermost radius of one scattering in the age)
    once = np.flatnonzero(scatterings >= 1)
    if once.size == 0:
        return None
    last = once[-1]
    if last + 1 == radii.size:
        raise ValueError(
            "the halo scatters within its age at every radius: it has no "
            "SIDM core of an outer halo")
    t = np.log(scatterings[last]) / np.log(
        scatterings[last] / scatterings[last + 1])
    ln_r1 = lnr[last] + t * (lnr[last + 1] - lnr[last])
    r1 = float(np.exp(ln_r1))
    rho1 = float(np.exp(np.interp(ln_r1, lnr, np.log(density))))
    mass1 = float(mass(ln_r1))
    v1 = float(np.interp(ln_r1, lnr, dispersion))
    core = _IsothermalCore(r1, baryons_mass)
    rho0, v0 = np.exp(_sidm_match(core, rho1, mass1, v1))
    return SIDMCore(
        r1, float(rho0), float(v0), core.mass_function(rho0, v0))


class _IsothermalCore:
    """
    The isothermal cores of SIDM to the radius r1 in the potential of the
    baryons (of their mass within spheres): their density at r1 and
    their mass within the radii, of their central density rho0 and
    velocity dispersion v0 (arrays of several), by Runge-Kutta in ln r.
    """

    def __init__(self, r1: float, baryons_mass: _Mass):
        nsteps = int(np.ceil(-np.log(_SIDM_START) / _SIDM_STEP))
        # (the radii of the steps and of their middles)
        self.radii = r1 * np.exp(np.linspace(
            np.log(_SIDM_START), 0, 2 * nsteps + 1))
        self.baryons = baryons_mass(self.radii)
        self.step = -np.log(_SIDM_START) / nsteps

    def integrate(
            self, rho0: np.ndarray, v0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the density at r1 and the masses within the radii of the
        steps (one row each) of cores of rho0 and v0.
        """
        r, mb = self.radii, self.baryons

        def slopes(i: int, y: np.ndarray, m: np.ndarray):
            # (y = (Phi - Phi(0)) / v0^2, m the mass of the core)
            return (G * (m + mb[i]) / (r[i] * v0 ** 2),
                    4 * np.pi * r[i] ** 3 * rho0 * np.exp(-y))

        # (uniform within the first radius: Phi - Phi(0) = G M / (2 r))
        m = 4 * np.pi / 3 * rho0 * r[0] ** 3
        y = G * (m + mb[0]) / (2 * r[0] * v0 ** 2)
        masses = [m]
        h = self.step
        for i in range(0, r.size - 1, 2):
            k1 = slopes(i, y, m)
            k2 = slopes(i + 1, y + h / 2 * k1[0], m + h / 2 * k1[1])
            k3 = slopes(i + 1, y + h / 2 * k2[0], m + h / 2 * k2[1])
            k4 = slopes(i + 2, y + h * k3[0], m + h * k3[1])
            y = y + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            m = m + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            masses.append(m)
        return rho0 * np.exp(-y), np.array(masses)

    def mass_function(
            self, rho0: float, v0: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return the mass within the radii (< r1) of the core of rho0 and v0:
        uniform within the first step, interpolated in ln r beyond.
        """
        _, masses = self.integrate(np.array([rho0]), np.array([v0]))
        steps = self.radii[::2]
        spline = scipy.interpolate.CubicSpline(
            np.log(steps), np.log(masses[:, 0]))

        def mass(R: np.ndarray) -> np.ndarray:
            return np.where(
                R < steps[0], 4 * np.pi / 3 * rho0 * R ** 3,
                np.exp(spline(np.log(np.maximum(R, steps[0])))))

        return mass


def _sidm_match(
        core: _IsothermalCore, rho1: float, mass1: float, v1: float
) -> tuple[float, float]:
    """
    Return ln rho0 and ln v0 of the isothermal core whose density and
    mass at r1 are those of the halo before (rho1, mass1; v1 its velocity
    dispersion there), of the lowest rho0 that matches: on a grid, for
    each rho0, the v0 of the density at r1 (which grows with v0), and the
    first rho0 up from rho1 whose mass at r1 matches too; then refined by
    Newton iterations.

    Raises
    ------
    ValueError
        If no core matches.
    """
    def misfits(ln_rho0: np.ndarray, ln_v0: np.ndarray) -> np.ndarray:
        density, masses = core.integrate(np.exp(ln_rho0), np.exp(ln_v0))
        return np.array([np.log(density / rho1),
                         np.log(masses[-1] / mass1)])

    ln_rho = np.log(rho1) + np.linspace(
        0, np.log(_SIDM_DENSITIES[0]), _SIDM_DENSITIES[1])
    ln_v = np.log(v1) + np.linspace(
        *np.log(_SIDM_DISPERSIONS[:2]), _SIDM_DISPERSIONS[2])
    grid_rho, grid_v = np.meshgrid(ln_rho, ln_v, indexing='ij')
    density = misfits(grid_rho.ravel(), grid_v.ravel())[0].reshape(
        grid_rho.shape)
    matched = np.full(ln_rho.size, np.nan)
    for j, row in enumerate(density):
        up = np.flatnonzero((row[:-1] < 0) & (row[1:] >= 0))
        if up.size:
            k = up[0]
            matched[j] = ln_v[k] + (ln_v[k + 1] - ln_v[k]) * (
                -row[k] / (row[k + 1] - row[k]))
    valid = np.flatnonzero(np.isfinite(matched))
    mass = np.full(ln_rho.size, np.nan)
    mass[valid] = misfits(ln_rho[valid], matched[valid])[1]
    crossing = np.flatnonzero(
        np.isfinite(mass[:-1]) & np.isfinite(mass[1:])
        & (mass[:-1] < 0) & (mass[1:] >= 0))
    if crossing.size == 0:
        raise ValueError(
            "no isothermal SIDM core matches the density and the mass of "
            "the halo at r1")
    j = crossing[0]
    t = mass[j] / (mass[j] - mass[j + 1])
    x = np.array([ln_rho[j] + t * (ln_rho[j + 1] - ln_rho[j]),
                  matched[j] + t * (matched[j + 1] - matched[j])])
    # (Newton, of the Jacobian by forward differences, in one integration)
    h = 1e-6
    for _ in range(_SIDM_ITERATIONS):
        f = misfits(x[0] + np.array([0, h, 0]), x[1] + np.array([0, 0, h]))
        jacobian = (f[:, 1:] - f[:, :1]) / h
        step = np.linalg.solve(jacobian, -f[:, 0])
        x = x + step
        if np.max(np.abs(step)) < _SIDM_TOLERANCE:
            return float(x[0]), float(x[1])
    raise ValueError(
        "the isothermal SIDM core did not match the halo at r1 to the "
        "tolerance")


def _contracted(
        R: np.ndarray,
        halo_vcirc2: _VCirc2,
        baryons_mass: _Mass,
        a0: float = 1.0,
        w: float = 1.0,
        r_d: float = 1.0
) -> np.ndarray:
    """
    Return the squared circular velocity at the radii R of a halo
    contracted adiabatically by baryons, of the orbit-averaged radii
    r_bar = a0 r_D (r / r_D)^w (Gnedin et al. 2004; r_bar = r, a0 = w =
    1: Blumenthal et al. 1986): the dark matter within r_bar' of the shell
    of the radius r' ends within r_bar of r, with r' M(r_bar') = r (M(
    r_bar') + M_b(r_bar)) (Burkert et al. 2010, Eq. 19, with the radii of
    Gnedin et al.), of the masses within spheres of the halo before (M =
    v^2 r / G, of halo_vcirc2) and of the baryons (baryons_mass), functions
    of the radius: at R = r_bar, v^2 = G M(r_bar') / R.
    """
    out = np.zeros_like(R)
    valid = R > 0
    averaged = R[valid]
    # (G M of both)
    baryons = G * baryons_mass(averaged)
    # (the radius of the shell whose orbit-averaged radius is R)
    r = r_d * (averaged / (a0 * r_d)) ** (1 / w)
    lo = np.log(r)
    hi = lo + _CONTRACTION_RANGE
    for _ in range(_CONTRACTION_BISECTIONS):
        mid = (lo + hi) / 2
        x = np.exp(mid)
        x_averaged = a0 * r_d * (x / r_d) ** w
        halo = halo_vcirc2(x_averaged) * x_averaged
        inside = x * halo < r * (halo + baryons)
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    x_averaged = a0 * r_d * (np.exp((lo + hi) / 2) / r_d) ** w
    out[valid] = halo_vcirc2(x_averaged) * x_averaged / averaged
    return out


halo_response_parser = parseutils.TypedParser(HaloResponse, [
    ResponseBlumenthal86,
    ResponseGnedin04,
    ResponseCautun20,
    ResponseSIDM])
