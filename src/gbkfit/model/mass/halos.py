"""
The profiles of dark matter halos (see spheroids.Halo, and halo_params
for their parameters): spherical densities of the values of their
parameters and of the overdensity of their halo (see HaloProfile.at).
Most are a density rho_s f(r / r_s) of a given shape f, with its mass
shape m(x) = int_0^x f(t) t^2 dt (see ScaledHaloProfile); the nodewise
halo is a density given at radial nodes (ProfileNodewiseHalo).
"""

import abc
from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import scipy.special

from gbkfit.config import TypedParser, TypedSerializable, Unit, check_options
from gbkfit.params import ModelEvaluationError, ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from . import _numerics
from .base import MassContext
from .halo_params import (
    DARK_FRACTIONS, ParameterisationChoice, halo_scale_radius, halo_scales,
    parameterisation_pdescs)
from .units import G


__all__ = [
    'HaloProfile',
    'ProfileBurkert',
    'ProfileCoreNFW',
    'ProfileDekelZhao',
    'ProfileEinasto',
    'ProfileNFW',
    'ProfileNodewiseHalo',
    'ProfilePseudoIsothermal',
    'ProfileTruncatedNFW',
    'ProfileZhao',
    'ScaledHaloProfile',
    'SphericalHalo',
    'halo_profile_parser'
]


# Below this x, the mass shapes use their series (the closed forms lose
# their precision by cancellation)
_X_SERIES = 1e-3

# Below this |y|, expm1(y) / y is taken as its series (see _power_integral)
_Y_SERIES = 1e-8


class SphericalHalo(abc.ABC):
    """
    A spherical halo at the values of its parameters (see HaloProfile.at):
    its density, its mass within spheres and its potential at the
    spherical radii r (kpc). It is the profile of its spheroid (see
    shapes): it takes the values of the parameters of the spheroid and
    its context, as profiles do (see profiles.SpheroidProfile), and uses
    neither, since it has no parameters left.
    """

    @abc.abstractmethod
    def density(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        """Return its density (Msun/kpc^3) at the spherical radii r."""
        pass

    @abc.abstractmethod
    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        """Return its mass (Msun) within the spherical radii r."""
        pass

    def potential(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        """
        Return its potential ((km/s)^2) at the spherical radii r: 0 at
        infinity, or at the centre if it is infinite at infinity (see
        zero_at_infinity); here by quadrature.
        """
        return _numerics.spherical_potential(
            self.density, self.enclosed_mass, np.asarray(r, dtype=float))

    def zero_at_infinity(self) -> bool:
        """
        Check whether its potential is 0 at infinity (here), or at the
        centre (of a mass that grows so fast that its potential at
        infinity is infinite: the pseudo-isothermal halo).
        """
        return True


class HaloProfile(TypedSerializable, abc.ABC):
    """
    The profile of a dark matter halo (see spheroids.Halo): a spherical
    density of the values of its parameters and of the overdensity of
    its halo (see at).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    def parameterisation(self) -> str | None:
        """
        Return its parameterisation (see halo_params), if it has one;
        here, it has none.
        """
        return None

    @abc.abstractmethod
    def at(
            self, params: ParamValues, overdensity_density: float
    ) -> SphericalHalo:
        """
        Return the halo of the values of its parameters and of D rho_ref
        (Msun/kpc^3), the density of the overdensity of its halo at the
        epoch of its galaxy (see halo_params.Overdensity).

        Raises
        ------
        ModelEvaluationError
            If the values give no halo.
        """
        pass


class ScaledHaloProfile(HaloProfile, abc.ABC):
    """
    A halo of density rho_s f(r / r_s): its parameters are those of its
    parameterisation, and those of the shape of its profile.

    Parameters
    ----------
    parameterisation : str, optional
        Its parameterisation: 'mass_concentration' (the default),
        'velocity_concentration', 'density_radius',
        'dark_fraction_concentration' or 'dark_fraction_radius' (see
        halo_params; its halo solves those of a dark fraction, see
        spheroids.Halo).

    Raises
    ------
    ValueError
        If the parameterisation is not valid.
    """

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(parameterisation=self._parameterisation)

    def __init__(
            self,
            parameterisation: ParameterisationChoice = 'mass_concentration'
    ):
        check_options(ScaledHaloProfile.__init__, locals())
        self._parameterisation = parameterisation

    def pdescs(self) -> dict[str, ParamDesc]:
        return parameterisation_pdescs(self._parameterisation) \
            | self._shape_pdescs()

    def parameterisation(self) -> str:
        return self._parameterisation

    def scales(
            self,
            params: ParamValues,
            overdensity_density: float,
            parameterisation: str | None = None
    ) -> tuple[float, float]:
        """
        Return its scale density (Msun/kpc^3) and radius (kpc), of the
        values of the parameters of its parameterisation, or of another
        one (not of a dark fraction), and D rho_ref (see at).
        """
        parameterisation = parameterisation or self._parameterisation
        r_s = halo_scale_radius(parameterisation, params, overdensity_density)
        return halo_scales(
            parameterisation, params, overdensity_density,
            lambda c: self._scaling_mass(c, params, r_s))

    def at(
            self, params: ParamValues, overdensity_density: float
    ) -> SphericalHalo:
        """
        See HaloProfile.at; not of a dark fraction, which its halo solves
        (see at_scales).

        Raises
        ------
        ValueError
            If its parameterisation is that of a dark fraction.
        """
        if self._parameterisation in DARK_FRACTIONS:
            raise ValueError(
                f"the halo of the parameterisation "
                f"{self._parameterisation!r} is solved for by its halo")
        return self.at_scales(
            *self.scales(params, overdensity_density), params)

    def at_scales(
            self, rho_s: float, r_s: float, params: ParamValues
    ) -> SphericalHalo:
        """
        Return the halo of the scale density rho_s (Msun/kpc^3) and
        radius r_s (kpc), and of the values of the parameters of the shape
        of its profile.
        """
        return _ScaledHalo(self, rho_s, r_s, self._shape_values(params, r_s))

    def _scaling_mass(
            self, c: float, params: ParamValues, r_s: float) -> float:
        """
        Return m(c), the mass shape at the concentration c of the halo of
        the scale radius r_s (see halo_params): that of its profile.
        """
        return float(self._mass_shape(
            np.array(c, dtype=float), self._shape_values(params, r_s)))

    def _shape_values(
            self, params: ParamValues, r_s: float) -> dict[str, float]:
        """
        Return the values of the parameters of the shape of its profile,
        by name, of a halo of the scale radius r_s (kpc; for the shapes
        with radii, which take them in units of r_s).
        """
        return {name: params[name] for name in self._shape_pdescs()}

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return the parameters of the shape of its profile, by name."""
        return {}

    @abc.abstractmethod
    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        """Return f(x), for the values of the shape parameters."""
        pass

    @abc.abstractmethod
    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        """Return m(x), for the values of the shape parameters."""
        pass

    def _potential_shape(
            self, x: np.ndarray, shape: dict[str, float]
    ) -> np.ndarray | None:
        """
        Return the potential in units of 4 pi G rho_s r_s^2 at x, if it
        has a closed form (None: computed by quadrature).
        """
        return None

    def _zero_at_infinity(self) -> bool:
        """
        Check whether its potential is 0 at infinity (here), or at the
        centre (see SphericalHalo.zero_at_infinity).
        """
        return True


class _ScaledHalo(SphericalHalo):
    """
    A halo of density rho_s f(r / r_s) at the values of its parameters
    (see ScaledHaloProfile.at_scales).
    """

    def __init__(
            self,
            profile: ScaledHaloProfile,
            rho_s: float,
            r_s: float,
            shape: dict[str, float]
    ):
        self._profile = profile
        self._rho_s = rho_s
        self._r_s = r_s
        self._shape = shape

    def density(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        return self._rho_s * self._profile._density_shape(
            np.asarray(r, dtype=float) / self._r_s, self._shape)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        return 4 * np.pi * self._rho_s * self._r_s ** 3 \
            * self._profile._mass_shape(
                np.asarray(r, dtype=float) / self._r_s, self._shape)

    def potential(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        phi = self._profile._potential_shape(r / self._r_s, self._shape)
        if phi is None:
            return super().potential(r)
        return 4 * np.pi * G * self._rho_s * self._r_s ** 2 * phi

    def zero_at_infinity(self) -> bool:
        return self._profile._zero_at_infinity()


class ProfileNFW(ScaledHaloProfile):
    """
    The halo of Navarro, Frenk & White (1997): a density that falls as
    r^-1 inside its scale radius and as r^-3 outside. Its shape is f(x) =
    1 / (x (1 + x)^2), m(x) = ln(1 + x) - x / (1 + x), and its potential
    -4 pi G rho_s r_s^2 ln(1 + x) / x.

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'nfw'

    @staticmethod
    def type_title() -> str:
        return 'NFW'

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        return _nfw_density_shape(x)

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        return _nfw_mass_shape(x)

    def _potential_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        x_safe = np.maximum(x, _X_SERIES)
        return -np.where(
            x < _X_SERIES, 1 - x / 2 + x ** 2 / 3 - x ** 3 / 4,
            np.log1p(x_safe) / x_safe)


class ProfileBurkert(ScaledHaloProfile):
    """
    The cored halo of Burkert (1995): a density that is flat inside its
    core radius and falls as r^-3 outside. Its shape is f(x) = 1 / ((1 +
    x) (1 + x^2)), m(x) = (ln((1 + x)^2 (1 + x^2)) - 2 arctan(x)) / 4. Its
    scale radius r_s is the core radius r_0, and its concentration r_D /
    r_0.

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'burkert'

    @staticmethod
    def type_title() -> str:
        return 'Burkert'

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        return 1 / ((1 + x) * (1 + x ** 2))

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        x_safe = np.maximum(x, _X_SERIES)
        return np.where(
            x < _X_SERIES,
            x ** 3 / 3 - x ** 4 / 4,
            (2 * np.log1p(x_safe) + np.log1p(x_safe ** 2)
             - 2 * np.arctan(x_safe)) / 4)


class ProfilePseudoIsothermal(ScaledHaloProfile):
    """
    The pseudo-isothermal sphere: a density that is flat inside its core
    radius and falls as r^-2 outside (of a flat rotation curve). Its shape
    is f(x) = 1 / (1 + x^2), m(x) = x - arctan(x). Its scale radius r_s is
    the core radius r_c, and its concentration r_D / r_c. Its mass grows
    without bound, so its potential at infinity is infinite: it is given
    relative to the centre, 4 pi G rho_s r_s^2 (ln(1 + x^2) / 2 +
    arctan(x) / x - 1).

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'pseudo_isothermal'

    @staticmethod
    def type_title() -> str:
        return 'Pseudo-isothermal'

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        return 1 / (1 + x ** 2)

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        x_safe = np.maximum(x, _X_SERIES)
        return np.where(
            x < _X_SERIES,
            x ** 3 / 3 - x ** 5 / 5 + x ** 7 / 7,
            x_safe - np.arctan(x_safe))

    def _potential_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        x_safe = np.maximum(x, _X_SERIES)
        return np.where(
            x < _X_SERIES, x ** 2 / 6 - x ** 4 / 20,
            np.log1p(x_safe ** 2) / 2 + np.arctan(x_safe) / x_safe - 1)

    def _zero_at_infinity(self) -> bool:
        return False


class ProfileEinasto(ScaledHaloProfile):
    """
    The halo of Einasto (1965): a density whose logarithmic slope steepens
    with radius, as r^alpha. Its shape is f(x) = exp(-(2 / a) (x^a - 1)),
    with a its shape (parameter alpha), and m(x) = e^(2/a) / a (a /
    2)^(3/a) gamma(3 / a, (2 / a) x^a) (gamma: the lower incomplete gamma
    function). Its scale radius r_s is r_-2, where the logarithmic slope
    of its density is -2; its concentration is r_D / r_-2. Its potential
    has a closed form, with the upper incomplete gamma function.

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'einasto'

    @staticmethod
    def type_title() -> str:
        return 'Einasto'

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(alpha=ParamScalarDesc(
            title='Shape parameter', desc="the shape of the profile", min=0,
            unit='', default=0.17))

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        a = shape['alpha']
        return np.exp(-(2 / a) * (x ** a - 1))

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        a = shape['alpha']
        s = 3 / a
        return (np.exp(2 / a) / a * (a / 2) ** s * scipy.special.gamma(s)
                * scipy.special.gammainc(s, (2 / a) * x ** a))

    def _potential_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        # -(m(x) / x + int_x^inf f(t) t dt)
        a = shape['alpha']
        s = 2 / a
        outer = (np.exp(2 / a) / a * (a / 2) ** s * scipy.special.gamma(s)
                 * scipy.special.gammaincc(s, (2 / a) * x ** a))
        mass_term = np.divide(
            self._mass_shape(x, shape), x, out=np.zeros_like(x), where=x > 0)
        return -(mass_term + outer)


class ProfileZhao(ScaledHaloProfile):
    """
    The halo of Zhao (1996): a density of a free inner slope, outer slope
    and sharpness of the transition between them. Its shape is f(x) =
    x^-g (1 + x^a)^(-(b - g) / a), of the shape parameters alpha (a, the
    sharpness of the transition), beta (b, the outer slope) and gamma (g,
    the inner slope, below 3), and m(x) = x^(3 - g) / (3 - g) 2F1((3 - g)
    / a, (b - g) / a; (3 - g) / a + 1; -x^a). The generalised NFW halo is
    the case a = 1, b = 3, and NFW a = 1, b = 3, g = 1. Its scale radius
    r_s is that of this formula (not r_-2), and its concentration r_D /
    r_s. Its potential is finite for b > 2.

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'zhao'

    @staticmethod
    def type_title() -> str:
        return 'Zhao'

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            alpha=ParamScalarDesc(
                title='Transition sharpness',
                desc="the sharpness of the transition", min=0,
                unit='', default=1),
            beta=ParamScalarDesc(
                title='Outer slope', desc="the outer slope", unit='',
                default=3),
            gamma=ParamScalarDesc(
                title='Inner slope', desc="the inner slope", max=3,
                unit='', default=1))

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        a, b, g = shape['alpha'], shape['beta'], shape['gamma']
        return x ** -g * (1 + x ** a) ** (-(b - g) / a)

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        a, b, g = shape['alpha'], shape['beta'], shape['gamma']
        p = (3 - g) / a
        return x ** (3 - g) / (3 - g) * scipy.special.hyp2f1(
            p, (b - g) / a, p + 1, -x ** a)


class ProfileDekelZhao(ScaledHaloProfile):
    """
    The halo of Dekel et al. 2017 and Freundlich et al. 2020: a cusp or a
    core of a free inner slope, falling as r^-3.5 outside. Its mean
    density within r is rho_s / (x^g (1 + x^(1/2))^(2 (3 - g))): f(x) = (1
    - g / 3) / (x^g (1 + x^(1/2))^(7 - 2 g)), of the inner slope g (its
    shape parameter gamma, below 3; a of the papers) and the outer slope
    -3.5, and m(x) = chi^(2 (3 - g)) / 3, with chi = x^(1/2) / (1 +
    x^(1/2)). Its scale radius r_s is r_c of the papers (its slope is -2
    at ((2 - g) / 1.5)^2 r_s), and its concentration r_D / r_s. Its
    potential has a closed form for g < 2: -(2 / 3) ((1 - chi^(4 - 2 g)) /
    (4 - 2 g) - (1 - chi^(5 - 2 g)) / (5 - 2 g)) in units of 4 pi G rho_s
    r_s^2.

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'dekel_zhao'

    @staticmethod
    def type_title() -> str:
        return 'Dekel-Zhao'

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(gamma=ParamScalarDesc(
            title='Inner slope', desc="the inner slope", max=3, unit='',
            default=1))

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        a = shape['gamma']
        return (1 - a / 3) / (x ** a * (1 + np.sqrt(x)) ** (7 - 2 * a))

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        chi = np.sqrt(x) / (1 + np.sqrt(x))
        return chi ** (2 * (3 - shape['gamma'])) / 3

    def _potential_shape(
            self, x: np.ndarray, shape: dict[str, float]
    ) -> np.ndarray | None:
        a = shape['gamma']
        if not a < 2:
            return None
        chi = np.sqrt(x) / (1 + np.sqrt(x))
        return -2 / 3 * ((1 - chi ** (4 - 2 * a)) / (4 - 2 * a)
                         - (1 - chi ** (5 - 2 * a)) / (5 - 2 * a))


class ProfileCoreNFW(ScaledHaloProfile):
    """
    The cored NFW halo of Read, Agertz & Collins (2016, coreNFW): an NFW
    halo whose central cusp is flattened into a core. Its mass is that of
    the NFW halo of its parameters times t^n, t = tanh(r / r_c), of the
    core radius r_c (kpc; parameter rc) and the strength of the core n (0:
    the NFW halo, 1: a full core). With x_c = r_c / r_s: m(x) = m_NFW(x)
    t^n, and f(x) = f_NFW(x) t^n + n t^(n - 1) (1 - t^2) m_NFW(x) / (x^2
    x_c). Its scale radius and concentration are those of the NFW halo,
    whose mass within r_D is M_D (the cored halo has t(r_D)^n of it).

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'core_nfw'

    @staticmethod
    def type_title() -> str:
        return 'Cored NFW'

    def _scaling_mass(
            self, c: float, params: ParamValues, r_s: float) -> float:
        # (that of the NFW halo it cores)
        return float(_nfw_mass_shape(np.array(c, dtype=float)))

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            rc=ParamScalarDesc(
                title='Core radius', desc="the core radius r_c", min=0,
                unit='kpc', default=1),
            n=ParamScalarDesc(
                title='Core strength', desc="the strength of the core",
                min=0, max=1, unit='', default=1))

    def _shape_values(
            self, params: ParamValues, r_s: float) -> dict[str, float]:
        return dict(xc=params['rc'] / r_s, n=params['n'])

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        xc, n = shape['xc'], shape['n']
        if xc == 0:
            return _nfw_density_shape(x)
        t = np.tanh(x / xc)
        # (m_NFW(x) / x^2, 1 / 2 at the centre)
        mass = np.divide(
            _nfw_mass_shape(x), x ** 2, out=np.full_like(x, 0.5),
            where=x > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            core = np.where(n > 0, n * t ** (n - 1), 0) * (1 - t ** 2)
        return _nfw_density_shape(x) * t ** n + core * mass / xc

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        if shape['xc'] == 0:
            return _nfw_mass_shape(x)
        return _nfw_mass_shape(x) * np.tanh(x / shape['xc']) ** shape['n']


class ProfileTruncatedNFW(ScaledHaloProfile):
    """
    An NFW halo cut off exponentially beyond a truncation radius, so that
    its mass is finite. It is the NFW halo of its parameters truncated as
    galpy's ExpTruncNFWPotential: f(x) = f_NFW(x) exp(-x / x_t), of the
    truncation radius r_t (kpc; parameter rt), x_t = r_t / r_s. With a = 1
    / x_t, its mass is m(x) = (1 + a) e^a (E1(a) - E1(a (1 + x))) - 1 +
    e^(-a x) / (1 + x) (E1: the exponential integral), finite as x grows;
    for r_t above r_s / 700. Its scale radius and concentration are those
    of the NFW halo, whose mass within r_D is M_D (more than that of the
    truncated halo).

    Parameters
    ----------
    parameterisation : optional
        See ScaledHaloProfile.
    """

    @staticmethod
    def type() -> str:
        return 'truncated_nfw'

    @staticmethod
    def type_title() -> str:
        return 'Truncated NFW'

    def _scaling_mass(
            self, c: float, params: ParamValues, r_s: float) -> float:
        # (that of the NFW halo it truncates)
        return float(_nfw_mass_shape(np.array(c, dtype=float)))

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(rt=ParamScalarDesc(
            title='Truncation radius', desc="the truncation radius r_t",
            min=0, unit='kpc', default=200))

    def _shape_values(
            self, params: ParamValues, r_s: float) -> dict[str, float]:
        return dict(a=r_s / params['rt'])

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        return _nfw_density_shape(x) * np.exp(-shape['a'] * x)

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        a = shape['a']
        x_safe = np.maximum(x, _X_SERIES)
        closed = ((1 + a) * np.exp(a) * (
            scipy.special.exp1(a) - scipy.special.exp1(a * (1 + x_safe)))
            - 1 + np.exp(-a * x_safe) / (1 + x_safe))
        # (the integral of t (1 - (2 + a) t + (3 + 2 a + a^2 / 2) t^2))
        series = (x ** 2 / 2 - (2 + a) * x ** 3 / 3
                  + (3 + 2 * a + a ** 2 / 2) * x ** 4 / 4)
        return np.where(x < _X_SERIES, series, closed)


def _nfw_density_shape(x: np.ndarray) -> np.ndarray:
    """Return f(x) of the NFW halo (see ProfileNFW)."""
    return 1 / (x * (1 + x) ** 2)


def _nfw_mass_shape(x: np.ndarray) -> np.ndarray:
    """Return m(x) of the NFW halo (see ProfileNFW)."""
    x_safe = np.maximum(x, _X_SERIES)
    return np.where(
        x < _X_SERIES,
        x ** 2 / 2 - 2 * x ** 3 / 3 + 3 * x ** 4 / 4 - 4 * x ** 5 / 5,
        np.log1p(x_safe) - x_safe / (1 + x_safe))


class ProfileNodewiseHalo(HaloProfile):
    """
    A halo of a density given at radial nodes: 10^log_rho (Msun/kpc^3) at
    each node (kpc), a power law between them (linear in log r and log
    rho), continued within the first node and beyond the last; positive
    by construction. Its mass and its potential are in closed form. Its
    mass at the centre must be finite: its density there must fall less
    steeply than r^-3 (or an error, ModelEvaluationError). Its potential
    is 0 at infinity if its density beyond the last node falls more
    steeply than r^-2, and from the centre otherwise (see
    SphericalHalo.zero_at_infinity).

    Parameters
    ----------
    nodes : Sequence of float
        The radii (kpc) of its nodes: at least two, positive, increasing.

    Raises
    ------
    ValueError
        If the nodes are not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nodewise'

    @staticmethod
    def type_title() -> str:
        return 'Nodewise'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(nodes=self._nodes.tolist())

    def __init__(self, nodes: Annotated[Sequence[float], Unit('kpc')]):
        check_options(ProfileNodewiseHalo.__init__, locals())
        nodes = np.asarray(nodes, dtype=float)
        if (nodes.ndim != 1 or len(nodes) < 2 or not np.all(np.isfinite(nodes))
                or nodes[0] <= 0 or np.any(np.diff(nodes) <= 0)):
            raise ValueError(
                f"the nodes of a nodewise halo must be at least two radii, "
                f"positive, increasing; they are {nodes.tolist()}")
        self._nodes = nodes

    def nodes(self) -> np.ndarray:
        """Return the radii (kpc) of its nodes."""
        return self._nodes

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(log_rho=ParamVectorDesc(
            len(self._nodes), title='Log density',
            desc="log10 of the density at each node",
            unit='dex(solMass / kpc3)', default=7))

    def at(
            self, params: ParamValues, overdensity_density: float
    ) -> SphericalHalo:
        log_rho = np.asarray(params['log_rho'], dtype=float)
        slope = (log_rho[1] - log_rho[0]) \
            / np.log10(self._nodes[1] / self._nodes[0])
        if not slope > -3:
            raise ModelEvaluationError(
                f"the nodewise halo has an infinite mass at its centre: its "
                f"density within its first node falls as r^{slope:.3g}, "
                f"r^-3 or steeper")
        return _NodewiseHalo(self._nodes, 10 ** log_rho)


class _NodewiseHalo(SphericalHalo):
    """
    A halo of a density given at radial nodes, at the values of its
    parameters (see ProfileNodewiseHalo): on each segment [a, b], the
    density rho_ref (r / r_ref)^k, of the density rho_ref at the node
    r_ref; the first segment is [0, nodes[0]] and the last [nodes[-1],
    inf], of the power laws next to them.
    """

    def __init__(self, nodes: np.ndarray, rho: np.ndarray):
        slopes = np.diff(np.log(rho)) / np.diff(np.log(nodes))
        self._a = np.concatenate(([0.0], nodes))
        self._b = np.concatenate((nodes, [np.inf]))
        self._ref = np.concatenate((nodes[:1], nodes))
        self._rho = np.concatenate((rho[:1], rho))
        self._k = np.concatenate((slopes[:1], slopes, slopes[-1:]))

    def density(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        rr = np.asarray(r, dtype=float)[..., None]
        on = (rr >= self._a) & (rr < self._b)
        with np.errstate(divide='ignore'):
            values = self._rho * (rr / self._ref) ** self._k
        return np.sum(np.where(on, values, 0), axis=-1)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        # 4 pi rho_ref r_ref^3 int t^(k + 2) dt over each segment, from
        # a / r_ref to x / r_ref, x = r within it (a before, b beyond)
        x = np.clip(np.asarray(r, dtype=float)[..., None], self._a, self._b)
        return 4 * np.pi * np.sum(
            self._rho * self._ref ** 3 * self._spans(self._a, x, 3), axis=-1)

    def potential(
            self,
            r: npt.ArrayLike,
            params: ParamValues | None = None,
            context: MassContext | None = None
    ) -> np.ndarray:
        # -G (M(<r) / r + 4 pi int_r^inf rho s ds) from infinity, and G (4
        # pi int_0^r rho s ds - M(<r) / r) from the centre (of the same
        # gradient G M(<r) / r^2, and 0 at the centre), each integral 4 pi
        # rho_ref r_ref^2 int t^(k + 1) dt over the segments
        r = np.asarray(r, dtype=float)
        x = np.clip(r[..., None], self._a, self._b)
        if self.zero_at_infinity():
            spans = self._spans(x, self._b, 2)
        else:
            spans = self._spans(self._a, x, 2)
        outer = 4 * np.pi * np.sum(
            self._rho * self._ref ** 2 * spans, axis=-1)
        mass = self.enclosed_mass(r)
        inner = np.divide(mass, r, out=np.zeros_like(r), where=r > 0)
        if self.zero_at_infinity():
            return -G * (inner + outer)
        return G * (outer - inner)

    def zero_at_infinity(self) -> bool:
        # (of a finite potential at infinity: beyond its last node, a
        # density that falls more steeply than r^-2)
        return self._k[-1] < -2

    def _spans(
            self, lower: np.ndarray, upper: np.ndarray, power: int
    ) -> np.ndarray:
        """
        Return int t^(k + power - 1) dt from lower / r_ref to upper / r_ref
        on each segment (0 where they meet).
        """
        p = self._k + power
        with np.errstate(divide='ignore', invalid='ignore'):
            span = _power_integral(np.log(upper / self._ref), p) \
                - _power_integral(np.log(lower / self._ref), p)
        return np.where(np.broadcast_to(lower == upper, span.shape), 0, span)


def _power_integral(L: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Return int_1^(e^L) t^(p - 1) dt = (e^(p L) - 1) / p, of L, the log
    of the upper limit, and p (broadcast together): L (e^y - 1) / y, y =
    p L, which tends to L as p goes to 0; at L = -inf (an upper limit of
    0), -1 / p for p > 0 (and -inf otherwise), and at L = inf, -1 / p for
    p < 0 (and inf otherwise).
    """
    y = p * L
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        ratio = np.where(
            np.abs(y) < _Y_SERIES, 1 + y / 2, np.expm1(y) / y)
        return np.select(
            [np.isneginf(L), np.isposinf(L)],
            [np.where(p > 0, -1 / p, -np.inf),
             np.where(p < 0, -1 / p, np.inf)], L * ratio)


halo_profile_parser = TypedParser(HaloProfile, [
    ProfileNFW,
    ProfileBurkert,
    ProfilePseudoIsothermal,
    ProfileEinasto,
    ProfileZhao,
    ProfileDekelZhao,
    ProfileCoreNFW,
    ProfileTruncatedNFW,
    ProfileNodewiseHalo], family='halo_profile')
