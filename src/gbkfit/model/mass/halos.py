"""
The profiles of spherical dark matter halos (see spheroids.MassSpheroid,
and halo_params for their parameters): a density rho_s f(r / r_s) of a
given shape f, and its mass shape m(x) = int_0^x f(t) t^2 dt.
"""

import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.special

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils.parseutils import ConfigError
from .base import MassContext
from .halo_params import (
    Overdensity, check_parameterisation, halo_scales,
    parameterisation_pdescs)
from .profiles import SpheroidProfile
from .units import G


__all__ = [
    'ProfileBurkert',
    'ProfileEinasto',
    'ProfileHalo',
    'ProfileNFW',
    'ProfilePseudoIsothermal',
    'ProfileZhao'
]


# Below this x, the mass shapes use their series (the closed forms lose
# their precision by cancellation)
_X_SERIES = 1e-3


class ProfileHalo(SpheroidProfile, abc.ABC):
    """
    A spherical halo of density rho_s f(r / r_s): its parameters are those
    of its parameterisation, and those of the shape of its profile.

    Parameters
    ----------
    parameters : str, optional
        Its parameterisation: 'mass_concentration' (the default),
        'velocity_concentration' or 'density_radius' (see halo_params).
    overdensity : str, optional
        The overdensity of its mass and radius, for the parameterisations
        of a mass or a velocity: e.g. '200c' (the default), '200m' or
        'vir' (see halo_params.Overdensity).

    Raises
    ------
    ConfigError
        If the parameterisation or the overdensity is not valid, or an
        overdensity is given for 'density_radius'.
    """

    def dump(self) -> dict[str, Any]:
        info = super().dump() | dict(parameters=self._parameters)
        if self._overdensity is not None:
            info.update(overdensity=str(self._overdensity))
        return info

    def __init__(
            self,
            parameters: str = 'mass_concentration',
            overdensity: str | None = None
    ):
        check_parameterisation(parameters)
        if parameters == 'density_radius':
            if overdensity is not None:
                raise ConfigError(
                    "the parameters 'density_radius' take no overdensity")
            self._overdensity = None
        else:
            self._overdensity = Overdensity(
                '200c' if overdensity is None else overdensity)
        self._parameters = parameters

    def pdescs(self) -> dict[str, ParamDesc]:
        return parameterisation_pdescs(self._parameters) | self._shape_pdescs()

    def scales(
            self, params: ParamValues, context: MassContext
    ) -> tuple[float, float]:
        """Return its scale density (Msun/kpc^3) and radius (kpc)."""
        shape = self._shape_values(params)
        overdensity = None if self._overdensity is None \
            else self._overdensity.density(context)
        return halo_scales(
            self._parameters, params, overdensity,
            lambda c: float(self._mass_shape(np.array(c, dtype=float), shape)))

    def density(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        rho_s, r_s = self.scales(params, context)
        return rho_s * self._density_shape(
            np.asarray(r, dtype=float) / r_s, self._shape_values(params))

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        rho_s, r_s = self.scales(params, context)
        return 4 * np.pi * rho_s * r_s ** 3 * self._mass_shape(
            np.asarray(r, dtype=float) / r_s, self._shape_values(params))

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        shape = self._shape_values(params)
        rho_s, r_s = self.scales(params, context)
        phi = self._potential_shape(r / r_s, shape)
        if phi is None:
            return super().potential(r, params, context)
        return 4 * np.pi * G * rho_s * r_s ** 2 * phi

    def _shape_values(self, params: ParamValues) -> dict[str, float]:
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


class ProfileNFW(ProfileHalo):
    """
    The halo of Navarro, Frenk & White (1997): f(x) = 1 / (x (1 + x)^2),
    m(x) = ln(1 + x) - x / (1 + x), and the potential -4 pi G rho_s r_s^2
    ln(1 + x) / x.

    Parameters
    ----------
    parameters, overdensity : optional
        See ProfileHalo.
    """

    @staticmethod
    def type() -> str:
        return 'nfw'

    def _density_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        return 1 / (x * (1 + x) ** 2)

    def _mass_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        x_safe = np.maximum(x, _X_SERIES)
        return np.where(
            x < _X_SERIES,
            x ** 2 / 2 - 2 * x ** 3 / 3 + 3 * x ** 4 / 4 - 4 * x ** 5 / 5,
            np.log1p(x_safe) - x_safe / (1 + x_safe))

    def _potential_shape(
            self, x: np.ndarray, shape: dict[str, float]) -> np.ndarray:
        x_safe = np.maximum(x, _X_SERIES)
        return -np.where(
            x < _X_SERIES, 1 - x / 2 + x ** 2 / 3 - x ** 3 / 4,
            np.log1p(x_safe) / x_safe)


class ProfileBurkert(ProfileHalo):
    """
    The halo of Burkert (1995): f(x) = 1 / ((1 + x) (1 + x^2)), m(x) =
    (ln((1 + x)^2 (1 + x^2)) - 2 arctan(x)) / 4. Its scale radius r_s is
    the core radius r_0, and its concentration r_D / r_0.

    Parameters
    ----------
    parameters, overdensity : optional
        See ProfileHalo.
    """

    @staticmethod
    def type() -> str:
        return 'burkert'

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


class ProfilePseudoIsothermal(ProfileHalo):
    """
    The pseudo-isothermal sphere: f(x) = 1 / (1 + x^2), m(x) = x -
    arctan(x). Its scale radius r_s is the core radius r_c, and its
    concentration r_D / r_c. Its mass grows without bound, so its
    potential is infinite.

    Parameters
    ----------
    parameters, overdensity : optional
        See ProfileHalo.
    """

    @staticmethod
    def type() -> str:
        return 'pseudo_isothermal'

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

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        raise ValueError(
            "the potential of a pseudo-isothermal sphere is infinite: its "
            "mass grows without bound")


class ProfileEinasto(ProfileHalo):
    """
    The halo of Einasto (1965): f(x) = exp(-(2 / a) (x^a - 1)), with a its
    shape (parameter alpha), and m(x) = e^(2/a) / a (a / 2)^(3/a)
    gamma(3 / a, (2 / a) x^a) (gamma: the lower incomplete gamma
    function). Its scale radius r_s is r_-2, where the logarithmic slope
    of its density is -2; its concentration is r_D / r_-2. Its potential
    has a closed form, with the upper incomplete gamma function.

    Parameters
    ----------
    parameters, overdensity : optional
        See ProfileHalo.
    """

    @staticmethod
    def type() -> str:
        return 'einasto'

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(alpha=ParamScalarDesc(
            desc="the shape of the profile", minimum=0))

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


class ProfileZhao(ProfileHalo):
    """
    The halo of Zhao (1996): f(x) = x^-g (1 + x^a)^(-(b - g) / a), of the
    shape parameters alpha (a, the sharpness of the transition), beta
    (b, the outer slope) and gamma (g, the inner slope, below 3), and
    m(x) = x^(3 - g) / (3 - g) 2F1((3 - g) / a, (b - g) / a; (3 - g) / a +
    1; -x^a). The generalised NFW halo is the case a = 1, b = 3, and NFW
    a = 1, b = 3, g = 1. Its scale radius r_s is that of this formula
    (not r_-2), and its concentration r_D / r_s. Its potential is finite
    for b > 2.

    Parameters
    ----------
    parameters, overdensity : optional
        See ProfileHalo.
    """

    @staticmethod
    def type() -> str:
        return 'zhao'

    def _shape_pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            alpha=ParamScalarDesc(
                desc="the sharpness of the transition", minimum=0),
            beta=ParamScalarDesc(desc="the outer slope"),
            gamma=ParamScalarDesc(desc="the inner slope", maximum=3))

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
