"""
The profiles of spheroids (see spheroids.MassSpheroid): spherical
densities, given as functions of the spherical radius r with their
enclosed masses; the halos are in halos.
"""

import abc
from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import scipy.special

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import Unit
from . import _numerics
from .base import MassContext
from .units import G


__all__ = [
    'ProfileNodewise',
    'ProfileSersic',
    'SpheroidProfile'
]


class SpheroidProfile(parseutils.TypedSerializable, abc.ABC):
    """
    The profile of a spheroid: its density and its enclosed mass, as
    functions of the spherical radius r (kpc), and its potential
    (numerically, unless a profile gives it exactly).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def density(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its density (Msun/kpc^3) at the spherical radii r (kpc; see
        MassComponent.vcirc2 for the other arguments).
        """
        pass

    @abc.abstractmethod
    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its mass (Msun) within the spherical radii r (kpc; see
        MassComponent.vcirc2 for the other arguments).
        """
        pass

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its projected density (Msun/kpc^2) at the radii R (kpc; see
        MassComponent.vcirc2 for the other arguments).

        Raises
        ------
        NotImplementedError
            If the profile does not give it (here).
        """
        raise NotImplementedError(
            f"the profile '{self.type()}' does not give its projected "
            f"density")

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its potential ((km/s)^2), 0 at infinity, at the spherical
        radii r (kpc; see MassComponent.vcirc2 for the other arguments).

        Raises
        ------
        ValueError
            If its potential is infinite.
        """
        # -G (M(<r) / r + 4 pi int_r^inf rho r' dr'), by quadrature
        return _numerics.spherical_potential(
            lambda x: self.density(x, params, context),
            lambda x: self.enclosed_mass(x, params, context), r)


class ProfileSersic(SpheroidProfile):
    """
    The spherical deprojection of a Sersic profile of total mass 10^log_m
    (Msun), effective radius re (kpc; within which half of the projected
    mass is) and index n: the surface density S_e exp(-b ((R / re)^(1/n)
    - 1)), with b(n) exact. The density, enclosed mass and potential are
    computed numerically (see _numerics.sersic_tables), for n in [0.3,
    12].
    """

    @staticmethod
    def type() -> str:
        return 'sersic'

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(
            log_m=ParamScalarDesc(
                desc="log10 of the mass", unit='dex(solMass)'),
            re=ParamScalarDesc(
                desc="the effective radius", min=0, unit='kpc'),
            n=ParamScalarDesc(
                desc="the Sersic index", min=0.3, max=12,
                unit=''))

    def density(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, re, tables = self._scales(params)
        return mass / re ** 3 * tables.density(np.asarray(r) / re)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, re, tables = self._scales(params)
        return mass * tables.mass(np.asarray(r, dtype=float) / re)

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # S_e exp(-b ((R / re)^(1/n) - 1)), of the total mass 2 pi n re^2
        # S_e e^b b^(-2n) Gamma(2n)
        mass, re, _ = self._scales(params)
        n = float(params['n'])
        b = _numerics.sersic_b(n)
        surface_e = mass * b ** (2 * n) / (
            2 * np.pi * n * re ** 2 * np.exp(b) * scipy.special.gamma(2 * n))
        return surface_e * np.exp(
            -b * ((np.asarray(R, dtype=float) / re) ** (1 / n) - 1))

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, re, tables = self._scales(params)
        return G * mass / re * tables.potential(r / re)

    @staticmethod
    def _scales(
            params: ParamValues
    ) -> tuple[float, float, _numerics.SphericalTables]:
        """Return its mass, its effective radius and its unit tables."""
        return (10 ** params['log_m'], params['re'],
                _numerics.sersic_tables(float(params['n'])))


class ProfileGaussian(SpheroidProfile):
    """
    A spherical Gaussian of total mass 10^log_m (Msun) and dispersion
    sigma (kpc): the density M / ((2 pi)^(3/2) sigma^3) exp(-r^2 / (2
    sigma^2)), its enclosed mass M (erf(x / sqrt(2)) - sqrt(2 / pi) x
    exp(-x^2 / 2)) and its potential -G M erf(x / sqrt(2)) / r (x = r /
    sigma), exactly: a Gaussian of a multi-Gaussian expansion that a
    mass follows (see spheroids.MassLightSpheroid). Not a profile of
    configurations, nor public.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(
            log_m=ParamScalarDesc(
                desc="log10 of the mass", unit='dex(solMass)'),
            sigma=ParamScalarDesc(
                desc="the dispersion", min=0, unit='kpc'))

    def density(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, sigma = 10 ** params['log_m'], params['sigma']
        return mass / ((2 * np.pi) ** 1.5 * sigma ** 3) * np.exp(
            -np.asarray(r, dtype=float) ** 2 / (2 * sigma ** 2))

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, sigma = 10 ** params['log_m'], params['sigma']
        x = np.asarray(r, dtype=float) / sigma
        return mass * (scipy.special.erf(x / np.sqrt(2))
                       - np.sqrt(2 / np.pi) * x * np.exp(-x ** 2 / 2))

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, sigma = 10 ** params['log_m'], params['sigma']
        return mass / (2 * np.pi * sigma ** 2) * np.exp(
            -np.asarray(R, dtype=float) ** 2 / (2 * sigma ** 2))

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # (at r = 0, its limit -G M sqrt(2 / pi) / sigma)
        mass, sigma = 10 ** params['log_m'], params['sigma']
        r = np.asarray(r, dtype=float)
        inner = np.divide(
            scipy.special.erf(r / (np.sqrt(2) * sigma)), r,
            out=np.full_like(r, np.sqrt(2 / np.pi) / sigma), where=r > 0)
        return -G * mass * inner


class ProfileNodewise(SpheroidProfile):
    """
    The spherical deprojection of a projected density given at radial
    nodes (kpc), as the surface density of a nodewise disk (see
    surfaces.SurfaceNodewise): 10^log_sigma (Msun/kpc^2) at each node but
    the last, where it is 0, linear in between, and 0 beyond; e.g. the
    surface density of a bulge, measured (times a mass-to-light ratio) or
    free. Of any shape, it keeps that projection face-on (see
    shapes.ShapeFlattened).

    Its density, enclosed mass and potential are exact for that profile:
    with the slope k of the projected density S on each segment [a, b] of
    the nodes, the Abel deprojection rho(r) = -1/pi int_r^inf S'(R) / sqrt(
    R^2 - r^2) dR is -1/pi sum k [acosh(b / r) - acosh(max(a, r) / r)] over
    the segments beyond r (a logarithmic cusp at the centre, where S
    falls), M(<r) = -4 int
    S'(R) I(R, r) dR with I = int_0^min(r, R) s^2 / sqrt(R^2 - s^2) ds,
    and 4 pi int_r^inf rho s ds = -4 int_r^inf S'(R) sqrt(R^2 - r^2) dR,
    each in closed form on each segment.

    Parameters
    ----------
    nodes : Sequence of float
        The radii (kpc) of its nodes: at least two, from 0, increasing (a
        projected density with a hole at the centre would need a negative
        density).

    Raises
    ------
    ConfigError
        If the nodes are not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nodewise'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(nodes=self._nodes.tolist())

    def __init__(self, nodes: Annotated[Sequence[float], Unit('kpc')]):
        check_options(ProfileNodewise.__init__, locals())
        nodes = np.asarray(nodes, dtype=float)
        if (nodes.ndim != 1 or len(nodes) < 2 or not np.all(np.isfinite(nodes))
                or nodes[0] != 0 or np.any(np.diff(nodes) <= 0)):
            raise ConfigError(
                f"the nodes of a nodewise profile must be at least two "
                f"radii, from 0, increasing; they are {nodes.tolist()}")
        self._nodes = nodes

    def nodes(self) -> np.ndarray:
        """Return the radii (kpc) of its nodes."""
        return self._nodes

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(log_sigma=ParamVectorDesc(
            len(self._nodes) - 1,
            desc="log10 of the projected density at each node but the "
                 "last", unit='dex(solMass / kpc2)'))

    def density(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)[..., None]
        a, b, k = self._segments(params)
        span, beyond = _segment_spans(a, b, r)
        with np.errstate(invalid='ignore'):
            terms = np.where(beyond & (k != 0), k * span, 0)
        return -terms.sum(axis=-1) / np.pi

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)[..., None]
        a, b, k = self._segments(params)
        moments = _deprojected_mass_moment(b, r) \
            - _deprojected_mass_moment(a, r)
        return -4 * np.sum(k * moments, axis=-1)

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        sigma = np.append(10 ** np.asarray(params['log_sigma']), 0)
        return np.interp(R, self._nodes, sigma, right=0)

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # -G (M(<r) / r + 4 pi int_r^inf rho s ds), the second term -4 sum
        # k [R sqrt(R^2 - r^2) / 2 - r^2 / 2 ln(R + sqrt(R^2 - r^2))] from
        # max(a, r) to b
        r = np.asarray(r, dtype=float)
        mass = self.enclosed_mass(r, params, context)
        a, b, k = self._segments(params)
        rr = r[..., None]
        span, beyond = _segment_spans(a, b, rr)
        lower = np.maximum(a, rr)
        with np.errstate(invalid='ignore'):
            outer = (b * np.sqrt(np.maximum(b ** 2 - rr ** 2, 0))
                     - lower * np.sqrt(np.maximum(lower ** 2 - rr ** 2, 0))
                     - np.where(rr > 0, rr ** 2 * span, 0)) / 2
            outer = -4 * np.sum(np.where(beyond, k * outer, 0), axis=-1)
        inner = np.divide(mass, r, out=np.zeros_like(r), where=r > 0)
        return -G * (inner + outer)

    def _segments(
            self,
            params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the inner and outer radii of the segments of the nodes, and
        the slope of the projected density on each.
        """
        sigma = np.append(10 ** np.asarray(params['log_sigma']), 0)
        return (self._nodes[:-1], self._nodes[1:],
                np.diff(sigma) / np.diff(self._nodes))


def _segment_spans(
        a: np.ndarray,
        b: np.ndarray,
        r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return acosh(b / r) - acosh(max(a, r) / r) of the segments [a, b] at
    the radii r (broadcast), as ln((b + sqrt(b^2 - r^2)) / (L + sqrt(L^2 -
    r^2))), L = max(a, r), finite at r = 0 but on a segment from 0 (where
    it is infinite); and which segments reach beyond r (b > r).
    """
    lower = np.maximum(a, r)
    beyond = b > r
    with np.errstate(divide='ignore', invalid='ignore'):
        span = np.log(
            (b + np.sqrt(np.maximum(b ** 2 - r ** 2, 0)))
            / (lower + np.sqrt(np.maximum(lower ** 2 - r ** 2, 0))))
    return span, beyond


def _deprojected_mass_moment(R: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Return P(R, r), an antiderivative in R of I(R, r) = int_0^min(r, R)
    s^2 / sqrt(R^2 - s^2) ds (see ProfileNodewise), continuous at R = r:
    pi R^3 / 12 within r, and beyond it R^3 / 6 asin(r / R) - r R sqrt(R^2
    - r^2) / 6 + r^3 / 3 acosh(R / r) (0 at r = 0).
    """
    R, r = np.broadcast_arrays(R, r)
    inside = R <= r
    with np.errstate(divide='ignore', invalid='ignore'):
        outside = (
            R ** 3 / 6 * np.arcsin(np.where(inside, 1, r / R))
            - r * R * np.sqrt(np.maximum(R ** 2 - r ** 2, 0)) / 6
            + r ** 3 / 3 * np.arccosh(np.where(inside | (r == 0), 1, R / r)))
    return np.where(inside, np.pi * R ** 3 / 12, outside)
