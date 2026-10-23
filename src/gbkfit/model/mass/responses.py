"""
The responses of spherical halos to the gravity of their baryons (see
MassSpheroid): the adiabatic contractions of Blumenthal et al. (1986) and
Gnedin et al. (2004), and that of Cautun et al. (2020).
"""

import abc
from collections.abc import Callable
from typing import Any

import numpy as np

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import MassContext
from .halos import ProfileHalo
from .profiles import SpheroidProfile


__all__ = [
    'HaloResponse',
    'ResponseBlumenthal86',
    'ResponseCautun20',
    'ResponseGnedin04',
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

    @abc.abstractmethod
    def vcirc2(
            self,
            R: np.ndarray,
            halo_vcirc2: _VCirc2,
            baryons_vcirc2: _VCirc2,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the squared circular velocity ((km/s)^2) of the halo after
        its response at the radii R (kpc; 1-D), of those of the halo
        before and of the baryons in the plane (functions of the radius),
        the profile of the halo, the values of the parameters of its
        spheroid and its context (see MassComponent.vcirc2). NaN where the
        baryons push outwards (negative v^2).
        """


class ResponseBlumenthal86(HaloResponse):
    """
    The adiabatic contraction of Blumenthal et al. (1986): the dark matter
    within r' ends within r, with r' M(r') = r (M(r') + M_b(r)) (Burkert
    et al. 2010, Eq. 19), M = v^2 r / G of the halo before and of the
    baryons in the plane: v^2(r) = G M(r') / r = v^2(r') r' / r. (Burkert
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
            baryons_vcirc2: _VCirc2,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return _contracted(R, halo_vcirc2, baryons_vcirc2)


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
            baryons_vcirc2: _VCirc2,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return _contracted(
            R, halo_vcirc2, baryons_vcirc2, _GNEDIN04_A0, _GNEDIN04_W,
            profile.halo_radius(params, context))


class ResponseCautun20(HaloResponse):
    """
    The contraction of Cautun et al. (2020, Eq. 11), as they fit their
    simulations: the mass of the halo within r times 0.45 + 0.38 (eta +
    1.16)^0.53, eta = M_b (1 - f_b) / (f_b M) of the masses M = v^2 r / G
    within r of the halo before (of dark matter only) and of the baryons
    in the plane, f_b the cosmic baryon fraction of the cosmology of the
    galaxy (about 1 at eta = 1: baryons that follow the dark matter).
    """

    @staticmethod
    def type() -> str:
        return 'cautun20'

    def vcirc2(
            self,
            R: np.ndarray,
            halo_vcirc2: _VCirc2,
            baryons_vcirc2: _VCirc2,
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
        halo, baryons = halo_vcirc2(R), baryons_vcirc2(R)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = baryons * (1 - fraction) / (fraction * halo)
            out = halo * (0.45 + 0.38 * (eta + 1.16) ** 0.53)
        out = np.where(baryons < 0, np.nan, out)
        out[R == 0] = 0
        return out


def _contracted(
        R: np.ndarray,
        halo_vcirc2: _VCirc2,
        baryons_vcirc2: _VCirc2,
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
    Gnedin et al.), M = v^2 r / G of the halo before and of the baryons
    in the plane (halo_vcirc2 and baryons_vcirc2, functions of the
    radius): at R = r_bar, v^2 = G M(r_bar') / R. NaN where the baryons
    push outwards (negative v^2).
    """
    out = np.full_like(R, np.nan)
    baryons_all = baryons_vcirc2(R) * R
    valid = (R > 0) & (baryons_all >= 0)
    out[R == 0] = 0
    averaged = R[valid]
    baryons = baryons_all[valid]
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
    ResponseCautun20])
