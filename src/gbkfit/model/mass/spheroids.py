"""
Spheroids (e.g. a bulge, or a dark matter halo): a profile (see profiles
and halos) of a shape (see shapes), a sphere or a flattened spheroid; a
halo can respond to the gravity of the baryons (contracted adiabatically,
or given by its dark matter fraction).
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.optimize

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import MassComponent, MassContext
from .halos import (
    ProfileBurkert, ProfileCoreNFW, ProfileDekelZhao, ProfileEinasto,
    ProfileNFW, ProfileNFWTruncated, ProfilePseudoIsothermal, ProfileZhao)
from .profiles import ProfileNodewise, ProfileSersic, SpheroidProfile
from .shapes import ShapeSphere, SpheroidShape, spheroid_shape_parser
from .units import G


__all__ = [
    'DarkFraction',
    'MassSpheroid',
    'dark_fraction_parser',
    'spheroid_profile_parser'
]


spheroid_profile_parser = parseutils.TypedParser(SpheroidProfile, [
    ProfileSersic,
    ProfileNodewise,
    ProfileNFW,
    ProfileBurkert,
    ProfilePseudoIsothermal,
    ProfileEinasto,
    ProfileZhao,
    ProfileDekelZhao,
    ProfileCoreNFW,
    ProfileNFWTruncated])

# The contractions of halos (see MassSpheroid)
CONTRACTIONS = ('blumenthal86',)

# The bisections of the initial radius of a contracted shell, in ln r
# over [ln r, ln r + _CONTRACTION_RANGE] (to the precision of floats)
_CONTRACTION_RANGE = 50.0
_CONTRACTION_BISECTIONS = 80


def _contracted(
        R: np.ndarray,
        halo_vcirc2: Callable[[np.ndarray], np.ndarray],
        baryons_vcirc2: np.ndarray
) -> np.ndarray:
    """
    Return the squared circular velocity at the radii R of a halo
    contracted adiabatically by baryons (Blumenthal et al. 1986): the
    dark matter within r' ends within r, with r' M(r') = r (M(r') +
    M_b(r)) (Burkert et al. 2010, Eq. 19), M = v^2 r / G of the halo
    before (halo_vcirc2, a function of the radius) and of the baryons in
    the plane (baryons_vcirc2, at R): v^2 = G M(r') / r = v^2(r') r' / r.
    NaN where the baryons push outwards (negative v^2).
    """
    out = np.full_like(R, np.nan)
    valid = (R > 0) & (baryons_vcirc2 >= 0)
    out[R == 0] = 0
    r = R[valid]
    baryons = baryons_vcirc2[valid] * r
    lo = np.log(r)
    hi = lo + _CONTRACTION_RANGE
    for _ in range(_CONTRACTION_BISECTIONS):
        mid = (lo + hi) / 2
        x = np.exp(mid)
        halo = halo_vcirc2(x) * x
        inside = x * halo < r * (halo + baryons)
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    x = np.exp((lo + hi) / 2)
    out[valid] = halo_vcirc2(x) * x / r
    return out


class DarkFraction(parseutils.Serializable):
    """
    A spheroid given by its dark matter fraction (see MassSpheroid): the
    fraction fdm = v^2 / (v^2 + v_b^2) of its squared circular velocity v^2
    in that with its baryons (v_b^2) at the radius r_fdm (kpc), two
    parameters, sets one of its parameters (solve), to the value within
    the bounds that gives it.

    Parameters
    ----------
    solve : str
        The name of the parameter it sets (e.g. 'log_m', or 'c').
    bounds : Sequence of float
        The range of the values of that parameter, [lower, upper].

    Raises
    ------
    ConfigError
        If the name or the bounds are not valid.
    """

    def dump(self) -> dict[str, Any]:
        return dict(solve=self._solve, bounds=list(self._bounds))

    def __init__(self, solve: str, bounds: Sequence[float]):
        parseutils.check_name(solve)
        if not (isinstance(bounds, Sequence) and len(bounds) == 2 and all(
                isinstance(b, (int, float)) and not isinstance(b, bool)
                and math.isfinite(b) for b in bounds)
                and bounds[0] < bounds[1]):
            raise ConfigError(
                f"the bounds of a dark fraction must be two finite numbers, "
                f"increasing; they are {bounds!r}")
        self._solve = solve
        self._bounds = (float(bounds[0]), float(bounds[1]))

    def solve(self) -> str:
        """Return the name of the parameter it sets."""
        return self._solve

    def bounds(self) -> tuple[float, float]:
        """Return the range of the values of that parameter."""
        return self._bounds


dark_fraction_parser = parseutils.BasicParser(DarkFraction)


class MassSpheroid(MassComponent):
    """
    A spheroid of a profile and a shape: its density, circular velocity,
    forces and potential are those of the profile laid out in the shape
    (see shapes), and its parameters those of both.

    A spheroid (e.g. a dark matter halo) can respond to the gravity of
    baryons, mass components of its model (see MassComponent.responds_to):

    - contracted ('blumenthal86'; spherical only): the dark matter within
      r' ends within r, with r' M(r') = r (M(r') + M_b(r)) (Blumenthal et
      al. 1986; Burkert et al. 2010, Eq. 19), M = v^2 r / G of the halo
      before and of the baryons in the plane: v^2(r) = G M(r') / r =
      v^2(r') r' / r. (Burkert et al. 2010, Eq. 18, which DysmalPy's
      documentation quotes, writes v^2(r') for this term, smaller by r /
      r': it does not keep the mass that Eq. 19 moves within r.) Its
      forces are those of its spherical shells, -v^2(r) / r^2 toward the
      centre (r the spherical radius); its density and potential are not
      given.
    - given by its dark matter fraction (see DarkFraction): fdm = v^2 /
      (v^2 + v_b^2) at r_fdm (two parameters, instead of the parameter it
      sets), of its circular velocity, contracted if it is, and that of
      the baryons.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    profile : SpheroidProfile
        Its profile.
    shape : SpheroidShape, optional
        Its shape; by default, a sphere.
    baryons : Sequence of str, optional
        The names of the mass components of its model that it responds to.
    contraction : str, optional
        Its contraction by the baryons: none (the default) or
        'blumenthal86'.
    dark_fraction : DarkFraction, optional
        Its dark matter fraction, which sets one of its parameters.

    Raises
    ------
    ConfigError
        If the name or the names of the baryons are not valid, the
        contraction is not valid or the spheroid not spherical, there are
        no baryons for a contraction or a dark fraction, or a dark
        fraction sets a parameter that the spheroid does not have.
    """

    @staticmethod
    def type() -> str:
        return 'spheroid'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'MassSpheroid':
        parseutils.load_option_and_update_info(
            spheroid_profile_parser, info, 'profile', required=True)
        parseutils.load_option_and_update_info(
            spheroid_shape_parser, info, 'shape')
        parseutils.load_option_and_update_info(
            dark_fraction_parser, info, 'dark_fraction')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = super().dump() | dict(
            profile=spheroid_profile_parser.dump(self._profile),
            shape=spheroid_shape_parser.dump(self._shape))
        if self._baryons:
            info['baryons'] = list(self._baryons)
        if self._contraction is not None:
            info['contraction'] = self._contraction
        if self._dark_fraction is not None:
            info['dark_fraction'] = dark_fraction_parser.dump(
                self._dark_fraction)
        return info

    def __init__(
            self,
            name: str,
            profile: SpheroidProfile,
            shape: SpheroidShape | None = None,
            baryons: Sequence[str] = (),
            contraction: str | None = None,
            dark_fraction: DarkFraction | None = None
    ):
        super().__init__(name)
        shape = ShapeSphere() if shape is None else shape
        for baryon in baryons:
            parseutils.check_name(baryon)
        if repeated := sorted({b for b in baryons if baryons.count(b) > 1}):
            raise ConfigError(f"the baryons repeat: {repeated}")
        if contraction is not None:
            if contraction not in CONTRACTIONS:
                raise ConfigError(
                    f"contraction must be one of {list(CONTRACTIONS)}; it "
                    f"is {contraction!r}")
            if not shape.spherical():
                raise ConfigError(
                    "the contraction of a spheroid is that of spherical "
                    "shells: it needs a spherical shape")
        if (contraction is not None or dark_fraction is not None) \
                and not baryons:
            raise ConfigError(
                "a contraction or a dark fraction needs the baryons that "
                "the spheroid responds to (baryons)")
        own = profile.pdescs() | shape.pdescs()
        if dark_fraction is not None and dark_fraction.solve() not in own:
            raise ConfigError(
                f"the dark fraction sets {dark_fraction.solve()!r}, which "
                f"is not a parameter of the spheroid; its parameters: "
                f"{list(own)}")
        self._profile = profile
        self._shape = shape
        self._baryons = tuple(baryons)
        self._contraction = contraction
        self._dark_fraction = dark_fraction

    def profile(self) -> SpheroidProfile:
        """Return its profile."""
        return self._profile

    def shape(self) -> SpheroidShape:
        """Return its shape."""
        return self._shape

    def baryons(self) -> tuple[str, ...]:
        """Return the names of the baryons it responds to."""
        return self._baryons

    def contraction(self) -> str | None:
        """Return its contraction, if any."""
        return self._contraction

    def dark_fraction(self) -> DarkFraction | None:
        """Return its dark fraction, if any."""
        return self._dark_fraction

    def spherical(self) -> bool:
        return self._shape.spherical()

    def responds_to(self) -> tuple[str, ...]:
        return self._baryons

    def takes_shares(self) -> bool:
        return self._dark_fraction is not None

    def pdescs(self) -> dict[str, ParamDesc]:
        pdescs = self._profile.pdescs() | self._shape.pdescs()
        if self._dark_fraction is not None:
            del pdescs[self._dark_fraction.solve()]
            pdescs |= dict(
                fdm=ParamScalarDesc(
                    desc="the dark matter fraction (of v^2) at r_fdm",
                    minimum=0, maximum=1),
                r_fdm=ParamScalarDesc(
                    desc="the radius of the dark matter fraction (kpc)",
                    minimum=0))
        return pdescs

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if self._contraction is None and self._dark_fraction is None:
            return self._shape.vcirc2(R, self._profile, params, context)
        params = self._given(params, context)
        return self._responding_vcirc2(
            np.asarray(R, dtype=float), params, context)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if self._contraction is not None:
            # (spherical: the mass of its circular velocity, v^2 r / G)
            r = np.asarray(r, dtype=float)
            return self.vcirc2(r, params, context) * r / G
        return self._shape.enclosed_mass(
            r, self._profile, self._given(params, context), context)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        self._check_not_contracted('density')
        return self._shape.density(
            R, z, self._profile, self._given(params, context), context)

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # (that of its profile, whatever its shape: squashing a sphere
        # with the mass of each shell kept keeps its face-on projection)
        self._check_not_contracted('surface density')
        return self._profile.surface_density(
            R, self._given(params, context), context)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._contraction is not None:
            # (spherical: the pull of its mass within the spherical radius)
            R, z = np.broadcast_arrays(
                np.asarray(R, dtype=float), np.asarray(z, dtype=float))
            r = np.hypot(R, z)
            scale = -np.divide(
                self.vcirc2(r, params, context), r ** 2,
                out=np.zeros_like(r), where=r > 0)
            return scale * R, scale * z
        return self._shape.forces(
            R, z, self._profile, self._given(params, context), context)

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        self._check_not_contracted('potential')
        return self._shape.potential(
            R, z, self._profile, self._given(params, context), context)

    def _check_not_contracted(self, what: str) -> None:
        """Raise NotImplementedError if it is contracted."""
        if self._contraction is not None:
            raise NotImplementedError(
                f"the {what} of a contracted spheroid is not given")

    def _baryons_vcirc2(
            self, R: np.ndarray, context: MassContext) -> np.ndarray:
        """Return the squared circular velocity of its baryons at R."""
        return sum(context.vcirc2s[name](R) for name in self._baryons)

    def _responding_vcirc2(
            self, R: np.ndarray, params: ParamValues, context: MassContext
    ) -> np.ndarray:
        """
        Return its squared circular velocity at R, contracted if it is,
        for the values of all the parameters of its profile and shape.
        """
        def vcirc2(radii: np.ndarray) -> np.ndarray:
            return self._shape.vcirc2(radii, self._profile, params, context)
        if self._contraction is None:
            return vcirc2(R)
        return _contracted(
            R.ravel(), vcirc2, self._baryons_vcirc2(R.ravel(), context)
        ).reshape(R.shape)

    def _given(
            self, params: ParamValues, context: MassContext) -> ParamValues:
        """
        Return the values of the parameters of its profile and shape: with
        a dark fraction, that of the parameter it sets, solved for (to the
        precision of floats).

        Raises
        ------
        ValueError
            If no value within the bounds gives the dark fraction.
        """
        if self._dark_fraction is None:
            return params
        name = self._dark_fraction.solve()
        radius = np.array([float(params['r_fdm'])])
        fdm = float(params['fdm'])
        target = fdm / (1 - fdm) * self._baryons_vcirc2(radius, context)[0]

        def excess(value: float) -> float:
            return float(self._responding_vcirc2(
                radius, dict(params) | {name: value}, context)[0] - target)
        lower, upper = self._dark_fraction.bounds()
        if not np.sign(excess(lower)) * np.sign(excess(upper)) <= 0:
            raise ValueError(
                f"no value of {name!r} within {[lower, upper]} gives the "
                f"spheroid {self.name()!r} the dark matter fraction {fdm} "
                f"at {radius[0]} kpc")
        value = scipy.optimize.brentq(
            excess, lower, upper, xtol=1e-14, rtol=4 * np.finfo(float).eps)
        return dict(params) | {name: value}
