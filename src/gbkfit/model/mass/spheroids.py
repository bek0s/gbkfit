"""
Spheroids (e.g. a bulge, or a dark matter halo): a profile (see profiles
and halos) of a shape (see shapes), a sphere or a flattened spheroid; a
halo can respond to the gravity of the baryons (see responses: e.g.
contracted adiabatically), or be given by its dark matter fraction.
"""

import math
from collections.abc import Sequence
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
from .responses import HaloResponse, halo_response_parser
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

    - by a response (see HaloResponse; spherical only): e.g. contracted
      adiabatically, with the parameters of the response. Its forces are
      those of its spherical shells, -v^2(r) / r^2 toward the centre (r
      the spherical radius); its density and potential are not given.
    - given by its dark matter fraction (see DarkFraction): fdm = v^2 /
      (v^2 + v_b^2) at r_fdm (two parameters, instead of the parameter it
      sets), of its circular velocity, after its response if it has one,
      and that of the baryons.

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
    response : HaloResponse, optional
        Its response to the baryons; none by default.
    dark_fraction : DarkFraction, optional
        Its dark matter fraction, which sets one of its parameters.

    Raises
    ------
    ConfigError
        If the name or the names of the baryons are not valid, it has a
        response but is not spherical, or of a profile that the response
        does not take, or parameters of the response repeat those of its
        profile and shape, there are no baryons for a response or a dark
        fraction, or a dark fraction sets a parameter that the spheroid
        does not have.
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
            halo_response_parser, info, 'response')
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
        if self._response is not None:
            info['response'] = halo_response_parser.dump(self._response)
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
            response: HaloResponse | None = None,
            dark_fraction: DarkFraction | None = None
    ):
        super().__init__(name)
        shape = ShapeSphere() if shape is None else shape
        for baryon in baryons:
            parseutils.check_name(baryon)
        if repeated := sorted({b for b in baryons if baryons.count(b) > 1}):
            raise ConfigError(f"the baryons repeat: {repeated}")
        own = profile.pdescs() | shape.pdescs()
        if response is not None:
            if not shape.spherical():
                raise ConfigError(
                    "the response of a spheroid is that of spherical "
                    "shells: it needs a spherical shape")
            response.check_profile(profile)
            if repeated := sorted(own.keys() & response.pdescs().keys()):
                raise ConfigError(
                    f"the parameters of the response repeat those of the "
                    f"profile and the shape: {repeated}")
        if (response is not None or dark_fraction is not None) \
                and not baryons:
            raise ConfigError(
                "a response or a dark fraction needs the baryons that the "
                "spheroid responds to (baryons)")
        if dark_fraction is not None and dark_fraction.solve() not in own:
            raise ConfigError(
                f"the dark fraction sets {dark_fraction.solve()!r}, which "
                f"is not a parameter of the spheroid; its parameters: "
                f"{list(own)}")
        self._profile = profile
        self._shape = shape
        self._baryons = tuple(baryons)
        self._response = response
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

    def response(self) -> HaloResponse | None:
        """Return its response to the baryons, if any."""
        return self._response

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
        if self._response is not None:
            pdescs |= self._response.pdescs()
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
        if self._response is None and self._dark_fraction is None:
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
        if self._response is not None:
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
        self._check_no_response('density')
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
        self._check_no_response('surface density')
        return self._profile.surface_density(
            R, self._given(params, context), context)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._response is not None:
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
        self._check_no_response('potential')
        return self._shape.potential(
            R, z, self._profile, self._given(params, context), context)

    def _check_no_response(self, what: str) -> None:
        """Raise NotImplementedError if it responds to its baryons."""
        if self._response is not None:
            raise NotImplementedError(
                f"the {what} of a spheroid that responds to its baryons is "
                f"not given")

    def _baryons_vcirc2(
            self, R: np.ndarray, context: MassContext) -> np.ndarray:
        """Return the squared circular velocity of its baryons at R."""
        return sum(context.vcirc2s[name](R) for name in self._baryons)

    def _responding_vcirc2(
            self, R: np.ndarray, params: ParamValues, context: MassContext
    ) -> np.ndarray:
        """
        Return its squared circular velocity at R, after its response if
        it has one, for the values of all the parameters of its profile,
        shape and response.
        """
        def vcirc2(radii: np.ndarray) -> np.ndarray:
            return self._shape.vcirc2(radii, self._profile, params, context)

        def baryons(radii: np.ndarray) -> np.ndarray:
            return self._baryons_vcirc2(radii, context)
        if self._response is None:
            return vcirc2(R)
        return self._response.vcirc2(
            R.ravel(), vcirc2, baryons, self._profile, params, context
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
