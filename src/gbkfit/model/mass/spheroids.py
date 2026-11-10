"""
Spheroids: dark matter halos (Halo), of a profile (see halos) laid out
in a shape (see shapes), which can respond to the gravity of their
baryons (see responses) or be given by their dark matter fraction (see
halo_params); spheroids of luminous mass, measured or free
(MassSpheroid); and the mass that follows the light of a spheroid of
the galaxy (TracedSpheroid).
"""

import abc
from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.interpolate
import scipy.optimize

from gbkfit.config import TypedParser, check_name, check_options
from gbkfit.params import ModelEvaluationError, ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from .base import MassComponent, MassContext, SpheroidDensity
from .halo_params import DARK_FRACTIONS, Overdensity
from .halos import (
    HaloProfile, ScaledHaloProfile, SphericalHalo, halo_profile_parser)
from .profiles import (
    ProfileNodewise, ProfileSersic, SphericalProfile, SpheroidProfile)
from .responses import (
    SIDM_PDESCS, Contraction, contraction_parser, sidm_vcirc2)
from .shapes import ShapeSphere, SpheroidShape, spheroid_shape_parser
from .units import G


__all__ = [
    'Halo',
    'MassSpheroid',
    'TracedSpheroid',
    'spheroid_profile_parser'
]


spheroid_profile_parser = TypedParser(SpheroidProfile, [
    ProfileSersic,
    ProfileNodewise], family='spheroid_profile')

# The density of a halo that responds to its baryons: its mass within
# spheres differentiated by central differences of this step in ln r; at
# the centre, that at this radius (kpc)
_DENSITY_STEP = 1e-4
_DENSITY_CENTRE = 1e-6

# The potential of a halo that responds to its baryons: the difference of
# its squared circular velocity from that of the halo before, integrated
# in ln r on this grid (kpc), by Simpson's rule
_POTENTIAL_RADII = np.geomspace(1e-4, 1e6, 1001)

# The dark fraction of a halo: the bracket of the scale it sets (log10
# of its mass or scale density), widened from its start by these steps
# (dex), at most this many times on each side
_DARK_FRACTION_STARTS = dict(mass_concentration=12.0, density_radius=7.0)
_DARK_FRACTION_STEP = 2.0
_DARK_FRACTION_WIDENINGS = 8


class _ShapedMass(MassComponent, abc.ABC):
    """
    A spheroid of a spherical profile laid out in a shape (a sphere, or
    flattened; see shapes): its density, circular velocity, forces and
    potential are those of the profile in the shape (see _profile_at).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    shape : SpheroidShape, optional
        Its shape; by default, a sphere.
    """

    def __init__(self, name: str, shape: SpheroidShape | None):
        super().__init__(name)
        self._shape = ShapeSphere() if shape is None else shape

    def shape(self) -> SpheroidShape:
        """Return its shape."""
        return self._shape

    def spherical(self) -> bool:
        return self._shape.spherical()

    @abc.abstractmethod
    def _profile_at(
            self, params: ParamValues, context: MassContext
    ) -> SphericalProfile:
        """
        Return its profile, which its shape lays out at the values of its
        parameters (see MassComponent.vcirc2 for the arguments).
        """
        pass

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._shape.vcirc2(
            R, self._profile_at(params, context), params, context)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._shape.enclosed_mass(
            r, self._profile_at(params, context), params, context)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._shape.density(
            R, z, self._profile_at(params, context), params, context)

    def gives_density(self, thick: Collection[str]) -> bool:
        return True

    def gives_cell_masses(self, thick: Collection[str]) -> bool:
        return True

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._shape.forces(
            R, z, self._profile_at(params, context), params, context)

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._shape.potential(
            R, z, self._profile_at(params, context), params, context)


class MassSpheroid(_ShapedMass):
    """
    A spheroid of luminous mass (e.g. a bulge, from its photometry, or
    free): a profile (see profiles; Sersic, or nodewise) laid out in a
    shape (a sphere, or flattened; see shapes). Its density, circular
    velocity, forces and potential are those of the profile in the shape,
    and its parameters those of both. (The mass that follows the light
    of a spheroid of the galaxy is a TracedSpheroid.)

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    profile : SpheroidProfile
        Its profile.
    shape : SpheroidShape, optional
        Its shape; by default, a sphere.

    Raises
    ------
    ValueError
        If the name is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'spheroid'

    @staticmethod
    def type_title() -> str:
        return 'Spheroid'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            profile=spheroid_profile_parser.dump(self._profile),
            shape=spheroid_shape_parser.dump(self._shape))

    def __init__(
            self,
            name: str,
            profile: SpheroidProfile,
            shape: SpheroidShape | None = None
    ):
        check_options(MassSpheroid.__init__, locals())
        super().__init__(name, shape)
        self._profile = profile

    def profile(self) -> SpheroidProfile:
        """Return its profile."""
        return self._profile

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._profile.pdescs() | self._shape.pdescs()

    def gives_surface_density(self) -> bool:
        return self._profile.gives_surface_density()

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # (that of its profile, whatever its shape: squashing a sphere
        # with the mass of each shell kept keeps its face-on projection)
        return self._profile.surface_density(R, params, context)

    def _profile_at(
            self, params: ParamValues, context: MassContext
    ) -> SphericalProfile:
        return self._profile


class Halo(_ShapedMass):
    """
    A dark matter halo: a profile (see halos), of the overdensity that
    defines its mass M_D, radius r_D and concentration c = r_D / r_s (see
    halo_params), laid out in a shape (a sphere, or flattened; see
    shapes). Its density, circular velocity, forces and potential are
    those of the profile in the shape, and its parameters those of both.

    A spherical halo can respond to the gravity of baryons, mass
    components of its model that give their masses within spheres (see
    MassComponent.responds_to and gives_enclosed_mass):

    - it contracts (see responses.Contraction), with the parameters of
      its contraction;
    - it is of self-interacting dark matter (sidm), of an isothermal core
      (see responses.sidm_vcirc2) after its contraction if it has one,
      with the parameters sigma_m and age; its baryons shape the core,
      but it needs none.

    Responding, its forces are those of its spherical shells, -v^2(r) /
    r^2 toward the centre (r the spherical radius), as is its mass in
    cells (see MassComponent.cell_masses); its density is that of its
    mass within spheres, dM/dr / (4 pi r^2), by central differences (at
    the centre, that at 1e-6 kpc); its potential is that of the halo
    before plus the integral of the difference of v^2 / r from it (in ln
    r, between 1e-4 and 1e6 kpc; from infinity, or from the centre if
    the potential of the halo is given from there, see
    halos.SphericalHalo.zero_at_infinity).

    Of a parameterisation of a dark matter fraction (see halo_params), it
    is solved for the scale that the fraction sets: fdm = v^2 / (v^2 +
    v_b^2) at rfdm, of its squared circular velocity v^2 (after its
    responses) and that of its baryons v_b^2, which the gravity of its
    galaxy must give (see Gravity.gives). Its mass M_D at its
    concentration (dark_fraction_concentration), by Brent's method in
    log M_D, which v^2 grows with; its scale density at its scale radius
    (dark_fraction_radius), in closed form (v^2 is proportional to it),
    or by Brent's method in its log if it responds.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    profile : HaloProfile
        Its profile.
    overdensity : str, optional
        The overdensity of its mass and radius: e.g. '200c' (the
        default), '200m' or 'vir' (see halo_params.Overdensity).
    shape : SpheroidShape, optional
        Its shape; by default, a sphere.
    baryons : Sequence of str, optional
        The names of the mass components of its model that it responds to
        (that it contracts in, or that shape its SIDM core) or whose
        share sets its scale (of a dark fraction).
    contraction : Contraction, optional
        Its contraction by the baryons; none by default.
    sidm : bool, optional
        Whether it is of self-interacting dark matter; not by default.

    Raises
    ------
    ValueError
        If the name, the overdensity or the names of the baryons are not
        valid, or the baryons repeat; it contracts or is of SIDM but is
        not spherical; it contracts or is of a dark fraction without
        baryons, or has baryons for neither of these nor SIDM; or the
        parameters of its profile, shape, contraction and SIDM repeat.
    """

    @staticmethod
    def type() -> str:
        return 'halo'

    @staticmethod
    def type_title() -> str:
        return 'Dark-matter halo'

    def dump(self) -> dict[str, Any]:
        info = super().dump() | dict(
            profile=halo_profile_parser.dump(self._profile),
            overdensity=str(self._overdensity),
            shape=spheroid_shape_parser.dump(self._shape))
        if self._baryons:
            info['baryons'] = list(self._baryons)
        if self._contraction is not None:
            info['contraction'] = contraction_parser.dump(self._contraction)
        if self._sidm:
            info['sidm'] = True
        return info

    def __init__(
            self,
            name: str,
            profile: HaloProfile,
            overdensity: str = '200c',
            shape: SpheroidShape | None = None,
            baryons: Sequence[str] = (),
            contraction: Contraction | None = None,
            sidm: bool = False
    ):
        check_options(Halo.__init__, locals())
        super().__init__(name, shape)
        for baryon in baryons:
            check_name(baryon)
        if repeated := sorted({b for b in baryons if baryons.count(b) > 1}):
            raise ValueError(f"the baryons repeat: {repeated}")
        dark_fraction = profile.parameterisation() in DARK_FRACTIONS
        if (contraction is not None or sidm) and not self._shape.spherical():
            raise ValueError(
                "a halo that contracts or is of SIDM responds as spherical "
                "shells: it needs a spherical shape")
        if contraction is not None and not baryons:
            raise ValueError(
                "a contraction needs the baryons that the halo contracts in "
                "(baryons)")
        if dark_fraction and not baryons:
            raise ValueError(
                "a dark fraction needs the baryons whose share sets the halo "
                "(baryons)")
        if baryons and contraction is None and not sidm and not dark_fraction:
            raise ValueError(
                "the baryons of a halo are those that it responds to "
                "(contraction, sidm) or whose share sets its scale (a dark "
                "fraction); it has none of these: leave them out")
        parts = [profile.pdescs(), self._shape.pdescs(),
                 {} if contraction is None else contraction.pdescs(),
                 SIDM_PDESCS if sidm else {}]
        names = [name for part in parts for name in part]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(
                f"the parameters of the profile, the shape, the contraction "
                f"and SIDM of the halo repeat: {repeated}")
        self._profile = profile
        self._overdensity = Overdensity(overdensity)
        self._baryons = tuple(baryons)
        self._contraction = contraction
        self._sidm = sidm

    def profile(self) -> HaloProfile:
        """Return its profile."""
        return self._profile

    def overdensity(self) -> Overdensity:
        """Return the overdensity of its mass and radius."""
        return self._overdensity

    def baryons(self) -> tuple[str, ...]:
        """Return the names of its baryons."""
        return self._baryons

    def contraction(self) -> Contraction | None:
        """Return its contraction, if any."""
        return self._contraction

    def sidm(self) -> bool:
        """Check whether it is of self-interacting dark matter."""
        return self._sidm

    def responds_to(self) -> tuple[str, ...]:
        return self._baryons

    def takes_shares(self) -> bool:
        return self._profile.parameterisation() in DARK_FRACTIONS

    def takes_enclosed_masses(self) -> bool:
        return self._responds()

    def pdescs(self) -> dict[str, ParamDesc]:
        pdescs = self._profile.pdescs() | self._shape.pdescs()
        if self._contraction is not None:
            pdescs |= self._contraction.pdescs()
        if self._sidm:
            pdescs |= SIDM_PDESCS
        return pdescs

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._responds():
            return super().vcirc2(R, params, context)
        R = np.asarray(R, dtype=float)
        return self._responding_vcirc2(
            self._profile_at(params, context), R.ravel(), params, context
        ).reshape(R.shape)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._responds():
            return super().enclosed_mass(r, params, context)
        # (spherical: the mass of its circular velocity, v^2 r / G)
        r = np.asarray(r, dtype=float)
        return self.vcirc2(r, params, context) * r / G

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._responds():
            return super().density(R, z, params, context)
        # dM/dr / (4 pi r^2), of M = v^2 r / G at r e^(+-h)
        r = np.maximum(np.hypot(R, z), _DENSITY_CENTRE)
        h = _DENSITY_STEP
        shifted = np.concatenate([(r * np.exp(h)).ravel(),
                                  (r * np.exp(-h)).ravel()])
        mass = self.enclosed_mass(shifted, params, context)
        upper, lower = mass.reshape(2, *r.shape)
        return (upper - lower) / (2 * np.sinh(h) * r) / (4 * np.pi * r ** 2)

    def cell_masses(
            self,
            r_edges: np.ndarray,
            b_edges: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._responds():
            return super().cell_masses(r_edges, b_edges, params, context)
        # (spherical: of its enclosed mass, as MassSpherical.cell_masses)
        return np.outer(
            np.diff(self.enclosed_mass(r_edges, params, context)),
            np.diff(np.sin(b_edges)))

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._responds():
            return super().forces(R, z, params, context)
        # (spherical: the pull of its mass within the spherical radius)
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        r = np.hypot(R, z)
        scale = -np.divide(
            self.vcirc2(r, params, context), r ** 2,
            out=np.zeros_like(r), where=r > 0)
        return scale * R, scale * z

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._responds():
            return super().potential(R, z, params, context)
        # (spherical: that of the halo before, plus the integral of the
        # difference of v^2 / r from it, see the class docstring)
        r = np.hypot(R, z)
        halo = self._profile_at(params, context)
        radii = _POTENTIAL_RADII
        ln_radii = np.log(radii)
        difference = self._shape.vcirc2(radii, halo, params, context) \
            - self._responding_vcirc2(halo, radii, params, context)
        integral = scipy.interpolate.CubicSpline(
            ln_radii, scipy.integrate.cumulative_simpson(
                difference, x=ln_radii, initial=0))
        inner = integral(np.log(np.clip(r, radii[0], radii[-1])))
        before = halo.potential(r)
        if halo.zero_at_infinity():
            return before + (integral(ln_radii[-1]) - inner)
        return before - inner

    def _responds(self) -> bool:
        """Check whether it responds to its baryons (see the class)."""
        return self._contraction is not None or self._sidm

    def _profile_at(
            self, params: ParamValues, context: MassContext
    ) -> SphericalHalo:
        if self._profile.parameterisation() in DARK_FRACTIONS:
            return self._solved(params, context)
        return self._profile.at(params, self._overdensity.density(context))

    def _halo_vcirc2(
            self,
            halo: SphericalHalo,
            R: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the squared circular velocity at the radii R (1-D) of the
        halo, after its responses if it has any.
        """
        if self._responds():
            return self._responding_vcirc2(halo, R, params, context)
        return self._shape.vcirc2(R, halo, params, context)

    def _responding_vcirc2(
            self,
            halo: SphericalHalo,
            R: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the squared circular velocity at the radii R (1-D) of the
        halo after its responses: its contraction, then its SIDM core.
        """
        shape = self._shape
        contraction = self._contraction

        def before(radii: np.ndarray) -> np.ndarray:
            return shape.vcirc2(radii, halo, params, context)

        def contracted(radii: np.ndarray) -> np.ndarray:
            return contraction.vcirc2(
                radii, before, baryons, params, context)

        def baryons(radii: np.ndarray) -> np.ndarray:
            return sum((context.enclosed_masses[name](radii)
                        for name in self._baryons), np.zeros_like(radii))

        vcirc2 = before if contraction is None else contracted
        if self._sidm:
            return sidm_vcirc2(R, vcirc2, baryons, params)
        return vcirc2(R)

    def _solved(
            self, params: ParamValues, context: MassContext
    ) -> SphericalHalo:
        """
        Return the halo of a dark fraction at the values of its
        parameters: of the scale that the fraction sets (see the class).

        Raises
        ------
        ModelEvaluationError
            If the fraction is not within (0, 1), or no scale gives it.
        """
        profile: ScaledHaloProfile = self._profile
        base = DARK_FRACTIONS[profile.parameterisation()]
        radius = np.array([float(params['rfdm'])])
        fdm = float(params['fdm'])
        if not 0 < fdm < 1:
            raise ModelEvaluationError(
                f"the dark matter fraction {fdm} of the halo "
                f"{self.name()!r} at {radius[0]} kpc gives no halo: it must "
                f"be within (0, 1)")
        target = fdm / (1 - fdm) * sum(
            context.vcirc2s[name](radius)[0] for name in self._baryons)
        if base == 'density_radius':
            rs = params['rs']
            if not self._responds():
                unit = profile.at_scales(1.0, rs, params)
                return profile.at_scales(
                    target / self._shape.vcirc2(
                        radius, unit, params, context)[0], rs, params)

            def halo_at(log_scale: float) -> SphericalHalo:
                return profile.at_scales(10 ** log_scale, rs, params)
        else:
            density = self._overdensity.density(context)

            def halo_at(log_scale: float) -> SphericalHalo:
                return profile.at_scales(*profile.scales(
                    dict(params) | dict(log_m=log_scale), density, base),
                    params)

        def excess(log_scale: float) -> float:
            return float(self._halo_vcirc2(
                halo_at(log_scale), radius, params, context)[0]) - target

        lower, upper = _bracket(excess, _DARK_FRACTION_STARTS[base])
        if lower is None:
            raise ModelEvaluationError(
                f"no halo gives the halo {self.name()!r} the dark matter "
                f"fraction {fdm} at {radius[0]} kpc")
        return halo_at(scipy.optimize.brentq(
            excess, lower, upper, xtol=1e-12, rtol=4 * np.finfo(float).eps))


def _bracket(
        excess: Callable[[float], float], start: float
) -> tuple[float, float] | tuple[None, None]:
    """
    Return the bounds of a root of an increasing function of x (excess),
    widened from start by _DARK_FRACTION_STEP, at most
    _DARK_FRACTION_WIDENINGS times; (None, None) if none brackets it.
    """
    step = _DARK_FRACTION_STEP if excess(start) < 0 else -_DARK_FRACTION_STEP
    for i in range(_DARK_FRACTION_WIDENINGS):
        x = start + (i + 1) * step
        if (excess(x) < 0) != (step > 0):
            return tuple(sorted((x - step, x)))
    return None, None


class TracedSpheroid(MassComponent):
    """
    The mass that traces the light of a spheroid of the galaxy, times a
    mass-to-light ratio. It follows it to any radius (see
    components.Spheroid; rmax only bounds how far its light is drawn):
    upsilon times the light of each term of its profile, its density,
    circular velocity, forces and potential those of the term (see
    SpheroidDensity; exactly, or as the shapes give them, see shapes).

    upsilon is the mass per unit of its light, as of the mass disks that
    follow disks (see surfaces.SurfaceComponent): without a unit of its
    emission, in Msun/kpc^2 per unit of its surface brightness; with one,
    in Msun/pc^2 per unit (a mass-to-light ratio, in its first band).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    component : str
        The name of the spheroid of its galaxy that it follows.

    Raises
    ------
    ValueError
        If the name or the name of the component is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'traced_spheroid'

    @staticmethod
    def type_title() -> str:
        return 'Traced spheroid'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(component=self._component)

    def __init__(self, name: str, component: str):
        check_options(TracedSpheroid.__init__, locals())
        super().__init__(name)
        check_name(component)
        self._component = component

    def component(self) -> str:
        """Return the name of the spheroid it follows."""
        return self._component

    def followed_components(self) -> tuple[str, ...]:
        return (self._component,)

    @classmethod
    def followed_kind(cls) -> 'type[SpheroidDensity]':
        return SpheroidDensity

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(upsilon=ParamScalarDesc(
            title='Mass-to-light ratio',
            desc="the mass per unit of the light of the spheroid", min=0,
            default=1))

    def mass_ratios(self) -> dict[str, str]:
        return dict(upsilon=self._component)

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._sum(
            lambda shape, profile, values: shape.vcirc2(
                R, profile, values, context), params, context)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._sum(
            lambda shape, profile, values: shape.enclosed_mass(
                r, profile, values, context), params, context)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._sum(
            lambda shape, profile, values: shape.density(
                R, z, profile, values, context), params, context)

    def gives_density(self, thick: Collection[str]) -> bool:
        return True

    def gives_surface_density(self) -> bool:
        return True

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # (that of the profile of each term, whatever its shape, see
        # MassSpheroid.surface_density)
        return self._sum(
            lambda shape, profile, values: profile.surface_density(
                R, values, context), params, context)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        forces = [
            shape.forces(R, z, profile, values, context)
            for shape, profile, values in self._terms(context)]
        scale = self._upsilon(params, context)
        return (scale * sum(force[0] for force in forces),
                scale * sum(force[1] for force in forces))

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._sum(
            lambda shape, profile, values: shape.potential(
                R, z, profile, values, context), params, context)

    def _terms(self, context: MassContext) -> tuple[tuple[
            SpheroidShape, SpheroidProfile, ParamValues], ...]:
        """Return the terms of the density of the spheroid it follows."""
        density = context.density(self._component)
        if not isinstance(density, SpheroidDensity):
            raise ValueError(
                f"the mass component {self.name()!r} follows the density of "
                f"a spheroid, but that of {self._component!r} is not one")
        return density.terms()

    def _upsilon(self, params: ParamValues, context: MassContext) -> float:
        """
        Return the mass of one unit of the mass of the light of its terms
        (see SpheroidDensity): upsilon, times the mass of a unit of the
        density if it has a physical unit.
        """
        upsilon = params['upsilon']
        mass = context.density(self._component).mass_per_unit()
        return upsilon if mass is None else upsilon * mass

    def _sum(
            self,
            of_term: Callable[
                [SpheroidShape, SpheroidProfile, ParamValues], np.ndarray],
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the sum over its terms of what of_term gives of each, times
        upsilon (see _upsilon).
        """
        return self._upsilon(params, context) * sum(
            of_term(shape, profile, values)
            for shape, profile, values in self._terms(context))
