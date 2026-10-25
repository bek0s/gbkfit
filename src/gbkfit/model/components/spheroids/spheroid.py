"""
The spheroid component: the light of a spheroid of stars (e.g. a bulge,
or an elliptical galaxy), a light profile (LightProfile) laid out in a
shape.
"""

import abc
import functools
import math
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
import scipy.special

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import (
    ParamDesc, ParamScalarDesc, ParamValues, ParamVectorDesc)
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ...mass._numerics import (
    sersic_b, sersic_density_table, sersic_tables)
from ...mass import SpheroidDensity
from ...mass.profiles import ProfileGaussian, ProfileSersic, SpheroidProfile
from ...mass.shapes import (
    ShapeFlattened, ShapeSphere, SpheroidShape, spheroid_shape_parser)
from .._detail import dump_geometry, dump_name, geometry_desc
from .._needs import Need, NeedsForces, NeedsMass
from .._native import (
    NativeComponentPlan, NativePlan, ParamBuffer, emission_setup,
    evaluate_native, make_bands)
from .._units import DataUnit, mass_per_unit
from ..base import (
    Component, ComponentState, FieldsLayout, FrameParams, NativeGrid)
from ..geometries import Geometry
from ..lines import Band, LineCatalogue
from ..roles import (
    STARS, BandEmission, Emission, LineEmission, emission_parser)
from ._jeans import JeansKinematics
from ._kinematics import (
    SpheroidKinematics, SpheroidLight, SpheroidMotion, TraitsKinematics)
from ._stage import SpheroidStage


__all__ = [
    'LightMGE',
    'LightProfile',
    'LightSersic',
    'Spheroid',
    'light_profile_parser',
    'spheroid_kinematics_parser'
]


class LightProfile(parseutils.TypedSerializable, abc.ABC):
    """
    The light profile of a spheroid (see Spheroid): its density as a sum
    of terms, each a function of its own spheroidal radius m, m^2 = R^2 +
    z^2 / q^2 (R the distance from the axis of the spheroid and z the
    height along it, arcsec; q the axis ratio of the term, its height over
    its width): an amplitude times f(m / s), of the scale s, f(u) =
    exp(T(ln u) - c u^p), with c and p of the profile (see function) and
    T its table (see table), or 0 without one. Seen on the sky, a term
    projects to a term of the axis ratio q' (see projected_terms) of the
    function exp(-c u^p): its profile is a sum of Sersic profiles.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def takes_shape(self) -> bool:
        """
        Check whether it is laid out in a shape (see Spheroid), or has the
        axis ratios of its terms itself.
        """
        pass

    @abc.abstractmethod
    def nterms(self) -> int:
        """Return the number of its terms."""
        pass

    @abc.abstractmethod
    def ntable(self) -> int:
        """
        Return the number of points of the table of its function (see
        table); 0 without one.
        """
        pass

    @abc.abstractmethod
    def function(self, params: ParamValues) -> tuple[float, float]:
        """
        Return c and p of the function of its terms at the values of its
        parameters (see LightProfile).
        """
        pass

    @abc.abstractmethod
    def terms(self, params: ParamValues, q: float) -> np.ndarray:
        """
        Return its terms at the values of its parameters, laid out in a
        shape of the axis ratio q (1 for a sphere; ignored by a profile
        that does not take a shape): the amplitude (the density, in the
        unit of the emission per arcsec), the scale (arcsec) and the axis
        ratio of each, an array of shape (nterms, 3); NaN for a term of a
        scale or an axis ratio that is not greater than zero.
        """
        pass

    @abc.abstractmethod
    def projected_terms(
            self, params: ParamValues, q: float, incl: float
    ) -> np.ndarray:
        """
        Return the terms of its projection on the sky at the inclination
        incl (degrees; see terms): the amplitude (the surface density, in
        the unit of the emission), the scale and the projected axis ratio
        q' of each, q'^2 = cos^2(incl) + q^2 sin^2(incl), of the function
        exp(-c u^p) (without the table).
        """
        pass

    @abc.abstractmethod
    def cut(self, params: ParamValues, q: float, rmax: float) -> float:
        """
        Return the share of its light that the sphere of radius rmax
        (arcsec) around its centre cuts (see Spheroid), at the values of
        its parameters, laid out in a shape of axis ratio q (see terms):
        that of each term beyond its spheroidal radius rmax / max(1, q).
        """
        pass

    @abc.abstractmethod
    def mass_terms(
            self,
            params: ParamValues,
            shape: SpheroidShape | None,
            kpc_per_arcsec: float
    ) -> list[tuple[SpheroidShape, SpheroidProfile, ParamValues]]:
        """
        Return the terms of its density that the mass of the galaxy
        follows (see mass.SpheroidDensity), at the values of its
        parameters, laid out in the given shape (none for a profile that
        takes none), in a galaxy of the given length of an arcsec (kpc):
        the shape, the profile and the values of their parameters of each,
        of the total mass of its light (that times kpc^2), to any radius.
        """
        pass

    def table(self, params: ParamValues) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the table T of its function at the values of its
        parameters (see the native SpheroidArgs::table): T at ln u = zero
        + i * step for the ntable points i, and the range (zero, step,
        slope), where slope is that of T in ln u below the first point.

        Raises
        ------
        NotImplementedError
            If it has no table (see ntable).
        """
        raise NotImplementedError(
            f"the light profile '{self.type()}' has no table")


class LightSersic(LightProfile):
    """
    The spherical deprojection of a Sersic profile: seen face-on, the
    surface brightness ie exp(-b ((R / re)^(1/n) - 1)) of a sphere, of
    its brightness ie at the effective radius re (arcsec; within which
    half of its light is) and the index n, with b(n) exact; laid out in a
    shape (see Spheroid). Its density, infinite at the centre for n > 1,
    is computed numerically (see mass._numerics.sersic_density_table),
    for n in [0.3, 12]: f(u) = exp(T(ln u) - b u^(1/n)), the table T of
    ln f + b u^(1/n) is smooth. Its projection is exact.
    """

    @staticmethod
    def type() -> str:
        return 'sersic'

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(
            ie=ParamScalarDesc(
                desc="the surface brightness at the effective radius, "
                     "face-on (the unit of the emission)"),
            re=ParamScalarDesc(
                desc="the effective radius (arcsec)", minimum=0),
            n=ParamScalarDesc(
                desc="the Sersic index", minimum=0.3, maximum=12))

    def takes_shape(self) -> bool:
        return True

    def nterms(self) -> int:
        return 1

    def ntable(self) -> int:
        return _SERSIC_TABLE_SIZE

    def function(self, params: ParamValues) -> tuple[float, float]:
        n = float(params['n'])
        return sersic_b(n), 1 / n

    def terms(self, params: ParamValues, q: float) -> np.ndarray:
        # The face-on surface brightness ie at re is that of the density
        # ie / re f(r / re), with f the deprojection of exp(-b (u^(1/n) -
        # 1)); squashed into the axis ratio q, its density is that / q
        re, q = np.float64(params['re']), np.float64(q)
        with np.errstate(divide='ignore', invalid='ignore'):
            return _checked_terms(
                np.array([[params['ie'] / (re * q), re, q]]))

    def projected_terms(
            self, params: ParamValues, q: float, incl: float
    ) -> np.ndarray:
        # ie exp(-b (u^(1/n) - 1)) = ie e^b exp(-b u^(1/n)), of q' (the
        # projection of rho_sph(m) / q is Sigma_sph(m') / q')
        b = sersic_b(float(params['n']))
        qp = _projected_ratio(q, incl)
        with np.errstate(divide='ignore', invalid='ignore'):
            return _checked_terms(np.array([[
                params['ie'] * np.exp(b) / qp, params['re'], qp]]))

    def cut(self, params: ParamValues, q: float, rmax: float) -> float:
        tables = sersic_tables(float(params['n']))
        scale = max(1.0, q) * np.float64(params['re'])
        with np.errstate(divide='ignore', invalid='ignore'):
            return float(1 - tables.mass(rmax / scale))

    def mass_terms(
            self,
            params: ParamValues,
            shape: SpheroidShape | None,
            kpc_per_arcsec: float
    ) -> list[tuple[SpheroidShape, SpheroidProfile, ParamValues]]:
        # (a Sersic profile of the mass of its light, 2 pi n re^2 ie e^b
        # b^-2n Gamma(2n), in kpc)
        n, re = float(params['n']), params['re']
        b = sersic_b(n)
        light = 2 * np.pi * n * re ** 2 * params['ie'] * np.exp(b) \
            * b ** (-2 * n) * scipy.special.gamma(2 * n)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_m = np.log10(light * kpc_per_arcsec ** 2)
        values = dict(log_m=log_m, re=re * kpc_per_arcsec, n=n) \
            | {name: params[name] for name in shape.pdescs()}
        return [(shape, ProfileSersic(), values)]

    def table(self, params: ParamValues) -> tuple[np.ndarray, np.ndarray]:
        return _sersic_table(float(params['n']))


# The number of points of the tables of the Sersic profiles (see
# mass._numerics.sersic_density_table)
_SERSIC_TABLE_SIZE = 512


@functools.lru_cache(maxsize=64)
def _sersic_table(n: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the table of the Sersic profile of index n (see table): ln f
    + b u^(1/n), of f its deprojection (see
    mass._numerics.sersic_density_table).
    """
    log_f, (zero, step, _) = sersic_density_table(n, _SERSIC_TABLE_SIZE)
    u = np.exp(zero + step * np.arange(len(log_f)))
    table = log_f + sersic_b(n) * u ** (1 / n)
    return table, np.array([zero, step, (table[1] - table[0]) / step])


class LightMGE(LightProfile):
    """
    A multi-Gaussian expansion (MGE): a sum of ngauss flattened Gaussians,
    each of its own axis ratio: Gaussian k is, seen face-on, of the peak
    surface brightness i0[k] and the dispersion sigma[k] (arcsec), with
    the intrinsic axis ratio q[k] (its height over its width, flattened
    for q < 1): the density i0 / (sqrt(2 pi) sigma q) exp(-m^2 / (2
    sigma^2)), m^2 = R^2 + z^2 / q^2, of the total light 2 pi i0 sigma^2,
    exactly (the voxels sample it, see Spheroid: a Gaussian much thinner
    than a voxel needs finer ones). It has its own axis ratios: it takes
    no shape. Its parameters are vectors of ngauss elements.

    Parameters
    ----------
    ngauss : int
        The number of its Gaussians.

    Raises
    ------
    ConfigError
        If ngauss is not a positive integer.
    """

    @staticmethod
    def type() -> str:
        return 'mge'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(ngauss=self._ngauss)

    def __init__(self, ngauss: int):
        if isinstance(ngauss, bool) or not isinstance(ngauss, int) \
                or ngauss < 1:
            raise ConfigError(
                f"ngauss must be a positive integer; it is {ngauss!r}")
        self._ngauss = ngauss

    def ngauss(self) -> int:
        """Return the number of its Gaussians."""
        return self._ngauss

    def pdescs(self) -> dict[str, ParamDesc]:
        n = self._ngauss
        return dict(
            i0=ParamVectorDesc(
                n, desc="the peak surface brightness of each Gaussian, "
                        "face-on (the unit of the emission)"),
            sigma=ParamVectorDesc(
                n, desc="the dispersion of each Gaussian (arcsec)",
                minimum=0),
            q=ParamVectorDesc(
                n, desc="the intrinsic axis ratio of each Gaussian "
                        "(height over width)", minimum=0))

    def takes_shape(self) -> bool:
        return False

    def nterms(self) -> int:
        return self._ngauss

    def ntable(self) -> int:
        return 0

    def function(self, params: ParamValues) -> tuple[float, float]:
        return 0.5, 2.0

    def terms(self, params: ParamValues, q: float) -> np.ndarray:
        i0, sigma, ratio = (
            np.asarray(params[name], dtype=float)
            for name in ('i0', 'sigma', 'q'))
        with np.errstate(divide='ignore', invalid='ignore'):
            return _checked_terms(np.stack(
                (i0 / (np.sqrt(2 * np.pi) * sigma * ratio), sigma, ratio),
                axis=-1))

    def projected_terms(
            self, params: ParamValues, q: float, incl: float
    ) -> np.ndarray:
        i0, sigma, ratio = (
            np.asarray(params[name], dtype=float)
            for name in ('i0', 'sigma', 'q'))
        qp = _projected_ratio(ratio, incl)
        with np.errstate(divide='ignore', invalid='ignore'):
            return _checked_terms(
                np.stack((i0 / qp, sigma, qp), axis=-1))

    def cut(self, params: ParamValues, q: float, rmax: float) -> float:
        # The share of the light of a 3d Gaussian beyond x dispersions,
        # 1 - erf(x / sqrt(2)) + sqrt(2 / pi) x exp(-x^2 / 2), of each
        # Gaussian, weighted by its light
        i0, sigma, ratio = (
            np.asarray(params[name], dtype=float)
            for name in ('i0', 'sigma', 'q'))
        with np.errstate(divide='ignore', invalid='ignore'):
            x = rmax / (np.maximum(1, ratio) * sigma)
            beyond = scipy.special.erfc(x / np.sqrt(2)) \
                + np.sqrt(2 / np.pi) * x * np.exp(-x ** 2 / 2)
            light = i0 * sigma ** 2
            return float(np.sum(light * beyond) / np.sum(light))

    def mass_terms(
            self,
            params: ParamValues,
            shape: SpheroidShape | None,
            kpc_per_arcsec: float
    ) -> list[tuple[SpheroidShape, SpheroidProfile, ParamValues]]:
        # (a flattened spherical Gaussian of the mass of its light, 2 pi i0
        # sigma^2, in kpc, of each)
        flattened, gaussian = ShapeFlattened(), ProfileGaussian()
        terms = []
        for i0, sigma, q in zip(*(
                np.asarray(params[name], dtype=float)
                for name in ('i0', 'sigma', 'q'))):
            with np.errstate(divide='ignore', invalid='ignore'):
                log_m = np.log10(
                    2 * np.pi * i0 * (sigma * kpc_per_arcsec) ** 2)
            terms.append((flattened, gaussian, dict(
                log_m=log_m, sigma=sigma * kpc_per_arcsec, q=q)))
        return terms


def _projected_ratio(q: npt.ArrayLike, incl: float) -> np.ndarray:
    """
    Return the axis ratio q' of the projection of a spheroid of the axis
    ratio q at the inclination incl (degrees): q'^2 = cos^2(incl) + q^2
    sin^2(incl), complete the square of m^2 along the line of sight
    (face-on 1, edge-on q).
    """
    i = np.radians(incl)
    return np.sqrt(np.cos(i) ** 2 + np.asarray(q) ** 2 * np.sin(i) ** 2)


def _checked_terms(terms: np.ndarray) -> np.ndarray:
    """
    Return the terms of a light profile (see LightProfile.terms), with
    those of a scale or an axis ratio that is not greater than zero (e.g.
    at the bound 0 of its parameter) all NaN: they make the density NaN,
    so that they are seen.
    """
    invalid = ~((terms[:, 1] > 0) & (terms[:, 2] > 0))
    terms[invalid] = np.nan
    return terms


light_profile_parser = parseutils.TypedParser(LightProfile, [
    LightSersic,
    LightMGE])

spheroid_kinematics_parser = parseutils.TypedParser(SpheroidKinematics, [
    JeansKinematics,
    TraitsKinematics])


# The geometric parameters of a spheroid: its centre, the orientation of
# its axis unless it is a sphere without kinematics, and with kinematics,
# its systemic velocity
GEOMETRY_PDESCS = dict(
    xpos=ParamScalarDesc(),
    ypos=ParamScalarDesc(),
    posa=ParamScalarDesc(),
    incl=ParamScalarDesc(),
    vsys=ParamScalarDesc())


class Spheroid(Component):
    """
    A spheroid of stars: the density of its light profile, laid out in
    its shape (unless the profile has its own axis ratios, see LightMGE),
    and its kinematics, if any (see SpheroidKinematics; with a systemic
    velocity vsys). Its axis is the spin axis of a disk of its posa and
    incl; a sphere without kinematics has none (no posa or incl). It emits
    in bands, seen by images, or as stars (see StarsEmission), seen by
    images and by the spectral cubes that ask for them, with kinematics.

    On a grid without a depth axis (e.g. the line of sight flat), it is
    its projection on the sky, exactly, to any radius; its kinematics need
    one. On a grid with a depth axis (where dust can dim it, or for its
    kinematics), it is evaluated at every voxel,
    within the sphere of radius rmax around its centre: each term of the
    profile (see LightProfile) up to its spheroidal radius m = rmax /
    max(1, q), the largest spheroid of its axis ratio q in the sphere; a
    warning says when that cuts more than 1% of its light. The voxels
    (or pixels) near its centre hold the mean of its density over them
    (where it can be infinite, e.g. a Sersic profile of n > 1), the
    others the density at their centre.

    Parameters
    ----------
    profile : LightProfile
        Its light profile.
    rmax : float
        How far it is drawn on a grid with a depth axis: the radius
        (arcsec) of the sphere around its centre that holds its light
        there, which sets the depth axis (see Component.bounding_radius).
    emission : Emission
        The light it emits, in bands (BandEmission).
    shape : SpheroidShape, optional
        Its shape (see mass.shapes): a sphere (the default), or flattened
        of the axis ratio q (a parameter), with the face-on projection of
        the sphere; none for a profile with its own axis ratios.
    kinematics : SpheroidKinematics, optional
        How its stars move: in the gravity of its galaxy (JeansKinematics)
        or as traits give (TraitsKinematics); none by default.
    geometry : Geometry, 'galaxy' or 'own', optional
        The geometry whose centre and axis it shares: that of its galaxy
        (by default), a named one (not a loose or tilted one), or none
        ('own'; see Component).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If rmax is not positive, the emission is not in bands, the profile
        has its own axis ratios and a shape is given, the geometry is
        loose or tilted, or the kinematics cannot move in it (see
        SpheroidKinematics.motion).
    """

    @staticmethod
    def type() -> str:
        return 'spheroid'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        parseutils.load_option_and_update_info(
            light_profile_parser, info, 'profile', required=True)
        parseutils.load_option_and_update_info(
            spheroid_shape_parser, info, 'shape')
        parseutils.load_option_and_update_info(
            spheroid_kinematics_parser, info, 'kinematics')
        parseutils.load_option_and_update_info(
            emission_parser, info, 'emission', required=True,
            catalogue=catalogue)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = dict(type=self.type()) | dump_name(self) \
            | dump_geometry(self) | dict(
            profile=light_profile_parser.dump(self._profile),
            rmax=self._rmax)
        if self._profile.takes_shape():
            info.update(shape=spheroid_shape_parser.dump(self._shape))
        if self._kinematics is not None:
            info.update(kinematics=spheroid_kinematics_parser.dump(
                self._kinematics))
        info.update(emission=emission_parser.dump(self.emission()))
        return info

    def __init__(
            self,
            profile: LightProfile,
            rmax: float,
            emission: Emission,
            shape: SpheroidShape | None = None,
            kinematics: SpheroidKinematics | None = None,
            geometry: Geometry | Literal['galaxy', 'own'] = 'galaxy',
            name: str | None = None
    ):
        if isinstance(geometry, Geometry) and (
                geometry.loose() or geometry.tilted()):
            raise ConfigError(
                f"a spheroid has one centre and axis, but "
                f"{geometry_desc(geometry)} is loose or tilted; give it "
                f"geometry 'own'")
        if not isinstance(emission, BandEmission):
            raise ConfigError(
                "a spheroid emits in bands (emission type: band)")
        if isinstance(rmax, bool) or not isinstance(rmax, (int, float)) \
                or not (math.isfinite(rmax) and rmax > 0):
            raise ConfigError(
                f"rmax must be a finite number greater than zero; it is "
                f"{rmax!r}")
        if not profile.takes_shape() and shape is not None:
            raise ConfigError(
                f"the light profile '{profile.type()}' has the axis ratios "
                f"of its terms: it takes no shape")
        super().__init__(name, geometry, emission, None)
        if shape is None and profile.takes_shape():
            shape = ShapeSphere()
        self._profile = profile
        self._rmax = float(rmax)
        self._shape = shape
        self._kinematics = kinematics
        self._motion = kinematics.motion(self._rmax) \
            if kinematics is not None else None
        axis = kinematics is not None \
            or not (shape is not None and shape.spherical())
        self._geometry_pdescs = {
            name: pdesc for name, pdesc in GEOMETRY_PDESCS.items()
            if name in ('xpos', 'ypos')
            or (name in ('posa', 'incl') and axis)
            or (name == 'vsys' and kinematics is not None)}
        own = self._geometry_pdescs | profile.pdescs() \
            | (shape.pdescs() if shape is not None else {}) \
            | (self._motion.pdescs() if self._motion is not None else {})
        self._bands = make_bands(own, emission)
        self._pdescs = own | self._bands.pdescs()

    def profile(self) -> LightProfile:
        """Return its light profile."""
        return self._profile

    def shape(self) -> SpheroidShape | None:
        """Return its shape; none for a profile with its axis ratios."""
        return self._shape

    def kinematics(self) -> SpheroidKinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def motion(self) -> SpheroidMotion | None:
        """Return the motion of its kinematics, if any."""
        return self._motion

    def rmax(self) -> float:
        """Return the radius (arcsec) beyond which it has no light."""
        return self._rmax

    def axis_ratio(self, params: ParamValues) -> float:
        """
        Return the axis ratio of its shape at the values of its parameters
        (1 for a sphere, or a profile with its own axis ratios).
        """
        shape = self._shape
        return params['q'] if shape is not None and not shape.spherical() \
            else 1.0

    def light(self, params: ParamValues) -> SpheroidLight:
        """
        Return its light at the values of its parameters, as its kinematics
        take it (see _jeans.light): the terms of its profile, their
        function and its table.
        """
        profile = self._profile
        return (profile.terms(params, self.axis_ratio(params)),
                profile.function(params),
                profile.table(params) if profile.ntable() else None)

    def thick(self, seen_by: 'type[Emission]') -> bool:
        # (an image is its projection; in a spectral cube, its velocities
        # change along the line of sight)
        return issubclass(seen_by, LineEmission)

    def frame_params(self) -> FrameParams:
        names = [
            name if name in self._geometry_pdescs else None
            for name in ('vsys', 'posa', 'incl')]
        return FrameParams('xpos', 'ypos', *names)

    def uses_mass(self) -> bool:
        return self._kinematics is not None and self._kinematics.uses_mass()

    def needs(self) -> tuple[Need, ...]:
        if not self.uses_mass():
            return ()
        label = (
            f"the {self._kinematics.type()} kinematics of the spheroid "
            f"{self.name()!r}")
        return (
            NeedsMass(f"{label} need the mass of its galaxy"),
            NeedsForces(f"{label} take the forces of its gravity off its "
                        f"plane"))

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def bounding_radius(self) -> float:
        return self._rmax

    def extent_cut(self, params: ParamValues) -> tuple[str, float]:
        return 'rmax', self._profile.cut(
            params, self.axis_ratio(params), self._rmax)

    def density_kind(self) -> 'type[SpheroidDensity]':
        return SpheroidDensity

    def plan_stage(
            self,
            driver: Driver,
            sky_origin: tuple[float, float] | None,
            followed: bool,
            label: str
    ) -> SpheroidStage | None:
        # (it has a part when the mass follows it, or for its kinematics in
        # the gravity)
        if not followed and not self.uses_mass():
            return None
        return SpheroidStage(self, mass_per_unit(self.emission()))

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            data_unit: DataUnit | None,
            redshift: float
    ) -> NativeComponentPlan:
        if lines is not None and STARS in lines and self._kinematics is None:
            raise ConfigError(
                "a spectral cube sees the stars of a spheroid with "
                "kinematics only (e.g. kinematics: {type: jeans}, or "
                "{type: traits})")
        emission = emission_setup(
            None, self._bands, self.emission().unit(), spectral, lines,
            band, emits, data_unit, redshift, ())
        return NativeComponentPlan(
            driver, SpheroidPlan(self, driver, emission.nlines, dtype),
            emission, opacity_ratios)


class SpheroidPlan(NativePlan):
    """
    The evaluation of a Spheroid on a driver and dtype: its native
    description, and the values of its geometric parameters, of the terms
    of its profile and their function (and table, if any), and of the
    motion of its kinematics, if any (see SpheroidMotion), in one buffer
    (see ParamBuffer), written at each evaluation: those of its density on
    a grid with a depth axis, and those of its projection on a grid
    without one.
    """

    def __init__(
            self, spheroid: Spheroid, driver: Driver, nlines: int,
            dtype: np.dtype
    ):
        """
        nlines is the number of the lines it adds to spectral cubes (0
        without them; see the native GridArgs::lines).
        """
        # The groups of the buffer: those of the density, those of the
        # motion, if any, and the lines
        profile = spheroid.profile()
        motion = spheroid.motion()
        density = dict.fromkeys(GEOMETRY_PDESCS, 1) \
            | dict(terms=3 * profile.nterms(), function=2)
        if ntable := profile.ntable():
            density |= dict(table=ntable, table_range=3)
        sizes = density | (motion.sizes() if motion is not None else {}) \
            | dict(lines=3 * nlines)
        buffer = ParamBuffer(driver, dtype, sizes)
        self._native_spheroid = driver.native_class('Spheroid', dtype)(
            **{name: buffer.device(name) for name in density},
            rmax=spheroid.rmax(),
            **(motion.native(driver, dtype, buffer)
               if motion is not None else {}))
        self._lines_h = buffer.host('lines').reshape(nlines, 3)
        self._lines_d = buffer.device('lines').reshape(nlines, 3) \
            if nlines else None
        self._spheroid = spheroid
        self._driver = driver
        self._dtype = dtype
        self._buffer = buffer
        self._backend = driver.native_class('GModel', dtype)()

    def evaluate(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None,
            density_key: str,
            lines: np.ndarray | None = None,
            rdata_scale: float = 1.0,
            image_scale: float = 1.0,
            state: ComponentState | None = None
    ) -> None:
        # (without a depth axis, the z axis of the grid has size 1 and step
        # 0, see NativeGrid)
        projected = grid['spat_step'][2] == 0
        self._write_params(params, projected, lines, state)
        moves = self._spheroid.kinematics() is not None
        evaluate_native(
            lambda kwargs: self._backend.smspheroid_evaluate(
                self._native_spheroid, projected=projected, **kwargs),
            self._driver, self._dtype, grid, outputs, out_extra,
            density_key, self._lines_d, rdata_scale, image_scale,
            velocity=moves, dispersion=moves)

    def fields_layout(self) -> FieldsLayout:
        # (a spheroid fills the voxels with j, v and s)
        return FieldsLayout(layer=False, nslots=3)

    def _write_params(
            self,
            params: ParamValues,
            projected: bool,
            lines: np.ndarray | None,
            state: ComponentState | None
    ) -> None:
        """
        Write the parameter values into the host memory: the geometry (a
        sphere without kinematics has posa and incl 0), the terms of the
        profile, or of its projection, their function and table, the values
        of the lines, if given, and those of the motion of its kinematics,
        if any (with its state), and copy them to the device at once.
        """
        buffer = self._buffer
        spheroid = self._spheroid
        for name in GEOMETRY_PDESCS:
            buffer.host(name)[:] = params.get(name, 0.0)
        q = spheroid.axis_ratio(params)
        profile = spheroid.profile()
        terms = profile.projected_terms(params, q, params.get('incl', 0.0)) \
            if projected else profile.terms(params, q)
        buffer.host('terms')[:] = terms.ravel()
        buffer.host('function')[:] = profile.function(params)
        if profile.ntable():
            table, table_range = profile.table(params)
            buffer.host('table')[:] = table
            buffer.host('table_range')[:] = table_range
        if lines is not None:
            self._lines_h[:] = lines
        if (motion := spheroid.motion()) is not None:
            motion.write(buffer, params, state)
        buffer.copy_to_device()
