"""
The traits of the components made of nodes (disks and cones): the
profiles of their properties, as functions of the radius r and the
azimuth theta on the surface of a component (polar traits), or of the
height above it (height traits).

The surface of a disk is its plane, and its nodes are rings: r is the
radius in the plane, and the height is z, the height above the plane
(above the displaced plane with vertical distortion traits). The surface
of a cone is the mid-surface of its walls, at the angle wall from its
axis, and its nodes are shells, spheres around its apex: r is the
distance from the apex, and the height is d, the distance from the
mid-surface across the wall. The axis of a cone is the spin axis of a
disk of its inclination incl and position angle posa, and theta is the
azimuth around the axis, as in that disk.

The kinds of traits, and how the traits of one kind combine:

- density (rpt, rht): the density per unit area of the surface (of a
  disk, seen face-on): the brightness of a component that emits, the
  optical depth of dust; the sum of the polar traits, each times its
  height trait in a thick component (a probability density of the
  height, which integrates to 1).
- velocity (vpt, vht): the line-of-sight velocity; vsys plus the sum of
  the polar traits, each a speed along its direction projected on the
  line of sight, times its height trait (a factor of the height, 1 at 0
  for most).
- dispersion (dpt, dht): the velocity dispersion; the absolute value of
  the sum of the polar traits, each times its height trait.
- vertical distortion (zpt): the sum displaces the plane of a disk (its
  midplane) along z, each within its zmax.
- selection (spt): 1 where a trait selects the component, 0 elsewhere;
  a part of the component is kept if every trait selects it.

The height traits pair with the polar traits of their kind by their
order, and a component has at most four traits of a kind. Each type of
component takes its own list of traits (see _nodal.TraitTable): the
cones, e.g., take no vertical distortion, mixture or map traits.

Coordinates and units: theta is from the major axis on the side of the
position angle posa (the receding side when the rotation is positive),
increasing towards the far side (theta = 90 degrees, on the sky at the
position angle posa + 90 degrees). z is positive away from the viewer
for inclinations below 90 degrees, as is the positive side of the axis
of a cone; the height traits depend on |z| or |d| only. Lengths are in
arcsec, angles in degrees and velocities in km/s. The traits are
evaluated between the first and the last radius of the nodes of a
component only.

A velocity polar trait is a speed v along its direction (its option
direction; the trait mass is tangential), and its speed parameter is
named after it: tangential ('tan', vt, around the axis), radial ('rad',
vr, in the plane of a disk, positive outwards), vertical ('ver', vv,
along z, positive along it), outward ('out', vo, along the radius of
the sphere about the centre of a disk or the apex of a cone, positive
away from it: on a disk, away from its plane on either side, e.g. a
wind of its gas) or the line of sight ('los', vl). On a disk, the
speeds are projected on the line of sight as v cos(theta) sin(incl), v
sin(theta) sin(incl), v cos(incl), v (sin(a) sin(theta) sin(incl) +
cos(a) cos(incl)) and v, a the angle of the point from the positive
side of the axis (sin(a) = R / r and cos(a) = z / r, at the radius R,
the height z and r = sqrt(R^2 + z^2)); on a cone, the tangential and
outward ones as on a disk. The tangential and radial ones are 0 on the
axis of a disk, the tangential one on the axis of a cone, and the
outward ones at the centre of a disk and the apex of a cone, where
their directions are undefined.

The planar velocity polar traits of a disk (e.g. the flow along a bar)
are not speeds along one direction, but velocities in its plane, with a
radial and a tangential part: they have no option direction, and their
velocity is projected on the line of sight as v_y sin(incl), v_y its
part along the azimuth 90 degrees (see VPTraitPlanar).

Node-wise parameters (those of the nw_ traits, and of the height traits
with rnodes) have a value for each node: at its radii (sampling
'rnodes'), interpolated to the nodes the component is evaluated on (its
subnodes, see rstep) with the interpolation of the component (interp),
or at those subnodes (sampling 'subrings'); between the subnodes, they
are interpolated linearly.

The mixture traits are sums of elliptical blobs. Blob i is centred at
the radius r_i and the azimuth t_i, has the amplitude a_i, the scale s_i
along its major axis and q_i s_i along its minor axis, and its major
axis at the azimuth t_i + p_i + 90 degrees (p_i = 0: tangential, 90:
radial); rho is the elliptical distance from its centre, sqrt(x^2 + y^2
/ q_i^2) along its axes.

The map trait is a grid of square cells in the plane of the disk,
centred on its centre, x along the azimuth 0 and y along the azimuth 90
degrees, with one density for each cell (a free map).

The Monte Carlo disks draw their clouds from the density traits: from
each subring, where the trait is flat, for most, and exactly for the
uniform, mixture and map traits.
"""

import abc
import dataclasses
import numbers
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.special

import gbkfit.math
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


# Density polar traits
RPT_UID_UNIFORM = 1
RPT_UID_EXP = 2
RPT_UID_GAUSS = 3
RPT_UID_GGAUSS = 4
RPT_UID_LORENTZ = 5
RPT_UID_MOFFAT = 6
RPT_UID_SECH2 = 7
RPT_UID_MIXTURE_EXP = 51
RPT_UID_MIXTURE_GAUSS = 52
RPT_UID_MIXTURE_GGAUSS = 53
RPT_UID_MIXTURE_MOFFAT = 54
RPT_UID_NW_UNIFORM = 101
RPT_UID_NW_HARMONIC = 102
RPT_UID_NW_DISTORTION = 103
RPT_UID_MAP = 151

# Density height traits
RHT_UID_UNIFORM = 1
RHT_UID_EXP = 2
RHT_UID_GAUSS = 3
RHT_UID_GGAUSS = 4
RHT_UID_LORENTZ = 5
RHT_UID_MOFFAT = 6
RHT_UID_SECH2 = 7

# Velocity polar traits: the shapes of their speeds
VPT_UID_UNIFORM = 1
VPT_UID_ARCTAN = 2
VPT_UID_BOISSIER = 3
VPT_UID_EPINAT = 4
VPT_UID_LRAMP = 5
VPT_UID_TANH = 6
VPT_UID_POLYEX = 7
VPT_UID_RIX = 8
VPT_UID_COURTEAU = 9
VPT_UID_BRANDT = 10
VPT_UID_ISO = 11
VPT_UID_NFW = 12
VPT_UID_WEDGE = 13
VPT_UID_BAR = 14
VPT_UID_NW_UNIFORM = 101
VPT_UID_NW_HARMONIC = 102
VPT_UID_NW_BAR = 103

# Velocity polar traits: the directions of their speeds
VPT_DIRECTION_TAN = 1
VPT_DIRECTION_RAD = 2
VPT_DIRECTION_VER = 3
VPT_DIRECTION_LOS = 4
VPT_DIRECTION_OUT = 5
VPT_DIRECTION_PLANE = 6

# Velocity height traits
VHT_UID_ONE = 1
VHT_UID_LINEAR = 2
VHT_UID_EXP = 3
VHT_UID_GAUSS = 4
VHT_UID_MASS = 5

# The heights of the table of the velocity height trait 'mass' (see
# VHTraitMass): this many, from 0 to its zmax
VHT_MASS_NHEIGHTS = 33

# Dispersion polar traits
DPT_UID_UNIFORM = 1
DPT_UID_EXP = 2
DPT_UID_GAUSS = 3
DPT_UID_GGAUSS = 4
DPT_UID_LORENTZ = 5
DPT_UID_MOFFAT = 6
DPT_UID_SECH2 = 7
DPT_UID_MIXTURE_EXP = 51
DPT_UID_MIXTURE_GAUSS = 52
DPT_UID_MIXTURE_GGAUSS = 53
DPT_UID_MIXTURE_MOFFAT = 54
DPT_UID_NW_UNIFORM = 101
DPT_UID_NW_HARMONIC = 102
DPT_UID_NW_DISTORTION = 103

# Dispersion height traits
DHT_UID_ONE = 1
DHT_UID_LINEAR = 2
DHT_UID_EXP = 3
DHT_UID_GAUSS = 4

# Vertical distortion polar traits
ZPT_UID_NW_UNIFORM = 101
ZPT_UID_NW_HARMONIC = 102

# Selection polar traits
SPT_UID_AZRANGE = 1
SPT_UID_RRANGE = 2
SPT_UID_NW_AZRANGE = 101

# Where the values of the node-wise parameters of a trait are given: at
# the radial nodes of its disk, which interpolates them to the subnodes,
# or at the subnodes themselves (see _disk.Disk)
SAMPLINGS = ('rnodes', 'subrings')
SAMPLING_DEFAULT = 'rnodes'


@dataclasses.dataclass(frozen=True)
class VPTDirection:
    """
    A direction of the speeds of the velocity polar traits: its uid in the
    native kernels, and the name of the speed parameter of the traits
    along it.
    """
    uid: int
    speed: str


# The directions of the speeds of the velocity polar traits, by name (see
# the module docstring)
VPT_DIRECTIONS = dict(
    tan=VPTDirection(VPT_DIRECTION_TAN, 'vt'),
    rad=VPTDirection(VPT_DIRECTION_RAD, 'vr'),
    ver=VPTDirection(VPT_DIRECTION_VER, 'vv'),
    los=VPTDirection(VPT_DIRECTION_LOS, 'vl'),
    out=VPTDirection(VPT_DIRECTION_OUT, 'vo'))


def _ptrait_params_mixture_6p(nblobs: int) -> dict[str, ParamDesc]:
    return dict(
        r=ParamVectorDesc(nblobs),  # polar coord (radius)
        t=ParamVectorDesc(nblobs),  # polar coord (angle)
        a=ParamVectorDesc(nblobs),  # amplitude
        s=ParamVectorDesc(nblobs),  # size
        q=ParamVectorDesc(nblobs),  # axis ratio (minor/major)
        p=ParamVectorDesc(nblobs))  # position angle relative to t


def _ptrait_params_mixture_7p(nblobs: int) -> dict[str, ParamDesc]:
    return dict(
        r=ParamVectorDesc(nblobs),  # polar coord (radius)
        t=ParamVectorDesc(nblobs),  # polar coord (angle)
        a=ParamVectorDesc(nblobs),  # amplitude
        s=ParamVectorDesc(nblobs),  # size
        b=ParamVectorDesc(nblobs),  # shape
        q=ParamVectorDesc(nblobs),  # axis ratio (minor/major)
        p=ParamVectorDesc(nblobs))  # position angle relative to t


def _ptrait_params_nw_harmonic(
        order: int, nnodes: int
) -> dict[str, ParamDesc]:
    params = dict(a=ParamVectorDesc(nnodes))
    if order > 0:
        params.update(p=ParamVectorDesc(nnodes))
    return params


def _ptrait_params_nw_distortion(nnodes: int) -> dict[str, ParamDesc]:
    return dict(
        a=ParamVectorDesc(nnodes),
        p=ParamVectorDesc(nnodes),
        s=ParamVectorDesc(nnodes))


def _integrate_rings(
        rings: np.ndarray,
        fun: Callable[..., np.ndarray],
        *args: np.ndarray
) -> np.ndarray:
    # ring centers, equally spaced, same width
    rsep = rings[1] - rings[0]
    # Minimum radius of the rings
    rmin = rings - rsep * 0.5
    # Maximum radius of the rings
    rmax = rings + rsep * 0.5
    # Calculate the amplitude of each ring
    ampl = fun(rings, *args)
    return ampl * np.pi * (rmax * rmax - rmin * rmin)


def _ptrait_integrate_uniform(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    rsep = rings[1] - rings[0]
    rmin = rings[0] - 0.5 * rsep
    rmax = rings[-1] + 0.5 * rsep
    return np.pi * a * (rmax * rmax - rmin * rmin)


def _ptrait_integrate_exponential(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.expon_1d_fun, a, 0, s)


def _ptrait_integrate_gauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.gauss_1d_fun, a, 0, s)


def _ptrait_integrate_ggauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    b = params['b']
    return _integrate_rings(rings, gbkfit.math.ggauss_1d_fun, a, 0, s, b)


def _ptrait_integrate_lorentz(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.lorentz_1d_fun, a, 0, s)


def _ptrait_integrate_moffat(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    b = params['b']
    return _integrate_rings(rings, gbkfit.math.moffat_1d_fun, a, 0, s, b)


def _ptrait_integrate_sech2(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.sech2_1d_fun, a, 0, s)


# The azimuthal means split the circles around narrow structure at
# distances along them of at least this fraction of its size, and up to
# this multiple of the larger of its size and its distance (see
# _splits_around)
_SPLIT_FINEST = 2.0 ** -8
_SPLIT_WIDEST = 8


def _splits_around(
        r: np.ndarray, t: np.ndarray, size: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """
    Return the azimuths (degrees) at which the azimuthal means split the
    circles of the radii r around structures at the azimuths t (degrees),
    of the sizes size (arcsec, along the circles), whose centres are at
    the distances distance (arcsec) from the circles: at t (where a cusp
    can be), and on both sides at the distances along the circle (up to
    half a turn) that double from the larger of the distance and
    _SPLIT_FINEST times the size up to _SPLIT_WIDEST times the larger of
    the size and the distance (on a circle, a structure varies on the
    larger of its size and its distance). Between two, the structure is
    smooth. t, size and distance are arrays of shape (len(r), n); the
    result is of shape (len(r), m), NaN where a circle has fewer splits.
    """
    size = np.abs(size)
    finest = np.maximum(distance, _SPLIT_FINEST * size)
    widest = _SPLIT_WIDEST * np.maximum(size, distance)
    levels = int(np.ceil(np.log2(_SPLIT_WIDEST / _SPLIT_FINEST))) + 1
    lengths = finest[..., None] * 2.0 ** np.arange(levels)
    r = np.asarray(r, dtype=float)[:, None, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.rad2deg(lengths / r)
    w = np.where((lengths <= 2 * widest[..., None]) & (w < 180), w, np.nan)
    t = np.asarray(t, dtype=float)[..., None]
    return np.concatenate((t, t - w, t + w), axis=-1).reshape(len(r), -1)


def _splits_bar(r: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """
    Return the azimuths (degrees) at which the azimuthal means split the
    circles of the radii r across a flow along a bar (see VPTraitBar):
    where they cross the edges of its strip, at p - a, p + a, p + 180 - a
    and p + 180 + a (sin(a) = s / (2 r)), or within it, its axis (a = 90
    degrees), where the flow turns.
    """
    r = np.asarray(r, dtype=float)[:, None]
    p = params['p']
    ratio = np.divide(
        np.abs(params['s']) / 2, r, out=np.full_like(r, np.inf), where=r > 0)
    a = np.rad2deg(np.arcsin(np.minimum(ratio, 1)))
    return np.concatenate((p - a, p + a, p + 180 - a, p + 180 + a), axis=1)


def _splits_nw_distortion(
        r: np.ndarray,
        params: dict[str, np.ndarray],
        subrnodes: Sequence[float]
) -> np.ndarray:
    """
    Return the azimuths (degrees) at which the azimuthal means split the
    circles of the radii r around a distortion (see RPTraitNWDistortion):
    around its azimuth p, by its width s, and opposite (where its
    distance to p turns).
    """
    p = np.interp(r, subrnodes, params['p'])[:, None]
    s = np.interp(r, subrnodes, params['s'])[:, None]
    return np.concatenate(
        (_splits_around(r, p, s, np.zeros_like(s)), p + 180), axis=1)


def _ptrait_cloud_flux_mixture(
        params: dict[str, np.ndarray], norm: np.ndarray
) -> np.ndarray:
    """
    Return the flux of the clouds of a mixture of blobs (see the native
    rp_trait_mixture_rnd): the sum of the |amplitude| times the integral
    (norm, of a blob of amplitude 1) times the axis ratio of each blob.
    The kernel gives each cloud the sign of the amplitude of its blob.
    """
    a = np.abs(np.asarray(params['a']) * np.asarray(params['q']))
    return np.sum(a * norm)


def _ptrait_integrate_mixture_exponential(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    return _ptrait_cloud_flux_mixture(params, 2 * np.pi * s * s)


def _ptrait_integrate_mixture_gauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    return _ptrait_cloud_flux_mixture(params, 2 * np.pi * s * s)


def _ptrait_integrate_mixture_ggauss(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    b = np.asarray(params['b'])
    return _ptrait_cloud_flux_mixture(
        params, 2 * np.pi * s * s * scipy.special.gamma(2 / b) / b)


def _ptrait_integrate_mixture_moffat(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:  # noqa
    s = np.asarray(params['s'])
    b = np.asarray(params['b'])
    if np.any(b <= 1):
        raise RuntimeError(
            "the blobs of mixture_moffat of the Monte Carlo disk need b > 1 "
            "(their flux is infinite otherwise)")
    return _ptrait_cloud_flux_mixture(params, np.pi * s * s / (b - 1))


def _ptrait_integrate_nw_uniform(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    a = params['a']
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_cloud_flux_nw_harmonic(
        params: dict[str, np.ndarray], rings: np.ndarray, order: int
) -> np.ndarray:
    # The integral of |a cos(k (t - p))| around a ring is 2 / pi of that of
    # |a| for k > 0
    a = params['a'] * (2 / np.pi if order else 1)
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_integrate_nw_distortion(
        params: dict[str, np.ndarray], rings: np.ndarray
) -> np.ndarray:
    """
    Return the flux of each ring of a distortion: its area times the mean
    around it of a exp(-(t r)^2 / (2 s^2)), for the azimuths t within half
    a turn of the centre of the distortion.
    """
    a = params['a']
    s = np.abs(params['s'])

    def mean(r):
        width = s / r
        erf = scipy.special.erf(
            np.pi / (np.sqrt(2) * np.maximum(width, 1e-300)))
        return np.where(
            width > 0, np.sqrt(2 * np.pi) * width * erf / (2 * np.pi), 0) * a
    return _integrate_rings(rings, mean)


def trait_desc(cls: type['Trait']) -> str:
    descs = {
        RPTrait: 'density polar trait',
        RHTrait: 'density height trait',
        VPTrait: 'velocity polar trait',
        VHTrait: 'velocity height trait',
        DPTrait: 'velocity dispersion polar trait',
        DHTrait: 'velocity dispersion height trait',
        ZPTrait: 'vertical distortion polar trait',
        SPTrait: 'selection polar trait'}
    label = None
    for k, v in descs.items():
        if issubclass(cls, k):
            label = v
    assert label
    return parseutils.make_typed_desc(cls, label)


@dataclasses.dataclass(frozen=True)
class PressureSupport:
    """
    How the rotation of a disk in the gravity of its galaxy is corrected
    for its pressure (see VPTraitMass): mode, 'jeans' or
    'self_gravitating', and density, the names of the mass components of
    the galaxy whose densities it takes (their sum), or None for that of
    the disk.
    """
    mode: str
    density: tuple[str, ...] | None = None


class Trait(parseutils.TypedSerializable, abc.ABC):
    """
    A trait of a component made of nodes: a profile of one of its
    properties (see the module docstring). A trait declares its type (its name in configurations),
    its uid (its identifier in the native kernels), its parameters
    (params_sm, params_rnw) and its constants (consts).
    """

    @staticmethod
    @abc.abstractmethod
    def uid() -> int:
        """Return the identifier of the trait in the native kernels."""
        pass

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def consts(self) -> tuple[float, ...]:
        """
        Return the constants of the trait, in the order the kernels read
        them.
        """
        return ()

    def params_sm(self) -> dict[str, ParamDesc]:
        """Return its parameters that are not node-wise; none here."""
        return {}

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        """
        Return its node-wise parameters, of nnodes values each; none
        here.
        """
        return {}

    def sampling(self) -> str:
        """Return where the values of the node-wise parameters are given."""
        return SAMPLING_DEFAULT

    def circular_velocity_params(self) -> tuple[str, ...]:
        """
        Return the names of the node-wise parameters whose values are not
        given, but are the circular velocity of the mass of their galaxy
        at the radii where they are sampled (see VPTraitMass); none
        here.
        """
        return ()

    def pressure_support(self) -> PressureSupport | None:
        """
        Return how the values of its circular velocity parameters are
        corrected for the pressure of their disk, if they are (see
        VPTraitMass); not here.
        """
        return None

    def rotation_factor_params(self) -> dict[str, float]:
        """
        Return the node-wise parameters whose values are not given, but
        are the factors of the rotation in the gravity of their galaxy at
        a height, v(R, z) / v(R, 0), at the radii where they are sampled
        (see VHTraitMass): their heights (arcsec), by name; none here.
        """
        return {}


class TraitFeatureZmax:
    """
    The option zmax of a density height trait or a vertical distortion
    trait: a mixin.
    """

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(zmax=self.zmax())  # noqa

    def __init__(self, **kwargs: Any):
        zmax = kwargs.pop('zmax')
        if isinstance(zmax, bool) or not (
                isinstance(zmax, numbers.Real) and 0 < zmax < np.inf):
            raise ConfigError(
                f"zmax must be a positive length; it is {zmax!r}")
        self._zmax = float(zmax)
        super().__init__(**kwargs)

    def zmax(self) -> float:
        """
        Return its zmax (arcsec): a bound along z (see RHTrait and
        ZPTrait).
        """
        return self._zmax


class TraitFeatureDirection:
    """
    The option direction of a velocity polar trait: a mixin.
    """

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(direction=self.direction())  # noqa

    def __init__(self, **kwargs: Any):
        direction = kwargs.pop('direction')
        if direction not in VPT_DIRECTIONS:
            raise ConfigError(
                f"direction must be one of {list(VPT_DIRECTIONS)}; "
                f"it is {direction!r}")
        self._direction = direction
        super().__init__(**kwargs)

    def direction(self) -> str:
        """Return the direction of its speed (see VPT_DIRECTIONS)."""
        return self._direction


class TraitFeatureSampling:
    """
    The option sampling of a trait with node-wise parameters: a mixin.
    """

    def dump(self) -> dict[str, Any]:
        dump_sampling = self.sampling() != SAMPLING_DEFAULT
        info = dict(sampling=self.sampling()) if dump_sampling else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs: Any):
        sampling = kwargs.pop('sampling')
        if sampling not in SAMPLINGS:
            raise ConfigError(
                f"sampling must be one of {list(SAMPLINGS)}; "
                f"it is '{sampling}'")
        self._sampling = sampling
        super().__init__(**kwargs)

    def sampling(self) -> str:
        """Return where the values of its node-wise parameters are given."""
        return self._sampling


class TraitFeatureRNodes:
    """
    The option rnodes of a height trait, whose parameters can be
    node-wise: a mixin.
    """

    def dump(self) -> dict[str, Any]:
        dump_rnodes = self.rnodes()
        info = dict(rnodes=self.rnodes()) if dump_rnodes else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs: Any):
        self._rnodes = kwargs.pop('rnodes')
        super().__init__(**kwargs)

    def rnodes(self) -> bool:
        """Return whether its parameters are node-wise."""
        return self._rnodes


class TraitFeatureNBlobs:
    """
    The option nblobs of a mixture trait: a mixin.
    """

    def dump(self) -> dict[str, Any]:
        info = dict(nblobs=self.nblobs())
        return super().dump() | info  # noqa

    def __init__(self, **kwargs: Any):
        nblobs = kwargs.pop('nblobs')
        if not nblobs >= 1:
            raise ConfigError(f"nblobs must be at least 1; it is {nblobs}")
        self._nblobs = nblobs
        super().__init__(**kwargs)

    def nblobs(self) -> int:
        """Return the number of blobs."""
        return self._nblobs

    def consts(self) -> tuple[float, ...]:
        return (self.nblobs(),)

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        # Around each blob, by its smaller scale, at the distance of its
        # centre from each circle
        size = np.abs(params['s']) * np.minimum(1, np.abs(params['q']))
        shape = (len(r), self.nblobs())
        distance = np.abs(np.asarray(r, dtype=float)[:, None] - params['r'])
        return _splits_around(
            r, np.broadcast_to(params['t'], shape),
            np.broadcast_to(size, shape), distance)


class TraitFeatureOrder:
    """
    The option order of a harmonic trait: a mixin.
    """

    def dump(self) -> dict[str, Any]:
        info = dict(order=self.order())
        return super().dump() | info  # noqa

    def __init__(self, **kwargs: Any):
        order = kwargs.pop('order')
        if isinstance(order, bool) or not isinstance(order, numbers.Integral) \
                or order < 0:
            raise ConfigError(
                f"order must be an integer of at least 0; it is {order!r}")
        self._order = order
        super().__init__(**kwargs)

    def order(self) -> int:
        """Return the order of the harmonic."""
        return self._order

    def consts(self) -> tuple[float, ...]:
        return (self.order(),)


class PTrait(Trait, abc.ABC):
    """
    A polar trait: a function of the radius and the azimuth on the
    surface of its component.
    """

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        """
        Return the azimuths (degrees) at which the azimuthal means of the
        gravity of its galaxy split the circles of the radii r (arcsec, an
        array), so that the trait is smooth between two of them (see
        _profiles): where it jumps, and around its narrow structure (e.g.
        blobs). An array of shape (len(r), n), NaN where a circle has
        fewer than n; None if the trait is smooth on every circle (here).
        params has the values of its parameters, the node-wise ones at the
        subnodes subrnodes, between which the disk interpolates them
        linearly.
        """
        return None


class HTrait(
        TraitFeatureRNodes, TraitFeatureSampling, Trait,
        abc.ABC):
    """
    A height trait: a function of the height above the surface of its
    component, whose parameters can have a value for each node (rnodes). Raise
    ConfigError for a sampling without rnodes.
    """

    def __init__(self, rnodes: bool, sampling: str, **kwargs: Any):
        kwargs.update(rnodes=rnodes, sampling=sampling)
        super().__init__(**kwargs)
        if not self.rnodes() and self.sampling() != SAMPLING_DEFAULT:
            raise ConfigError(
                f"sampling is '{self.sampling()}', but rnodes is False: "
                f"the trait has no node-wise parameters")


class RPTrait(PTrait, abc.ABC):
    """
    A density polar trait: the density per unit area of the surface of
    its component (of a disk, seen face-on): light, or the optical depth
    of dust. The Monte Carlo disks draw their
    clouds from it (see has_analytical_integral and cloud_flux).
    """

    @abc.abstractmethod
    def has_analytical_integral(self) -> bool:
        """
        Check whether the Monte Carlo disk makes the clouds of the trait for
        the whole disk at once (True), or for each ring.
        """
        pass

    @abc.abstractmethod
    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        """
        Return the flux the Monte Carlo disk shares among the clouds of
        the trait: for the whole disk (one value, with an analytical
        integral) or for each of the given rings. It is the integral of
        |B| (B is the trait) with the sign of the amplitude of the trait.
        Where B changes sign along a ring, the kernel draws the clouds
        from |B| and gives each the sign of B relative to the amplitude
        (e.g. the sign of the cosine of a harmonic).
        """
        pass

    @abc.abstractmethod
    def axisymmetric(self) -> bool:
        """
        Check whether its value is the same at every azimuth (at each
        radius): then the azimuthal means of it that the gravity of its
        galaxy takes are its value at one azimuth (see DiskDensity and
        pressure_support).
        """
        pass


class RHTrait(TraitFeatureZmax, HTrait, abc.ABC):
    """
    A density height trait: the probability density of the height of
    the density above the surface (the midplane of a disk), which
    integrates to 1.

    The density lies within zmax of the surface (its option zmax), which
    vertical distortion traits displace: the profile is truncated there
    and renormalised, so that each column keeps its density (none where
    less than 0.001 of the profile is within zmax). With the zmax of the
    vertical distortion traits, it bounds a disk along z, so that the
    depth axis of an observation can hold all of it (see Depth).
    """

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(), self.rnodes())

    @abc.abstractmethod
    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        """
        Return the cumulative distribution of the profile, not truncated,
        at the heights z above the midplane, for the values of its
        parameters (s, or s and b), broadcast together.
        """
        pass


class VPTrait(PTrait, abc.ABC):
    """
    A velocity polar trait: a speed along a direction, projected on the
    line of sight (see the module docstring).
    """

    @abc.abstractmethod
    def direction(self) -> str:
        """Return the direction of its speed (see VPT_DIRECTIONS)."""
        pass

    def direction_uid(self) -> int:
        """Return the identifier of its direction in the native kernels."""
        return VPT_DIRECTIONS[self.direction()].uid

    def speed(self) -> str:
        """Return the name of its speed parameter, by its direction."""
        return VPT_DIRECTIONS[self.direction()].speed

    def axisymmetric(self) -> bool:
        """
        Check whether its speed is the same at every azimuth (at each
        radius): then the azimuthal means of it that the vertical
        equilibrium of its disk takes are its value at one azimuth (see
        parts.Kinematics); here, a speed of the radius only.
        """
        return True


class VPTraitPlanar(VPTrait, abc.ABC):
    """
    A planar velocity polar trait: a velocity in the plane of a disk, with
    a radial and a tangential part (see the module docstring). Its kernel
    gives v_y, its part along the azimuth 90 degrees, which is projected
    on the line of sight as v_y sin(incl).
    """

    def direction(self) -> str:
        return 'plane'

    def direction_uid(self) -> int:
        return VPT_DIRECTION_PLANE

    def axisymmetric(self) -> bool:
        return False


class VHTrait(HTrait, abc.ABC):
    """
    A velocity height trait: a factor of the height, which multiplies its
    velocity polar trait.
    """

    def consts(self) -> tuple[float, ...]:
        return (self.rnodes(),)


class DPTrait(PTrait, abc.ABC):
    """
    A velocity dispersion polar trait.
    """

    @abc.abstractmethod
    def axisymmetric(self) -> bool:
        """
        Check whether its value is the same at every azimuth (at each
        radius): then the azimuthal means of it that pressure support
        takes are its value at one azimuth (see pressure_support).
        """
        pass


class DHTrait(HTrait, abc.ABC):
    """
    A velocity dispersion height trait: a factor of the height, which
    multiplies its dispersion polar trait.
    """

    def consts(self) -> tuple[float, ...]:
        return (self.rnodes(),)


class ZPTrait(TraitFeatureZmax, PTrait, abc.ABC):
    """
    A vertical distortion polar trait: a displacement of the plane of the
    disk (its midplane) along z, within zmax of the plane of the galaxy
    (its option zmax), the valid range of its amplitude a. A larger one
    (e.g. of an interpolation of a between the rings that overshoots) is
    limited to zmax; with extra outputs, a warning says where. With the
    zmax of the density height traits, it bounds the disk along z, so that
    the depth axis of an observation can hold all of it (see Depth).
    """

    def _amplitude_desc(self, nnodes: int) -> ParamVectorDesc:
        """Return the description of its node-wise amplitude, a."""
        return ParamVectorDesc(
            nnodes, desc="the amplitude of the displacement (arcsec)",
            minimum=-self.zmax(), maximum=self.zmax())


class SPTrait(PTrait, abc.ABC):
    """
    A selection polar trait: 1 where it selects its component, 0
    elsewhere.
    """

    @abc.abstractmethod
    def axisymmetric(self) -> bool:
        """
        Check whether its value is the same at every azimuth (at each
        radius): then the azimuthal means of the density it selects, which
        the gravity of its galaxy takes, can be its value at one azimuth
        (see DiskDensity and pressure_support).
        """
        pass


class RPTraitUniform(RPTrait):
    """
    A uniform surface density: a. Its parameter: a, the density.
    Mainly for tests: its use is discouraged (a warning says so).
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def uid() -> int:
        return RPT_UID_UNIFORM

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_uniform(params, rings)


class RPTraitExponential(RPTrait):
    """
    An exponential surface density: a exp(-r / s). Its parameters: a,
    the density at the centre, and s, the scale length.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return RPT_UID_EXP

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_exponential(params, rings)


class RPTraitGauss(RPTrait):
    """
    A Gaussian surface density: a exp(-r^2 / (2 s^2)). Its parameters:
    a, the density at the centre, and s, the dispersion of the
    Gaussian.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return RPT_UID_GAUSS

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_gauss(params, rings)


class RPTraitGGauss(RPTrait):
    """
    A generalised Gaussian surface density: a exp(-(r / s)^b). Its
    parameters: a, the density at the centre, s, the scale length, and
    b, the shape (1 is the exponential, and 2 a Gaussian of dispersion s /
    sqrt(2)).
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def uid() -> int:
        return RPT_UID_GGAUSS

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc(),
            b=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_ggauss(params, rings)


class RPTraitLorentz(RPTrait):
    """
    A Lorentzian surface density: a s^2 / (r^2 + s^2). Its parameters:
    a, the density at the centre, and s, the half width at half
    maximum.
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def uid() -> int:
        return RPT_UID_LORENTZ

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_lorentz(params, rings)


class RPTraitMoffat(RPTrait):
    """
    A Moffat surface density: a (1 + (r / s)^2)^-b. Its parameters: a,
    the density at the centre, s, the core radius, and b, the power of
    the fall.
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MOFFAT

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc(),
            b=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_moffat(params, rings)


class RPTraitSech2(RPTrait):
    """
    A sech^2 surface density: a sech^2(r / s). Its parameters: a, the
    density at the centre, and s, the scale length.
    """

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def uid() -> int:
        return RPT_UID_SECH2

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_sech2(params, rings)


class RPTraitMixtureExponential(TraitFeatureNBlobs, RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
    docstring), each an exponential of the elliptical distance rho from
    its centre: a exp(-rho / s). Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, q, its axis ratio (minor
    over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_exponential'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MIXTURE_EXP

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_6p(self.nblobs())

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_exponential(params, rings)


class RPTraitMixtureGauss(TraitFeatureNBlobs, RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
    docstring), each a Gaussian of the elliptical distance rho from its
    centre: a exp(-rho^2 / (2 s^2)). Its parameters, vectors of a value
    for each blob: r and t, the radius and the azimuth of its centre, a,
    its amplitude, s, its scale along its major axis, q, its axis ratio
    (minor over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_gauss'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_6p(self.nblobs())

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_gauss(params, rings)


class RPTraitMixtureGGauss(TraitFeatureNBlobs, RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
    docstring), each a generalised Gaussian of the elliptical distance rho
    from its centre: a exp(-(rho / s)^b). Its parameters, vectors of a
    value for each blob: r and t, the radius and the azimuth of its
    centre, a, its amplitude, s, its scale along its major axis, b, its
    shape, q, its axis ratio (minor over major), and p, the angle of its
    major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_ggauss'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_7p(self.nblobs())

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_ggauss(params, rings)


class RPTraitMixtureMoffat(TraitFeatureNBlobs, RPTrait):
    """
    A surface density made of nblobs elliptical blobs (see the module
    docstring), each a Moffat of the elliptical distance rho from its
    centre: a (1 + (rho / s)^2)^-b. Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, b, its shape, q, its
    axis ratio (minor over major), and p, the angle of its major axis. The
    Monte Carlo disks need b > 1 (a finite flux).

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_moffat'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_7p(self.nblobs())

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_mixture_moffat(params, rings)


class RPTraitNWUniform(TraitFeatureSampling, RPTrait):
    """
    A surface density given at each ring: a(r). Its parameter: a,
    node-wise.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_uniform'

    @staticmethod
    def uid() -> int:
        return RPT_UID_NW_UNIFORM

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def axisymmetric(self) -> bool:
        return True

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(a=ParamVectorDesc(nnodes))

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_nw_uniform(params, rings)


class RPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, RPTrait):
    """
    A harmonic density: a(r) cos(m (theta - p(r))), of the order m. Its
    parameters, node-wise: a, the amplitude, and p (but for order 0), the
    azimuth of a maximum; a p that changes with r winds it into a spiral.
    It is negative in parts: it is meant to perturb an axisymmetric trait.

    Parameters
    ----------
    order : int
        The order m of the harmonic: the pattern repeats every 360 / m
        degrees, and order 0 is an axisymmetric a(r).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_harmonic'

    @staticmethod
    def uid() -> int:
        return RPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, sampling=sampling)

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self.order() == 0

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes)

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_cloud_flux_nw_harmonic(
            params, rings, self.order())


class RPTraitNWDistortion(TraitFeatureSampling, RPTrait):
    """
    A surface density around one azimuth at each ring: a(r)
    exp(-(dtheta r)^2 / (2 s(r)^2)), with dtheta the azimuth from p(r)
    (radians, within half a turn). Its parameters, node-wise: a, the peak,
    p, its azimuth, and s, its width along the ring.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_distortion'

    @staticmethod
    def uid() -> int:
        return RPT_UID_NW_DISTORTION

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def axisymmetric(self) -> bool:
        return False

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_distortion(nnodes)

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_nw_distortion(r, params, subrnodes)

    def has_analytical_integral(self) -> bool:
        return False

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        return _ptrait_integrate_nw_distortion(params, rings)


class RPTraitMap(RPTrait):
    """
    A free map: a surface density of nx by ny square cells of side step
    (arcsec) in the plane of the disk, centred on its centre, x along the
    azimuth 0 and y along the azimuth 90 degrees (see the module
    docstring). The cell (i, j) covers [(i - nx / 2) step, (i + 1 - nx /
    2) step) along x and likewise along y, and has the density a[j nx +
    i], of any sign, all over it; 0 outside the grid. Its parameter: a,
    the nx ny densities, row by row (as a numpy image of shape (ny, nx),
    x along its columns). A warped disk has the map in the plane of each
    of its rings, as it has its blobs.

    Parameters
    ----------
    size : Sequence of int
        The number of cells along x and along y, [nx, ny].
    step : float
        The side of the cells (arcsec).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'map'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MAP

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(size=list(self._size), step=self._step)

    def __init__(self, size: Sequence[int], step: float):
        if not (isinstance(size, Sequence) and len(size) == 2 and all(
                isinstance(n, numbers.Integral) and not isinstance(n, bool)
                and n >= 1 for n in size)):
            raise ConfigError(
                f"size must be the numbers of cells along x and y, [nx, "
                f"ny], each at least 1; it is {size!r}")
        if isinstance(step, bool) or not (
                isinstance(step, numbers.Real) and 0 < step < np.inf):
            raise ConfigError(
                f"step must be a positive length; it is {step!r}")
        self._size = (int(size[0]), int(size[1]))
        self._step = float(step)

    def size(self) -> tuple[int, int]:
        """Return the numbers of cells along x and y, (nx, ny)."""
        return self._size

    def step(self) -> float:
        """Return the side of the cells (arcsec)."""
        return self._step

    def consts(self) -> tuple[float, ...]:
        return (*self._size, self._step)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        nx, ny = self._size
        return dict(a=ParamVectorDesc(
            nx * ny, desc="the density of each cell, row by row"))

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], rings: np.ndarray
    ) -> np.ndarray:
        # The kernel draws the cells by |a| and gives each cloud the sign
        # of the density of its cell
        return np.sum(np.abs(params['a'])) * self._step ** 2

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        # The circles cross the lines x = xe at the azimuths +-acos(xe /
        # r), and the lines y = ye at asin(ye / r) and 180 - asin(ye / r);
        # jumps where they cross them on the grid
        nx, ny = self._size
        half_x, half_y = nx * self._step / 2, ny * self._step / 2
        xe = np.arange(nx + 1) * self._step - half_x
        ye = np.arange(ny + 1) * self._step - half_y
        r = np.asarray(r, dtype=float)[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            ax = np.rad2deg(np.arccos(xe / r))
            ay = np.rad2deg(np.arcsin(ye / r))
            cross_x = np.concatenate((ax, -ax), axis=1)
            cross_y = np.concatenate((ay, 180 - ay), axis=1)
            on_x = np.abs(r * np.sin(np.deg2rad(cross_x))) <= half_y
            on_y = np.abs(r * np.cos(np.deg2rad(cross_y))) <= half_x
        return np.concatenate(
            (np.where(on_x, cross_x, np.nan), np.where(on_y, cross_y, np.nan)),
            axis=1)


class RHTraitP1(RHTrait, abc.ABC):
    """
    A density height trait of one parameter, s.
    """

    def __init__(
            self,
            zmax: float,
            rnodes: bool = False,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(zmax=zmax, rnodes=rnodes, sampling=sampling)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else dict(s=ParamScalarDesc())

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else dict(s=ParamVectorDesc(nrnodes))


class RHTraitP2(RHTrait, abc.ABC):
    """
    A density height trait of two parameters, s and b.
    """

    def __init__(
            self,
            zmax: float,
            rnodes: bool = False,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(zmax=zmax, rnodes=rnodes, sampling=sampling)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else dict(
            s=ParamScalarDesc(),
            b=ParamScalarDesc())

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else dict(
            s=ParamVectorDesc(nrnodes),
            b=ParamVectorDesc(nrnodes))


class RHTraitUniform(RHTraitP1):
    """
    A uniform vertical distribution of the density: 1 / (2 s) for |z|
    <= s, 0 beyond. Its parameter: s, the half thickness. It integrates to
    1. Its use is discouraged: its sharp edges alias on the voxels (a
    warning says so).

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def uid() -> int:
        return RHT_UID_UNIFORM

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return np.clip((z + s) / (2 * s), 0, 1)


class RHTraitExponential(RHTraitP1):
    """
    An exponential vertical distribution of the density: exp(-|z| / s)
    / (2 s). Its parameter: s, the scale height. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return RHT_UID_EXP

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 + 0.5 * np.sign(z) * -np.expm1(-np.abs(z) / s)


class RHTraitGauss(RHTraitP1):
    """
    A Gaussian vertical distribution of the density: exp(-z^2 / (2
    s^2)) / (s sqrt(2 pi)). Its parameter: s, the dispersion of the
    Gaussian. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return RHT_UID_GAUSS

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 * (1 + scipy.special.erf(z / (s * np.sqrt(2))))


class RHTraitGGauss(RHTraitP2):
    """
    A generalised Gaussian vertical distribution of the density: b / (2
    s Gamma(1 / b)) exp(-(|z| / s)^b). Its parameters: s, the scale
    height, and b, the shape. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def uid() -> int:
        return RHT_UID_GGAUSS

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, b = params
        p = scipy.special.gammainc(1 / b, (np.abs(z) / s) ** b)
        return 0.5 + 0.5 * np.sign(z) * p

    def __init__(
            self,
            zmax: float,
            rnodes: bool = False,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(zmax=zmax, rnodes=rnodes, sampling=sampling)


class RHTraitLorentz(RHTraitP1):
    """
    A Lorentzian vertical distribution of the density: s / (pi (z^2 +
    s^2)). Its parameter: s, the half width at half maximum; its tails are
    heavy (zmax cuts them). It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def uid() -> int:
        return RHT_UID_LORENTZ

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 + np.arctan(z / s) / np.pi


class RHTraitMoffat(RHTraitP2):
    """
    A Moffat vertical distribution of the density: Gamma(b) / (Gamma(b
    - 1/2) s sqrt(pi)) (1 + (z / s)^2)^-b, for b > 1/2. Its parameters: s,
    the core height, and b, the power of the fall. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def uid() -> int:
        return RHT_UID_MOFFAT

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        # (the share beyond |z|, of the incomplete beta function)
        s, b = params
        u2 = (z / s) ** 2
        beyond = scipy.special.betainc(b - 0.5, 0.5, 1 / (1 + u2))
        return np.where(z < 0, beyond / 2, 1 - beyond / 2)


class RHTraitSech2(RHTraitP1):
    """
    A sech^2 vertical distribution of the density: sech^2(z / s) / (2
    s). Its parameter: s, the scale height. It integrates to 1.

    Parameters
    ----------
    zmax : float
        The height above the midplane (arcsec) within which the density
        lies (see RHTrait).
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def uid() -> int:
        return RHT_UID_SECH2

    def cdf(self, z: np.ndarray, *params: np.ndarray) -> np.ndarray:
        s, = params
        return 0.5 * (1 + np.tanh(z / s))


class VPTraitUniform(TraitFeatureDirection, VPTrait):
    """
    A uniform speed: v (e.g. a flat rotation curve). Its parameter: v,
    the speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_UNIFORM

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {self.speed(): ParamScalarDesc()}


class VPTraitArctan(TraitFeatureDirection, VPTrait):
    """
    An arctan profile of the speed: v (2 / pi) atan(r / rt). Its
    parameters: rt, the turnover radius, and v, the asymptotic speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'arctan'

    @staticmethod
    def uid() -> int:
        return VPT_UID_ARCTAN

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitBoissier(TraitFeatureDirection, VPTrait):
    """
    An exponential profile of the speed: v (1 - exp(-r / rt)). Its
    parameters: rt, the scale radius, and v, the asymptotic speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'boissier'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BOISSIER

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitEpinat(TraitFeatureDirection, VPTrait):
    """
    The profile of the rotation curves of Epinat et al. (2008): v (r /
    rt)^g / (1 + (r / rt)^a). Its parameters: rt, the turnover radius, v,
    the speed scale, a, the outer slope, and g, the inner slope.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'epinat'

    @staticmethod
    def uid() -> int:
        return VPT_UID_EPINAT

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'a': ParamScalarDesc(),
            'g': ParamScalarDesc()}


class VPTraitLRamp(TraitFeatureDirection, VPTrait):
    """
    A speed that rises linearly to v at rt and is flat beyond: v min(r /
    rt, 1). Its parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'lramp'

    @staticmethod
    def uid() -> int:
        return VPT_UID_LRAMP

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitTanh(TraitFeatureDirection, VPTrait):
    """
    A tanh profile of the speed: v tanh(r / rt). Its parameters: rt, the
    turnover radius, and v, the asymptotic speed.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'tanh'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TANH

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitPolyex(TraitFeatureDirection, VPTrait):
    """
    The polyex profile of the rotation curves of Giovanelli and Haynes
    (2002): v (1 - exp(-r / rt)) (1 + a r / rt). Its parameters: rt, the
    scale radius, v, the speed scale, and a, the slope of its outer part.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'polyex'

    @staticmethod
    def uid() -> int:
        return VPT_UID_POLYEX

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'a': ParamScalarDesc()}


class VPTraitRix(TraitFeatureDirection, VPTrait):
    """
    The profile of the rotation curves of Rix et al. (1997): v (1 + r /
    rt)^b (1 + (r / rt)^-g)^(-1 / g). Its parameters: rt, the turnover
    radius, v, the speed scale, g, the sharpness of the turnover, and b,
    the power law of the profile at large radii.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'rix'

    @staticmethod
    def uid() -> int:
        return VPT_UID_RIX

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'b': ParamScalarDesc(),
            'g': ParamScalarDesc()}


class VPTraitCourteau(TraitFeatureDirection, VPTrait):
    """
    The profile of the rotation curves of Courteau (1997): v (1 + x)^b /
    (1 + x^g)^(1 / g), with x = rt / r. Its parameters: rt, the turnover
    radius, v, the speed scale, and b and g, its shape.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'courteau'

    @staticmethod
    def uid() -> int:
        return VPT_UID_COURTEAU

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'b': ParamScalarDesc(),
            'g': ParamScalarDesc()}


class VPTraitBrandt(TraitFeatureDirection, VPTrait):
    """
    The profile of the rotation curves of Brandt (1960), of maximum v at
    the radius rt: v (r / rt) / (1/3 + 2/3 (r / rt)^n)^(3 / (2 n)). Its
    parameters: rt, v and n, the sharpness of the turnover.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'brandt'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BRANDT

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            'rt': ParamScalarDesc(),
            self.speed(): ParamScalarDesc(),
            'n': ParamScalarDesc()}


class VPTraitIso(TraitFeatureDirection, VPTrait):
    """
    The profile of the rotation curve of a pseudo-isothermal sphere of
    core radius rt and asymptotic speed v: v sqrt(1 - (rt / r) atan(r /
    rt)). Its parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'iso'

    @staticmethod
    def uid() -> int:
        return VPT_UID_ISO

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitNFW(TraitFeatureDirection, VPTrait):
    """
    The profile of the rotation curve of an NFW halo of scale radius rt,
    of maximum v (at 2.163 rt): v sqrt(f(r / rt) / f_max), with f(u) =
    (ln(1 + u) - u / (1 + u)) / u. Its parameters: rt and v.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nfw'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NFW

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {'rt': ParamScalarDesc(), self.speed(): ParamScalarDesc()}


class VPTraitNWUniform(TraitFeatureSampling, TraitFeatureDirection, VPTrait):
    """
    A speed given at each ring: v(r). Its parameter: v, node-wise.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_UNIFORM

    def __init__(
            self,
            direction: str,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(direction=direction, sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return {self.speed(): ParamVectorDesc(nnodes)}


class VPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, TraitFeatureDirection,
        VPTrait):
    """
    A harmonic speed: a(r) cos(m (theta - p(r))), of the order m. Its
    parameters, node-wise: a, the amplitude, and p (but for order 0), the
    azimuth of a maximum.

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).
    order : int
        The order m of the harmonic: the pattern repeats every 360 / m
        degrees, and order 0 is an axisymmetric a(r).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_harmonic'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_HARMONIC

    def __init__(
            self,
            direction: str,
            order: int,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(direction=direction, order=order, sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes)

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self.order() == 0


class VPTraitWedge(TraitFeatureDirection, VPTrait):
    """
    A uniform speed within two opposite wedges of a disk (e.g. a radial
    flow, as DysmalPy's UniformWedgeFlow): v at the azimuths within s / 2
    of p or of p + 180 degrees, |cos(theta - p)| >= cos(s / 2), and 0 at
    the others. Its parameters: v, p (degrees) and s (degrees, the full
    width of each wedge).

    Parameters
    ----------
    direction : str
        The direction of its speed: 'tan', 'rad', 'ver', 'out' or 'los'
        (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'wedge'

    @staticmethod
    def uid() -> int:
        return VPT_UID_WEDGE

    def __init__(self, direction: str):
        super().__init__(direction=direction)

    def params_sm(self) -> dict[str, ParamDesc]:
        return {
            self.speed(): ParamScalarDesc(),
            'p': ParamScalarDesc(),
            's': ParamScalarDesc()}

    def axisymmetric(self) -> bool:
        return False

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        p, s = params['p'], params['s']
        return np.tile(
            [p - s / 2, p + s / 2, p + 180 - s / 2, p + 180 + s / 2],
            (len(r), 1))


class VPTraitBar(VPTraitPlanar):
    """
    A uniform flow along a bar, in the plane of a disk (as DysmalPy's
    UniformBarFlow): the speed v along the axis of the bar, away from its
    centre on both sides (inwards if v < 0), within the strip of the full
    width s along the axis, and 0 outside. The axis is at the azimuth p
    (and p + 180 degrees); at the distance x along it (positive towards
    p) and y across it, the velocity is v sign(x) along the axis where
    |y| < s / 2, with v_y = v sign(x) sin(p). The strip reaches the edge
    of the disk. Its parameters: v, p (degrees) and s (arcsec).
    """

    @staticmethod
    def type() -> str:
        return 'bar'

    @staticmethod
    def uid() -> int:
        return VPT_UID_BAR

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            v=ParamScalarDesc(),
            p=ParamScalarDesc(),
            s=ParamScalarDesc())

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_bar(r, params)


class VPTraitNWBar(TraitFeatureSampling, VPTraitPlanar):
    """
    A flow along a bar whose speed is given at distances along it (as
    DysmalPy's VariableXBarFlow): as VPTraitBar, with the speed v(|x|) at
    the distance |x| from the centre along the axis, of a value at each
    node (at the distances of the radii of the nodes); 0 within the first
    or beyond the last (e.g. a bar of a length: speeds of 0 beyond it).
    Its parameters: p (degrees) and s (arcsec); v, node-wise.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_bar'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_BAR

    def __init__(self, sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            p=ParamScalarDesc(),
            s=ParamScalarDesc())

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(v=ParamVectorDesc(nnodes))

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_bar(r, params)


class VPTraitMass(VPTrait):
    """
    The rotation curve of the mass of its galaxy (see gbkfit.model.mass):
    the circular velocity at the rings it is evaluated on, which its
    galaxy computes from the parameters of its distance and its mass at
    each evaluation, and with pressure support, the
    rotation of the gas of its disk in that gravity; 0 where its square is
    negative (the mass pulls outwards, e.g. inside a ring, or the pressure
    exceeds the gravity: no rotation, as DysmalPy takes it), NaN where it
    is infinite (there is no circular orbit, e.g. at a jump of the surface
    density of a thin disk). A tangential speed (it has no option
    direction). It has no parameters of its own, and its galaxy must have
    mass.

    With pressure_support 'jeans', the rotation of an isotropic gas at the
    midplane of its disk (the radial Jeans equation): v^2 = v_c^2 + (R /
    rho) d(rho sigma^2)/dR, with the azimuthal means of the density rho of
    the disk at its midplane (its density traits times its density
    height traits at z = 0, or its density traits if thin: the equation
    of a thin disk) and of rho sigma^2, sigma its velocity dispersion at
    the midplane (with the dispersion of the vertical equilibrium of its
    disk in quadrature, if any, see parts.Kinematics; azimuthal means as
    those of mass.SurfaceComponent); the derivative by the second-order
    three-point formula on the rings; NaN without density. With
    'self_gravitating', the same for a disk whose thickness its own
    gravity sets, h ~ sigma^2 / S (S its surface density, its density
    traits): rho ~ S^2 / sigma^2 and rho sigma^2 ~ S^2, so that v^2 =
    v_c^2 + 2 sigma^2 dln S / dln R for an axisymmetric disk (Burkert et
    al. 2010), whatever its density height traits (DysmalPy's pressure
    support of types 1 and 2, for an exponential and a Sersic S).

    With pressure_density, the names of mass components of its galaxy
    (one, or several: e.g. its atomic and its molecular gas), the
    pressure takes the sum of their densities instead of that of its
    disk: at their midplane ('jeans'; e.g. the gas of thick mass disks,
    as DysmalPy's pressure support of type 3, which adds its gas
    components), or their surface densities ('self_gravitating'; e.g.
    that of the mass disk, as DysmalPy's types 1 and 2 take its
    effective radius), with the dispersion of its disk.

    The rotation is the same at every height, or with the velocity height
    trait 'mass', that of the gravity at each height (see VHTraitMass).

    Parameters
    ----------
    pressure_support : str, optional
        The pressure support: none (the default), 'jeans' or
        'self_gravitating'.
    pressure_density : str or Sequence of str, optional
        The name of the mass component whose density the pressure takes,
        or the names of several, whose densities add up; by default, that
        of its disk.

    Raises
    ------
    ConfigError
        If pressure_support or pressure_density is not valid (names that
        repeat, or none), or pressure_density is given without
        pressure_support.
    """

    @staticmethod
    def type() -> str:
        return 'mass'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_UNIFORM

    def dump(self) -> dict[str, Any]:
        info = {}
        if (support := self._pressure_support) is not None:
            info['pressure_support'] = support.mode
            if (names := support.density) is not None:
                info['pressure_density'] = \
                    names[0] if len(names) == 1 else list(names)
        return super().dump() | info

    def __init__(
            self,
            pressure_support: str | None = None,
            pressure_density: str | Sequence[str] | None = None
    ):
        if pressure_support not in (None, 'jeans', 'self_gravitating'):
            raise ConfigError(
                f"pressure_support must be 'jeans', 'self_gravitating' or "
                f"none; it is {pressure_support!r}")
        if pressure_density is not None:
            if pressure_support is None:
                raise ConfigError(
                    "pressure_density is the density of the pressure "
                    "support: it needs pressure_support")
            pressure_density = (pressure_density,) \
                if isinstance(pressure_density, str) \
                else tuple(pressure_density)
            if not pressure_density \
                    or len(set(pressure_density)) != len(pressure_density):
                raise ConfigError(
                    f"pressure_density names mass components, at least one "
                    f"and each once; it is {list(pressure_density)}")
            for name in pressure_density:
                parseutils.check_name(name)
        self._pressure_support = None if pressure_support is None \
            else PressureSupport(pressure_support, pressure_density)

    def direction(self) -> str:
        return 'tan'

    def sampling(self) -> str:
        return 'subrings'

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vt=ParamVectorDesc(nnodes))

    def circular_velocity_params(self) -> tuple[str, ...]:
        return ('vt',)

    def pressure_support(self) -> PressureSupport | None:
        return self._pressure_support


class VHTraitOne(VHTrait):
    """
    No change of the velocity with the height: the factor 1 (the default).
    """

    @staticmethod
    def type() -> str:
        return 'one'

    @staticmethod
    def uid() -> int:
        return VHT_UID_ONE

    def __init__(self):
        super().__init__(
            rnodes=False, sampling=SAMPLING_DEFAULT)


class VHTraitMass(TraitFeatureZmax, VHTrait):
    """
    The rotation in the gravity of the galaxy at each height: the factor
    v(R, z) / v(R, 0) of its velocity polar trait 'mass' (see
    VPTraitMass), with which it pairs, of the rotation of a circular orbit
    at the height z in the radial force of the mass of the galaxy there,
    v(R, z)^2 = -R F_R(R, z) (see gravity.Gravity), plus the pressure
    support of the trait 'mass' at z, if any: 'jeans', with the density of
    its disk at z (its density traits times its density height traits) or
    that of the mass components of pressure_density (their sum), and its
    dispersion at z (with its dispersion height traits);
    'self_gravitating', in the isothermal sheet of the surface density S
    of the mass components of pressure_density (required: the disk has
    no mass) and the dispersion
    sigma at the midplane, of the density ~ sech^2(z / z0), z0 = sigma^2 /
    (pi G S). v(R, z) is 0 where its square is negative, and NaN where it
    is infinite, as at the midplane; the factor is 1 where v(R, 0) is 0
    (no rotation at any height). Its galaxy computes it at each
    evaluation, at the subnodes of the disk and at VHT_MASS_NHEIGHTS
    heights from 0 to zmax, and its disk interpolates it linearly between
    them; above zmax, it is the factor at zmax. The rings of the disk are
    taken flat (coplanar and concentric), as for the rotation at its
    midplane. It has no parameters of its own.

    Parameters
    ----------
    zmax : float
        The highest height (arcsec) of its table: that of the density
        height traits of its disk.

    Raises
    ------
    ConfigError
        If zmax is not a positive length.
    """

    @staticmethod
    def type() -> str:
        return 'mass'

    @staticmethod
    def uid() -> int:
        return VHT_UID_MASS

    def dump(self) -> dict[str, Any]:
        # (its rnodes and sampling are not options)
        return dict(type=self.type(), zmax=self.zmax())

    def __init__(self, zmax: float):
        super().__init__(zmax=zmax, rnodes=True, sampling='subrings')

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(), VHT_MASS_NHEIGHTS)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return {
            name: ParamVectorDesc(nnodes)
            for name in self.rotation_factor_params()}

    def rotation_factor_params(self) -> dict[str, float]:
        heights = np.linspace(0, self.zmax(), VHT_MASS_NHEIGHTS)
        return {f'f{k}': float(z) for k, z in enumerate(heights)}


class VHTraitP1(VHTrait, abc.ABC):
    """
    A velocity height trait: a factor of the height, of one parameter
    (param_name), node-wise if rnodes.
    """

    def __init__(
            self,
            rnodes: bool = False,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(rnodes=rnodes, sampling=sampling)

    @staticmethod
    @abc.abstractmethod
    def param_name() -> str:
        """Return the name of its parameter."""
        pass

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else {
            self.param_name(): ParamScalarDesc()}

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else {
            self.param_name(): ParamVectorDesc(nrnodes)}


class VHTraitLinear(VHTraitP1):
    """
    The factor max(0, |z| - z0), which grows linearly above the height z0:
    with a polar trait of the change per unit height (e.g. a lag), and
    another of the velocity in the plane. Its parameter: z0.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'linear'

    @staticmethod
    def uid() -> int:
        return VHT_UID_LINEAR

    @staticmethod
    def param_name() -> str:
        return 'z0'


class VHTraitExponential(VHTraitP1):
    """
    The factor exp(-|z| / s), which falls with the scale height s. Its
    parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return VHT_UID_EXP

    @staticmethod
    def param_name() -> str:
        return 's'


class VHTraitGauss(VHTraitP1):
    """
    The factor exp(-z^2 / (2 s^2)), which falls with the scale height s.
    Its parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return VHT_UID_GAUSS

    @staticmethod
    def param_name() -> str:
        return 's'


class DPTraitUniform(DPTrait):
    """
    A uniform velocity dispersion: a. Its parameter: a, the dispersion
    (km/s).
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def uid() -> int:
        return DPT_UID_UNIFORM

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=ParamScalarDesc())


class DPTraitExponential(DPTrait):
    """
    An exponential velocity dispersion: a exp(-r / s). Its parameters: a,
    the dispersion (km/s) at the centre, and s, the scale length.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return DPT_UID_EXP

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())


class DPTraitGauss(DPTrait):
    """
    A Gaussian velocity dispersion: a exp(-r^2 / (2 s^2)). Its parameters:
    a, the dispersion (km/s) at the centre, and s, the dispersion of the
    Gaussian.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return DPT_UID_GAUSS

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())


class DPTraitGGauss(DPTrait):
    """
    A generalised Gaussian velocity dispersion: a exp(-(r / s)^b). Its
    parameters: a, the dispersion (km/s) at the centre, s, the scale
    length, and b, the shape (1 is the exponential, and 2 a Gaussian of
    dispersion s / sqrt(2)).
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def uid() -> int:
        return DPT_UID_GGAUSS

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc(),
            b=ParamScalarDesc())


class DPTraitLorentz(DPTrait):
    """
    A Lorentzian velocity dispersion: a s^2 / (r^2 + s^2). Its parameters:
    a, the dispersion (km/s) at the centre, and s, the half width at half
    maximum.
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def uid() -> int:
        return DPT_UID_LORENTZ

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())


class DPTraitMoffat(DPTrait):
    """
    A Moffat velocity dispersion: a (1 + (r / s)^2)^-b. Its parameters: a,
    the dispersion (km/s) at the centre, s, the core radius, and b, the
    power of the fall.
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MOFFAT

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc(),
            b=ParamScalarDesc())


class DPTraitSech2(DPTrait):
    """
    A sech^2 velocity dispersion: a sech^2(r / s). Its parameters: a, the
    dispersion (km/s) at the centre, and s, the scale length.
    """

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def uid() -> int:
        return DPT_UID_SECH2

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=ParamScalarDesc(),
            s=ParamScalarDesc())


class DPTraitMixtureExponential(TraitFeatureNBlobs, DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each an exponential of the elliptical distance rho from
    its centre: a exp(-rho / s). Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, q, its axis ratio (minor
    over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_exponential'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_EXP

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_6p(self.nblobs())


class DPTraitMixtureGauss(TraitFeatureNBlobs, DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each a Gaussian of the elliptical distance rho from its
    centre: a exp(-rho^2 / (2 s^2)). Its parameters, vectors of a value
    for each blob: r and t, the radius and the azimuth of its centre, a,
    its amplitude, s, its scale along its major axis, q, its axis ratio
    (minor over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_gauss'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_6p(self.nblobs())


class DPTraitMixtureGGauss(TraitFeatureNBlobs, DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each a generalised Gaussian of the elliptical distance rho
    from its centre: a exp(-(rho / s)^b). Its parameters, vectors of a
    value for each blob: r and t, the radius and the azimuth of its
    centre, a, its amplitude, s, its scale along its major axis, b, its
    shape, q, its axis ratio (minor over major), and p, the angle of its
    major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_ggauss'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_7p(self.nblobs())


class DPTraitMixtureMoffat(TraitFeatureNBlobs, DPTrait):
    """
    A velocity dispersion made of nblobs elliptical blobs (see the module
    docstring), each a Moffat of the elliptical distance rho from its
    centre: a (1 + (rho / s)^2)^-b. Its parameters, vectors of a value for
    each blob: r and t, the radius and the azimuth of its centre, a, its
    amplitude, s, its scale along its major axis, b, its shape, q, its
    axis ratio (minor over major), and p, the angle of its major axis.

    Parameters
    ----------
    nblobs : int
        The number of blobs.

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'mixture_moffat'

    @staticmethod
    def uid() -> int:
        return DPT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return _ptrait_params_mixture_7p(self.nblobs())


class DPTraitNWUniform(TraitFeatureSampling, DPTrait):
    """
    A velocity dispersion given at each ring: a(r). Its parameter: a,
    node-wise.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_uniform'

    @staticmethod
    def uid() -> int:
        return DPT_UID_NW_UNIFORM

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def axisymmetric(self) -> bool:
        return True

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(a=ParamVectorDesc(nnodes))


class DPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, DPTrait):
    """
    A harmonic dispersion: a(r) cos(m (theta - p(r))), of the order m. Its
    parameters, node-wise: a, the amplitude, and p (but for order 0), the
    azimuth of a maximum; a p that changes with r winds it into a spiral.
    It is negative in parts: it is meant to perturb an axisymmetric trait.

    Parameters
    ----------
    order : int
        The order m of the harmonic: the pattern repeats every 360 / m
        degrees, and order 0 is an axisymmetric a(r).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_harmonic'

    @staticmethod
    def uid() -> int:
        return DPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, sampling=sampling)

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self.order() == 0

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes)


class DPTraitNWDistortion(TraitFeatureSampling, DPTrait):
    """
    A velocity dispersion around one azimuth at each ring: a(r)
    exp(-(dtheta r)^2 / (2 s(r)^2)), with dtheta the azimuth from p(r)
    (radians, within half a turn). Its parameters, node-wise: a, the peak,
    p, its azimuth, and s, its width along the ring.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_distortion'

    @staticmethod
    def uid() -> int:
        return DPT_UID_NW_DISTORTION

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def axisymmetric(self) -> bool:
        return False

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_distortion(nnodes)

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _splits_nw_distortion(r, params, subrnodes)


class DHTraitOne(DHTrait):
    """
    No change of the dispersion with the height: the factor 1 (the
    default).
    """

    @staticmethod
    def type() -> str:
        return 'one'

    @staticmethod
    def uid() -> int:
        return DHT_UID_ONE

    def __init__(self):
        super().__init__(
            rnodes=False, sampling=SAMPLING_DEFAULT)


class DHTraitP1(DHTrait, abc.ABC):
    """
    A dispersion height trait: a factor of the height, of one parameter
    (param_name), node-wise if rnodes.
    """

    def __init__(
            self,
            rnodes: bool = False,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(rnodes=rnodes, sampling=sampling)

    @staticmethod
    @abc.abstractmethod
    def param_name() -> str:
        """Return the name of its parameter."""
        pass

    def params_sm(self) -> dict[str, ParamDesc]:
        return {} if self.rnodes() else {
            self.param_name(): ParamScalarDesc()}

    def params_rnw(self, nrnodes: int) -> dict[str, ParamDesc]:
        return {} if not self.rnodes() else {
            self.param_name(): ParamVectorDesc(nrnodes)}


class DHTraitLinear(DHTraitP1):
    """
    The factor max(0, |z| - z0), which grows linearly above the height z0:
    with a polar trait of the change per unit height (e.g. a lag), and
    another of the dispersion in the plane. Its parameter: z0.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'linear'

    @staticmethod
    def uid() -> int:
        return DHT_UID_LINEAR

    @staticmethod
    def param_name() -> str:
        return 'z0'


class DHTraitExponential(DHTraitP1):
    """
    The factor exp(-|z| / s), which falls with the scale height s. Its
    parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def uid() -> int:
        return DHT_UID_EXP

    @staticmethod
    def param_name() -> str:
        return 's'


class DHTraitGauss(DHTraitP1):
    """
    The factor exp(-z^2 / (2 s^2)), which falls with the scale height s.
    Its parameter: s.

    Parameters
    ----------
    rnodes : bool, optional
        Whether its parameters have a value for each ring (e.g. the scale
        height of a flaring disk).
    sampling : str, optional
        Where the values of its parameters are given, if they have a value
        for each ring (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def uid() -> int:
        return DHT_UID_GAUSS

    @staticmethod
    def param_name() -> str:
        return 's'


class ZPTraitNWUniform(TraitFeatureSampling, ZPTrait):
    """
    A displacement of the plane given at each ring: a(r), along z. Its
    parameter: a, node-wise.

    Parameters
    ----------
    zmax : float
        The largest displacement (arcsec; see ZPTrait).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_uniform'

    @staticmethod
    def uid() -> int:
        return ZPT_UID_NW_UNIFORM

    def __init__(
            self,
            zmax: float,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(zmax=zmax, sampling=sampling)

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(),)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(a=self._amplitude_desc(nnodes))


class ZPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, ZPTrait):
    """
    A harmonic displacement of the plane: a(r) cos(m (theta - p(r))), of
    the order m; order 1 is a warp, whose line of nodes is 90 degrees from
    p. Its parameters, node-wise: a, the amplitude, and p (but for order
    0), the azimuth of a maximum.

    Parameters
    ----------
    order : int
        The order m of the harmonic: the pattern repeats every 360 / m
        degrees, and order 0 is an axisymmetric a(r).
    zmax : float
        The largest displacement (arcsec; see ZPTrait).
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_harmonic'

    @staticmethod
    def uid() -> int:
        return ZPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            zmax: float,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, zmax=zmax, sampling=sampling)

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(), self.order())

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes) | dict(
            a=self._amplitude_desc(nnodes))


class SPTraitAzimuthalRange(SPTrait):
    """
    A wedge of the component: the azimuths within s / 2 of p. Its parameters:
    p, the azimuth of its centre, and s, its full width.
    """

    @staticmethod
    def type() -> str:
        return 'azrange'

    @staticmethod
    def uid() -> int:
        return SPT_UID_AZRANGE

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            p=ParamScalarDesc(),
            s=ParamScalarDesc())

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        p, s = params['p'], params['s']
        return np.tile([p - s / 2, p + s / 2], (len(r), 1))


class SPTraitRadialRange(SPTrait):
    """
    A range of radii of the component (a ring of a disk): the radii from
    rmin (inclusive) to rmax (exclusive). Its parameters: rmin and rmax.
    """

    @staticmethod
    def type() -> str:
        return 'rrange'

    @staticmethod
    def uid() -> int:
        return SPT_UID_RRANGE

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rmin=ParamScalarDesc(),
            rmax=ParamScalarDesc())


class SPTraitNWAzimuthalRange(
        TraitFeatureSampling, SPTrait):
    """
    A wedge of the component at each node: the azimuths within s(r) / 2
    of p(r). Its parameters, node-wise: p and s.

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the module docstring).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_azrange'

    @staticmethod
    def uid() -> int:
        return SPT_UID_NW_AZRANGE

    def __init__(self, sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def axisymmetric(self) -> bool:
        return False

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(
            p=ParamVectorDesc(nnodes),
            s=ParamVectorDesc(nnodes))

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        p = np.interp(r, subrnodes, params['p'])
        s = np.interp(r, subrnodes, params['s'])
        return np.column_stack((p - s / 2, p + s / 2))


# Density polar traits parser
rpt_parser = parseutils.TypedParser(RPTrait, [
    RPTraitUniform,
    RPTraitExponential,
    RPTraitGauss,
    RPTraitGGauss,
    RPTraitLorentz,
    RPTraitMoffat,
    RPTraitSech2,
    RPTraitMixtureExponential,
    RPTraitMixtureGauss,
    RPTraitMixtureGGauss,
    RPTraitMixtureMoffat,
    RPTraitNWUniform,
    RPTraitNWHarmonic,
    RPTraitNWDistortion,
    RPTraitMap])

# Density height traits parser
rht_parser = parseutils.TypedParser(RHTrait, [
    RHTraitUniform,
    RHTraitExponential,
    RHTraitGauss,
    RHTraitGGauss,
    RHTraitLorentz,
    RHTraitMoffat,
    RHTraitSech2])


# Velocity polar traits parser
vpt_parser = parseutils.TypedParser(VPTrait, [
    VPTraitUniform,
    VPTraitArctan,
    VPTraitBoissier,
    VPTraitEpinat,
    VPTraitLRamp,
    VPTraitTanh,
    VPTraitPolyex,
    VPTraitRix,
    VPTraitCourteau,
    VPTraitBrandt,
    VPTraitIso,
    VPTraitNFW,
    VPTraitNWUniform,
    VPTraitNWHarmonic,
    VPTraitWedge,
    VPTraitBar,
    VPTraitNWBar,
    VPTraitMass])

# Velocity height traits parser
vht_parser = parseutils.TypedParser(VHTrait, [
    VHTraitOne,
    VHTraitLinear,
    VHTraitExponential,
    VHTraitGauss,
    VHTraitMass])

# Dispersion polar traits parser
dpt_parser = parseutils.TypedParser(DPTrait, [
    DPTraitUniform,
    DPTraitExponential,
    DPTraitGauss,
    DPTraitGGauss,
    DPTraitLorentz,
    DPTraitMoffat,
    DPTraitSech2,
    DPTraitMixtureExponential,
    DPTraitMixtureGauss,
    DPTraitMixtureGGauss,
    DPTraitMixtureMoffat,
    DPTraitNWUniform,
    DPTraitNWHarmonic,
    DPTraitNWDistortion])

# Dispersion height traits parser
dht_parser = parseutils.TypedParser(DHTrait, [
    DHTraitOne,
    DHTraitLinear,
    DHTraitExponential,
    DHTraitGauss])

# Vertical polar distortion traits parser
zpt_parser = parseutils.TypedParser(ZPTrait, [
    ZPTraitNWUniform,
    ZPTraitNWHarmonic])

# Selection polar traits parser
spt_parser = parseutils.TypedParser(SPTrait, [
    SPTraitAzimuthalRange,
    SPTraitRadialRange,
    SPTraitNWAzimuthalRange])
