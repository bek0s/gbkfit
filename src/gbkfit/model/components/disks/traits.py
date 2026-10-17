"""
The traits of the disks: the profiles of their properties, as functions of
the radius r and the azimuth theta in the plane of a disk (polar traits),
or of the height z above it (height traits).

The kinds of traits, and how the traits of one kind combine:

- density (rpt, rht): the surface density seen face-on (the brightness
  of a component that emits, the optical depth of dust); the sum of
  the polar traits, each times its height trait in a thick disk (a
  probability density of z, which integrates to 1).
- velocity (vpt, vht): the line-of-sight velocity; vsys plus the sum of
  the polar traits, each projected on the line of sight and times its
  height trait (a factor of z, 1 at z = 0 for most).
- dispersion (dpt, dht): the velocity dispersion; the absolute value of
  the sum of the polar traits, each times its height trait.
- vertical distortion (zpt): the sum displaces the plane of the disk
  (its midplane) along z, each within its zmax.
- selection (spt): 1 where a trait selects the disk, 0 elsewhere; a part
  of the disk is kept if every trait selects it.

The height traits pair with the polar traits of their kind by their
order, and a disk has at most four traits of a kind.

Coordinates and units: r is the radius in the plane of the disk, and
theta the azimuth in it, from the major axis on the side of the position
angle posa (the receding side when the rotation is positive), increasing
towards the far side (theta = 90 degrees, on the sky at the position
angle posa + 90 degrees). z is the height above the plane (above the
displaced plane with vertical distortion traits), positive away from the
viewer for inclinations below 90 degrees, and the height traits depend
on |z| only. Lengths are in arcsec, angles in degrees and velocities in
km/s. The traits are evaluated between the first and the last radius of
the rings of the disk only.

The tangential velocities are projected on the line of sight as v
cos(theta) sin(incl), the radial ones (positive outwards) as v
sin(theta) sin(incl), the vertical ones (positive along z) as v
cos(incl), and the line-of-sight ones as they are.

Node-wise parameters (those of the nw_ traits, and of the height traits
with rnodes) have a value for each ring: at its radii (sampling
'rnodes'), interpolated to the rings the disk is evaluated on (its
subrings, see rstep) with the interpolation of the disk (interp), or at
those subrings (sampling 'subrings'); between the subrings, they are
interpolated linearly.

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

# Velocity polar traits
VPT_UID_TAN_UNIFORM = 1
VPT_UID_TAN_ARCTAN = 2
VPT_UID_TAN_BOISSIER = 3
VPT_UID_TAN_EPINAT = 4
VPT_UID_TAN_LRAMP = 5
VPT_UID_TAN_TANH = 6
VPT_UID_TAN_POLYEX = 7
VPT_UID_TAN_RIX = 8
VPT_UID_TAN_COURTEAU = 9
VPT_UID_TAN_BRANDT = 10
VPT_UID_TAN_ISO = 11
VPT_UID_TAN_NFW = 12
VPT_UID_NW_TAN_UNIFORM = 101
VPT_UID_NW_TAN_HARMONIC = 102
VPT_UID_NW_RAD_UNIFORM = 103
VPT_UID_NW_RAD_HARMONIC = 104
VPT_UID_NW_VER_UNIFORM = 105
VPT_UID_NW_VER_HARMONIC = 106
VPT_UID_NW_LOS_UNIFORM = 107
VPT_UID_NW_LOS_HARMONIC = 108

# Velocity height traits
VHT_UID_ONE = 1
VHT_UID_LINEAR = 2
VHT_UID_EXP = 3
VHT_UID_GAUSS = 4

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


class Trait(parseutils.TypedSerializable, abc.ABC):
    """
    A trait of a disk: a profile of one of its properties (see the module
    docstring). A trait declares its type (its name in configurations),
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
        at the radii where they are sampled (see VPTraitTanMass); none
        here.
        """
        return ()

    def pressure_support(self) -> str | None:
        """
        Return how the values of its circular velocity parameters are
        corrected for the pressure of their disk, if they are (see
        VPTraitTanMass); not here.
        """
        return None


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
    A polar trait: a function of the radius and the azimuth in the plane
    of the disk.
    """

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        """
        Return the azimuths (degrees) at which the trait jumps on the
        circles of the radii r (arcsec, an array), for the azimuthal means
        of the gravity of its galaxy (see _profiles): an array of shape
        (len(r), n), NaN where a circle has fewer than n; None if it does
        not jump (here). params has the values of its parameters, the
        node-wise ones at the subnodes subrnodes, between which the disk
        interpolates them linearly.
        """
        return None


class HTrait(
        TraitFeatureRNodes, TraitFeatureSampling, Trait,
        abc.ABC):
    """
    A height trait: a function of the height above the plane of the disk,
    whose parameters can have a value for each ring (rnodes). Raise
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
    A density polar trait: the density of the disk seen face-on (light,
    or the optical depth of dust). The Monte Carlo disks draw their
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
    the density above the midplane, which integrates to 1.

    The density lies within zmax of the midplane (its option zmax), which
    vertical distortion traits displace: the profile is truncated there
    and renormalised, so that each column keeps its density (none where
    less than 0.001 of the profile is within zmax). With the zmax of the
    vertical distortion traits, it bounds the disk along z, so that the
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
    A velocity polar trait: a velocity, projected on the line of sight
    (see the module docstring).
    """


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
    A selection polar trait: 1 where it selects the disk, 0 elsewhere.
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


class VPTraitTanUniform(VPTrait):
    """
    A flat rotation curve: vt. Its parameter: vt, the rotation velocity. A
    tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_UNIFORM

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(vt=ParamScalarDesc())


class VPTraitTanArctan(VPTrait):
    """
    An arctan rotation curve: vt (2 / pi) atan(r / rt). Its parameters:
    rt, the turnover radius, and vt, the asymptotic velocity. A tangential
    velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_arctan'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_ARCTAN

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc())


class VPTraitTanBoissier(VPTrait):
    """
    An exponential rotation curve: vt (1 - exp(-r / rt)). Its parameters:
    rt, the scale radius, and vt, the asymptotic velocity. A tangential
    velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_boissier'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_BOISSIER

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc())


class VPTraitTanEpinat(VPTrait):
    """
    The rotation curve of Epinat et al. (2008): vt (r / rt)^g / (1 + (r /
    rt)^a). Its parameters: rt, the turnover radius, vt, the velocity
    scale, a, the outer slope, and g, the inner slope. A tangential
    velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_epinat'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_EPINAT

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc(),
            a=ParamScalarDesc(),
            g=ParamScalarDesc())


class VPTraitTanLRamp(VPTrait):
    """
    A rotation curve that rises linearly to vt at rt and is flat beyond:
    vt min(r / rt, 1). Its parameters: rt and vt. A tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_lramp'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_LRAMP

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc())


class VPTraitTanTanh(VPTrait):
    """
    A tanh rotation curve: vt tanh(r / rt). Its parameters: rt, the
    turnover radius, and vt, the asymptotic velocity. A tangential
    velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_tanh'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_TANH

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc())


class VPTraitTanPolyex(VPTrait):
    """
    The polyex rotation curve of Giovanelli and Haynes (2002): vt (1 -
    exp(-r / rt)) (1 + a r / rt). Its parameters: rt, the scale radius,
    vt, the velocity scale, and a, the slope of its outer part. A
    tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_polyex'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_POLYEX

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc(),
            a=ParamScalarDesc())


class VPTraitTanRix(VPTrait):
    """
    The rotation curve of Rix et al. (1997): vt (1 + r / rt)^b (1 + (r /
    rt)^-g)^(-1 / g). Its parameters: rt, the turnover radius, vt, the
    velocity scale, g, the sharpness of the turnover, and b, the power law
    of the curve at large radii. A tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_rix'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_RIX

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc(),
            b=ParamScalarDesc(),
            g=ParamScalarDesc())


class VPTraitTanCourteau(VPTrait):
    """
    The rotation curve of Courteau (1997): vt (1 + x)^b / (1 + x^g)^(1 /
    g), with x = rt / r. Its parameters: rt, the turnover radius, vt, the
    velocity scale, and b and g, its shape. A tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_courteau'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_COURTEAU

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc(),
            b=ParamScalarDesc(),
            g=ParamScalarDesc())


class VPTraitTanBrandt(VPTrait):
    """
    The rotation curve of Brandt (1960), of maximum vt at the radius rt:
    vt (r / rt) / (1/3 + 2/3 (r / rt)^n)^(3 / (2 n)). Its parameters: rt,
    vt and n, the sharpness of the turnover. A tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_brandt'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_BRANDT

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc(),
            n=ParamScalarDesc())


class VPTraitTanIso(VPTrait):
    """
    The rotation curve of a pseudo-isothermal sphere of core radius rt and
    asymptotic velocity vt: vt sqrt(1 - (rt / r) atan(r / rt)). A
    tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_iso'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_ISO

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc())


class VPTraitTanNFW(VPTrait):
    """
    The rotation curve of an NFW halo of scale radius rt, of maximum vt
    (at 2.163 rt): vt sqrt(f(r / rt) / f_max), with f(u) = (ln(1 + u) - u
    / (1 + u)) / u. A tangential velocity.
    """

    @staticmethod
    def type() -> str:
        return 'tan_nfw'

    @staticmethod
    def uid() -> int:
        return VPT_UID_TAN_NFW

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rt=ParamScalarDesc(),
            vt=ParamScalarDesc())


class VPTraitNWTanUniform(TraitFeatureSampling, VPTrait):
    """
    A rotation curve given at each ring: vt(r). Its parameter: vt,
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
        return 'nw_tan_uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_TAN_UNIFORM

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vt=ParamVectorDesc(nnodes))


class VPTraitTanMass(VPTrait):
    """
    The rotation curve of the mass of its galaxy (see gbkfit.model.mass):
    the circular velocity at the rings it is evaluated on, which its
    galaxy computes from the parameters of its distance and its mass at
    each evaluation, and with pressure support, the
    rotation of the gas of its disk in that gravity; NaN where its square
    is negative or infinite (there is no circular orbit, e.g. at a jump of
    the surface density of a thin disk, or where the pressure exceeds the
    gravity). A tangential velocity. It has no parameters of its own, and
    its galaxy must have mass.

    With pressure_support 'jeans', the rotation of an isotropic gas at the
    midplane of its disk (the radial Jeans equation): v^2 = v_c^2 + (R /
    rho) d(rho sigma^2)/dR, with the azimuthal means of the density rho of
    the disk at its midplane (its density traits times its density
    height traits at z = 0, or its density traits if thin: the equation
    of a thin disk) and of rho sigma^2, sigma its velocity dispersion at
    the midplane (azimuthal means as those of mass.SurfaceComponent); the
    derivative by the second-order three-point formula on the rings; NaN
    without density.

    Parameters
    ----------
    pressure_support : str, optional
        The pressure support: none (the default), or 'jeans'.

    Raises
    ------
    ConfigError
        If pressure_support is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'tan_mass'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_TAN_UNIFORM

    def dump(self) -> dict[str, Any]:
        info = dict(pressure_support=self._pressure_support) \
            if self._pressure_support is not None else {}
        return super().dump() | info

    def __init__(self, pressure_support: str | None = None):
        if pressure_support not in (None, 'jeans'):
            raise ConfigError(
                f"pressure_support must be 'jeans' or none; it is "
                f"{pressure_support!r}")
        self._pressure_support = pressure_support

    def sampling(self) -> str:
        return 'subrings'

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vt=ParamVectorDesc(nnodes))

    def circular_velocity_params(self) -> tuple[str, ...]:
        return ('vt',)

    def pressure_support(self) -> str | None:
        return self._pressure_support


class VPTraitNWTanHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, VPTrait):
    """
    A harmonic tangential velocity: a(r) cos(m (theta - p(r))), of the
    order m. Its parameters, node-wise: a, the amplitude, and p (but for
    order 0), the azimuth of a maximum.

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
        return 'nw_tan_harmonic'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_TAN_HARMONIC

    def __init__(
            self,
            order: int,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes)


class VPTraitNWRadUniform(TraitFeatureSampling, VPTrait):
    """
    A radial velocity given at each ring: vr(r), positive outwards. Its
    parameter: vr, node-wise.

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
        return 'nw_rad_uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_RAD_UNIFORM

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vr=ParamVectorDesc(nnodes))


class VPTraitNWRadHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, VPTrait):
    """
    A harmonic radial velocity: a(r) cos(m (theta - p(r))), of the order
    m. Its parameters, node-wise: a, the amplitude, and p (but for order
    0), the azimuth of a maximum.

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
        return 'nw_rad_harmonic'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_RAD_HARMONIC

    def __init__(
            self,
            order: int,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes)


class VPTraitNWVerUniform(TraitFeatureSampling, VPTrait):
    """
    A vertical velocity given at each ring: vv(r), positive along z on
    both sides of the plane. Its parameter: vv, node-wise.

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
        return 'nw_ver_uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_VER_UNIFORM

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vv=ParamVectorDesc(nnodes))


class VPTraitNWVerHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, VPTrait):
    """
    A harmonic vertical velocity: a(r) cos(m (theta - p(r))), of the order
    m. Its parameters, node-wise: a, the amplitude, and p (but for order
    0), the azimuth of a maximum.

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
        return 'nw_ver_harmonic'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_VER_HARMONIC

    def __init__(
            self,
            order: int,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes)


class VPTraitNWLOSUniform(TraitFeatureSampling, VPTrait):
    """
    A line-of-sight velocity given at each ring: vl(r), not projected. Its
    parameter: vl, node-wise.

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
        return 'nw_los_uniform'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_LOS_UNIFORM

    def __init__(
            self,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(vl=ParamVectorDesc(nnodes))


class VPTraitNWLOSHarmonic(
        TraitFeatureOrder, TraitFeatureSampling, VPTrait):
    """
    A harmonic line-of-sight velocity: a(r) cos(m (theta - p(r))), of the
    order m, not projected. Its parameters, node-wise: a, the amplitude,
    and p (but for order 0), the azimuth of a maximum.

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
        return 'nw_los_harmonic'

    @staticmethod
    def uid() -> int:
        return VPT_UID_NW_LOS_HARMONIC

    def __init__(
            self,
            order: int,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, sampling=sampling)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return _ptrait_params_nw_harmonic(self.order(), nnodes)


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
    A wedge of the disk: the azimuths within s / 2 of p. Its parameters:
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
    A ring of the disk: the radii from rmin (inclusive) to rmax
    (exclusive). Its parameters: rmin and rmax.
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
    A wedge of the disk at each ring: the azimuths within s(r) / 2 of
    p(r). Its parameters, node-wise: p and s.

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
    VPTraitTanUniform,
    VPTraitTanArctan,
    VPTraitTanBoissier,
    VPTraitTanEpinat,
    VPTraitTanLRamp,
    VPTraitTanTanh,
    VPTraitTanPolyex,
    VPTraitTanRix,
    VPTraitTanCourteau,
    VPTraitTanBrandt,
    VPTraitTanIso,
    VPTraitTanNFW,
    VPTraitNWTanUniform,
    VPTraitTanMass,
    VPTraitNWTanHarmonic,
    VPTraitNWRadUniform,
    VPTraitNWRadHarmonic,
    VPTraitNWVerUniform,
    VPTraitNWVerHarmonic,
    VPTraitNWLOSUniform,
    VPTraitNWLOSHarmonic])

# Velocity height traits parser
vht_parser = parseutils.TypedParser(VHTrait, [
    VHTraitOne,
    VHTraitLinear,
    VHTraitExponential,
    VHTraitGauss])

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
