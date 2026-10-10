
import abc
import logging

import numpy as np
import scipy.special

import gbkfit.math
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import parseutils
from .common import NWMode, nwmode_parser


_log = logging.getLogger(__name__)

# Density polar traits
# The density polar trait uids are used to generate uids for
# the surface brightness and opacity polar traits.
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

# Density height traits
# The density height trait uids are used to generate uids for
# the surface brightness and opacity height traits.
RHT_UID_UNIFORM = 1
RHT_UID_EXP = 2
RHT_UID_GAUSS = 3
RHT_UID_GGAUSS = 4
RHT_UID_LORENTZ = 5
RHT_UID_MOFFAT = 6
RHT_UID_SECH2 = 7

# Surface Brightness polar traits
BPT_UID_OFFSET = 0
BPT_UID_UNIFORM = BPT_UID_OFFSET + RPT_UID_UNIFORM
BPT_UID_EXP = BPT_UID_OFFSET + RPT_UID_EXP
BPT_UID_GAUSS = BPT_UID_OFFSET + RPT_UID_GAUSS
BPT_UID_GGAUSS = BPT_UID_OFFSET + RPT_UID_GGAUSS
BPT_UID_LORENTZ = BPT_UID_OFFSET + RPT_UID_LORENTZ
BPT_UID_MOFFAT = BPT_UID_OFFSET + RPT_UID_MOFFAT
BPT_UID_SECH2 = BPT_UID_OFFSET + RPT_UID_SECH2
BPT_UID_MIXTURE_EXP = BPT_UID_OFFSET + RPT_UID_MIXTURE_EXP
BPT_UID_MIXTURE_GAUSS = BPT_UID_OFFSET + RPT_UID_MIXTURE_GAUSS
BPT_UID_MIXTURE_GGAUSS = BPT_UID_OFFSET + RPT_UID_MIXTURE_GGAUSS
BPT_UID_MIXTURE_MOFFAT = BPT_UID_OFFSET + RPT_UID_MIXTURE_MOFFAT
BPT_UID_NW_UNIFORM = BPT_UID_OFFSET + RPT_UID_NW_UNIFORM
BPT_UID_NW_HARMONIC = BPT_UID_OFFSET + RPT_UID_NW_HARMONIC
BPT_UID_NW_DISTORTION = BPT_UID_OFFSET + RPT_UID_NW_DISTORTION

# Surface Brightness height traits
BHT_UID_OFFSET = 0
BHT_UID_UNIFORM = BHT_UID_OFFSET + RHT_UID_UNIFORM
BHT_UID_EXP = BHT_UID_OFFSET + RHT_UID_EXP
BHT_UID_GAUSS = BHT_UID_OFFSET + RHT_UID_GAUSS
BHT_UID_GGAUSS = BHT_UID_OFFSET + RHT_UID_GGAUSS
BHT_UID_LORENTZ = BHT_UID_OFFSET + RHT_UID_LORENTZ
BHT_UID_MOFFAT = BHT_UID_OFFSET + RHT_UID_MOFFAT
BHT_UID_SECH2 = BHT_UID_OFFSET + RHT_UID_SECH2

# Opacity polar traits
OPT_UID_OFFSET = 1000
OPT_UID_UNIFORM = OPT_UID_OFFSET + RPT_UID_UNIFORM
OPT_UID_EXP = OPT_UID_OFFSET + RPT_UID_EXP
OPT_UID_GAUSS = OPT_UID_OFFSET + RPT_UID_GAUSS
OPT_UID_GGAUSS = OPT_UID_OFFSET + RPT_UID_GGAUSS
OPT_UID_LORENTZ = OPT_UID_OFFSET + RPT_UID_LORENTZ
OPT_UID_MOFFAT = OPT_UID_OFFSET + RPT_UID_MOFFAT
OPT_UID_SECH2 = OPT_UID_OFFSET + RPT_UID_SECH2
OPT_UID_MIXTURE_EXP = OPT_UID_OFFSET + RPT_UID_MIXTURE_EXP
OPT_UID_MIXTURE_GAUSS = OPT_UID_OFFSET + RPT_UID_MIXTURE_GAUSS
OPT_UID_MIXTURE_GGAUSS = OPT_UID_OFFSET + RPT_UID_MIXTURE_GGAUSS
OPT_UID_MIXTURE_MOFFAT = OPT_UID_OFFSET + RPT_UID_MIXTURE_MOFFAT
OPT_UID_NW_UNIFORM = OPT_UID_OFFSET + RPT_UID_NW_UNIFORM
OPT_UID_NW_HARMONIC = OPT_UID_OFFSET + RPT_UID_NW_HARMONIC
OPT_UID_NW_DISTORTION = OPT_UID_OFFSET + RPT_UID_NW_DISTORTION

# Opacity height traits
OHT_UID_OFFSET = 1000
OHT_UID_UNIFORM = OHT_UID_OFFSET + RHT_UID_UNIFORM
OHT_UID_EXP = OHT_UID_OFFSET + RHT_UID_EXP
OHT_UID_GAUSS = OHT_UID_OFFSET + RHT_UID_GAUSS
OHT_UID_GGAUSS = OHT_UID_OFFSET + RHT_UID_GGAUSS
OHT_UID_LORENTZ = OHT_UID_OFFSET + RHT_UID_LORENTZ
OHT_UID_MOFFAT = OHT_UID_OFFSET + RHT_UID_MOFFAT
OHT_UID_SECH2 = OHT_UID_OFFSET + RHT_UID_SECH2

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

# Weight polar traits
WPT_UID_AXIS_RANGE = 1

TRUNC_DEFAULT = 0

# Where the values of the node-wise parameters of a trait are given: at
# the radial nodes of its disk, which interpolates them to the subnodes,
# or at the subnodes themselves (see _disk.Disk)
SAMPLINGS = ('rnodes', 'subrings')
SAMPLING_DEFAULT = 'rnodes'


def _ptrait_params_mixture_6p(nblobs):
    return (
        ParamVectorDesc('r', nblobs),  # polar coord (radius)
        ParamVectorDesc('t', nblobs),  # polar coord (angle)
        ParamVectorDesc('a', nblobs),  # amplitude
        ParamVectorDesc('s', nblobs),  # size
        ParamVectorDesc('q', nblobs),  # axis ratio (minor/major)
        ParamVectorDesc('p', nblobs))  # position angle relative to t


def _ptrait_params_mixture_7p(nblobs):
    return (
        ParamVectorDesc('r', nblobs),  # polar coord (radius)
        ParamVectorDesc('t', nblobs),  # polar coord (angle)
        ParamVectorDesc('a', nblobs),  # amplitude
        ParamVectorDesc('s', nblobs),  # size
        ParamVectorDesc('b', nblobs),  # shape
        ParamVectorDesc('q', nblobs),  # axis ratio (minor/major)
        ParamVectorDesc('p', nblobs))  # position angle relative to t


def _ptrait_params_nw_harmonic(order, nnodes, nwmode):
    params = []
    params += [(ParamVectorDesc('a', nnodes), nwmode)]
    params += [(ParamVectorDesc('p', nnodes), nwmode)] * (order > 0)
    return tuple(params)


def _ptrait_params_nw_distortion(nnodes, nwmode):
    return (
        (ParamVectorDesc('a', nnodes), nwmode),
        (ParamVectorDesc('p', nnodes), nwmode),
        (ParamVectorDesc('s', nnodes), nwmode))


def _integrate_rings(rings, fun, *args):
    # ring centers, equally spaced, same width
    rsep = rings[1] - rings[0]
    # Minimum radius of the rings
    rmin = rings - rsep * 0.5
    # Maximum radius of the rings
    rmax = rings + rsep * 0.5
    # Calculate the amplitude of each ring
    ampl = fun(rings, *args)
    return ampl * np.pi * (rmax * rmax - rmin * rmin)


def _ptrait_integrate_uniform(params, rings):
    a = params['a']
    rsep = rings[1] - rings[0]
    rmin = rings[0] - 0.5 * rsep
    rmax = rings[-1] + 0.5 * rsep
    return np.pi * a * (rmax * rmax - rmin * rmin)


def _ptrait_integrate_exponential(params, rings):
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.expon_1d_fun, a, 0, s)


def _ptrait_integrate_gauss(params, rings):
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.gauss_1d_fun, a, 0, s)


def _ptrait_integrate_ggauss(params, rings):
    a = params['a']
    s = params['s']
    b = params['b']
    return _integrate_rings(rings, gbkfit.math.ggauss_1d_fun, a, 0, s, b)


def _ptrait_integrate_lorentz(params, rings):
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.lorentz_1d_fun, a, 0, s)


def _ptrait_integrate_moffat(params, rings):
    a = params['a']
    s = params['s']
    b = params['b']
    return _integrate_rings(rings, gbkfit.math.moffat_1d_fun, a, 0, s, b)


def _ptrait_integrate_sech2(params, rings):
    a = params['a']
    s = params['s']
    return _integrate_rings(rings, gbkfit.math.sech2_1d_fun, a, 0, s)


def _ptrait_cloud_flux_mixture(params, norm):
    """
    The flux of the clouds of a mixture of blobs (see the native
    rp_trait_mixture_rnd): the sum of the |amplitude| times the integral
    (norm, of a blob of amplitude 1) times the axis ratio of each blob.
    The kernel gives each cloud the sign of the amplitude of its blob.
    """
    a = np.abs(np.asarray(params['a']) * np.asarray(params['q']))
    return np.sum(a * norm)


def _ptrait_integrate_mixture_exponential(params, rings):  # noqa
    s = np.asarray(params['s'])
    return _ptrait_cloud_flux_mixture(params, 2 * np.pi * s * s)


def _ptrait_integrate_mixture_gauss(params, rings):  # noqa
    s = np.asarray(params['s'])
    return _ptrait_cloud_flux_mixture(params, 2 * np.pi * s * s)


def _ptrait_integrate_mixture_ggauss(params, rings):  # noqa
    s = np.asarray(params['s'])
    b = np.asarray(params['b'])
    return _ptrait_cloud_flux_mixture(
        params, 2 * np.pi * s * s * scipy.special.gamma(2 / b) / b)


def _ptrait_integrate_mixture_moffat(params, rings):  # noqa
    s = np.asarray(params['s'])
    b = np.asarray(params['b'])
    if np.any(b <= 1):
        raise RuntimeError(
            "the blobs of mixture_moffat of the Monte Carlo disk need b > 1 "
            "(their flux is infinite otherwise)")
    return _ptrait_cloud_flux_mixture(params, np.pi * s * s / (b - 1))


def _ptrait_integrate_nw_uniform(params, rings):
    a = params['a']
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_cloud_flux_nw_harmonic(params, rings, order):
    # The integral of |a cos(k (t - p))| around a ring is 2 / pi of that of
    # |a| for k > 0
    a = params['a'] * (2 / np.pi if order else 1)
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_integrate_nw_distortion(params, rings):
    """
    The flux of each ring of a distortion: its area times the mean around
    it of a exp(-(t r)^2 / (2 s^2)), for the azimuths t within half a turn
    of the centre of the distortion.
    """
    a = params['a']
    s = np.abs(params['s'])

    def mean(r):
        width = s / r
        return np.where(
            width > 0,
            np.sqrt(2 * np.pi) * width * scipy.special.erf(
                np.pi / (np.sqrt(2) * np.maximum(width, 1e-300))) / (2 * np.pi),
            0) * a
    return _integrate_rings(rings, mean)


def trait_desc(cls):
    descs = {
        BPTrait: 'surface brightness polar trait',
        BHTrait: 'surface brightness height trait',
        VPTrait: 'velocity polar trait',
        VHTrait: 'velocity height trait',
        DPTrait: 'velocity dispersion polar trait',
        DHTrait: 'velocity dispersion height trait',
        ZPTrait: 'vertical distortion polar trait',
        SPTrait: 'selection polar trait',
        WPTrait: 'weight polar trait',
        OPTrait: 'opacity polar trait',
        OHTrait: 'opacity height trait'}
    label = None
    for k, v in descs.items():
        if issubclass(cls, k):
            label = v
    assert label
    return parseutils.make_typed_desc(cls, label)


class Trait(parseutils.TypedSerializable, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def uid():
        pass

    def dump(self):
        return dict(type=self.type())

    def consts(self):
        """The constants of the trait, in the order the kernels read them."""
        return ()

    def params_sm(self):
        return tuple()

    def params_rnw(self, nnodes):
        return tuple()

    def sampling(self):
        """Where the values of the node-wise parameters are given."""
        return SAMPLING_DEFAULT

    def circular_velocity_params(self):
        """
        The names of the node-wise parameters whose values are not given,
        but are the circular velocity of the mass model of the gmodel at
        the radii where they are sampled (see VPTraitMass); none here.
        """
        return ()


class TraitFeatureTrunc:

    def dump(self):
        dump_trunc = self.trunc() > 0
        info = dict(trunc=self.trunc()) if dump_trunc else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        trunc = kwargs.pop('trunc')
        if not trunc >= 0:
            raise RuntimeError(f"trunc must be at least 0; it is {trunc}")
        self._trunc = trunc
        super().__init__(**kwargs)

    def trunc(self):
        return self._trunc


class TraitFeatureNWMode:

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(
            nwmode_parser, info, 'nwmode')
        return super().load(info)  # noqa

    def dump(self):
        nwmode = self.nwmode()
        info = dict(nwmode=nwmode_parser.dump(nwmode)) if nwmode else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        self._nwmode = kwargs.pop('nwmode')
        super().__init__(**kwargs)

    def nwmode(self):
        return self._nwmode


class TraitFeatureSampling:

    def dump(self):
        dump_sampling = self.sampling() != SAMPLING_DEFAULT
        info = dict(sampling=self.sampling()) if dump_sampling else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        sampling = kwargs.pop('sampling')
        if sampling not in SAMPLINGS:
            raise RuntimeError(
                f"sampling must be one of {list(SAMPLINGS)}; "
                f"it is '{sampling}'")
        self._sampling = sampling
        super().__init__(**kwargs)

    def sampling(self):
        return self._sampling


class TraitFeatureRNodes:

    def dump(self):
        dump_rnodes = self.rnodes()
        info = dict(rnodes=self.rnodes()) if dump_rnodes else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        self._rnodes = kwargs.pop('rnodes')
        super().__init__(**kwargs)

    def rnodes(self):
        return self._rnodes


class TraitFeatureNBlobs:

    def dump(self):
        info = dict(nblobs=self.nblobs())
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        nblobs = kwargs.pop('nblobs')
        if not nblobs >= 1:
            raise RuntimeError(f"nblobs must be at least 1; it is {nblobs}")
        self._nblobs = nblobs
        super().__init__(**kwargs)

    def nblobs(self):
        return self._nblobs

    def consts(self):
        return (self.nblobs(),)


class TraitFeatureOrder:

    def dump(self):
        info = dict(order=self.order())
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        order = kwargs.pop('order')
        if not order >= 0:
            raise RuntimeError(f"order must be at least 0; it is {order}")
        self._order = order
        super().__init__(**kwargs)

    def order(self):
        return self._order

    def consts(self):
        return (self.order(),)


class PTrait(Trait, abc.ABC):
    pass


class HTrait(
        TraitFeatureRNodes, TraitFeatureNWMode, TraitFeatureSampling, Trait,
        abc.ABC):

    def __init__(self, rnodes, nwmode, sampling, **kwargs):
        kwargs.update(rnodes=rnodes, nwmode=nwmode, sampling=sampling)
        super().__init__(**kwargs)
        if not self.rnodes() and self.nwmode():
            _log.warning(
                f"rnodes is set to {self.rnodes()}; "
                f"nwmode {self.nwmode()} will be ignored")
        if not self.rnodes() and self.sampling() != SAMPLING_DEFAULT:
            raise RuntimeError(
                f"sampling is '{self.sampling()}', but rnodes is False: "
                f"the trait has no node-wise parameters")


class BPTrait(PTrait, abc.ABC):

    @abc.abstractmethod
    def has_analytical_integral(self):
        """
        Whether the Monte Carlo disk makes the clouds of the trait for the
        whole disk at once (True), or for each ring.
        """
        pass

    @abc.abstractmethod
    def cloud_flux(self, params, rings):
        """
        The flux the Monte Carlo disk shares among the clouds of the trait:
        for the whole disk (one value, with an analytical integral) or for
        each of the given rings. It is the integral of |B| (B is the trait)
        with the sign of the amplitude of the trait. Where B changes sign
        along a ring, the kernel draws the clouds from |B| and gives each
        the sign of B relative to the amplitude (e.g. the sign of the
        cosine of a harmonic).
        """
        pass


class BHTrait(TraitFeatureTrunc, HTrait, abc.ABC):

    def consts(self):
        return (self.trunc(), self.rnodes())


class VPTrait(PTrait, abc.ABC):
    pass


class VHTrait(HTrait, abc.ABC):

    def consts(self):
        return (self.rnodes(),)


class DPTrait(PTrait, abc.ABC):
    pass


class DHTrait(HTrait, abc.ABC):

    def consts(self):
        return (self.rnodes(),)


class ZPTrait(PTrait, abc.ABC):
    pass


class SPTrait(PTrait, abc.ABC):
    pass


class WPTrait(PTrait, abc.ABC):
    pass


class OPTrait(PTrait, abc.ABC):

    @abc.abstractmethod
    def has_analytical_integral(self):
        """
        Whether the Monte Carlo disk makes the clouds of the trait for the
        whole disk at once (True), or for each ring.
        """
        pass

    @abc.abstractmethod
    def cloud_flux(self, params, rings):
        """
        The flux the Monte Carlo disk shares among the clouds of the trait:
        for the whole disk (one value, with an analytical integral) or for
        each of the given rings. It is the integral of |B| (B is the trait)
        with the sign of the amplitude of the trait. Where B changes sign
        along a ring, the kernel draws the clouds from |B| and gives each
        the sign of B relative to the amplitude (e.g. the sign of the
        cosine of a harmonic).
        """
        pass


class OHTrait(TraitFeatureTrunc, HTrait, abc.ABC):

    def consts(self):
        return (self.trunc(), self.rnodes())


class BPTraitUniform(BPTrait):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return BPT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('a'),)

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_uniform(params, rings)


class BPTraitExponential(BPTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return BPT_UID_EXP

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_exponential(params, rings)


class BPTraitGauss(BPTrait):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return BPT_UID_GAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_gauss(params, rings)


class BPTraitGGauss(BPTrait):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return BPT_UID_GGAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_ggauss(params, rings)


class BPTraitLorentz(BPTrait):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return BPT_UID_LORENTZ

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_lorentz(params, rings)


class BPTraitMoffat(BPTrait):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return BPT_UID_MOFFAT

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_moffat(params, rings)


class BPTraitSech2(BPTrait):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return BPT_UID_SECH2

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_sech2(params, rings)


class BPTraitMixtureExponential(TraitFeatureNBlobs, BPTrait):

    @staticmethod
    def type():
        return 'mixture_exponential'

    @staticmethod
    def uid():
        return BPT_UID_MIXTURE_EXP

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_exponential(params, rings)


class BPTraitMixtureGauss(TraitFeatureNBlobs, BPTrait):

    @staticmethod
    def type():
        return 'mixture_gauss'

    @staticmethod
    def uid():
        return BPT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_gauss(params, rings)


class BPTraitMixtureGGauss(TraitFeatureNBlobs, BPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return BPT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_ggauss(params, rings)


class BPTraitMixtureMoffat(TraitFeatureNBlobs, BPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return BPT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_moffat(params, rings)


class BPTraitNWUniform(TraitFeatureNWMode, TraitFeatureSampling, BPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return BPT_UID_NW_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_nw_uniform(params, rings)


class BPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, BPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return BPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_cloud_flux_nw_harmonic(
            params, rings, self.order())


class BPTraitNWDistortion(TraitFeatureNWMode, TraitFeatureSampling, BPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return BPT_UID_NW_DISTORTION

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_nw_distortion(params, rings)


class BHTraitP1(BHTrait, abc.ABC):

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT,
            trunc: float = TRUNC_DEFAULT):
        super().__init__(
            rnodes=rnodes, nwmode=nwmode, sampling=sampling, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('s'),)

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc('s', nrnodes), self.nwmode()),)


class BHTraitP2(BHTrait, abc.ABC):

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT,
            trunc: float = TRUNC_DEFAULT):
        super().__init__(
            rnodes=rnodes, nwmode=nwmode, sampling=sampling, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc('s', nrnodes), self.nwmode()),
            (ParamVectorDesc('b', nrnodes), self.nwmode()))


class BHTraitUniform(BHTraitP1):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return BHT_UID_UNIFORM


class BHTraitExponential(BHTraitP1):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return BHT_UID_EXP


class BHTraitGauss(BHTraitP1):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return BHT_UID_GAUSS


class BHTraitGGauss(BHTraitP2):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return BHT_UID_GGAUSS

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT,
            trunc: float = TRUNC_DEFAULT):
        super().__init__(
            rnodes=rnodes, nwmode=nwmode, sampling=sampling, trunc=trunc)


class BHTraitLorentz(BHTraitP1):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return BHT_UID_LORENTZ


class BHTraitMoffat(BHTraitP2):
    """The Moffat profile (1 + (z / s)^2)^-b, for b > 1/2."""

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return BHT_UID_MOFFAT


class BHTraitSech2(BHTraitP1):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return BHT_UID_SECH2


class VPTraitTanUniform(VPTrait):

    @staticmethod
    def type():
        return 'tan_uniform'

    @staticmethod
    def uid():
        return VPT_UID_TAN_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('vt'),)


class VPTraitTanArctan(VPTrait):

    @staticmethod
    def type():
        return 'tan_arctan'

    @staticmethod
    def uid():
        return VPT_UID_TAN_ARCTAN

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'))


class VPTraitTanBoissier(VPTrait):

    @staticmethod
    def type():
        return 'tan_boissier'

    @staticmethod
    def uid():
        return VPT_UID_TAN_BOISSIER

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'))


class VPTraitTanEpinat(VPTrait):

    @staticmethod
    def type():
        return 'tan_epinat'

    @staticmethod
    def uid():
        return VPT_UID_TAN_EPINAT

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'),
            ParamScalarDesc('a'),
            ParamScalarDesc('g'))


class VPTraitTanLRamp(VPTrait):

    @staticmethod
    def type():
        return 'tan_lramp'

    @staticmethod
    def uid():
        return VPT_UID_TAN_LRAMP

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'))


class VPTraitTanTanh(VPTrait):

    @staticmethod
    def type():
        return 'tan_tanh'

    @staticmethod
    def uid():
        return VPT_UID_TAN_TANH

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'))


class VPTraitTanPolyex(VPTrait):

    @staticmethod
    def type():
        return 'tan_polyex'

    @staticmethod
    def uid():
        return VPT_UID_TAN_POLYEX

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'),
            ParamScalarDesc('a'))


class VPTraitTanRix(VPTrait):

    @staticmethod
    def type():
        return 'tan_rix'

    @staticmethod
    def uid():
        return VPT_UID_TAN_RIX

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'),
            ParamScalarDesc('b'),
            ParamScalarDesc('g'))


class VPTraitTanCourteau(VPTrait):
    """
    The rotation curve of Courteau (1997): vt (1 + x)^b / (1 + x^g)^(1 / g),
    with x = rt / r.
    """

    @staticmethod
    def type():
        return 'tan_courteau'

    @staticmethod
    def uid():
        return VPT_UID_TAN_COURTEAU

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'),
            ParamScalarDesc('b'),
            ParamScalarDesc('g'))


class VPTraitTanBrandt(VPTrait):
    """
    The rotation curve of Brandt (1960), of maximum vt at the radius rt:
    vt (r / rt) / (1/3 + 2/3 (r / rt)^n)^(3 / (2 n)).
    """

    @staticmethod
    def type():
        return 'tan_brandt'

    @staticmethod
    def uid():
        return VPT_UID_TAN_BRANDT

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'),
            ParamScalarDesc('n'))


class VPTraitTanIso(VPTrait):
    """
    The rotation curve of a pseudo-isothermal sphere of core radius rt and
    asymptotic velocity vt: vt sqrt(1 - (rt / r) atan(r / rt)).
    """

    @staticmethod
    def type():
        return 'tan_iso'

    @staticmethod
    def uid():
        return VPT_UID_TAN_ISO

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'))


class VPTraitTanNFW(VPTrait):
    """
    The shape of the rotation curve of an NFW halo of scale radius rt, with
    its maximum vt (at 2.163 rt).
    """

    @staticmethod
    def type():
        return 'tan_nfw'

    @staticmethod
    def uid():
        return VPT_UID_TAN_NFW

    def params_sm(self):
        return (
            ParamScalarDesc('rt'),
            ParamScalarDesc('vt'))


class VPTraitNWTanUniform(TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_TAN_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vt', nnodes), self.nwmode()),)


class VPTraitMass(VPTrait):
    """
    The circular velocity of the mass model of the gmodel (see
    gmodels.mass): a node-wise tangential velocity whose values at the
    subnodes of the disk the gmodel computes from the parameters of its
    mass model at each evaluation. It has no parameters of its own.
    """

    @staticmethod
    def type():
        return 'mass'

    @staticmethod
    def uid():
        return VPT_UID_NW_TAN_UNIFORM

    def sampling(self):
        return 'subrings'

    def params_rnw(self, nnodes):
        return ((ParamVectorDesc('vt', nnodes), None),)

    def circular_velocity_params(self):
        return ('vt',)


class VPTraitNWTanHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_TAN_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWRadUniform(TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_RAD_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vr', nnodes), self.nwmode()),)


class VPTraitNWRadHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_RAD_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWVerUniform(TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_VER_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vv', nnodes), self.nwmode()),)


class VPTraitNWVerHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_VER_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWLOSUniform(TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_los_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_LOS_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vl', nnodes), self.nwmode()),)


class VPTraitNWLOSHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, VPTrait):

    @staticmethod
    def type():
        return 'nw_los_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_LOS_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VHTraitOne(VHTrait):

    @staticmethod
    def type():
        return 'one'

    @staticmethod
    def uid():
        return VHT_UID_ONE

    def __init__(self):
        super().__init__(
            rnodes=False, nwmode=None, sampling=SAMPLING_DEFAULT)


class VHTraitP1(VHTrait, abc.ABC):
    """
    A velocity height trait: a factor of the height, of one parameter
    (param_name), node-wise if rnodes.
    """

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, sampling=sampling)

    @staticmethod
    @abc.abstractmethod
    def param_name() -> str:
        pass

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc(self.param_name()),)

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc(self.param_name(), nrnodes), self.nwmode()),)


class VHTraitLinear(VHTraitP1):
    """
    The factor max(0, |z| - z0), which grows linearly above the height
    z0: with a polar trait of the change per unit height, e.g. a lag.
    """

    @staticmethod
    def type():
        return 'linear'

    @staticmethod
    def uid():
        return VHT_UID_LINEAR

    @staticmethod
    def param_name():
        return 'z0'


class VHTraitExponential(VHTraitP1):
    """The factor exp(-|z| / s), which falls with the scale height s."""

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return VHT_UID_EXP

    @staticmethod
    def param_name():
        return 's'


class VHTraitGauss(VHTraitP1):
    """The factor exp(-z^2 / (2 s^2)), which falls with the scale height s."""

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return VHT_UID_GAUSS

    @staticmethod
    def param_name():
        return 's'


class DPTraitUniform(DPTrait):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return DPT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('a'),)


class DPTraitExponential(DPTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return DPT_UID_EXP

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))


class DPTraitGauss(DPTrait):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return DPT_UID_GAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))


class DPTraitGGauss(DPTrait):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return DPT_UID_GGAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))


class DPTraitLorentz(DPTrait):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return DPT_UID_LORENTZ

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))


class DPTraitMoffat(DPTrait):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return DPT_UID_MOFFAT

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))


class DPTraitSech2(DPTrait):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return DPT_UID_SECH2

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))


class DPTraitMixtureExponential(TraitFeatureNBlobs, DPTrait):

    @staticmethod
    def type():
        return 'mixture_exponential'

    @staticmethod
    def uid():
        return DPT_UID_MIXTURE_EXP

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())


class DPTraitMixtureGauss(TraitFeatureNBlobs, DPTrait):

    @staticmethod
    def type():
        return 'mixture_gauss'

    @staticmethod
    def uid():
        return DPT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())


class DPTraitMixtureGGauss(TraitFeatureNBlobs, DPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return DPT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())


class DPTraitMixtureMoffat(TraitFeatureNBlobs, DPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return DPT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())


class DPTraitNWUniform(TraitFeatureNWMode, TraitFeatureSampling, DPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return DPT_UID_NW_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)


class DPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, DPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return DPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class DPTraitNWDistortion(TraitFeatureNWMode, TraitFeatureSampling, DPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return DPT_UID_NW_DISTORTION

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())


class DHTraitOne(DHTrait):

    @staticmethod
    def type():
        return 'one'

    @staticmethod
    def uid():
        return DHT_UID_ONE

    def __init__(self):
        super().__init__(
            rnodes=False, nwmode=None, sampling=SAMPLING_DEFAULT)


class DHTraitP1(DHTrait, abc.ABC):
    """
    A dispersion height trait: a factor of the height, of one parameter
    (param_name), node-wise if rnodes.
    """

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, sampling=sampling)

    @staticmethod
    @abc.abstractmethod
    def param_name() -> str:
        pass

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc(self.param_name()),)

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc(self.param_name(), nrnodes), self.nwmode()),)


class DHTraitLinear(DHTraitP1):
    """
    The factor max(0, |z| - z0), which grows linearly above the height
    z0: with a polar trait of the change per unit height, e.g. a lag.
    """

    @staticmethod
    def type():
        return 'linear'

    @staticmethod
    def uid():
        return DHT_UID_LINEAR

    @staticmethod
    def param_name():
        return 'z0'


class DHTraitExponential(DHTraitP1):
    """The factor exp(-|z| / s), which falls with the scale height s."""

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return DHT_UID_EXP

    @staticmethod
    def param_name():
        return 's'


class DHTraitGauss(DHTraitP1):
    """The factor exp(-z^2 / (2 s^2)), which falls with the scale height s."""

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return DHT_UID_GAUSS

    @staticmethod
    def param_name():
        return 's'


class ZPTraitNWUniform(TraitFeatureNWMode, TraitFeatureSampling, ZPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return ZPT_UID_NW_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)


class ZPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, ZPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return ZPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class SPTraitAzimuthalRange(SPTrait):

    @staticmethod
    def type():
        return 'azrange'

    @staticmethod
    def uid():
        return SPT_UID_AZRANGE

    def params_sm(self):
        return (
            ParamScalarDesc('p'),
            ParamScalarDesc('s'))


class SPTraitRadialRange(SPTrait):
    """The radii from rmin (inclusive) to rmax (exclusive)."""

    @staticmethod
    def type():
        return 'rrange'

    @staticmethod
    def uid():
        return SPT_UID_RRANGE

    def params_sm(self):
        return (
            ParamScalarDesc('rmin'),
            ParamScalarDesc('rmax'))


class SPTraitNWAzimuthalRange(
        TraitFeatureNWMode, TraitFeatureSampling, SPTrait):

    @staticmethod
    def type():
        return 'nw_azrange'

    @staticmethod
    def uid():
        return SPT_UID_NW_AZRANGE

    def __init__(self, nwmode=None, sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('p', nnodes), self.nwmode()),
            (ParamVectorDesc('s', nnodes), self.nwmode()))


class WPTraitAxisRange(WPTrait):

    @staticmethod
    def type():
        return 'axis_range'

    @staticmethod
    def uid():
        return WPT_UID_AXIS_RANGE

    def __init__(self, axis, angle: float, weight: float):
        if axis not in [0, 1]:
            raise RuntimeError(
                f"invalid axis value; "
                f"choose between 0 (minor axis) and 1 (major axis); "
                f"supplied value: {axis}")
        if not 0 <= angle <= 180:
            raise RuntimeError(
                f"invalid angle value; "
                f"angle must be between 0 and 180; "
                f"supplied value: {angle}")
        super().__init__()
        self._axis = axis
        self._angle = angle
        self._weight = weight

    def dump(self):
        return super().dump() | dict(
            axis=self._axis, angle=self._angle, weight=self._weight)

    def consts(self):
        return (self._axis, self._angle, self._weight)


class OPTraitUniform(OPTrait):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return OPT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('a'),)

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_uniform(params, rings)


class OPTraitExponential(OPTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return OPT_UID_EXP

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_exponential(params, rings)


class OPTraitGauss(OPTrait):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return OPT_UID_GAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_gauss(params, rings)


class OPTraitGGauss(OPTrait):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return OPT_UID_GGAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_ggauss(params, rings)


class OPTraitLorentz(OPTrait):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return OPT_UID_LORENTZ

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_lorentz(params, rings)


class OPTraitMoffat(OPTrait):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return OPT_UID_MOFFAT

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_moffat(params, rings)


class OPTraitSech2(OPTrait):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return OPT_UID_SECH2

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_sech2(params, rings)


class OPTraitMixtureExponential(TraitFeatureNBlobs, OPTrait):

    @staticmethod
    def type():
        return 'mixture_exponential'

    @staticmethod
    def uid():
        return OPT_UID_MIXTURE_EXP

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_exponential(params, rings)


class OPTraitMixtureGauss(TraitFeatureNBlobs, OPTrait):

    @staticmethod
    def type():
        return 'mixture_gauss'

    @staticmethod
    def uid():
        return OPT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_gauss(params, rings)


class OPTraitMixtureGGauss(TraitFeatureNBlobs, OPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return OPT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_ggauss(params, rings)


class OPTraitMixtureMoffat(TraitFeatureNBlobs, OPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return OPT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs: int):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_mixture_moffat(params, rings)


class OPTraitNWUniform(TraitFeatureNWMode, TraitFeatureSampling, OPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return OPT_UID_NW_UNIFORM

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_nw_uniform(params, rings)


class OPTraitNWHarmonic(
        TraitFeatureOrder, TraitFeatureNWMode, TraitFeatureSampling, OPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return OPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(order=order, nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_cloud_flux_nw_harmonic(
            params, rings, self.order())


class OPTraitNWDistortion(TraitFeatureNWMode, TraitFeatureSampling, OPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return OPT_UID_NW_DISTORTION

    def __init__(
            self,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT):
        super().__init__(nwmode=nwmode, sampling=sampling)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def cloud_flux(self, params, rings):
        return _ptrait_integrate_nw_distortion(params, rings)


class OHTraitP1(OHTrait, abc.ABC):

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT,
            trunc: float = TRUNC_DEFAULT):
        super().__init__(
            rnodes=rnodes, nwmode=nwmode, sampling=sampling, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('s'),)

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc('s', nrnodes), self.nwmode()),)


class OHTraitP2(OHTrait, abc.ABC):

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT,
            trunc: float = TRUNC_DEFAULT):
        super().__init__(
            rnodes=rnodes, nwmode=nwmode, sampling=sampling, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc('s', nrnodes), self.nwmode()),
            (ParamVectorDesc('b', nrnodes), self.nwmode()))


class OHTraitUniform(OHTraitP1):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return OHT_UID_UNIFORM


class OHTraitExponential(OHTraitP1):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return OHT_UID_EXP


class OHTraitGauss(OHTraitP1):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return OHT_UID_GAUSS


class OHTraitGGauss(OHTraitP2):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return OHT_UID_GGAUSS

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            sampling: str = SAMPLING_DEFAULT,
            trunc: float = TRUNC_DEFAULT):
        super().__init__(
            rnodes=rnodes, nwmode=nwmode, sampling=sampling, trunc=trunc)


class OHTraitLorentz(OHTraitP1):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return OHT_UID_LORENTZ


class OHTraitMoffat(OHTraitP2):
    """The Moffat profile (1 + (z / s)^2)^-b, for b > 1/2."""

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return OHT_UID_MOFFAT


class OHTraitSech2(OHTraitP1):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return OHT_UID_SECH2


# Surface Brightness polar traits parser
bpt_parser = parseutils.TypedParser(BPTrait, [
    BPTraitUniform,
    BPTraitExponential,
    BPTraitGauss,
    BPTraitGGauss,
    BPTraitLorentz,
    BPTraitMoffat,
    BPTraitSech2,
    BPTraitMixtureExponential,
    BPTraitMixtureGauss,
    BPTraitMixtureGGauss,
    BPTraitMixtureMoffat,
    BPTraitNWUniform,
    BPTraitNWHarmonic,
    BPTraitNWDistortion])

# Surface Brightness height traits parser
bht_parser = parseutils.TypedParser(BHTrait, [
    BHTraitUniform,
    BHTraitExponential,
    BHTraitGauss,
    BHTraitGGauss,
    BHTraitLorentz,
    BHTraitMoffat,
    BHTraitSech2])

# Opacity polar traits parser
opt_parser = parseutils.TypedParser(OPTrait, [
    OPTraitUniform,
    OPTraitExponential,
    OPTraitGauss,
    OPTraitGGauss,
    OPTraitLorentz,
    OPTraitMoffat,
    OPTraitSech2,
    OPTraitMixtureExponential,
    OPTraitMixtureGauss,
    OPTraitMixtureGGauss,
    OPTraitMixtureMoffat,
    OPTraitNWUniform,
    OPTraitNWHarmonic,
    OPTraitNWDistortion])

# Opacity height traits parser
oht_parser = parseutils.TypedParser(OHTrait, [
    OHTraitUniform,
    OHTraitExponential,
    OHTraitGauss,
    OHTraitGGauss,
    OHTraitLorentz,
    OHTraitMoffat,
    OHTraitSech2])

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
    VPTraitMass,
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

# Weight polar traits parser
wpt_parser = parseutils.TypedParser(WPTrait, [
    WPTraitAxisRange])
