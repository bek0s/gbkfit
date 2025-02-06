
import abc
import copy
import logging

import numpy as np

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

# Vertical distortion polar traits
ZPT_UID_NW_UNIFORM = 101
ZPT_UID_NW_HARMONIC = 102

# Selection polar traits
SPT_UID_AZRANGE = 1
SPT_UID_NW_AZRANGE = 101

# Weight polar traits
WPT_UID_AXIS_RANGE = 1

TRUNC_DEFAULT = 0


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


def _ptrait_integrate_mixture_exponential(params, rings):  # noqa
    raise NotImplementedError()


def _ptrait_integrate_mixture_gauss(params, rings):  # noqa
    raise NotImplementedError()


def _ptrait_integrate_mixture_ggauss(params, rings):  # noqa
    raise NotImplementedError()


def _ptrait_integrate_mixture_moffat(params, rings):  # noqa
    raise NotImplementedError()


def _ptrait_integrate_nw_uniform(params, rings):
    a = params['a']
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_integrate_nw_harmonic(params, rings):
    a = params['a']
    c = np.inf
    return _integrate_rings(rings, gbkfit.math.uniform_1d_fun, a, 0, c)


def _ptrait_integrate_nw_distortion(params, rings):  # noqa
    raise NotImplementedError()


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


class Trait(parseutils.TypedParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def uid():
        pass

    @classmethod
    def load(cls, info):
        desc = trait_desc(cls)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(type=self.type())

    def __init__(self, **kwargs):
        self._kwargs = copy.deepcopy(kwargs)

    def consts(self):
        return tuple(self._kwargs.values())

    def params_sm(self):
        return tuple()

    def params_rnw(self, nnodes):
        return tuple()


class TraitFeatureTrunc:

    def dump(self):
        dump_trunc = self.trunc() > 0
        info = dict(trunc=self.trunc()) if dump_trunc else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        self._trunc = kwargs['trunc']
        super().__init__(**kwargs)

    def trunc(self):
        return self._trunc


class TraitFeatureNWMode:

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(
            nwmode_parser, info, 'nwmode', required=False)
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


class TraitFeatureRNodes:

    def dump(self):
        dump_rnodes = self.rnodes()
        info = dict(rnodes=self.rnodes()) if dump_rnodes else dict()
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        self._rnodes = kwargs['rnodes']
        super().__init__(**kwargs)

    def rnodes(self):
        return self._rnodes


class TraitFeatureNBlobs:

    def dump(self):
        info = dict(nblobs=self.nblobs())
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        self._nblobs = kwargs['nblobs']
        super().__init__(**kwargs)

    def nblobs(self):
        return self._nblobs


class TraitFeatureOrder:

    def dump(self):
        info = dict(order=self.order())
        return super().dump() | info  # noqa

    def __init__(self, **kwargs):
        self._order = kwargs['order']
        super().__init__(**kwargs)

    def order(self):
        return self._order


class PTrait(Trait, abc.ABC):
    pass


class HTrait(TraitFeatureRNodes, TraitFeatureNWMode, Trait, abc.ABC):

    def __init__(self, rnodes, nwmode, **kwargs):
        kwargs.update(rnodes=rnodes, nwmode=nwmode)
        super().__init__(**kwargs)
        if not self.rnodes() and self.nwmode():
            _log.warning(
                f"rnodes is set to {self.rnodes()}; "
                f"nwmode {self.nwmode()} will be ignored")


class BPTrait(PTrait, abc.ABC):

    @abc.abstractmethod
    def has_analytical_integral(self):
        pass

    @abc.abstractmethod
    def integrate(self, params, rings):
        pass


class BHTrait(TraitFeatureTrunc, HTrait, abc.ABC):

    def integrate(self, params):  # noqa
        # All surface brightness height traits are pdfs
        return 1


class VPTrait(PTrait, abc.ABC):
    pass


class VHTrait(HTrait, abc.ABC):
    pass


class DPTrait(PTrait, abc.ABC):
    pass


class DHTrait(HTrait, abc.ABC):
    pass


class ZPTrait(PTrait, abc.ABC):
    pass


class SPTrait(PTrait, abc.ABC):
    pass


class WPTrait(PTrait, abc.ABC):
    pass


class OPTrait(PTrait, abc.ABC):

    @abc.abstractmethod
    def has_analytical_integral(self):
        pass

    @abc.abstractmethod
    def integrate(self, params, rings):
        pass


class OHTrait(TraitFeatureTrunc, HTrait, abc.ABC):

    @abc.abstractmethod
    def integrate(self, params):
        pass


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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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
        return _ptrait_params_mixture_7p(self.nblobs())

    def has_analytical_integral(self):
        return True

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
        return _ptrait_integrate_mixture_moffat(params, rings)


class BPTraitNWUniform(TraitFeatureNWMode, BPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return BPT_UID_NW_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_uniform(params, rings)


class BPTraitNWHarmonic(TraitFeatureOrder, TraitFeatureNWMode, BPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return BPT_UID_NW_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_harmonic(params, rings)


class BPTraitNWDistortion(TraitFeatureNWMode, BPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return BPT_UID_NW_DISTORTION

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_distortion(params, rings)


class BHTraitP1(BHTrait, abc.ABC):

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            trunc: int | float = TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

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
            trunc: int | float = TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

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


class BHTraitLorentz(BHTraitP1):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return BHT_UID_LORENTZ


class BHTraitMoffat(BHTraitP2):

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


class VPTraitNWTanUniform(TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_TAN_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vt', nnodes), self.nwmode()),)


class VPTraitNWTanHarmonic(TraitFeatureOrder, TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_TAN_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWRadUniform(TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_RAD_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vr', nnodes), self.nwmode()),)


class VPTraitNWRadHarmonic(TraitFeatureOrder, TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_RAD_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWVerUniform(TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_VER_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vv', nnodes), self.nwmode()),)


class VPTraitNWVerHarmonic(TraitFeatureOrder, TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_VER_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWLOSUniform(TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_los_uniform'

    @staticmethod
    def uid():
        return VPT_UID_NW_LOS_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vl', nnodes), self.nwmode()),)


class VPTraitNWLOSHarmonic(TraitFeatureOrder, TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_los_harmonic'

    @staticmethod
    def uid():
        return VPT_UID_NW_LOS_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

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
        super().__init__(rnodes=False, nwmode=None)


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


class DPTraitNWUniform(TraitFeatureNWMode, DPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return DPT_UID_NW_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)


class DPTraitNWHarmonic(TraitFeatureOrder, TraitFeatureNWMode, DPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return DPT_UID_NW_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class DPTraitNWDistortion(TraitFeatureNWMode, DPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return DPT_UID_NW_DISTORTION

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

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
        super().__init__(rnodes=False, nwmode=None)


class ZPTraitNWUniform(TraitFeatureNWMode, ZPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return ZPT_UID_NW_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)


class ZPTraitNWHarmonic(TraitFeatureOrder, TraitFeatureNWMode, ZPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return ZPT_UID_NW_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

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


class SPTraitNWAzimuthalRange(TraitFeatureNWMode, SPTrait):

    @staticmethod
    def type():
        return 'nw_azrange'

    @staticmethod
    def uid():
        return SPT_UID_NW_AZRANGE

    def __init__(self, nwmode=None):
        super().__init__(nwmode=nwmode)

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

    def __init__(self, axis, angle: int | float, weight: int | float):
        if axis not in [0, 1]:
            raise RuntimeError(
                f"invalid axis value; "
                f"choose between 0 (minor axis) and 1 (major axis); "
                f"supplied value: {axis}")
        if 0 > angle > 180:
            raise RuntimeError(
                f"invalid angle value; "
                f"angle must be between 0 and 180; "
                f"supplied value: {angle}")
        super().__init__(axis=axis, angle=angle, weight=weight)


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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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

    def integrate(self, params, rings):
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
        return False

    def integrate(self, params, rings):
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
        return False

    def integrate(self, params, rings):
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
        return False

    def integrate(self, params, rings):
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
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_mixture_moffat(params, rings)


class OPTraitNWUniform(TraitFeatureNWMode, OPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return OPT_UID_NW_UNIFORM

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self.nwmode()),)

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_uniform(params, rings)


class OPTraitNWHarmonic(TraitFeatureOrder, TraitFeatureNWMode, OPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return OPT_UID_NW_HARMONIC

    def __init__(self, order: int, nwmode: NWMode | None = None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_harmonic(params, rings)


class OPTraitNWDistortion(TraitFeatureNWMode, OPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return OPT_UID_NW_DISTORTION

    def __init__(self, nwmode: NWMode | None = None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_distortion(params, rings)


class OHTraitP2(OHTrait, abc.ABC):

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            trunc: int | float = TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc('a', nrnodes), self.nwmode()),
            (ParamVectorDesc('s', nrnodes), self.nwmode()))


class OHTraitP3(OHTrait, abc.ABC):

    def __init__(
            self,
            rnodes: bool = False,
            nwmode: NWMode | None = None,
            trunc: int | float = TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            (ParamVectorDesc('a', nrnodes), self.nwmode()),
            (ParamVectorDesc('s', nrnodes), self.nwmode()),
            (ParamVectorDesc('b', nrnodes), self.nwmode()))


class OHTraitUniform(OHTraitP2):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return OHT_UID_UNIFORM

    def integrate(self, params):
        a = params['a']
        s = params['s']
        trunc = self.trunc() * s if self.trunc() else s
        return gbkfit.math.uniform_1d_int(a, -trunc, +trunc)


class OHTraitExponential(OHTraitP2):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return OHT_UID_EXP

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitGauss(OHTraitP2):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return OHT_UID_GAUSS

    def integrate(self, params):
        a = params['a']
        s = params['s']
        trunc = self.trunc() * s
        fun_f = gbkfit.math.gauss_1d_int
        fun_t = gbkfit.math.gauss_trunc_1d_int
        return fun_f(a, s) if not trunc else fun_t(a, s, -trunc, +trunc)


class OHTraitGGauss(OHTraitP3):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return OHT_UID_GGAUSS

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitLorentz(OHTraitP2):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return OHT_UID_LORENTZ

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitMoffat(OHTraitP3):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return OHT_UID_MOFFAT

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitSech2(OHTraitP2):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return OHT_UID_SECH2

    def integrate(self, params):
        raise NotImplementedError()


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
    VPTraitNWTanUniform,
    VPTraitNWTanHarmonic,
    VPTraitNWRadUniform,
    VPTraitNWRadHarmonic,
    VPTraitNWVerUniform,
    VPTraitNWVerHarmonic,
    VPTraitNWLOSUniform,
    VPTraitNWLOSHarmonic])

# Velocity height traits parser
vht_parser = parseutils.TypedParser(VHTrait, [
    VHTraitOne])

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
    DHTraitOne])

# Vertical polar distortion traits parser
zpt_parser = parseutils.TypedParser(ZPTrait, [
    ZPTraitNWUniform,
    ZPTraitNWHarmonic])

# Selection polar traits parser
spt_parser = parseutils.TypedParser(SPTrait, [
    SPTraitAzimuthalRange,
    SPTraitNWAzimuthalRange])

# Weight polar traits parser
wpt_parser = parseutils.TypedParser(WPTrait, [
    WPTraitAxisRange])
