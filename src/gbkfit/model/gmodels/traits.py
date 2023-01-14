
import abc
import copy
import logging

import numpy as np

import gbkfit.math
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import parseutils


_log = logging.getLogger(__name__)


# Surface Brightness polar traits
BP_TRAIT_UID_UNIFORM = 1
BP_TRAIT_UID_EXPONENTIAL = 2
BP_TRAIT_UID_GAUSS = 3
BP_TRAIT_UID_GGAUSS = 4
BP_TRAIT_UID_LORENTZ = 5
BP_TRAIT_UID_MOFFAT = 6
BP_TRAIT_UID_SECH2 = 7
BP_TRAIT_UID_MIXTURE_EXPONENTIAL = 51
BP_TRAIT_UID_MIXTURE_GAUSS = 52
BP_TRAIT_UID_MIXTURE_GGAUSS = 53
BP_TRAIT_UID_MIXTURE_MOFFAT = 54
BP_TRAIT_UID_NW_UNIFORM = 101
BP_TRAIT_UID_NW_HARMONIC = 102
BP_TRAIT_UID_NW_DISTORTION = 103

# Surface Brightness height traits
BH_TRAIT_UID_UNIFORM = 1
BH_TRAIT_UID_EXPONENTIAL = 2
BH_TRAIT_UID_GAUSS = 3
BH_TRAIT_UID_GGAUSS = 4
BH_TRAIT_UID_LORENTZ = 5
BH_TRAIT_UID_MOFFAT = 6
BH_TRAIT_UID_SECH2 = 7

# Velocity polar traits
VP_TRAIT_UID_TAN_UNIFORM = 1
VP_TRAIT_UID_TAN_ARCTAN = 2
VP_TRAIT_UID_TAN_BOISSIER = 3
VP_TRAIT_UID_TAN_EPINAT = 4
VP_TRAIT_UID_TAN_LRAMP = 5
VP_TRAIT_UID_TAN_TANH = 6
VP_TRAIT_UID_TAN_POLYEX = 7
VP_TRAIT_UID_TAN_RIX = 8
VP_TRAIT_UID_NW_TAN_UNIFORM = 101
VP_TRAIT_UID_NW_TAN_HARMONIC = 102
VP_TRAIT_UID_NW_RAD_UNIFORM = 103
VP_TRAIT_UID_NW_RAD_HARMONIC = 104
VP_TRAIT_UID_NW_VER_UNIFORM = 105
VP_TRAIT_UID_NW_VER_HARMONIC = 106
VP_TRAIT_UID_NW_LOS_UNIFORM = 107
VP_TRAIT_UID_NW_LOS_HARMONIC = 108

# Velocity height traits
VH_TRAIT_UID_ONE = 1

# Dispersion polar traits
DP_TRAIT_UID_UNIFORM = 1
DP_TRAIT_UID_EXPONENTIAL = 2
DP_TRAIT_UID_GAUSS = 3
DP_TRAIT_UID_GGAUSS = 4
DP_TRAIT_UID_LORENTZ = 5
DP_TRAIT_UID_MOFFAT = 6
DP_TRAIT_UID_SECH2 = 7
DP_TRAIT_UID_MIXTURE_EXPONENTIAL = 51
DP_TRAIT_UID_MIXTURE_GAUSS = 52
DP_TRAIT_UID_MIXTURE_GGAUSS = 53
DP_TRAIT_UID_MIXTURE_MOFFAT = 54
DP_TRAIT_UID_NW_UNIFORM = 101
DP_TRAIT_UID_NW_HARMONIC = 102
DP_TRAIT_UID_NW_DISTORTION = 103

# Dispersion height traits
DH_TRAIT_UID_ONE = 1

# Vertical distortion polar traits
ZP_TRAIT_UID_NW_UNIFORM = 101
ZP_TRAIT_UID_NW_HARMONIC = 102

# Selection polar traits
SP_TRAIT_UID_AZRANGE = 1
SP_TRAIT_UID_NW_AZRANGE = 101

# Weight polar traits
WP_TRAIT_UID_AXIS_RANGE = 1

# Opacity polar traits
OP_TRAIT_UID_UNIFORM = 1
OP_TRAIT_UID_EXPONENTIAL = 2
OP_TRAIT_UID_GAUSS = 3
OP_TRAIT_UID_GGAUSS = 4
OP_TRAIT_UID_LORENTZ = 5
OP_TRAIT_UID_MOFFAT = 6
OP_TRAIT_UID_SECH2 = 7
OP_TRAIT_UID_MIXTURE_EXPONENTIAL = 51
OP_TRAIT_UID_MIXTURE_GAUSS = 52
OP_TRAIT_UID_MIXTURE_GGAUSS = 53
OP_TRAIT_UID_MIXTURE_MOFFAT = 54
OP_TRAIT_UID_NW_UNIFORM = 101
OP_TRAIT_UID_NW_HARMONIC = 102
OP_TRAIT_UID_NW_DISTORTION = 103

# Opacity height traits
OH_TRAIT_UID_UNIFORM = 1
OH_TRAIT_UID_EXPONENTIAL = 2
OH_TRAIT_UID_GAUSS = 3
OH_TRAIT_UID_GGAUSS = 4
OH_TRAIT_UID_LORENTZ = 5
OH_TRAIT_UID_MOFFAT = 6
OH_TRAIT_UID_SECH2 = 7

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


def _htrait_params_fun_2p_sm(rnodes_enabled):
    return () if rnodes_enabled else (
        ParamScalarDesc('a'),
        ParamScalarDesc('s'))


def _htrait_params_fun_2p_nw(rnodes_enabled, nrnodes):
    return () if not rnodes_enabled else (
        ParamVectorDesc('a', nrnodes),
        ParamVectorDesc('s', nrnodes))


def _htrait_params_fun_3p_sm(rnodes_enabled):
    return () if rnodes_enabled else (
        ParamScalarDesc('a'),
        ParamScalarDesc('s'),
        ParamScalarDesc('b'))


def _htrait_params_fun_3p_nw(rnodes_enabled, nrnodes):
    return () if not rnodes_enabled else (
        ParamVectorDesc('a', nrnodes),
        ParamVectorDesc('s', nrnodes),
        ParamVectorDesc('b', nrnodes))


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

    def dump(self):
        dump_nwmode = self.nwmode() not in [None, 'absolute']
        info = dict(nwmode=self.nwmode()) if dump_nwmode else dict()
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
        if not self.rnodes() and self.nwmode() not in [None, 'absolute']:
            _log.warning(
                f"rnodes is set to {self.rnodes()}; "
                f"nwmode {self.nwmode()} will be ignored")


class BPTrait(PTrait, abc.ABC):

    @abc.abstractmethod
    def has_analytical_integral(self):
        pass

    @abc.abstractmethod
    def integrate(self, params, nodes):
        pass


class BHTrait(TraitFeatureTrunc, HTrait, abc.ABC):

    def integrate(self, params):  # noqa
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
    def integrate(self, params, nodes):
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
        return BP_TRAIT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('a'),)

    def has_analytical_integral(self):
        return True

    def integrate(self, params, rings):
        a = params['a']
        rsep = rings[1] - rings[0]
        rmin = rings[0] - 0.5 * rsep
        rmax = rings[-1] + 0.5 * rsep
        return np.pi * a * (rmax * rmax - rmin * rmin)


class BPTraitExponential(BPTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return BP_TRAIT_UID_EXPONENTIAL

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
        return BP_TRAIT_UID_GAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        a = params['a']
        s = params['s']
        return _integrate_rings(rings, gbkfit.math.gauss_1d_fun, a, 0, s)


class BPTraitGGauss(BPTrait):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return BP_TRAIT_UID_GGAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        a = params['a']
        s = params['s']
        b = params['b']
        return _integrate_rings(rings, gbkfit.math.ggauss_1d_fun, a, 0, s, b)


class BPTraitLorentz(BPTrait):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return BP_TRAIT_UID_LORENTZ

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        a = params['a']
        s = params['s']
        return _integrate_rings(rings, gbkfit.math.lorentz_1d_fun, a, 0, s)


class BPTraitMoffat(BPTrait):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return BP_TRAIT_UID_MOFFAT

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        a = params['a']
        s = params['s']
        b = params['b']
        return _integrate_rings(rings, gbkfit.math.moffat_1d_fun, a, 0, s, b)


class BPTraitSech2(BPTrait):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return BP_TRAIT_UID_SECH2

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        a = params['a']
        s = params['s']
        return _integrate_rings(rings, gbkfit.math.sech2_1d_fun, a, 0, s)


class BPTraitMixtureExponential(TraitFeatureNBlobs, BPTrait):

    @staticmethod
    def type():
        return 'mixture_exponential'

    @staticmethod
    def uid():
        return BP_TRAIT_UID_MIXTURE_EXPONENTIAL

    def __init__(self, nblobs):
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
        return BP_TRAIT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs):
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
        return BP_TRAIT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs):
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
        return BP_TRAIT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs):
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
        return BP_TRAIT_UID_NW_UNIFORM

    def __init__(self, nwmode=None):
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
        return BP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order, nwmode=None):
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
        return BP_TRAIT_UID_NW_DISTORTION

    def __init__(self, nwmode=None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_distortion(params, rings)


class BHTraitP1(BHTrait, abc.ABC):

    def __init__(self, rnodes=False, nwmode=None, trunc=TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('s'),)

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            ParamVectorDesc('s', nrnodes),)


class BHTraitP2(BHTrait, abc.ABC):

    def __init__(self, rnodes=False, nwmode=None, trunc=TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

    def params_sm(self):
        return () if self.rnodes() else (
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def params_rnw(self, nrnodes):
        return () if not self.rnodes() else (
            ParamVectorDesc('s', nrnodes),
            ParamVectorDesc('b', nrnodes))


class BHTraitUniform(BHTraitP1):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return BH_TRAIT_UID_UNIFORM


class BHTraitExponential(BHTraitP1):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return BH_TRAIT_UID_EXPONENTIAL


class BHTraitGauss(BHTraitP1):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return BH_TRAIT_UID_GAUSS


class BHTraitGGauss(BHTraitP2):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return BH_TRAIT_UID_GGAUSS


class BHTraitLorentz(BHTraitP1):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return BH_TRAIT_UID_LORENTZ


class BHTraitMoffat(BHTraitP2):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return BH_TRAIT_UID_MOFFAT


class BHTraitSech2(BHTraitP1):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return BH_TRAIT_UID_SECH2


class VPTraitTanUniform(VPTrait):

    @staticmethod
    def type():
        return 'tan_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_TAN_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('vt'),)


class VPTraitTanArctan(VPTrait):

    @staticmethod
    def type():
        return 'tan_arctan'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_TAN_ARCTAN

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
        return VP_TRAIT_UID_TAN_BOISSIER

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
        return VP_TRAIT_UID_TAN_EPINAT

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
        return VP_TRAIT_UID_TAN_LRAMP

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
        return VP_TRAIT_UID_TAN_TANH

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
        return VP_TRAIT_UID_TAN_POLYEX

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
        return VP_TRAIT_UID_TAN_RIX

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
        return VP_TRAIT_UID_NW_TAN_UNIFORM

    def __init__(self, nwmode=None):
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
        return VP_TRAIT_UID_NW_TAN_HARMONIC

    def __init__(self, order, nwmode=None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWRadUniform(TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_RAD_UNIFORM

    def __init__(self, nwmode=None):
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
        return VP_TRAIT_UID_NW_RAD_HARMONIC

    def __init__(self, order, nwmode=None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWVerUniform(TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_VER_UNIFORM

    def __init__(self, nwmode=None):
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
        return VP_TRAIT_UID_NW_VER_HARMONIC

    def __init__(self, order, nwmode=None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class VPTraitNWLOSUniform(TraitFeatureNWMode, VPTrait):

    @staticmethod
    def type():
        return 'nw_los_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_LOS_UNIFORM

    def __init__(self, nwmode=None):
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
        return VP_TRAIT_UID_NW_LOS_HARMONIC

    def __init__(self, order, nwmode=None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self._nwmode)


class VHTraitOne(VHTrait):

    @staticmethod
    def type():
        return 'one'

    @staticmethod
    def uid():
        return VH_TRAIT_UID_ONE

    def __init__(self):
        super().__init__(rnodes=False, nwmode=None)


class DPTraitUniform(DPTrait):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('a'),)


class DPTraitExponential(DPTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_EXPONENTIAL

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
        return DP_TRAIT_UID_GAUSS

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
        return DP_TRAIT_UID_GGAUSS

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
        return DP_TRAIT_UID_LORENTZ

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
        return DP_TRAIT_UID_MOFFAT

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
        return DP_TRAIT_UID_SECH2

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
        return DP_TRAIT_UID_MIXTURE_EXPONENTIAL

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())


class DPTraitMixtureGauss(TraitFeatureNBlobs, DPTrait):

    @staticmethod
    def type():
        return 'mixture_gauss'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_6p(self.nblobs())


class DPTraitMixtureGGauss(TraitFeatureNBlobs, DPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())


class DPTraitMixtureMoffat(TraitFeatureNBlobs, DPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _ptrait_params_mixture_7p(self.nblobs())


class DPTraitNWUniform(TraitFeatureNWMode, DPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_UNIFORM

    def __init__(self, nwmode=None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self._nwmode),)


class DPTraitNWHarmonic(TraitFeatureOrder, TraitFeatureNWMode, DPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order, nwmode=None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class DPTraitNWDistortion(TraitFeatureNWMode, DPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_DISTORTION

    def __init__(self, nwmode=None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())


class DHTraitOne(DHTrait):

    @staticmethod
    def type():
        return 'one'

    @staticmethod
    def uid():
        return DH_TRAIT_UID_ONE

    def __init__(self):
        super().__init__(rnodes=False, nwmode=None)


class ZPTraitNWUniform(TraitFeatureNWMode, ZPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return ZP_TRAIT_UID_NW_UNIFORM

    def __init__(self, nwmode=None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self._nwmode),)


class ZPTraitNWHarmonic(TraitFeatureOrder, TraitFeatureNWMode, ZPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return ZP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order, nwmode=None):
        super().__init__(order=order, nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_harmonic(self.order(), nnodes, self.nwmode())


class SPTraitAzimuthalRange(SPTrait):

    @staticmethod
    def type():
        return 'azrange'

    @staticmethod
    def uid():
        return SP_TRAIT_UID_AZRANGE

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
        return SP_TRAIT_UID_NW_AZRANGE

    def __init__(self, nwmode=None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('p', nnodes), self._nwmode),
            (ParamVectorDesc('s', nnodes), self._nwmode))


class WPTraitAxisRange(WPTrait):

    @staticmethod
    def type():
        return 'axis_range'

    @staticmethod
    def uid():
        return WP_TRAIT_UID_AXIS_RANGE

    def __init__(self, axis, angle, weight):
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
        return OP_TRAIT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('a'),)

    def has_analytical_integral(self):
        return True

    def integrate(self, params, rings):
        a = params['a']
        rsep = rings[1] - rings[0]
        rmin = rings[0] - 0.5 * rsep
        rmax = rings[-1] + 0.5 * rsep
        return np.pi * a * (rmax * rmax - rmin * rmin)


class OPTraitExponential(OPTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return OP_TRAIT_UID_EXPONENTIAL

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
        return OP_TRAIT_UID_GAUSS

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
        return OP_TRAIT_UID_GGAUSS

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
        return OP_TRAIT_UID_LORENTZ

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
        return OP_TRAIT_UID_MOFFAT

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
        return OP_TRAIT_UID_SECH2

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
        return OP_TRAIT_UID_MIXTURE_EXPONENTIAL

    def __init__(self, nblobs):
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
        return OP_TRAIT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs):
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
        return OP_TRAIT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs):
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
        return OP_TRAIT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs):
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
        return OP_TRAIT_UID_NW_UNIFORM

    def __init__(self, nwmode=None):
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
        return OP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order, nwmode=None):
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
        return OP_TRAIT_UID_NW_DISTORTION

    def __init__(self, nwmode=None):
        super().__init__(nwmode=nwmode)

    def params_rnw(self, nnodes):
        return _ptrait_params_nw_distortion(nnodes, self.nwmode())

    def has_analytical_integral(self):
        return False

    def integrate(self, params, rings):
        return _ptrait_integrate_nw_distortion(params, rings)


class OHTraitP2(OHTrait, abc.ABC):

    def __init__(self, rnodes=False, nwmode=None, trunc=TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

    def params_sm(self):
        return _htrait_params_fun_2p_sm(self.rnodes())

    def params_rnw(self, nrnodes):
        return _htrait_params_fun_2p_nw(self.rnodes(), nrnodes)


class OHTraitP3(OHTrait, abc.ABC):

    def __init__(self, rnodes=False, nwmode=None, trunc=TRUNC_DEFAULT):
        super().__init__(rnodes=rnodes, nwmode=nwmode, trunc=trunc)

    def params_sm(self):
        return _htrait_params_fun_3p_sm(self.rnodes())

    def params_rnw(self, nrnodes):
        return _htrait_params_fun_3p_nw(self.rnodes(), nrnodes)


class OHTraitUniform(OHTraitP2):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return OH_TRAIT_UID_UNIFORM

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
        return OH_TRAIT_UID_EXPONENTIAL

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitGauss(OHTraitP2):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return OH_TRAIT_UID_GAUSS

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
        return OH_TRAIT_UID_GGAUSS

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitLorentz(OHTraitP2):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return OH_TRAIT_UID_LORENTZ

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitMoffat(OHTraitP3):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return OH_TRAIT_UID_MOFFAT

    def integrate(self, params):
        raise NotImplementedError()


class OHTraitSech2(OHTraitP2):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return OH_TRAIT_UID_SECH2

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

# Velocity traits (polar) parser
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

# Velocity traits (height) parser
vht_parser = parseutils.TypedParser(VHTrait, [
    VHTraitOne])

# Dispersion traits (polar) parser
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

# Dispersion traits (height) parser
dht_parser = parseutils.TypedParser(DHTrait, [
    DHTraitOne])

# Vertical distortion traits (polar) parser
zpt_parser = parseutils.TypedParser(ZPTrait, [
    ZPTraitNWUniform,
    ZPTraitNWHarmonic])

# Selection traits (polar) parser
spt_parser = parseutils.TypedParser(SPTrait, [
    SPTraitAzimuthalRange,
    SPTraitNWAzimuthalRange])

# Weight traits (polar) parser
wpt_parser = parseutils.TypedParser(WPTrait, [
    WPTraitAxisRange])

# Opacity traits (polar) parser
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

# Opacity traits (height) parser
oht_parser = parseutils.TypedParser(OHTrait, [
    OHTraitUniform,
    OHTraitExponential,
    OHTraitGauss,
    OHTraitGGauss,
    OHTraitLorentz,
    OHTraitMoffat,
    OHTraitSech2])
