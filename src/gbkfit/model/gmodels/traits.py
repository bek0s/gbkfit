
import abc

import numpy as np

import gbkfit.math
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import parseutils


# Density traits (polar)
RP_TRAIT_UID_UNIFORM = 1
RP_TRAIT_UID_EXPONENTIAL = 2
RP_TRAIT_UID_GAUSS = 3
RP_TRAIT_UID_GGAUSS = 4
RP_TRAIT_UID_LORENTZ = 5
RP_TRAIT_UID_MOFFAT = 6
RP_TRAIT_UID_SECH2 = 7
RP_TRAIT_UID_MIXTURE_GGAUSS = 8
RP_TRAIT_UID_MIXTURE_MOFFAT = 9
RP_TRAIT_UID_NW_UNIFORM = 101
RP_TRAIT_UID_NW_HARMONIC = 102
RP_TRAIT_UID_NW_DISTORTION = 103

# Density traits (height)
RH_TRAIT_UID_UNIFORM = 1
RH_TRAIT_UID_EXPONENTIAL = 2
RH_TRAIT_UID_GAUSS = 3
RH_TRAIT_UID_GGAUSS = 4
RH_TRAIT_UID_LORENTZ = 5
RH_TRAIT_UID_MOFFAT = 6
RH_TRAIT_UID_SECH2 = 7

# Velocity traits (polar)
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

# Velocity traits (height)
VH_TRAIT_UID_ONE = 1

# Dispersion traits (polar)
DP_TRAIT_UID_UNIFORM = 1
DP_TRAIT_UID_EXPONENTIAL = 2
DP_TRAIT_UID_GAUSS = 3
DP_TRAIT_UID_GGAUSS = 4
DP_TRAIT_UID_LORENTZ = 5
DP_TRAIT_UID_MOFFAT = 6
DP_TRAIT_UID_SECH2 = 7
DP_TRAIT_UID_MIXTURE_GGAUSS = 8
DP_TRAIT_UID_MIXTURE_MOFFAT = 9
DP_TRAIT_UID_NW_UNIFORM = 101
DP_TRAIT_UID_NW_HARMONIC = 102
DP_TRAIT_UID_NW_DISTORTION = 103

# Dispersion traits (height)
DH_TRAIT_UID_ONE = 1

# Vertical distortion traits (polar)
ZP_TRAIT_UID_NW_UNIFORM = 101
ZP_TRAIT_UID_NW_HARMONIC = 102

# Selection traits (polar)
SP_TRAIT_UID_AZRANGE = 1
SP_TRAIT_UID_NW_AZRANGE = 101

# Weight traits (polar)
JP_TRAIT_UID_CRANGE = 1


TRUNC_DEFAULT = 0


def _params_mixture(n):
    return (
        ParamVectorDesc('r', n),  # polar coord (radius)
        ParamVectorDesc('t', n),  # polar coord (angle)
        ParamVectorDesc('a', n),  # amplitude
        ParamVectorDesc('s', n),  # size
        ParamVectorDesc('b', n),  # shape
        ParamVectorDesc('q', n),  # axis ratio (b/a)
        ParamVectorDesc('p', n))  # position angle relative to t


def _params_pw_harmonic(order, nnodes):
    params = []
    params += [ParamVectorDesc('a', nnodes)]
    params += [ParamVectorDesc('p', nnodes)] * (order > 0)
    return tuple(params)


def _integrate_nw(nodes, fun, *args):
    rsep = nodes[1] - nodes[0]
    rmin = nodes - rsep * 0.5
    rmax = nodes + rsep * 0.5
    ampl = fun(nodes, *args)
    return np.pi * ampl * (rmax * rmax - rmin * rmin)


def _trait_desc(cls):
    descs = {
        RPTrait: 'density polar',
        RHTrait: 'density height',
        VPTrait: 'velocity polar',
        VHTrait: 'velocity height',
        DPTrait: 'velocity dispersion polar',
        DHTrait: 'velocity dispersion height',
        ZPTrait: 'vertical distortion',
        SPTrait: 'selection polar'}
    postfix = None
    for k, v in descs.items():
        if issubclass(cls, k):
            postfix = v
    return f'{cls.type()} {postfix} trait ({cls.__name__})'


class Trait(parseutils.TypedParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def uid():
        pass

    @classmethod
    def load(cls, info):
        desc = _trait_desc(cls)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def __init__(self, **kwargs):
        self._kwargs = kwargs.copy()

    def dump(self):
        return dict(**self._kwargs)

    def consts(self):
        return tuple(self._kwargs.values())


class PTraitParams(abc.ABC):

    def params_sm(self):
        return tuple()

    def params_rnw(self, nrnodes):
        return tuple()


class HTraitParams(abc.ABC):

    def params_sm(self):
        return tuple()

    def params_rnw(self, nrnodes):
        return tuple()


class RPTrait(PTraitParams, Trait, abc.ABC):

    @abc.abstractmethod
    def has_ordinary_integral(self):
        pass

    @abc.abstractmethod
    def integrate(self, params, rnodes):
        pass


class RHTrait(HTraitParams, Trait, abc.ABC):
    pass


class VPTrait(PTraitParams, Trait, abc.ABC):
    pass


class VHTrait(HTraitParams, Trait, abc.ABC):
    pass


class DPTrait(PTraitParams, Trait, abc.ABC):
    pass


class DHTrait(HTraitParams, Trait, abc.ABC):
    pass


class ZPTrait(PTraitParams, Trait, abc.ABC):
    pass


class SPTrait(PTraitParams, Trait, abc.ABC):
    pass


class WPTrait(PTraitParams, Trait, abc.ABC):
    pass


class RPTraitUniform(RPTrait):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('a'),)

    def has_ordinary_integral(self):
        return True

    def integrate(self, params, nodes):
        ampl = params['a']
        rsep = nodes[1] - nodes[0]
        rmin = nodes[0] - 0.5 * rsep
        rmax = nodes[-1] + 0.5 * rsep
        return np.pi * ampl * (rmax * rmax - rmin * rmin)


class RPTraitExponential(RPTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_EXPONENTIAL

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        return _integrate_nw(nodes, gbkfit.math.expon_1d_fun, a, 0, s)


class RPTraitGauss(RPTrait):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_GAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        #return np.array([a * 2 * np.pi * s * s])
        return _integrate_nw(nodes, gbkfit.math.gauss_1d_fun, a, 0, s)


class RPTraitGGauss(RPTrait):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_GGAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        b = params['b']
        return _integrate_nw(nodes, gbkfit.math.ggauss_1d_fun, a, 0, s, b)


class RPTraitLorentz(RPTrait):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_LORENTZ

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        return _integrate_nw(nodes, gbkfit.math.lorentz_1d_fun, a, 0, s)


class RPTraitMoffat(RPTrait):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_MOFFAT

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        b = params['b']
        return _integrate_nw(nodes, gbkfit.math.moffat_1d_fun, a, 0, s, b)


class RPTraitSech2(RPTrait):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_SECH2

    def params_sm(self):
        return (
            ParamScalarDesc('a'),
            ParamScalarDesc('s'))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        return _integrate_nw(nodes, gbkfit.math.sech2_1d_pdf, a, 0, s)


class RPTraitMixtureGGauss(RPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs):
        Trait.__init__(self, nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'])

    def has_ordinary_integral(self):
        return True

    def integrate(self, params, nodes):
        raise NotImplementedError()


class RPTraitMixtureMoffat(RPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs):
        Trait.__init__(self, nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'])

    def has_ordinary_integral(self):
        return True

    def integrate(self, params, nodes):
        raise NotImplementedError()


class RPTraitNWUniform(RPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_NW_UNIFORM

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),)

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        return _integrate_nw(nodes, gbkfit.math.uniform_1d_fun, a, 0, np.inf)


class RPTraitNWHarmonic(RPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order):
        super().__init__(order=order)

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes)

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = nodes[-1] - nodes[0]
        return _integrate_nw(nodes, gbkfit.math.uniform_1d_fun, a, 0, s) * 2


class RPTraitNWDistortion(RPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_NW_DISTORTION

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),
            ParamVectorDesc('p', nnodes),
            ParamVectorDesc('s', nnodes))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        # foo = _integrate_nw(nodes, gbkfit.math.gauss_1d_fun, a, 0, s * nodes)
        # foo = a * np.sqrt(2 * np.pi) / np.sqrt(1 / s * s) * (nodes[1] - nodes[0])
        foo = a * np.sqrt(2 * np.pi) / np.sqrt((1) / (s * s)) * 1
        return foo


class _RHTraitSM(RHTrait, abc.ABC):

    def __init__(self, consts, params):
        Trait.__init__(self, **consts)
        self._params = tuple(params)

    def params_sm(self):
        if self._kwargs['rnodes']:
            return tuple()
        else:
            return tuple([ParamScalarDesc(p) for p in self._params])

    def params_rnw(self, nrnodes):
        if self._kwargs['rnodes']:
            return tuple([ParamVectorDesc(p, nrnodes) for p in self._params])
        else:
            return tuple()


class RHTraitUniform(_RHTraitSM):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_UNIFORM

    def __init__(self, rnodes=False):
        consts = dict(rnodes=rnodes)
        params = ['s']
        super().__init__(consts, params)


class RHTraitExponential(_RHTraitSM):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_EXPONENTIAL

    def __init__(self, rnodes=False, trunc=TRUNC_DEFAULT):
        consts = dict(rnodes=rnodes, trunc=trunc)
        params = ['s']
        super().__init__(consts, params)


class RHTraitGauss(_RHTraitSM):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_GAUSS

    def __init__(self, rnodes=False, trunc=TRUNC_DEFAULT):
        consts = dict(rnodes=rnodes, trunc=trunc)
        params = ['s']
        super().__init__(consts, params)


class RHTraitGGauss(_RHTraitSM):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_GGAUSS

    def __init__(self, rnodes=False, trunc=TRUNC_DEFAULT):
        consts = dict(rnodes=rnodes, trunc=trunc)
        params = ['s', 'b']
        super().__init__(consts, params)


class RHTraitLorentz(_RHTraitSM):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_LORENTZ

    def __init__(self, rnodes=False, trunc=TRUNC_DEFAULT):
        consts = dict(rnodes=rnodes, trunc=trunc)
        params = ['s']
        super().__init__(consts, params)


class RHTraitMoffat(_RHTraitSM):

    @staticmethod
    def type():
        return 'moffat'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_MOFFAT

    def __init__(self, rnodes=False, trunc=TRUNC_DEFAULT):
        consts = dict(rnodes=rnodes, trunc=trunc)
        params = ['s', 'b']
        super().__init__(consts, params)


class RHTraitSech2(_RHTraitSM):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_SECH2

    def __init__(self, rnodes=False, trunc=TRUNC_DEFAULT):
        consts = dict(rnodes=rnodes, trunc=trunc)
        params = ['s']
        super().__init__(consts, params)


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


class VPTraitNWTanUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_TAN_UNIFORM

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('vt', nnodes),)


class VPTraitNWTanHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_TAN_HARMONIC

    def __init__(self, order):
        Trait.__init__(self, order=order)

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes)


class VPTraitNWRadUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_RAD_UNIFORM

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('vr', nnodes),)


class VPTraitNWRadHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_RAD_HARMONIC

    def __init__(self, order):
        Trait.__init__(self, order=order)

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes)


class VPTraitNWVerUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_VER_UNIFORM

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('vv', nnodes),)


class VPTraitNWVerHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_VER_HARMONIC

    def __init__(self, order):
        Trait.__init__(self, order=order)

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes)


class VPTraitNWLOSUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_los_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_LOS_UNIFORM

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('vl', nnodes),)


class VPTraitNWLOSHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_los_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_LOS_HARMONIC

    def __init__(self, order):
        Trait.__init__(self, order=order)

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes)


class VHTraitOne(VHTrait):

    @staticmethod
    def type():
        return 'one'

    @staticmethod
    def uid():
        return VH_TRAIT_UID_ONE


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


class DPTraitMixtureGGauss(DPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs):
        Trait.__init__(self, nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'])


class DPTraitMixtureMoffat(DPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs):
        Trait.__init__(self, nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'])


class DPTraitNWUniform(DPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_UNIFORM

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),)


class DPTraitNWHarmonic(DPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order):
        Trait.__init__(self, order=order)

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes)


class DPTraitNWDistortion(DPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_DISTORTION

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),
            ParamVectorDesc('p', nnodes),
            ParamVectorDesc('s', nnodes))


class DHTraitOne(DHTrait):

    @staticmethod
    def type():
        return 'one'

    @staticmethod
    def uid():
        return DH_TRAIT_UID_ONE


class ZPTraitNWUniform(ZPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return ZP_TRAIT_UID_NW_UNIFORM

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),)


class ZPTraitNWHarmonic(ZPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return ZP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order):
        Trait.__init__(self, order=order)

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes)


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


class SPTraitNWAzimuthalRange(SPTrait):

    @staticmethod
    def type():
        return 'nw_azrange'

    @staticmethod
    def uid():
        return SP_TRAIT_UID_NW_AZRANGE

    def params_rnw(self, nnodes):
        return (
            ParamVectorDesc('p', nnodes),
            ParamVectorDesc('s', nnodes))


class WPTraitConstantRange(WPTrait):

    @staticmethod
    def type():
        return 'crange'

    @staticmethod
    def uid():
        return JP_TRAIT_UID_CRANGE

    def __init__(self, angle):
        Trait.__init__(self, angle=angle)


# Density traits (polar) parser
rpt_parser = parseutils.TypedParser(RPTrait, [
    RPTraitUniform,
    RPTraitExponential,
    RPTraitGauss,
    RPTraitGGauss,
    RPTraitLorentz,
    RPTraitMoffat,
    RPTraitSech2,
    RPTraitMixtureGGauss,
    RPTraitMixtureMoffat,
    RPTraitNWUniform,
    RPTraitNWHarmonic,
    RPTraitNWDistortion])

# Density traits (height) parser
rht_parser = parseutils.TypedParser(RHTrait, [
    RHTraitUniform,
    RHTraitExponential,
    RHTraitGauss,
    RHTraitGGauss,
    RHTraitLorentz,
    RHTraitSech2])

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
    WPTraitConstantRange])
