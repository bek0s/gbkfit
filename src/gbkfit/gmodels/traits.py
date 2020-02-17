
import abc

import numpy as np

import gbkfit.math
from gbkfit.params import ParamScalarDesc, ParamVectorDesc
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
RH_TRAIT_UID_SECH2 = 6

# Velocity traits (polar)
VP_TRAIT_UID_TAN_UNIFORM = 1
VP_TRAIT_UID_TAN_ARCTAN = 2
VP_TRAIT_UID_TAN_BOISSIER = 3
VP_TRAIT_UID_TAN_EPINAT = 4
VP_TRAIT_UID_TAN_LRAMP = 5
VP_TRAIT_UID_TAN_TANH = 6
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

# Warp traits (polar)
WP_TRAIT_UID_NW_UNIFORM = 101
WP_TRAIT_UID_NW_HARMONIC = 102

# Selection traits (polar)
SP_TRAIT_UID_AZRANGE = 1
SP_TRAIT_UID_NW_AZRANGE = 101


def _params_mixture(n):
    return (
        ParamVectorDesc('r', n),  # polar coord (radius)
        ParamVectorDesc('t', n),  # polar coord (angle)
        ParamVectorDesc('a', n),  # amplitude
        ParamVectorDesc('s', n),  # size
        ParamVectorDesc('b', n),  # shape
        ParamVectorDesc('q', n),  # axis ratio (b/a)
        ParamVectorDesc('p', n))  # position angle relative to t


def _params_pw_harmonic(k, nnodes):
    params = []
    params += [ParamVectorDesc(f'a', nnodes)]
    params += [ParamVectorDesc(f'p', nnodes)] * (k > 0)
    return tuple(params)


def _integrate_nw(nodes, fun, *args):
    rsep = nodes[1] - nodes[0]
    rmin = nodes - rsep * 0.5
    rmax = nodes + rsep * 0.5
    ampl = fun(nodes, *args)
    return np.pi * ampl * (rmax * rmax - rmin * rmin)


class Trait(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @staticmethod
    @abc.abstractmethod
    def uid():
        pass

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return {'type': self.type()}

    def consts(self):
        return tuple()

    def params_sm(self):
        return tuple()

    def params_nw(self, nnodes):
        return tuple()


class RPTrait(Trait, abc.ABC):

    @abc.abstractmethod
    def has_ordinary_integral(self):
        pass

    @abc.abstractmethod
    def integrate(self, params, nodes):
        pass


class RHTrait(Trait, abc.ABC):
    pass


class VPTrait(Trait, abc.ABC):
    pass


class VHTrait(Trait, abc.ABC):
    pass


class DPTrait(Trait, abc.ABC):
    pass


class DHTrait(Trait, abc.ABC):
    pass


class WPTrait(Trait, abc.ABC):
    pass


class SPTrait(Trait, abc.ABC):
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

    @classmethod
    def load(cls, info):
        return cls(info['n'])

    def dump(self):
        return {'type': self.type(), 'n': self._n}

    def __init__(self, n):
        self._n = n

    def consts(self):
        return self._n,

    def params_sm(self):
        return _params_mixture(self._n)

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

    @classmethod
    def load(cls, info):
        return cls(info['n'])

    def dump(self):
        return {'type': self.type(), 'n': self._n}

    def __init__(self, n):
        self._n = n

    def consts(self):
        return self._n,

    def params_sm(self):
        return _params_mixture(self._n)

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

    def params_nw(self, nnodes):
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

    @classmethod
    def load(cls, info):
        return cls(info['k'])

    def dump(self):
        return {'type': self.type(), 'k': self._k}

    def __init__(self, k):
        self._k = k

    def consts(self, nnodes=None):
        return self._k,

    def params_nw(self, nnodes):
        return _params_pw_harmonic(self._k, nnodes)

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

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),
            ParamVectorDesc('p', nnodes),
            ParamVectorDesc('s', nnodes))

    def has_ordinary_integral(self):
        return False

    def integrate(self, params, nodes):
        raise NotImplementedError()


class RHTraitUniform(RHTrait):

    @staticmethod
    def type():
        return 'uniform'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_UNIFORM

    def params_sm(self):
        return (
            ParamScalarDesc('s'),)


class RHTraitExponential(RHTrait):

    @staticmethod
    def type():
        return 'exponential'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_EXPONENTIAL

    def params_sm(self):
        return (
            ParamScalarDesc('s'),)


class RHTraitGauss(RHTrait):

    @staticmethod
    def type():
        return 'gauss'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_GAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('s'),)


class RHTraitGGauss(RHTrait):

    @staticmethod
    def type():
        return 'ggauss'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_GGAUSS

    def params_sm(self):
        return (
            ParamScalarDesc('s'),
            ParamScalarDesc('b'))


class RHTraitLorentz(RHTrait):

    @staticmethod
    def type():
        return 'lorentz'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_LORENTZ

    def params_sm(self):
        return (
            ParamScalarDesc('s'),)


class RHTraitSech2(RHTrait):

    @staticmethod
    def type():
        return 'sech2'

    @staticmethod
    def uid():
        return RH_TRAIT_UID_SECH2

    def params_sm(self):
        return (
            ParamScalarDesc('s'),)


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
            ParamScalarDesc('ea'),
            ParamScalarDesc('eb'))


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


class VPTraitNWTanUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_TAN_UNIFORM

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('vt', nnodes),)


class VPTraitNWTanHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_TAN_HARMONIC

    @classmethod
    def load(cls, info):
        return cls(info['k'])

    def dump(self):
        return {'type': self.type(), 'k': self._k}

    def __init__(self, k):
        self._k = k

    def consts(self):
        return self._k,

    def params_nw(self, nnodes):
        return _params_pw_harmonic(self._k, nnodes)


class VPTraitNWRadUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_RAD_UNIFORM

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('vr', nnodes),)


class VPTraitNWRadHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_RAD_HARMONIC

    @classmethod
    def load(cls, info):
        return cls(info['k'])

    def dump(self):
        return {'type': self.type(), 'k': self._k}

    def __init__(self, k):
        self._k = k

    def consts(self):
        return self._k,

    def params_nw(self, nnodes):
        return _params_pw_harmonic(self._k, nnodes)


class VPTraitNWVerUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_VER_UNIFORM

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('vv', nnodes),)


class VPTraitNWVerHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_VER_HARMONIC

    @classmethod
    def load(cls, info):
        return cls(info['k'])

    def dump(self):
        return {'type': self.type(), 'k': self._k}

    def __init__(self, k):
        self._k = k

    def consts(self):
        return self._k,

    def params_nw(self, nnodes):
        return _params_pw_harmonic(self._k, nnodes)


class VPTraitNWLOSUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_los_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_LOS_UNIFORM

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('vl', nnodes),)


class VPTraitNWLOSHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_los_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_LOS_HARMONIC

    @classmethod
    def load(cls, info):
        return cls(info['k'])

    def dump(self):
        return {'type': self.type(), 'k': self._k}

    def __init__(self, k):
        self._k = k

    def consts(self):
        return self._k,

    def params_nw(self, nnodes):
        return _params_pw_harmonic(self._k, nnodes)


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

    @classmethod
    def load(cls, info):
        return cls(info['n'])

    def dump(self):
        return {'type': self.type(), 'n': self._n}

    def __init__(self, n):
        self._n = n

    def consts(self):
        return self._n,

    def params_sm(self):
        return _params_mixture(self._n)


class DPTraitMixtureMoffat(DPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_MOFFAT

    @classmethod
    def load(cls, info):
        return cls(info['n'])

    def dump(self):
        return {'type': self.type(), 'n': self._n}

    def __init__(self, n):
        self._n = n

    def consts(self):
        return self._n,

    def params_sm(self):
        return _params_mixture(self._n)


class DPTraitNWUniform(DPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_UNIFORM

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),)


class DPTraitNWHarmonic(DPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_HARMONIC

    @classmethod
    def load(cls, info):
        return cls(info['k'])

    def dump(self):
        return {'type': self.type(), 'k': self._k}

    def __init__(self, k):
        self._k = k

    def consts(self, nnodes=None):
        return self._k,

    def params_nw(self, nnodes):
        return _params_pw_harmonic(self._k, nnodes)


class DPTraitNWDistortion(DPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_DISTORTION

    def params_nw(self, nnodes):
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


class WPTraitNWUniform(WPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_UNIFORM

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('a', nnodes),)


class WPTraitNWHarmonic(WPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_HARMONIC

    @classmethod
    def load(cls, info):
        return cls(info['k'])

    def dump(self):
        return {'type': self.type(), 'k': self._k}

    def __init__(self, k):
        self._k = k

    def consts(self, nnodes=None):
        return self._k,

    def params_nw(self, nnodes):
        return _params_pw_harmonic(self._k, nnodes)


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

    def params_nw(self, nnodes):
        return (
            ParamVectorDesc('p', nnodes),
            ParamVectorDesc('s', nnodes))


# Trait parsers
rpt_parser = parseutils.TypedParser(RPTrait)
rht_parser = parseutils.TypedParser(RHTrait)
vpt_parser = parseutils.TypedParser(VPTrait)
vht_parser = parseutils.TypedParser(VHTrait)
dpt_parser = parseutils.TypedParser(DPTrait)
dht_parser = parseutils.TypedParser(DHTrait)
wpt_parser = parseutils.TypedParser(WPTrait)
spt_parser = parseutils.TypedParser(SPTrait)

# Density traits (polar)
rpt_parser.register(RPTraitUniform)
rpt_parser.register(RPTraitExponential)
rpt_parser.register(RPTraitGauss)
rpt_parser.register(RPTraitGGauss)
rpt_parser.register(RPTraitLorentz)
rpt_parser.register(RPTraitMoffat)
rpt_parser.register(RPTraitSech2)
rpt_parser.register(RPTraitMixtureGGauss)
rpt_parser.register(RPTraitMixtureMoffat)
rpt_parser.register(RPTraitNWUniform)
rpt_parser.register(RPTraitNWHarmonic)
rpt_parser.register(RPTraitNWDistortion)

# Density traits (height)
rht_parser.register(RHTraitUniform)
rht_parser.register(RHTraitExponential)
rht_parser.register(RHTraitGauss)
rht_parser.register(RHTraitGGauss)
rht_parser.register(RHTraitLorentz)
rht_parser.register(RHTraitSech2)

# Velocity traits (polar)
vpt_parser.register(VPTraitTanUniform)
vpt_parser.register(VPTraitTanArctan)
vpt_parser.register(VPTraitTanBoissier)
vpt_parser.register(VPTraitTanEpinat)
vpt_parser.register(VPTraitTanLRamp)
vpt_parser.register(VPTraitTanTanh)
vpt_parser.register(VPTraitNWTanUniform)
vpt_parser.register(VPTraitNWTanHarmonic)
vpt_parser.register(VPTraitNWRadUniform)
vpt_parser.register(VPTraitNWRadHarmonic)
vpt_parser.register(VPTraitNWVerUniform)
vpt_parser.register(VPTraitNWVerHarmonic)
vpt_parser.register(VPTraitNWLOSUniform)
vpt_parser.register(VPTraitNWLOSHarmonic)

# Velocity traits (height)
vht_parser.register(VHTraitOne)

# Dispersion traits (polar)
dpt_parser.register(DPTraitUniform)
dpt_parser.register(DPTraitExponential)
dpt_parser.register(DPTraitGauss)
dpt_parser.register(DPTraitGGauss)
dpt_parser.register(DPTraitLorentz)
dpt_parser.register(DPTraitMoffat)
dpt_parser.register(DPTraitSech2)
dpt_parser.register(DPTraitMixtureGGauss)
dpt_parser.register(DPTraitMixtureMoffat)
dpt_parser.register(DPTraitNWUniform)
dpt_parser.register(DPTraitNWHarmonic)
dpt_parser.register(DPTraitNWDistortion)

# Dispersion traits (height)
dht_parser.register(DHTraitOne)

# Warp traits (polar)
wpt_parser.register(WPTraitNWUniform)
wpt_parser.register(WPTraitNWHarmonic)

# Selection traits (polar)
spt_parser.register(SPTraitAzimuthalRange)
spt_parser.register(SPTraitNWAzimuthalRange)
