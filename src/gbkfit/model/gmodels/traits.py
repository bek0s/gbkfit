
import abc
import copy

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
RP_TRAIT_UID_MIXTURE_EXPONENTIAL = 51
RP_TRAIT_UID_MIXTURE_GAUSS = 52
RP_TRAIT_UID_MIXTURE_GGAUSS = 53
RP_TRAIT_UID_MIXTURE_MOFFAT = 54
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
DP_TRAIT_UID_MIXTURE_EXPONENTIAL = 51
DP_TRAIT_UID_MIXTURE_GAUSS = 52
DP_TRAIT_UID_MIXTURE_GGAUSS = 53
DP_TRAIT_UID_MIXTURE_MOFFAT = 54
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
WP_TRAIT_UID_AXIS_RANGE = 1


TRUNC_DEFAULT = 0


def _params_mixture(n, include_shape):
    result = [
        ParamVectorDesc('r', n),  # polar coord (radius)
        ParamVectorDesc('t', n),  # polar coord (angle)
        ParamVectorDesc('a', n),  # amplitude
        ParamVectorDesc('s', n),  # size
        ParamVectorDesc('b', n),  # shape
        ParamVectorDesc('q', n),  # axis ratio (minor/major)
        ParamVectorDesc('p', n)]  # position angle relative to t
    if not include_shape:
        del result[4]
    return tuple(result)


def _params_pw_harmonic(order, nnodes, nwmode):
    params = []
    params += [(ParamVectorDesc('a', nnodes), nwmode)]
    params += [(ParamVectorDesc('p', nnodes), nwmode)] * (order > 0)
    return tuple(params)


def _integrate_rings(nodes, fun, *args):
    # Assume the nodes are equally spaced
    rsep = nodes[1] - nodes[0]
    # Minimum radius of the rings
    rmin = nodes - rsep * 0.5
    # Maximum radius of the rings
    rmax = nodes + rsep * 0.5
    # Calculate the amplitude of each ring
    ampl = fun(nodes, *args)
    return ampl * np.pi * (rmax * rmax - rmin * rmin)


def trait_desc(cls):
    descs = {
        RPTrait: 'density polar trait',
        RHTrait: 'density height trait',
        VPTrait: 'velocity polar trait',
        VHTrait: 'velocity height trait',
        DPTrait: 'velocity dispersion polar trait',
        DHTrait: 'velocity dispersion height trait',
        ZPTrait: 'vertical distortion polar trait',
        SPTrait: 'selection polar trait',
        WPTrait: 'weight polar trait'}
    label = None
    for k, v in descs.items():
        if issubclass(cls, k):
            label = v
    assert label
    return parseutils.make_typed_desc(cls, label)


def _dump_nwmode(nwmode):
    return dict(nwmode=nwmode) if nwmode is not None else dict()


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

    def __init__(self, **kwargs):
        self._kwargs = copy.deepcopy(kwargs)

    def dump(self):
        return dict(**self._kwargs)

    def consts(self):
        return tuple(self._kwargs.values())

    def params_sm(self):
        return tuple()

    def params_rnw(self, nnodes):
        return tuple()


class RPTrait(Trait, abc.ABC):

    @abc.abstractmethod
    def has_analytical_integral(self):
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


class ZPTrait(Trait, abc.ABC):
    pass


class SPTrait(Trait, abc.ABC):
    pass


class WPTrait(Trait, abc.ABC):
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

    def has_analytical_integral(self):
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

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        return _integrate_rings(nodes, gbkfit.math.expon_1d_fun, a, 0, s)


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

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        return _integrate_rings(nodes, gbkfit.math.gauss_1d_fun, a, 0, s)


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

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        b = params['b']
        return _integrate_rings(nodes, gbkfit.math.ggauss_1d_fun, a, 0, s, b)


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

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        return _integrate_rings(nodes, gbkfit.math.lorentz_1d_fun, a, 0, s)


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

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        b = params['b']
        return _integrate_rings(nodes, gbkfit.math.moffat_1d_fun, a, 0, s, b)


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

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        s = params['s']
        return _integrate_rings(nodes, gbkfit.math.sech2_1d_fun, a, 0, s)


class RPTraitMixtureExponential(RPTrait):

    @staticmethod
    def type():
        return 'mixture_exponential'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_MIXTURE_EXPONENTIAL

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], False)

    def has_analytical_integral(self):
        return True

    def integrate(self, params, nodes):
        raise NotImplementedError()


class RPTraitMixtureGauss(RPTrait):

    @staticmethod
    def type():
        return 'mixture_gauss'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], False)

    def has_analytical_integral(self):
        return True

    def integrate(self, params, nodes):
        raise NotImplementedError()


class RPTraitMixtureGGauss(RPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], True)

    def has_analytical_integral(self):
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
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], True)

    def has_analytical_integral(self):
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

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self._nwmode),)

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        c = np.inf
        return _integrate_rings(nodes, gbkfit.math.uniform_1d_fun, a, 0, c)


class RPTraitNWHarmonic(RPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_NW_HARMONIC

    def __init__(self, order, nwmode=None):
        super().__init__(order=order)
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes, self._nwmode)

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        a = params['a']
        c = np.inf
        return _integrate_rings(nodes, gbkfit.math.uniform_1d_fun, a, 0, c)


class RPTraitNWDistortion(RPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return RP_TRAIT_UID_NW_DISTORTION

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), None),
            (ParamVectorDesc('p', nnodes), None),
            (ParamVectorDesc('s', nnodes), None))

    def has_analytical_integral(self):
        return False

    def integrate(self, params, nodes):
        raise NotImplementedError()


class _RHTraitSM(RHTrait, abc.ABC):

    def __init__(self, consts, params):
        super().__init__(**consts)
        self._params = tuple(params)

    def params_sm(self):
        if not self._kwargs['rnodes']:
            return tuple([ParamScalarDesc(p) for p in self._params])
        else:
            return tuple()

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

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vt', nnodes), self._nwmode),)


class VPTraitNWTanHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_tan_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_TAN_HARMONIC

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, order, nwmode=None):
        super().__init__(order=order)
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes, self._nwmode)


class VPTraitNWRadUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_RAD_UNIFORM

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vr', nnodes), self._nwmode),)


class VPTraitNWRadHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_rad_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_RAD_HARMONIC

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, order, nwmode=None):
        super().__init__(order=order)
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes, self._nwmode)


class VPTraitNWVerUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_VER_UNIFORM

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vv', nnodes), self._nwmode),)


class VPTraitNWVerHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_ver_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_VER_HARMONIC

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, order, nwmode=None):
        super().__init__(order=order)
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes, self._nwmode)


class VPTraitNWLOSUniform(VPTrait):

    @staticmethod
    def type():
        return 'nw_los_uniform'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_LOS_UNIFORM

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('vl', nnodes), self._nwmode),)


class VPTraitNWLOSHarmonic(VPTrait):

    @staticmethod
    def type():
        return 'nw_los_harmonic'

    @staticmethod
    def uid():
        return VP_TRAIT_UID_NW_LOS_HARMONIC

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, order, nwmode=None):
        super().__init__(order=order)
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes, self._nwmode)


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


class DPTraitMixtureExponential(DPTrait):

    @staticmethod
    def type():
        return 'mixture_exponential'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_EXPONENTIAL

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], False)


class DPTraitMixtureGauss(DPTrait):

    @staticmethod
    def type():
        return 'mixture_gauss'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_GAUSS

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], False)


class DPTraitMixtureGGauss(DPTrait):

    @staticmethod
    def type():
        return 'mixture_ggauss'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_GGAUSS

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], True)


class DPTraitMixtureMoffat(DPTrait):

    @staticmethod
    def type():
        return 'mixture_moffat'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_MIXTURE_MOFFAT

    def __init__(self, nblobs):
        super().__init__(nblobs=nblobs)

    def params_sm(self):
        return _params_mixture(self._kwargs['nblobs'], True)


class DPTraitNWUniform(DPTrait):

    @staticmethod
    def type():
        return 'nw_uniform'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_UNIFORM

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self._nwmode),)


class DPTraitNWHarmonic(DPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_HARMONIC

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, order, nwmode=None):
        super().__init__(order=order)
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes, self._nwmode)


class DPTraitNWDistortion(DPTrait):

    @staticmethod
    def type():
        return 'nw_distortion'

    @staticmethod
    def uid():
        return DP_TRAIT_UID_NW_DISTORTION

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self._nwmode),
            (ParamVectorDesc('p', nnodes), self._nwmode),
            (ParamVectorDesc('s', nnodes), self._nwmode))


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

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return (
            (ParamVectorDesc('a', nnodes), self._nwmode),)


class ZPTraitNWHarmonic(ZPTrait):

    @staticmethod
    def type():
        return 'nw_harmonic'

    @staticmethod
    def uid():
        return ZP_TRAIT_UID_NW_HARMONIC

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, order, nwmode=None):
        super().__init__(order=order)
        self._nwmode = nwmode

    def params_rnw(self, nnodes):
        return _params_pw_harmonic(self._kwargs['order'], nnodes, self._nwmode)


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

    def dump(self):
        return super().dump() | _dump_nwmode(self._nwmode)

    def __init__(self, nwmode=None):
        super().__init__()
        self._nwmode = nwmode

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
        if weight < 0:
            raise RuntimeError(
                f"invalid weight value; "
                f"weight must be a positive number or zero; "
                f"supplied value: {weight}")
        super().__init__(axis=axis, angle=angle, weight=weight)


# Density traits (polar) parser
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
