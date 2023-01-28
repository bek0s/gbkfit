
import copy

import pygmo as pg

from gbkfit.utils import parseutils

from .core import FitterPygmo
from gbkfit.fitting.core import fitter_parser


__all__ = [
    'FitterPygmoGACO',
    'FitterPygmoDE',
    'FitterPygmoSADE',
    'FitterPygmoDE1220',
    'FitterPygmoGWO',
    'FitterPygmoIHS',
    'FitterPygmoPSO',
    'FitterPygmoGPSO',
    'FitterPygmoSEA',
    'FitterPygmoSGA',
    'FitterPygmoSA',
    'FitterPygmoABC',
    'FitterPygmoCMAES',
    'FitterPygmoXNES',
    'FitterPygmoNSGA2',
    'FitterPygmoMOAED',
    'FitterPygmoGMOAED',
    'FitterPygmoMACO',
    'FitterPygmoNSPSO',
    'FitterPygmoCompassSearch',
    'FitterPygmoNLopt',
    'FitterPygmoScipy',
    'FitterPygmoIpopt',
    'FitterPygmoSNOPT',
    'FitterPygmoWORHP',
    'FitterPygmoMBH'
]


def _locals_to_options(locals_):
    locals_ = copy.deepcopy(locals_)
    for key in ['__class__', 'self', 'size']:
        locals_.pop(key)
    return locals_


class FitterPygmoGACO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.gaco'

    def __init__(
            self, size, gen, ker=63, q=1.0, oracle=0.0, acc=0.01, threshold=1,
            n_gen_mark=7, impstop=100000, evalstop=100000, focus=0,
            memory=False, seed=0, verbosity=0):
        super().__init__(size)

    def _setup_algorithm(self, options, parameters):
        return pg.gaco(**options)


class FitterPygmoDE(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.de'

    def __init__(
            self, gen, f=0.8, cr=0.9, variant=2, ftol=1e-6, xtol=1e-6, seed=0,
            verbosity=0):
        super().__init__()


class FitterPygmoSADE(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sade'

    def __init__(
            self, gen, variant=2, variant_adptv=1, ftol=1e-6, xtol=1e-6,
            memory=False, seed=1):
        super().__init__()


class FitterPygmoDE1220(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.de1220'

    def __init__(
            self, gen, allowed_variants=(2, 3, 7, 10, 13, 14, 15, 16),
            variant_adptv=1, ftol=1e-6, xtol=1e-6, memory=False, seed=1):
        super().__init__()


class FitterPygmoGWO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.gwo'

    def __init__(self, gen, seed=1):
        super().__init__()


class FitterPygmoIHS(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.ihs'

    def __init__(
            self, gen, phmcr=0.85, ppar_min=0.35, ppar_max=0.99,
            bw_min=1e-5, bw_max=1.0, seed=1):
        super().__init__()


class FitterPygmoPSO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.pso'

    def __init__(
            self, gen, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5,
            variant=5, neighb_type=2, neighb_param=4, memory=False, seed=1):
        super().__init__()


class FitterPygmoGPSO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.pso_gen'

    def __init__(
            self, gen, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5,
            variant=5, neighb_type=2, neighb_param=4, memory = False, seed=1):
        super().__init__()


class FitterPygmoSEA(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sea'

    def __init__(self, gen, seed=1):
        super().__init__()


class FitterPygmoSGA(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sga'

    def __init__(
            self, gen, cr=0.9, eta_c=1.0, m=0.02, param_m=1.0, param_s=2,
            crossover='exponential', mutation='polynomial',
            selection='tournament', seed=1):
        super().__init__()


class FitterPygmoSA(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sa'

    def __init__(
            self, ts=10.0, tf=0.1, n_t_adj=10, n_range_adj=10, bin_size=10,
            start_range=1.0, seed=1):
        super().__init__()


class FitterPygmoABC(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.abc'

    def __init__(self, gen, limit=1, seed=1):
        super().__init__()


class FitterPygmoCMAES(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.cmaes'

    def __init__(
            self, gen, size, cc=-1, cs=-1, c1=-1, cmu=-1, sigma0=0.5, ftol=1e-6,
            xtol=1e-6, memory=False, force_bounds=False, seed=0, verbosity=0):
        foo = pygmo.algorithm(pygmo.cmaes())
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))


class FitterPygmoXNES(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.xnes'

    def __init__(
            self, gen, eta_mu=-1, eta_sigma=-1, eta_b=-1, sigma0=-1,
            ftol=1e-6, xtol=1e-6, memory=False, force_bounds=False, seed=1):
        super().__init__()


class FitterPygmoNSGA2(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.nsga2'

    def __init__(self, gen, cr=0.95, eta_c=10.0, m=0.01, eta_m=50.0, seed=1):
        super().__init__()


class FitterPygmoMOAED(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.moaed'

    def __init__(
            self, gen, weight_generation='grid', decomposition='tchebycheff',
            neighbours=20, cr=1, f=0.5, eta_m=20, realb=0.9, limit=2,
            preserve_diversity=True, seed=1):
        super().__init__()


class FitterPygmoGMOAED(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.moaed_gen'

    def __init__(
            self, gen, weight_generation='grid', decomposition='tchebycheff',
            neighbours=20, cr=1, f=0.5, eta_m=20, realb=0.9, limit=2,
            preserve_diversity=True, seed=1):
        super().__init__()


class FitterPygmoMACO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.maco'

    def __init__(
            self, gen, ker=63, q=1.0, threshold=1, n_gen_mark=7,
            evalstop=100000, focus=0.0, memory=False, seed=1):
        super().__init__()


class FitterPygmoNSPSO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.nspso'

    def __init__(
            self, gen, omega=0.6, c1=0.01, c2=0.5, chi=0.5, v_coeff=0.5,
            leader_selection_range=2, diversity_mechanism='crowding distance',
            memory=False, seed=1):
        super().__init__()


class FitterPygmoCompassSearch(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.compass_search'

    def __init__(
            self, max_fevals=1, start_range=0.1, stop_range=0.01,
            reduction_coeff=0.5):
        pass


class FitterPygmoNLopt(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.nlopt'

    def __init__(self, gen, size):
        pass


class FitterPygmoScipy(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.scipy'

    def __init__(self):
        pass


class FitterPygmoIpopt(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.ipopt'

    def __init__(self):
        pass


class FitterPygmoSNOPT(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.snopt'

    def __init__(self):
        pass


class FitterPygmoWORHP(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.worhp'

    def __init__(self):
        pass


class FitterPygmoMBH(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.mbh'

    @classmethod
    def load(cls, info, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'pygmo fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts['algo'] = fitter_parser.load(opts['algo'])
        return cls(**opts)

    def __init__(self, algo, stop=5, perturb=0.01, seed=0, verbosity=0):
        super().__init__()
