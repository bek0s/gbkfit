"""
The wrappers defined in this file are coded against pygmo 2.19.0
"""

import copy

import numpy as np
import pygmo as pg

from gbkfit.fitting.core import fitter_parser
from gbkfit.utils import iterutils, parseutils
from .core import FitterPygmo


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
    'FitterPygmoMOEAD',
    'FitterPygmoGMOEAD',
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
    for key in ['__class__', 'self', 'size', 'seed', 'verbosity']:
        locals_.pop(key)
    return locals_


def _locals_to_options_meta(locals_):
    locals_ = copy.deepcopy(locals_)
    for key in ['__class__', 'self', 'algo', 'seed', 'verbosity']:
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
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.gaco(**self.options())


class FitterPygmoDE(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.de'

    def __init__(
            self, size, gen, f=0.8, cr=0.9, variant=2, ftol=1e-6, xtol=1e-6,
            seed=0, verbosity=0):
        options = _locals_to_options(locals())
        options = iterutils.rename_key(options, 'f', 'F')
        options = iterutils.rename_key(options, 'cr', 'CR')
        super().__init__(size, seed, verbosity, **options)

    def create_algorithm(self, parameters):
        return pg.de(**self.options())


class FitterPygmoSADE(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sade'

    def __init__(
            self, size, gen, variant=2, variant_adptv=1, ftol=1e-6, xtol=1e-6,
            memory=False, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.sade(**self.options())


class FitterPygmoDE1220(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.de1220'

    def __init__(
            self, size, gen, allowed_variants=(2, 3, 7, 10, 13, 14, 15, 16),
            variant_adptv=1, ftol=1e-6, xtol=1e-6, memory=False, seed=0,
            verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.de1220(**self.options())


class FitterPygmoGWO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.gwo'

    def __init__(self, size, gen, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.gwo(**self.options())


class FitterPygmoIHS(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.ihs'

    def __init__(
            self, size, gen, phmcr=0.85, ppar_min=0.35, ppar_max=0.99,
            bw_min=1e-5, bw_max=1.0, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.ihs(**self.options())


class FitterPygmoPSO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.pso'

    def __init__(
            self, size, gen, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5,
            variant=5, neighb_type=2, neighb_param=4, memory=False, seed=0,
            verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.pso(**self.options())


class FitterPygmoGPSO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.pso_gen'

    def __init__(
            self, size, gen, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5,
            variant=5, neighb_type=2, neighb_param=4, memory=False, seed=0,
            verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.pso_gen(**self.options())


class FitterPygmoSEA(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sea'

    def __init__(self, size, gen, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.sea(**self.options())


class FitterPygmoSGA(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sga'

    def __init__(
            self, size, gen, cr=0.9, eta_c=1.0, m=0.02, param_m=1.0, param_s=2,
            crossover='exponential', mutation='polynomial',
            selection='tournament', seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.sga(**self.options())


class FitterPygmoSA(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sa'

    def __init__(
            self, size, ts=10.0, tf=0.1, n_t_adj=10, n_range_adj=10,
            bin_size=10, start_range=1.0, seed=0, verbosity=0):
        options = _locals_to_options(locals())
        options = iterutils.rename_key(options, 'ts', 'Ts')
        options = iterutils.rename_key(options, 'tf', 'Tf')
        options = iterutils.rename_key(options, 'n_t_adj', 'n_T_adj')
        super().__init__(size, seed, verbosity, **options)

    def create_algorithm(self, parameters):
        return pg.simulated_annealing(**self.options())


class FitterPygmoABC(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.abc'

    def __init__(self, size, gen, limit=1, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.bee_colony(**self.options())


class FitterPygmoCMAES(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.cmaes'

    def __init__(
            self, size, gen, cc=-1, cs=-1, c1=-1, cmu=-1, sigma0=0.5, ftol=1e-6,
            xtol=1e-6, memory=False, force_bounds=False, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.cmaes(**self.options())


class FitterPygmoXNES(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.xnes'

    def __init__(
            self, size, gen, eta_mu=-1, eta_sigma=-1, eta_b=-1, sigma0=-1,
            ftol=1e-6, xtol=1e-6, memory=False, force_bounds=False, seed=0,
            verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.xnes(**self.options())


class FitterPygmoNSGA2(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.nsga2'

    def __init__(
            self, size, gen, cr=0.95, eta_c=10.0, m=0.01, eta_m=50.0,
            seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def is_multi_objective(self):
        return True

    def create_algorithm(self, parameters):
        return pg.nsga2(**self.options())


class FitterPygmoMOEAD(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.moead'

    def __init__(
            self, size, gen, weight_generation='grid',
            decomposition='tchebycheff', neighbours=20, cr=1, f=0.5, eta_m=20,
            realb=0.9, limit=2, preserve_diversity=True, seed=0, verbosity=0):
        options = _locals_to_options(locals())
        options = iterutils.rename_key(options, 'cr', 'CR')
        options = iterutils.rename_key(options, 'f', 'F')
        super().__init__(size, seed, verbosity, **options)

    def is_multi_objective(self):
        return True

    def create_algorithm(self, parameters):
        return pg.moead(**self.options())


class FitterPygmoGMOEAD(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.moead_gen'

    def __init__(
            self, size, gen, weight_generation='grid',
            decomposition='tchebycheff', neighbours=20, cr=1, f=0.5, eta_m=20,
            realb=0.9, limit=2, preserve_diversity=True, seed=0, verbosity=0):
        options = _locals_to_options(locals())
        options = iterutils.rename_key(options, 'cr', 'CR')
        options = iterutils.rename_key(options, 'f', 'F')
        super().__init__(size, seed, verbosity, **options)

    def is_multi_objective(self):
        return True

    def create_algorithm(self, parameters):
        return pg.moead_gen(**self.options())


class FitterPygmoMACO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.maco'

    def __init__(
            self, size, gen, ker=63, q=1.0, threshold=1, n_gen_mark=7,
            evalstop=100000, focus=0.0, memory=False, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def is_multi_objective(self):
        return True

    def create_algorithm(self, parameters):
        return pg.maco(**self.options())


class FitterPygmoNSPSO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.nspso'

    def __init__(
            self, size, gen, omega=0.6, c1=0.01, c2=0.5, chi=0.5, v_coeff=0.5,
            leader_selection_range=2, diversity_mechanism='crowding distance',
            memory=False, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def is_multi_objective(self):
        return True

    def create_algorithm(self, parameters):
        return pg.nspso(**self.options())


class FitterPygmoCompassSearch(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.compass_search'

    def __init__(
            self, size, max_fevals=1, start_range=0.1, stop_range=0.01,
            reduction_coeff=0.5, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.compass_search(**self.options())


class FitterPygmoNLopt(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.nlopt'

    def __init__(
            self, solver, size, ftol_abs=0, ftol_rel=0, maxeval=0, maxtime=0,
            replacement='best', selection='worst', sr_seed=0,
            stopval=-np.inf, xtol_abs=0, xtol_rel=1e-08, local_optimizer=None,
            seed=0, verbosity=0):
        options = _locals_to_options(locals())
        auglag_solvers = ['auglag', 'auglag_eq']
        if local_optimizer in auglag_solvers:
            raise RuntimeError(
                "the local_optimizer cannot be set to 'auglag' or 'auglag_eq'")
        if solver in auglag_solvers and not local_optimizer:
            raise RuntimeError(
                f"when solver is '{solver}', "
                f"the local_optimizer must be set")
        if solver not in auglag_solvers and local_optimizer:
            raise RuntimeError(
                f"when solver is '{solver}', "
                f"the local_optimizer must not be set")
        super().__init__(size, seed, verbosity, **options)

    def create_algorithm(self, parameters):
        # Get a copy to allow modifications without touching anything
        options = copy.deepcopy(self.options())
        # Create a dictionary with all the options that need to be
        # set as pygmo algorithm properties. These options will be
        # removed from the 'options' dictionary.
        properties_list = [
            'ftol_abs', 'ftol_rel', 'maxeval', 'maxtime', 'replacement',
            'selection', 'sr_seed', 'stopval', 'xtol_abs', 'xtol_rel']
        properties_dict = {}
        for prop in properties_list:
            properties_dict[prop] = options.pop(prop)
        # Figure out which and how algorithm properties should be set
        properties_main = {}
        if local_optimizer := options.pop('local_optimizer', None):
            algo = pg.nlopt(local_optimizer)
            properties_local = properties_dict
            self._set_algorithm_properties(algo, properties_local)
            properties_main['local_optimizer'] = algo
        else:
            properties_main = properties_dict
        # Create the main pygmo algorithm
        algo = pg.nlopt(**options)
        # Set main algorithm properties
        self._set_algorithm_properties(algo, properties_main)
        return algo

    @staticmethod
    def _set_algorithm_properties(algo, properties):
        for key, val in properties.items():
            if key == 'sr_seed':
                algo.set_random_sr_seed(val)
            else:
                setattr(algo, key, val)


class FitterPygmoScipy(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.scipy'

    def __init__(
            self, method, size, tol=None, options=None, seed=0, verbosity=0):
        # Argument 'selection' is not used. At the time of writing,
        # only one selection policy was available from Python anyway.
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        return pg.scipy_optimize(**self.options())


class FitterPygmoIpopt(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.ipopt'

    def __init__(
            self, size, tol=1e-4, max_iter=3000, max_wall_time=3600,
            seed=0, verbosity=0):
        # todo: support more options (ipopt has A LOT of them)
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))

    def create_algorithm(self, parameters):
        # Get a copy to allow modifications without touching anything
        options = copy.deepcopy(self.options())
        algo = pg.ipopt()
        properties_integer = ['max_iter']
        properties_numeric = ['tol', 'max_wall_time']
        properties_string = []
        for prop in properties_integer:
            algo.set_integer_option(prop, options.pop(prop))
        for prop in properties_numeric:
            algo.set_numeric_option(prop, options.pop(prop))
        for prop in properties_string:
            algo.set_string_option(prop, options.pop(prop))
        return algo


class FitterPygmoSNOPT(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.snopt'

    def __init__(self, size, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))
        raise NotImplementedError()

    def create_algorithm(self, parameters):
        raise NotImplementedError()


class FitterPygmoWORHP(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.worhp'

    def __init__(self, size, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, **_locals_to_options(locals()))
        raise NotImplementedError()

    def create_algorithm(self, parameters):
        raise NotImplementedError()


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

    def dump(self):
        return dict()

    def __init__(self, algo, stop=5, perturb=0.01, seed=0, verbosity=0):
        super().__init__(
            algo.size(), seed, verbosity, **_locals_to_options_meta(locals()))
        self._algo = algo

    def is_multi_objective(self):
        return self._algo.is_multi_objective()

    def create_algorithm(self, parameters):
        wrapped_algo = self._algo.create_algorithm(parameters)
        options_with_wrapped_algo = self.options() | dict(algo=wrapped_algo)
        return pg.mbh(**options_with_wrapped_algo)
