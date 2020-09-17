
import abc
import copy

import numpy as np
import numpy.random as random
import pygmo as pg

from gbkfit.fitting.fitter import Fitter
from .params import FitParamsPygmo
from .problem import Problem

from gbkfit.fitting.result import FitterResultSolution, FitterResult, make_fitter_result

__all__ = [
    'FitterPygmo', 'FitterPygmoDE', 'FitterPygmoSADE', 'FitterPygmoDE1220',
    'FitterPygmoPSO', 'FitterPygmoCMAES', 'FitterPygmoXNES', 'FitterPygmoIpopt',
    'FitterPygmoNLopt', 'FitterPygmoScipy']


class FitterPygmo(Fitter, abc.ABC):

    @staticmethod
    def load_params(info, descs):
        return FitParamsPygmo.load(info, descs)

    def dump(self):
        alg_attrs = copy.deepcopy(self._alg_attrs)
        alg_attrs.pop('seed', None)
        return dict(
            **alg_attrs,
            size=self._size, seed=self._seed, verbosity=self._verbosity)

    def __init__(self, size, seed, verbosity, alg_attrs):
        super().__init__()
        self._size = size
        self._seed = seed
        self._verbosity = verbosity
        self._alg_attrs = alg_attrs

    def _fit_impl(self, objective, parameters):
        ndim = len(parameters.infos())
        minimums = ndim * [-np.inf]
        maximums = ndim * [+np.inf]
        initials = np.empty((ndim, self._size))
        for i, pinfo in enumerate(parameters.infos().values()):
            minimum = pinfo.minimum()
            maximum = pinfo.maximum()
            value = pinfo.initial_value()
            scale = pinfo.initial_scale()
            has_init = pinfo.has_initial()
            init_min = value - 0.5 * scale if not has_init else minimum
            init_max = value + 0.5 * scale if not has_init else maximum
            init_min = max(init_min, minimum)
            init_max = min(init_max, maximum)
            initials[i, :] = random.uniform(init_min, init_max, self._size)
            minimums[i] = minimum
            maximums[i] = maximum
        prb = pg.problem(Problem(objective, parameters, minimums, maximums))
        alg = pg.algorithm(self._setup_algorithm(self._alg_attrs, parameters))
        alg.set_verbosity(self._verbosity)
        pop = pg.population(prb, size=self._size, seed=self._seed)
        for i in range(self._size):
            pop.set_x(i, initials[:, i])
        pop = alg.evolve(pop)

        solutions = [FitterResultSolution(mode=pop.champion_x)]

        result = make_fitter_result(objective, parameters, solutions=solutions)

        print("------")
        exit()

        result = FitterResult()
        params_free = parameters.names()

        print("------")

        exit()


        datasets = objective.datasets()
        params_free = parameters.expressions().names(params=True, consts=False)
        params_fixed = parameters.expressions().names(params=False, consts=True)
        result = FitterResult(objective.datasets(), params_free, params_fixed)

        mode = pop.champion_x
        eparams_dict = dict(zip(params_free, mode))
        params_dict = parameters.expressions().evaluate(eparams_dict)
        models = objective.model().evaluate_h(params_dict)



        result.add_solution(mode=pop.champion_x, models=models)

        exit()
        return result

    @abc.abstractmethod
    def _setup_algorithm(self, attributes, parameters):
        pass





    pass


class FitterPygmoDE(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.de'

    def __init__(
            self, gen, size, F=0.8, CR=0.9, variant=2, ftol=1e-06, xtol=1e-06,
            seed=0, verbosity=0):
        super().__init__(
            size, seed, verbosity, dict(
                gen=gen, F=F, CR=CR, variant=variant, ftol=ftol, xtol=xtol,
                seed=seed))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.de(**attributes)
        return alg


class FitterPygmoSADE(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.sade'

    def __init__(
            self, gen, size, variant=2, variant_adptv=1, ftol=1e-06, xtol=1e-06,
            memory=False, seed=0, verbosity=0):
        super().__init__(
            size, seed, verbosity, dict(
                gen=gen, variant=variant, variant_adptv=variant_adptv,
                ftol=ftol, xtol=xtol, memory=memory, seed=seed))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.sade(**attributes)
        return alg


class FitterPygmoDE1220(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.de1220'

    def __init__(
            self, gen, size, allowed_variants=(2, 3, 7, 10, 13, 14, 15, 16),
            variant_adptv=1, ftol=1e-06, xtol=1e-06, memory=False, seed=0,
            verbosity=0):
        super().__init__(
            size, seed, verbosity, dict(
                gen=gen, allowed_variants=allowed_variants,
                variant_adptv=variant_adptv, ftol=ftol, xtol=xtol,
                memory=memory, seed=seed))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.de1220(**attributes)
        return alg


class FitterPygmoPSO(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.pso'

    def __init__(
            self, gen, size, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5,
            variant=5, neighb_type=2, neighb_param=4, memory=False, seed=0,
            verbosity=0):
        super().__init__(
            size, seed, verbosity, dict(
                gen=gen, omega=omega, eta1=eta1, eta2=eta2, max_vel=max_vel,
                variant=variant, neighb_type=neighb_type,
                neighb_param=neighb_param, memory=memory, seed=seed))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.pso(**attributes)
        return alg


class FitterPygmoCMAES(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.cmaes'

    def __init__(
            self, gen, size, cc=-1, cs=-1, c1=-1, cmu=-1, sigma0=0.5,
            ftol=1e-06, xtol=1e-06, memory=False, force_bounds=False, seed=0,
            verbosity=0):
        super().__init__(
            size, seed, verbosity, dict(
                gen=gen, cc=cc, cs=cs, c1=c1, cmu=cmu, sigma0=sigma0,
                ftol=ftol, xtol=xtol, memory=memory, force_bounds=force_bounds,
                seed=seed))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.cmaes(**attributes)
        return alg


class FitterPygmoXNES(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.xnes'

    def __init__(
            self, gen, size, eta_mu=-1, eta_sigma=-1, eta_b=-1, sigma0=-1,
            ftol=1e-06, xtol=1e-06, memory=False, force_bounds=False, seed=0,
            verbosity=0):
        super().__init__(
            size, seed, verbosity, dict(
                gen, eta_mu=eta_mu, eta_sigma=eta_sigma, eta_b=eta_b,
                sigma0=sigma0, ftol=ftol, xtol=xtol, memory=memory,
                force_bounds=force_bounds, seed=seed))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.xnes(**attributes)
        return alg


class FitterPygmoIpopt(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.ipopt'

    def __init__(self, size, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, dict())

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.ipopt()
        return alg


class FitterPygmoNLopt(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.nlopt'

    def __init__(self, method, size, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, dict(method=method))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.nlopt(attributes['method'])
        return alg


class FitterPygmoScipy(FitterPygmo):

    @staticmethod
    def type():
        return 'pygmo.scipy'

    def __init__(self, method, size, seed=0, verbosity=0):
        super().__init__(size, seed, verbosity, dict(method=method))

    def _setup_algorithm(self, attributes, parameters):
        alg = pg.scipy_optimize(attributes['method'])
        return alg
