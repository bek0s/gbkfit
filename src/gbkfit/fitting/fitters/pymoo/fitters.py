
import copy

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.pso import PSO

from . import FitterPymoo


def _locals_to_options(locals_):
    locals_ = copy.deepcopy(locals_)
    for key in ['__class__', 'self', 'termination', 'seed', 'verbose']:
        locals_.pop(key)
    return locals_


class FitterPymooPSO(FitterPymoo):

    @staticmethod
    def type():
        return 'pymoo.pso'

    def __init__(
            self, pop_size, w=0.9, c1=2.0, c2=2.0, adaptive=True,
            initial_velocity='random', max_velocity_rate=0.2,
            pertube_best=True, termination=None, seed=None, verbose=False):
        super().__init__(
            PSO,
            _locals_to_options(locals()),
            termination, seed, verbose)

    def _setup_options(self, options_init, options_setup):
        return options_init, options_setup


class FitterPymooNSGA2(FitterPymoo):

    @staticmethod
    def type():
        return 'pymoo.nsga2'

    def __init__(
            self, pop_size, selection=None, crossover=None, mutation=None,
            termination=None, seed=None, verbose=False):

        options = dict(
            pop_size=pop_size,
            selection=selection,
            crossover=crossover,
            mutation=mutation)
        super().__init__(NSGA2, termination, seed, verbose, options)

