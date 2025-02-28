import abc
import copy
import logging

import optuna

from .core import FitterOptuna
from gbkfit.utils import funcutils, iterutils


_log = logging.getLogger(__name__)


def _locals_to_sampler_options(locals_):
    locals_ = copy.deepcopy(locals_)
    for key in ('__class__', 'self', 'n_startup_trials', 'n_ei_candidates'):
        locals_.pop(key)
    return locals_


class FittingParamsProperty:
    pass


class FittingParams:
    pass


class FittingParamsPropertyOptuna(FittingParamsProperty):

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return dict()

    def __init__(self, minimum, maximum):
        pass


class FittingParamsOptuna(FittingParams):
    pass


class FitterOptunaGrid(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.grid'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            seed: int | None = None
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.GridSampler(**self.sampler_args())


class FitterOptunaRandom(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.random'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            seed: int | None = None
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.RandomSampler(**self.sampler_args())


class FitterOptunaTpe(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.tpe'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            n_startup_trials: int = 10,
            n_ei_candidates: int = 24,
            seed: int | None = None
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.TPESampler(**self.sampler_args())


class FitterOptunaCmaes(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.cmaes'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            seed: int | None = None,
            n_startup_trials: int = 1
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.CmaEsSampler(**self.sampler_args())


class FitterOptunaGp(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.gp'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            seed: int | None = None,
            n_startup_trials: int = 1
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.GPSampler(**self.sampler_args())


class FitterOptunaNsga2(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.nsga2'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            seed: int | None = None,
            n_startup_trials: int = 1
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.NSGAIISampler(**self.sampler_args())


class FitterOptunaNsga3(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.nsga3'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            population_size: int = 50,
            seed: int | None = None
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.NSGAIIISampler(**self.sampler_args())


class FitterOptunaQmc(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.qmc'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            # Defaults for now
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.QMCSampler(**self.sampler_args())


class FitterOptunaBruteForce(FitterOptuna, abc.ABC):

    @staticmethod
    def type():
        return 'optuna.bruteforce'

    def __init__(
            self,
            # Study.optimize() arguments
            n_trials: int | None = None,
            timeout: float | None = None,
            gc_after_trial: bool = False,
            # Sampler arguments
            seed: int | None = None
    ):
        super().__init__(
            n_trials, timeout, gc_after_trial,
            _locals_to_sampler_options(locals()))

    def create_sampler(self, params):
        return optuna.samplers.BruteForceSampler(**self.sampler_args())
