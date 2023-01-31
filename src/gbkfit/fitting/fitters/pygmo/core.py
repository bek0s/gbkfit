
import abc
import collections.abc
import logging

import numpy as np
import pygmo as pg

from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter
from gbkfit.params import parsers as param_parsers
from gbkfit.utils import iterutils, parseutils

from .problem import Problem

__all__ = [
    'FitParamPygmo',
    'FitParamsPygmo',
    'FitterPygmo'
]


_log = logging.getLogger(__name__)


class FitParamPygmo(FitParam):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='value',
                initial_width='width',
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        info = dict(
            value=self.initial_value(),
            width=self.initial_width(),
            min=self.minimum(),
            max=self.maximum())
        info = iterutils.remove_from_mapping_by_value(info, None)
        return info

    def __init__(
            self, minimum, maximum, initial_value=None, initial_width=None):
        super().__init__()
        initial_value, initial_width, initial_value_min, initial_value_max = \
            fitutils.prepare_optional_initial_value_min_max(
                initial_value, initial_width, minimum, maximum)
        self._initial_value = initial_value
        self._initial_width = initial_width
        self._initial_value_minimum = initial_value_min
        self._initial_value_maximum = initial_value_max
        self._minimum = minimum
        self._maximum = maximum

    def initial_value(self):
        return self._initial_value

    def initial_width(self):
        return self._initial_width

    def initial_value_minimum(self):
        return self._initial_value_minimum

    def initial_value_maximum(self):
        return self._initial_value_maximum

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum


class FitParamsPygmo(FitParams):

    @staticmethod
    def load_param(info):
        return FitParamPygmo.load(info)

    @classmethod
    def load(cls, info, pdescs):
        desc = parseutils.make_basic_desc(cls, 'fit params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        opts = param_parsers.load_params_parameters_conversions(
            opts, pdescs, collections.abc.Mapping, cls.load_param)
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return param_parsers.dump_params_parameters_conversions(
            self, FitParamPygmo, lambda x: x.dump(), conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        super().__init__(pdescs, parameters, conversions, FitParamPygmo)


class FitterPygmo(Fitter, abc.ABC):

    def load_params(self, info, descs):
        return FitParamsPygmo.load(info, descs)

    @classmethod
    def load(cls, info, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'pygmo fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict()

    def __init__(self, size, seed, verbosity, **kwargs):
        super().__init__(**kwargs)
        self._size = size
        self._seed = seed
        self._verbosity = verbosity
        self._rnd = np.random.default_rng(seed)
        self._options = kwargs

    def size(self):
        return self._size

    def options(self):
        return self._options

    def is_multi_objective(self):
        return False

    @abc.abstractmethod
    def create_algorithm(self, parameters):
        pass

    def _fit_impl(self, objective, parameters):

        ndim = len(parameters.infos())
        size = self._size
        seed = self._seed
        is_multi_objective = self.is_multi_objective()

        minimums = ndim * [-np.inf]
        maximums = ndim * [+np.inf]
        initial_values = np.empty((ndim, size))

        for i, (pname, pinfo) in enumerate(parameters.infos().items()):
            minimums[i] = pinfo.minimum()
            maximums[i] = pinfo.maximum()
            initial_values[i, :] = self._rnd.uniform(
                pinfo.initial_value_minimum(),
                pinfo.initial_value_maximum(),
                size)

        prb = pg.problem(Problem(objective, parameters, minimums, maximums, is_multi_objective))
        pop = pg.population(prb, size=size, seed=seed)

        for i in range(size):
            pop.set_x(i, initial_values[:, i])

        alg = pg.algorithm(self.create_algorithm(parameters))


        if alg.has_set_seed():
            alg.set_seed(self._seed)
        if alg.has_set_verbosity():
            alg.set_verbosity(self._verbosity)


        pop = alg.evolve(pop)
        exit()
        print("----- RESULTS -----")
        print(pg.sort_population_mo(pop.get_f()))
        print("get_f():", pop.get_f().tolist())
        print("get_x():", pop.get_x().tolist())
        print(pop.champion_x)
        print(pop.champion_f)
        print("BYE")
        exit()
