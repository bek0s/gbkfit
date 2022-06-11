import abc
import collections.abc
import copy
import logging

import pymoo

import numpy as np

from pymoo.optimize import minimize
from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter
from gbkfit.params import paramutils
from gbkfit.utils import iterutils, parseutils
from .problem import PymooProblem

from gbkfit.fitting.result import make_fitter_result


from . import pymooutils


class FitParamPymoo(FitParam, abc.ABC):

    @classmethod
    def load(cls, info, **kwargs):
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='value',
                initial_width='width',
                minimum='min',
                maximum='max'))
        return cls(**(opts | kwargs))

    def dump(self):
        return dict(
            value=self.initial_value(),
            width=self.initial_width(),
            minimum=self.minimum(),
            maximum=self.maximum())

    def __init__(
            self, initial_value=None, initial_width=None,
            minimum=None, maximum=None):
        super().__init__()
        initial_value, initial_value_min, initial_value_max = \
            fitutils.prepare_param_initial_value_range_from_value_width_min_max(
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


class FitParamsPymoo(FitParams):

    @staticmethod
    def load_param(info):
        return FitParamPymoo.load(info)

    @classmethod
    def load(cls, info, pdescs):
        desc = parseutils.make_basic_desc(cls, 'fit params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        opts = paramutils.load_params_parameters_conversions(
            opts, pdescs, collections.abc.Mapping, cls.load_param)
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return paramutils.dump_params_parameters_conversions(
            self, FitParamPymoo, lambda x: x.dump(), conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        super().__init__(
            pdescs, parameters, conversions, FitParamPymoo)


def _locals_to_options(locals_):
    locals_ = copy.deepcopy(locals_)
    locals_.pop('self')
    locals_.pop('__class__')
    options_setup = dict(
        termination=locals_.pop('termination'),
        seed=locals_.pop('seed'),
        verbose=locals_.pop('verbose'))
    options_init = dict(locals_)
    return options_init, options_setup


class FitterPymoo(Fitter):

    @staticmethod
    def type():
        return 'pymoo'

    @staticmethod
    def load_params(info, pdescs):
        return FitParamsPymoo.load(info, pdescs)

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        if 'crossover' in opts:
            opts['crossover'] = pymooutils.crossover_parser.load(
                opts['crossover'])
        if 'mutation' in opts:
            opts['mutation'] = pymooutils.mutation_parser.load(
                opts['mutation'])
        if 'selection' in opts:
            opts['selection'] = pymooutils.selection_parser.load(
                opts['selection'])
        if 'termination' in opts:
            opts['termination'] = pymooutils.termination_parser.load(
                opts['termination'])
        return cls(**opts)

    def dump(self):
        return dict()

    def __init__(
            self, algorithm_type, algorithm_kwargs,
            termination, seed, verbose):
        super().__init__()
        if seed is None:
            seed = 42
        self._algorithm_type = algorithm_type
        self._algorithm_options_init = algorithm_kwargs
        self._algorithm_options_setup = dict(
            termination=termination, seed=seed, verbose=verbose)

    def _fit_impl(self, objective, parameters):

        # Setup options
        algorithm_options_init, algorithm_options_setup = self._setup_options(
            copy.deepcopy(self._algorithm_options_init),
            copy.deepcopy(self._algorithm_options_setup))

        # Initialize algorithm
        algorithm = self._algorithm_type(**algorithm_options_init)

        # Setup problem
        problem = PymooProblem(objective, parameters)

        # Run optimization
        res = pymoo.optimize.minimize(
            problem, algorithm, **algorithm_options_setup)

        #
        solution = dict(mode=list(res.X))
        extra = dict(f=res.F)

        result = make_fitter_result(
            objective, parameters, extra=extra, solutions=solution)

        return result

    @abc.abstractmethod
    def _setup_options(self, options_init, options_setup):
        pass



