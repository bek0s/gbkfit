
import collections.abc
import copy
import logging

import emcee
import numpy as np
import numpy.random as random

from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter
from gbkfit.fitting.prior import prior_parser
from gbkfit.fitting.result import make_fitter_result
from gbkfit.params import paramutils
from gbkfit.utils import iterutils, parseutils

from .moves import *


__all__ = [
    'FitParamEmcee',
    'FitParamsEmcee',
    'FitterEmcee'
]


_log = logging.getLogger(__name__)


def _log_probability_wrapper(eparams, objective, parameters):
    params = parameters.evaluate(eparams)
    log_prior = 1 #parameters.priors()
    if np.isinf(log_prior):
        return -np.inf, [np.nan, np.nan]
    else:
        log_like = objective.log_likelihood(params)[0]
        return log_like + log_prior, [log_like, log_prior]


class FitParamEmcee(FitParam):

    @classmethod
    def load(cls, info):
        info['prior'] = fitutils.prepare_param_prior_info(info)
        info['prior'] = prior_parser.load(info['prior'], param_info=info)
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='value',
                initial_width='width'))
        return cls(**opts)

    def dump(self):
        return dict(
            value=self.initial_value(),
            width=self.initial_width(),
            prior=prior_parser.dump(self.prior()))

    def __init__(self, prior, initial_value=None, initial_width=None):
        super().__init__()
        initial_value, initial_value_min, initial_value_max = \
            fitutils.prepare_param_initial_value_range_from_value_width_min_max(
                initial_value, initial_width, prior.minimum, prior.maximum)
        self._prior = copy.deepcopy(prior)
        self._initial_value = initial_value
        self._initial_width = initial_width
        self._initial_value_minimum = initial_value_min
        self._initial_value_maximum = initial_value_max

    def prior(self):
        return self._prior

    def initial_value(self):
        return self._initial_value

    def initial_width(self):
        return self._initial_width

    def initial_value_minimum(self):
        return self._initial_value_minimum

    def initial_value_maximum(self):
        return self._initial_value_maximum


class FitParamsEmcee(FitParams):

    @staticmethod
    def load_param(info):
        return FitParamEmcee.load(info)

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
            self, FitParamEmcee, lambda x: x.dump(), conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        super().__init__(
            pdescs, parameters, conversions, FitParamEmcee)


class FitterEmcee(Fitter):

    @staticmethod
    def type():
        return 'emcee'

    @staticmethod
    def load_params(info, descs):
        return FitParamsEmcee.load(info, descs)

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(moves=load_moves_with_weights(opts.get('moves')))
        return cls(**opts)

    def dump(self):
        info = dict(type=self.type())
        options = dict(
            nwalkers=self._nwalkers,
            nsteps=self._nsteps,
            thin_by=self._thin_by,
            tune=self._tune,
            seed=self._seed,
            moves=dump_moves_with_weights(self._moves))
        parseutils.prepare_for_dump(
            options,
            remove_nones=True,
            remove_keys=())
        return info | options

    def __init__(
            self, nwalkers, nsteps, thin_by=1,
            moves=None, tune=False, seed=None):
        super().__init__()
        moves = [m if iterutils.is_sequence(m) else (m, 1.0) for m in moves]
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        self._thin_by = thin_by
        self._moves = moves
        self._tune = tune
        self._seed = seed

    def _fit_impl(self, objective, parameters):

        ndim = len(parameters.infos())
        # Build moves list
        moves = [(m.obj(), w) for m, w in self._moves] if self._moves else None
        # Create sampler
        sampler = emcee.EnsembleSampler(
            self._nwalkers, ndim, log_prob_fn=_log_probability_wrapper,
            moves=moves, args=[objective, parameters],
            parameter_names=parameters.enames(False, False, True))
        # Calculate the starting positions of all walkers
        initial_values = np.empty((self._nwalkers, ndim))
        for i, (pname, pinfo) in enumerate(parameters.infos().items()):
            initial_values[:, i] = pinfo.initial_value() + random.uniform(
                pinfo.initial_value_minimum(),
                pinfo.initial_value_maximum(),
                self._nwalkers)
        # Run mcmc sampling
        result = sampler.run_mcmc(
            initial_values, nsteps=self._nsteps, tune=self._tune,
            thin_by=self._thin_by, progress=True)

        print("initial values:", initial_values)
        # print("result:", result)

        samples = sampler.get_chain(discard=100, thin=1, flat=True)
        print(samples)

        exit()
