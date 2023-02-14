
import collections.abc
import copy
import logging

from collections.abc import Sequence
from typing import Tuple

import emcee
import numpy as np
import numpy.random as random

from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter
from gbkfit.fitting.prior import prior_parser, Prior, PriorDict, PriorUniform
from gbkfit.fitting.result import make_fitter_result
from gbkfit.params import parsers as param_parsers, utils as paramutils
from gbkfit.utils import iterutils, parseutils

from .moves import *


__all__ = [
    'FitParamEmcee',
    'FitParamsEmcee',
    'FitterEmcee'
]


_log = logging.getLogger(__name__)


def log_probability(eparams, parameters, objective):

    fmin = np.finfo(np.float32).min
    fmax = np.finfo(np.float32).max

    if eparams['xpos'] >= fmax:
        eparams['xpos'] = fmax - 1

    if eparams['xpos'] <= fmin:
        eparams['xpos'] = fmin + 1

    eparams['xpos'] = np.float32(eparams['xpos'])
    # print("func:", eparams['xpos'])
    params = parameters.evaluate(eparams)
    log_prior = parameters.priors().log_prob(eparams)
    # print( parameters.priors()['xpos'].maximum)
    if np.isinf(log_prior):
        return -np.inf, [np.nan, np.nan]
    else:

        log_like = sum(objective.log_likelihood(params))
        foo = log_like + log_prior, log_like, log_prior
        # print(params, foo)
        return foo


class FitParamEmcee(FitParam):

    @classmethod
    def load(cls, info):
        info['prior'] = fitutils.prepare_param_info_prior(info)
        info['prior'] = prior_parser.load_one(info['prior'], param_info=info)
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='value',
                initial_width='width'))
        return cls(**opts)

    def dump(self):
        info = dict(
            value=self.initial_value(),
            width=self.initial_width(),
            prior=prior_parser.dump(self.prior()))
        return iterutils.remove_from_mapping_by_value(info, None)

    def __init__(
            self,
            prior: Prior,
            initial_value: int | float | None = None,
            initial_width: int | float | None = None
    ):
        super().__init__()
        initial_value, initial_width = \
            fitutils.prepare_param_initial_value_width(
                initial_value, initial_width)
        self._prior = copy.deepcopy(prior)
        self._initial_value = initial_value
        self._initial_width = initial_width

    def prior(self):
        return self._prior

    def has_initial(self):
        return (self.initial_value() is not None and
                self.initial_width() is not None)

    def initial_value(self):
        return self._initial_value

    def initial_width(self):
        return self._initial_width


class FitParamsEmcee(FitParams):

    @staticmethod
    def load_param(info):
        return FitParamEmcee.load(info)

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
            self, FitParamEmcee, lambda x: x.dump(), conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        super().__init__(
            pdescs, parameters, conversions, FitParamEmcee)
        self._priors = PriorDict(
            {k: v.prior() for k, v in self.infos().items()})

    def priors(self):
        return self._priors


class FitterEmcee(Fitter):

    @staticmethod
    def type():
        return 'emcee'

    @staticmethod
    def load_params(info, descs):
        return FitParamsEmcee.load(info, descs)

    @classmethod
    def load(cls, info):
        info['moves'] = load_moves_with_weights(info.get('moves'))
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        info = dict(
            nwalkers=self._nwalkers,
            nsteps=self._nsteps,
            thin_by=self._thin_by,
            tune=self._tune,
            seed=self._seed,
            moves=dump_moves_with_weights(self._moves))
        info = iterutils.remove_from_mapping_by_value(info, None)
        return info

    def __init__(
            self,
            nwalkers: int,
            nsteps: int,
            thin_by: int = 1,
            moves:
            FitterEmceeMove |
            Sequence[FitterEmceeMove] |
            Sequence[Tuple[FitterEmceeMove, int | float]] | None = None,
            tune: bool = False,
            seed: int = 0
    ):
        super().__init__()
        moves = [m if iterutils.is_sequence(m) else (m, 1.0) for m in moves]
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        self._thin_by = thin_by
        self._moves = moves
        self._tune = tune
        self._seed = seed

    def _fit_impl(self, objective, parameters):

        nwalkers = self._nwalkers
        seed = self._seed
        rng = np.random.default_rng(self._seed)

        ndim = len(parameters.infos())
        # Build moves list
        moves = [(m.obj(), w) for m, w in self._moves] if self._moves else None
        # Create sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_fn=log_probability,
            moves=moves, args=[parameters, objective],
            parameter_names=parameters.enames(False, False, True),
            blobs_dtype=[('log_like', float), ('log_prior', float)])
            # blobs_dtype=[('log_like', float), ('log_prior', float)], backend=emcee.backends.HDFBackend('foo.h5'))

        # Calculate the starting positions of all walkers
        initial_values = np.empty((nwalkers, ndim))
        for i, (pname, pinfo) in enumerate(parameters.infos().items()):
            # Sample initial positions from a prior.
            # If an initial value/width is available,
            # use a uniform prior. Otherwise, use parameter's prior.
            prior = pinfo.prior()
            if pinfo.has_initial():
                value = pinfo.initial_value()
                width = pinfo.initial_width()
                prior = pinfo.prior()
                minimum = max(prior.minimum, value - width / 2)
                maximum = min(prior.maximum, value + width / 2)
                prior = PriorUniform(minimum, maximum)
            initial_values[:, i] = prior.sample(nwalkers)

        print(initial_values)
        # exit()

        # Run mcmc sampling
        sampler.run_mcmc(
            initial_values, nsteps=self._nsteps, tune=self._tune,
            thin_by=self._thin_by, progress=True)

        print(sampler.acceptance_fraction)
        print(sampler.get_autocorr_time())


        samples = sampler.get_chain(discard=0, thin=1, flat=True)
        log_like = sampler.get_blobs(flat=True)['log_like']
        log_prior = sampler.get_blobs(flat=True)['log_prior']

        print(samples)
        print(log_like)
        print(log_prior)

        # exit()

        posterior = dict(
            samples=samples,
            loglikes=log_like,
            logprior=log_prior
        )

        result = make_fitter_result(
            objective, parameters, posterior, solutions=())

        return result
