
import logging

import emcee
import numpy as np
import numpy.random as random

from gbkfit.fitting.prior import prior_parser
from gbkfit.fitting.core import FitParam, FitParams, Fitter

from gbkfit.fitting.utils import *

from gbkfit.params import utils as paramutils
from gbkfit.utils import iterutils, parseutils

from . moves import *


__all__ = [
    'FitParamEmcee',
    'FitParamsEmcee',
    'FitterEmcee']


_log = logging.getLogger(__name__)


# TODO Param and Val conversions
# only param dependencies?
class ParamPriorConversions:
    pass


def _log_probability_wrapper(evalues, objective, parameters):
    enames = parameters.interpreter().enames(free=True, tied=False, fixed=False)
    eparams = dict(zip(enames, evalues))
    params = parameters.interpreter().evaluate(eparams)
    log_prior = 0
    if np.isinf(log_prior):
        return -np.inf, [np.nan, np.nan]
    else:
        log_like = objective.log_likelihood(params)[0]
        return log_like + log_prior, [log_like, log_prior]


class FitParamEmcee(FitParam):

    @classmethod
    def load(cls, info):
        info['prior'] = prepare_param_prior_info(info)
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
            prepare_param_initial_value_and_range_from_value_width_min_max(
                initial_value, initial_width, prior.minimum, prior.maximum)
        self._prior = prior
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

    @classmethod
    def load(cls, info, descs):
        desc = parseutils.make_basic_desc(cls, 'fit params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        parameters = load_parameters(
            opts.get('parameters'), descs, cls.load_param)
        value_conversions = paramutils.load_parameter_value_conversions(
            opts.get('value_conversions'))
        prior_conversions = paramutils.load_parameter_prior_conversions(
            opts.get('prior_conversions'))
        return cls(descs, parameters, value_conversions, prior_conversions)

    @staticmethod
    def load_param(info):
        return FitParamEmcee.load(info)

    def dump(self):
        return dict()

    def __init__(
            self, descs, parameters,
            value_conversions=None, prior_conversions=None):
        super().__init__(descs, parameters, None, FitParamEmcee)


class FitterEmcee(Fitter):

    @staticmethod
    def type():
        return 'emcee'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(moves=load_moves_with_weights(opts.get('moves')))
        return cls(**opts)

    @staticmethod
    def load_params(info, descs):
        return FitParamsEmcee.load(info, descs)

    def dump(self):
        return dict(
            nwalkers=self._nwalkers,
            nsteps=self._nsteps,
            thin_by=self._thin_by,
            moves=dump_moves_with_weights(self._moves),
            tune=self._tune,
            seed=self._seed)

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
        moves = [(m.obj(), w) for m, w in self._moves] if self._moves else None

        sampler = emcee.EnsembleSampler(
            self._nwalkers, ndim, log_prob_fn=_log_probability_wrapper,
            moves=moves, args=[objective, parameters])

        initial_values = np.empty((self._nwalkers, ndim))
        for i, (pname, pinfo) in enumerate(parameters.infos().items()):
            initial_values[:, i] = pinfo.initial_value() + random.uniform(
                pinfo.initial_value_minimum(),
                pinfo.initial_value_maximum(),
                self._nwalkers)

        result = sampler.run_mcmc(
            initial_values,
            nsteps=self._nsteps, tune=self._tune, thin_by=self._thin_by,
            progress=True)

        print(initial_values)

        samples = sampler.get_chain(discard=100, thin=2, flat=True)
        print(samples)

        exit()
