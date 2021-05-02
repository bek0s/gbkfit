
import copy
import logging

import emcee
import numpy as np
import numpy.random as random

from gbkfit.utils import funcutils, iterutils, parseutils
import gbkfit.params.utils as paramutils
import gbkfit.fitting.fitter
import gbkfit.fitting.params
import gbkfit.params.utils

from gbkfit.fitting.prior import prior_parser

log = logging.getLogger(__name__)


def _log_probability_wrapper(evalues, objective, parameters, interpreter):
    enames = interpreter.get_param_names()
    eparams = dict(zip(enames, evalues))
    params = interpreter.evaluate(eparams)
    log_prior = 0
    if np.isinf(log_prior):
        return -np.inf, [np.nan, np.nan]
    else:
        log_like = objective.log_likelihood(params)
        return log_like + log_prior, [log_like, log_prior]


def foo(info, info_name, cls_name):
    prior_info = info['prior']
    prior_type = info['prior']['type']
    prior_cls = prior_parser.parsers().get(prior_type)
    if prior_cls:
        prior_cls_args = funcutils.extract_args(prior_cls.__init__)[0]
        if cls_name in prior_cls_args:
            if info_name in info and info_name not in prior_info:
                prior_info[info_name] = info[info_name]
            info.pop(info_name, None)


def ensure_prior(info):
    info['prior'] = info.get('prior', dict(type='uniform'))
    foo(info, 'min', 'minimum')
    foo(info, 'max', 'maximum')


class FitParamEmcee(gbkfit.fitting.params.FitParam):

    @classmethod
    def load(cls, info):
        info['prior'] = info.get('prior', dict(type='uniform'))
        info['prior'] = prior_parser.load_one(info['prior'], param_info=info)
        desc = ''
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_val='val',
                initial_std='std'))
        return cls(**opts)

    def dump(self):
        return dict(
            val=self.initial_val(),
            std=self.initial_std())

    def __init__(self, prior, initial_val, initial_std=None):
        super().__init__()
        self._prior = prior
        self._initial_val = initial_val
        self._initial_std = initial_std

    def prior(self):
        return self._prior

    def initial_val(self):
        return self._initial_val

    def initial_std(self):
        return self._initial_std


class FitParamsEmcee(gbkfit.fitting.params.FitParams):

    @classmethod
    def load(cls, info, descs):
        desc = 'fit params'
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        infos, exprs = paramutils.parse_param_info(
            opts['parameters'], descs)[4:]
        for k, v in infos.items():
            infos[k] = FitParamEmcee.load(v)
        return cls(descs, infos | exprs)

    def dump(self):
        pass

    def __init__(self, descs, parameters):
        super().__init__(descs, parameters, None)


class FitterEmcee(gbkfit.fitting.fitter.Fitter):

    @staticmethod
    def type():
        return 'emcee'

    @classmethod
    def load(cls, info):
        from . import moves
        desc = ''
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts_moves = opts.get('moves')
        if opts_moves:
            weights = [m.pop('weight') for m in opts_moves if 'weight' in m]
            moves = moves.parser.load(opts_moves)
            if len(moves) != len(weights) > 0:
                raise RuntimeError()
            opts.update(moves=tuple(zip(moves, weights)) if weights else moves)
        return cls(**opts)

    def dump(self):
        return tuple()

    @staticmethod
    def load_params(info, descs):
        return FitParamsEmcee.load(info, descs)

    def __init__(
            self, nwalkers, nsteps, nsteps_burnin=0, thin_by=1,
            moves=None, tune=False, seed=None):
        super().__init__()
        if moves is None:
            moves = tuple()
        for i, move in enumerate(moves):
            if not iterutils.is_sequence(move):
                moves[i] = (move, 1.0)
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        self._thin_by = thin_by
        self._moves = moves
        self._tune = tune
        self._seed = seed

        self._kwargs_sampler = ()

    def _fit_impl(self, objective, parameters):

        ndim = len(parameters.infos())
        minimums = ndim * [-np.inf]
        maximums = ndim * [+np.inf]
        initials = np.empty((ndim, 10))
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

        exit()


        pinfos = parameters.infos()

        ndim = len(pinfos)

        initial_vals = []

        # Build initial value arrays
        #
        for pname, pinfo in pinfos.items():
            val = pinfo.initial_val()

            # If an initial value is defined, use that and
            if val is not None:
                std = pinfo.initial_std()
                std = 1 if std is None else std
                val = std * random.randn(self._nwalkers)
            else:
                val = 0

            initial_vals.append(val)

        initial_vals = np.array(initial_vals).transpose()

        sampler = emcee.EnsembleSampler(
            self._nwalkers, ndim,
            log_prob_fn=_log_probability_wrapper,
            args=[objective, parameters, interpreter],
            moves=[m.move_obj() for m in self._moves] if self._moves else None)

        result = sampler.run_mcmc(
            initial_vals,
            nsteps=self._nsteps, tune=self._tune, thin_by=self._thin_by, progress=True)

        samples = sampler.get_chain(discard=100, thin=2, flat=True)
        print(samples)

        import corner
        import matplotlib.pyplot as plt
        plt.hist(samples)
        #fig = corner.corner(samples)
        plt.show()


