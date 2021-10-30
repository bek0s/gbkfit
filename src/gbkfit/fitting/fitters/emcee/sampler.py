
import copy
import logging

import emcee
import numpy as np
import numpy.random as random

from gbkfit.utils import funcutils, iterutils, parseutils
import gbkfit.params.utils as paramutils
import gbkfit.params.utils

from gbkfit.fitting.prior import prior_parser

from gbkfit.fitting.core import FitParam, FitParams, Fitter

from . moves import move_parser


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


def ensure_prior(info):
    if 'prior' not in info:
        if 'min' not in info or 'max' not in info:
            raise RuntimeError(
                f"no 'prior' found in the parameter description; "
                f"an attempt to set a default uniform prior failed because "
                f"'min' and/or 'max' are missing")
        info['prior'] = dict(
            type='uniform', min=info.pop('min'), max=info.pop('max'))
    return info


class FitParamEmcee(FitParam):

    @classmethod
    def load(cls, info):
        info = ensure_prior(info)
        info['prior'] = prior_parser.load_one(info['prior'], param_info=info)
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='val',
                initial_width='width'))
        return cls(**opts)

    def dump(self):
        return dict(
            prior=prior_parser.dump(self.prior()),
            initial_value=self.initial_value(),
            initial_width=self.initial_width())

    def __init__(self, prior, initial_value=None, initial_width=None):
        super().__init__()
        self._prior = prior
        self._initial_value = initial_value
        self._initial_width = initial_width

    def prior(self):
        return self._prior

    def initial_value(self):
        return self._initial_value

    def initial_width(self):
        return self._initial_width


class FitParamsEmcee(FitParams):

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
        super().__init__(descs, parameters, None, FitParamEmcee)


class FitterEmcee(Fitter):

    @staticmethod
    def type():
        return 'emcee'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        if moves := opts.get('moves'):
            moves = iterutils.tuplify(moves)
            weights = [m.pop('weight', 1.0) for m in moves]
            moves = move_parser.load(moves)
            opts.update(moves=tuple(zip(moves, weights)))
        return cls(**opts)

    def dump(self):
        return tuple()

    @staticmethod
    def load_params(info, descs):
        return FitParamsEmcee.load(info, descs)

    def __init__(
            self, nwalkers, nsteps, thin_by=1,
            moves=None, tune=False, seed=None):
        super().__init__()
        moves = iterutils.tuplify(moves, False)
        for i, move in enumerate(moves):
            if not iterutils.is_sequence(move):
                moves[i] = (move, 1.0)
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

        initial_vals = []
        for pname, pinfo in parameters.items():
            val = pinfo.initial_val()
            std = pinfo.initial_std()
            val = val + std * random.randn(self._nwalkers)
            initial_vals.append(val)

        initial_vals = np.array(initial_vals).transpose()

        result = sampler.run_mcmc(
            initial_vals,
            nsteps=self._nsteps, tune=self._tune, thin_by=self._thin_by,
            progress=True)

        exit()

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



