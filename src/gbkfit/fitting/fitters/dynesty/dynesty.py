
import abc
import copy
import logging

import dynesty
import dynesty.dynamicsampler
import numpy as np

from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter
from gbkfit.fitting.prior import prior_parser
from gbkfit.fitting.result import make_fitter_result
from gbkfit.params import paramutils
from gbkfit.utils import funcutils, iterutils, parseutils


from gbkfit.fitting.prior.prior_dict import PriorDict


_log = logging.getLogger(__name__)


__all__ = []


def _prior_transform_wrapper(theta, parameters):
    return parameters.priors().rescale(
        dict(zip(parameters.enames(fixed=False, tied=False, free=True), theta)))


def _log_likelihood_wrapper(theta, parameters, objective):

    eparams = dict(zip(parameters.enames(fixed=False, tied=False, free=True), theta))

    print(eparams)
    params = parameters.evaluate(eparams, check=False)

    return objective.log_likelihood(params)
    # import numpy.random
    # return numpy.random.uniform(0, 10.0)


class FitParamDynesty(FitParam):

    @classmethod
    def load(cls, info):
        info['prior'] = prepare_param_prior_info(info)
        info['prior'] = prior_parser.load(info['prior'], param_info=info)
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(
            prior=prior_parser.dump(self.prior()))

    def __init__(self, prior):
        super().__init__()
        self._prior = prior

    def prior(self):
        return self._prior


class FitParamsDynesty(FitParams):

    @classmethod
    def load(cls, info, descs):
        desc = parseutils.make_basic_desc(cls, 'fit params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        parameters = load_parameters(
            opts.get('parameters'), descs, cls.load_param)
        value_conversions = paramparsers.load_params_value_conversions(
            opts.get('value_conversions'))
        return cls(descs, parameters, value_conversions)

    @staticmethod
    def load_param(info):
        return FitParamDynesty.load(info)

    def dump(self):
        info = dict()
        return info

    def __init__(self, descs, parameters, value_conversions=None):
        super().__init__(descs, parameters, value_conversions, FitParamDynesty)

        self._priors = PriorDict(
            {k: v.prior() for k, v in self.infos().items()}, descs)

        # exit(priors)

    def priors(self):
        return self._priors




class FitterDynesty(Fitter, abc.ABC):

    @staticmethod
    def load_params(info, descs):
        return FitParamsDynesty.load(info, descs)

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return self._options_constructor | self._options_run_nested

    def __init__(self, options_constructor, options_run_nested):
        super().__init__()
        self._options_constructor = options_constructor
        self._options_run_nested = options_run_nested

    def _fit_impl(self, objective, parameters):
        result1 = self._fit_impl_impl(objective, parameters)
        result2 = result1
        return result2

    @abc.abstractmethod
    def _fit_impl_impl(self, objective, parameters):
        pass


class FitterDynestySNS(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.sns'

    def __init__(
            self,
            # dynesty.NestedSampler()
            nlive=500,
            bound='multi',
            sample='auto',
            update_interval=None,
            first_update=None,
            rstate=None,
            enlarge=None,
            bootstrap=None,
            walks=25,
            facc=0.5,
            slices=5,
            fmove=0.9,
            max_move=100,
            # dynesty.sampler.Sampler.run_nested()
            maxiter=None,
            maxcall=None,
            dlogz=None,
            logl_max=np.inf,
            n_effective=None,
            add_live=True,
            print_progress=True,
            print_func=None,
            save_bounds=True):
        # Extract dynesty.NestedSampler() arguments
        args_factory = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.DynamicNestedSampler)[0])
        # Extract dynesty.sampler.Sampler.run_nested() arguments
        args_run_nested = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.dynamicsampler.DynamicSampler.run_nested)[0])
        super().__init__(args_factory, args_run_nested)

    def _fit_impl_impl(self, objective, parameters):

        ndim = len(parameters.infos())

        sampler = dynesty.NestedSampler(
            _log_likelihood_wrapper, _prior_transform_wrapper, ndim,
            logl_args=(parameters, objective),
            ptform_args=(parameters,),
            **self._options_constructor)

        result = sampler.run_nested(**self._options_run_nested)

        return result


class FitterDynestyDNS(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.dns'

    def __init__(
            self,
            # dynesty.DynamicNestedSampler() arguments
            bound='multi',
            sample='auto',
            update_interval=None,
            first_update=None,
            rstate=None,
            enlarge=None,
            bootstrap=0,
            walks=25,
            facc=0.5,
            slices=None,
            fmove=0.9,
            max_move=100,
            # dynesty.dynamicsampler.DynamicSampler.run_nested() arguments
            nlive_init=500,
            maxiter_init=None,
            maxcall_init=None,
            dlogz_init=0.01,
            logl_max_init=np.inf,
            n_effective_init=np.inf,
            nlive_batch=500,
            wt_function=None,
            wt_kwargs=None,
            maxiter_batch=None,
            maxcall_batch=None,
            maxiter=None,
            maxcall=None,
            maxbatch=None,
            n_effective=None,
            stop_function=None,
            stop_kwargs=None,
            use_stop=True,
            save_bounds=True,
            print_progress=True,
            print_func=None):
        # Extract dynesty.DynamicNestedSampler() arguments
        args_factory = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.DynamicNestedSampler)[0])
        # Extract dynesty.dynamicsampler.DynamicSampler.run_nested() arguments
        args_run_nested = iterutils.extract_subdict(
            locals(), funcutils.extract_args(
                dynesty.dynamicsampler.DynamicSampler.run_nested)[0])
        super().__init__(args_factory, args_run_nested)

    def _fit_impl_impl(self, objective, parameters):
        ndim = 3

        sampler = dynesty.DynamicNestedSampler(
            _log_likelihood_wrapper, _prior_tansform_wrapper, ndim,
            logl_args=(objective, parameters),
            ptform_args=(),
            **self._options_constructor)

        print(self._options_run_nested)
        result = sampler.run_nested(**self._options_run_nested)

        return result
