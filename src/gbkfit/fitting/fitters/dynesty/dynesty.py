
import abc
import logging

import dynesty
import numpy as np

from gbkfit.fitting.core import FitParam, FitParams, Fitter


from gbkfit.utils import iterutils, parseutils

from gbkfit.fitting.prior import prior_parser

from gbkfit.fitting.utils import *



_log = logging.getLogger(__name__)


def _prior_tansform_wrapper(theta):
    pass


def _log_likelihood_wrapper(theta):
    pass


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
        return dict()

    def __init__(self, prior):
        super().__init__()

    def prior(self):
        return None


class FitParamsDynesty(FitParams):

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
        return FitParamDynesty.load(info)

    def dump(self):
        return ()

    def __init__(
            self, descs, parameters,
            value_conversions=None, prior_conversions=None):
        super().__init__(descs, parameters, None, FitParamDynesty)


class FitterDynesty(Fitter):

    @staticmethod
    def load_params(info, descs):
        return FitParamsDynesty.load(info, descs)

    def __init__(self):
        super().__init__()

    def _fit_impl(self, objective, parameters, interpreter):
        result1 = self._fit_impl_impl(objective, parameters, interpreter)
        result2 = result1
        return result2

    @abc.abstractmethod
    def _fit_impl_impl(self, objective, parameters, interpreter):
        pass


class FitterDynestySNS(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.sns'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                rstate='seed'))
        if 'rstate' in opts:
            opts['rstate'] = np.random.RandomState(opts['rstate'])
        return cls(**opts)

    def dump(self):
        return {'type': self.type(), **self._props}

    def __init__(
            self,
            # see dynesty.NestedSampler()
            nlive=500,
            bound='multi', sample='auto',
            update_interval=None, first_update=None, rstate=None,
            enlarge=None, bootstrap=0, vol_dec=0.5, vol_check=2.0,
            walks=25, facc=0.5, slices=5, fmove=0.9, max_move=100,
            # see dynesty.sampler.Sampler.run_nested()
            maxiter=None, maxcall=None, dlogz=None,
            logl_max=np.inf, n_effective=None,
            add_live=True, print_progress=True,
            print_func=None, save_bounds=True):

        super().__init__()
        self._props = locals()
        self._props.pop('self')
        self._props.pop('__class__')

    def _fit_impl_impl(self, objective, parameters, interpreter):

        ndim = 3

        sampler = dynesty.NestedSampler(
            _log_likelihood_wrapper, _prior_tansform_wrapper, ndim,
            logl_args=(objective, interpreter),
            ptform_args=(),
            **self._props)



        result = sampler.run_nested()


        pass



class FitterDynestyDNS(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.dns'

    @classmethod
    def load(cls, info):
        info['rstate'] = np.random.RandomState(info.get('seed'))
        cls_args = parseutils.parse_options(info, 'foo', fun=cls.__init__)
        return cls(**cls_args)

    def dump(self):
        return {'type': self.type(), **self._props}

    def __init__(
            self,
            # see dynesty.DynamicNestedSampler()
            bound='multi', sample='auto',
            update_interval=None, first_update=None, rstate=None,
            enlarge=None, bootstrap=0,
            walks=25, facc=0.5, slices=None, fmove=0.9, max_move=100,
            # dynesty.dynamicsampler.DynamicSampler.run_nested()
            nlive_init=500, maxiter_init=None,
            maxcall_init=None, dlogz_init=0.01, logl_max_init=np.inf,
            n_effective_init=np.inf, nlive_batch=500,
            wt_function=None, wt_kwargs=None,
            maxiter_batch=None, maxcall_batch=None,
            maxiter=None, maxcall=None, maxbatch=None,
            n_effective=None,
            stop_function=None, stop_kwargs=None, use_stop=True,
            save_bounds=True, print_progress=True, print_func=None):
        super().__init__()
        self._props = locals()
        self._props.pop('self')
        self._props.pop('__class__')

    def _fit_impl(self, objective, param_info, param_interp, **kwargs):
        pass
