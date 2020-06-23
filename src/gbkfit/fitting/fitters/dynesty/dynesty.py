
import abc
import logging

import dynesty
import numpy as np

import gbkfit.fitting.fitter
import gbkfit.fitting.params
from gbkfit.utils import parseutils


log = logging.getLogger(__name__)


class FitParamDynesty(gbkfit.fitting.params.FitParam):

    @classmethod
    def load(cls, info):
        return cls(**info)

    def __init__(self, prior, periodic=None, reflective=None):
        pass


class FitParamsDynesty(gbkfit.fitting.params.FitParams):

    @classmethod
    def load(cls, info, descs):
        return cls(info, descs)

    def __init__(self, params, descs, live_points=None):
        super().__init__(params, descs)


class FitterDynesty(gbkfit.fitting.fitter.Fitter):

    def __init__(self):
        super().__init__()

    def _impl_fit(self, data, model, params):
        return self._impl_impl_fit(data, model, params)

    @abc.abstractmethod
    def _impl_impl_fit(self, data, model, params):
        pass


class FitterDynestySNS(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.sns'

    @classmethod
    def load(cls, info):
        info['rstate'] = np.random.RandomState(info.get('seed'))
        return cls(**parseutils.parse_class_args(info, cls))

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

    def _fit_impl(self, objective, param_info, param_interp, **kwargs):

        pmins = []
        pmaxs = []

        for pname in model.get_param_names():
            pinfo = params[pname]
            pmins.append(pinfo['min'])
            pmaxs.append(pinfo['max'])

            if 'prior' in pinfo:
                prior_info = pinfo['prior']
            else:
                prior_info = dict(min=pinfo.get('min'), max=pinfo.get('max'))

        pmins = np.array(pmins)
        pmaxs = np.array(pmaxs)

        ndim = len(pmins)

        logl_args = [data, model]
        ptform_args = [pmins, pmaxs]

        sampler = dynesty.NestedSampler(
            residual_fun, ptform_func, ndim, nlive=1, logl_args=logl_args, ptform_args=ptform_args, **self._kwargs)



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
            enlarge=None, bootstrap=0, vol_dec=0.5, vol_check=2.0,
            walks=25, facc=0.5, slices=5, fmove=0.9, max_move=100,
            # dynesty.dynamicsampler.DynamicNestedSampler.run_nested()
            nlive_init=500, maxiter_init=None,
            maxcall_init=None, dlogz_init=0.01, logl_max_init=np.inf,
            n_effective_init=np.inf, nlive_batch=500,
            wt_function=None, wt_kwargs=None,
            maxiter_batch=None, maxcall_batch=None,
            maxiter=None, maxcall=None, maxbatch=None,
            n_effective=np.inf,
            stop_function=None, stop_kwargs=None, use_stop=True,
            save_bounds=True, print_progress=True, print_func=None):
        super().__init__()
        self._props = locals()
        self._props.pop('self')
        self._props.pop('__class__')

    def _fit_impl(self, objective, param_info, param_interp, **kwargs):
        pass
