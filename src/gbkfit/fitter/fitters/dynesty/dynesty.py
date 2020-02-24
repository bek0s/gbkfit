
import abc
import logging

import dynesty
import numpy as np
import ultranest

import gbkfit.fitter

import inspect
import dynesty
import dynesty.dynamicsampler
import dynesty.sampler

log = logging.getLogger(__name__)


def residual_fun(pvalues, data, model):
    params = dict(zip(model.get_param_names(), pvalues))
    #log.info(params)
    outputs = model.evaluate(params, False)
    chi2 = 0
    for dataset, output in zip(data, outputs):
        for name in output:
            dat = dataset[name].data()
            msk = dataset[name].mask()
            err = dataset[name].error()
            mdl = output[name]
            res = np.power((dat - mdl) / err, 2)
            #import astropy.io.fits as fits
            #fits.writeto('residual.fits', res, overwrite=True)
            res[np.isnan(res)] = 0
            #print(np.nansum(res))
            chi2 += np.nansum(res)
    #print("chi2: ", chi2)
    return -0.5 * chi2


def ptform_func(u, par_min, par_max):
    return par_min + (par_max - par_min) * u


class Likelihood:

    __name__ = 'Likelihood'

    def __init__(self, data, model):
        self._data = data
        self._model = model

    def __call__(self, pvalues, *args, **kwargs):
        data = self._data
        model = self._model
        params = dict(zip(model.get_param_names(), pvalues))
        #log.info(params)
        outputs = model.evaluate(params, False)
        chi2 = 0
        for dataset, output in zip(data, outputs):
            for name in output:
                dat = dataset[name].data()
                msk = dataset[name].mask()
                err = dataset[name].error()
                mdl = output[name]
                res = np.power((dat - mdl) / err, 2)
                # import astropy.io.fits as fits
                # fits.writeto('residual.fits', res, overwrite=True)
                res[np.isnan(res)] = 0
                #print(np.nansum(res))
                chi2 += np.nansum(res)
        #print("chi2: ", chi2)
        return -0.5 * chi2


class Prior:

    __name__ = 'Prior'

    def __init__(self, par_min, par_max):
        self.par_min = par_min
        self.par_max = par_max

    def __call__(self, cube, *args, **kwargs):
        par_min = self.par_min
        par_max = self.par_max
        params = cube.copy()
        params[:] = par_min + (par_max - par_min) * cube
        return params


class FitterDynesty(gbkfit.fitter.Fitter):

    def __init__(self):
        super().__init__()

    def _impl_fit(self, data, model, params):
        return self._impl_impl_fit(data, model, params)

    @abc.abstractmethod
    def _impl_impl_fit(self, data, model, params):
        pass


SSAMPLER_GLOBAL_ARGS_REQ = dict(
    # dynesty.NestedSampler()
    nlive=500
)

SSAMPLER_GLOBAL_ARGS_OPT = dict(
    # dynesty.NestedSampler()
    bound='multi', sample='auto',
    update_interval=None, first_update=None, rstate=None,
    enlarge=None, bootstrap=0, vol_dec=0.5, vol_check=2.0,
    walks=25, facc=0.5, slices=5, fmove=0.9, max_move=100,
    # dynesty.sampler.Sampler.run_nested()
    maxiter=None, maxcall=None, dlogz=None,
    logl_max=np.inf, n_effective=None,
    add_live=True, print_progress=True,
    print_func=None, save_bounds=True
)

SSAMPLER_PARAMS_ARGS_OPT = dict(
    # dynesty.NestedSampler()
    periodic=None,
    reflective=None,
    live_points=None
)

DSAMPLER_GLOBAL_ARGS_REQ = dict(
    # dynesty.dynamicsampler.DynamicNestedSampler.run_nested()
    nlive_init=100,
    nlive_batch=500
)

DSAMPLER_GLOBAL_ARGS_OPT = dict(
    # dynesty.DynamicNestedSampler()
    bound='multi', sample='auto',
    update_interval=None, first_update=None, rstate=None,
    enlarge=None, bootstrap=0, vol_dec=0.5, vol_check=2.0,
    walks=25, facc=0.5, slices=5, fmove=0.9, max_move=100,
    # dynesty.dynamicsampler.DynamicNestedSampler.run_nested()
    maxiter_init=None,
    maxcall_init=None, dlogz_init=0.01, logl_max_init=np.inf,
    n_effective_init=np.inf,
    wt_function=None, wt_kwargs=None,
    maxiter_batch=None, maxcall_batch=None,
    maxiter=None, maxcall=None, maxbatch=None,
    n_effective=np.inf,
    stop_function=None, stop_kwargs=None, use_stop=True,
    save_bounds=True, print_progress=True, print_func=None
)

DSAMPLER_PARAMS_ARGS_OPT = dict(
    # dynesty.DynamicNestedSampler()
    periodic=None,
    reflective=None,
    # dynesty.dynamicsampler.DynamicNestedSampler.run_nested()
    live_points=None
)


def check_args(args, required_args, optional_args):

    required_args = required_args.copy()
    optional_args = optional_args.copy()

    missing_args = set(required_args).difference(args)

    if missing_args:
        raise RuntimeError(
            f"The following Fitter arguments are "
            f"required: {missing_args}.")

    unknown_args = set(args).difference({**required_args, **optional_args})

    if unknown_args:
        raise RuntimeError(
            f"The following Fitter arguments are "
            f"not recognised and will be ignored: {unknown_args}.")





class FitterDynestySNestedSampling(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.sns'

    @classmethod
    def load(cls, info):
        return cls(**info)

    def dump(self):
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

        check_args(kwargs, SSAMPLER_GLOBAL_ARGS_REQ, SSAMPLER_GLOBAL_ARGS_OPT)

    def _impl_impl_fit(self, data, model, params):

        pmins = []
        pmaxs = []

        for pname in model.get_param_names():
            pinfo = params[pname]
            pmins.append(pinfo['min'])
            pmaxs.append(pinfo['max'])


        pmins = np.array(pmins)
        pmaxs = np.array(pmaxs)

        ndim = len(pmins)

        logl_args = [data, model]
        ptform_args = [pmins, pmaxs]



        sampler = dynesty.NestedSampler(
            residual_fun, ptform_func, ndim, nlive=1, logl_args=logl_args, ptform_args=ptform_args, **self._kwargs)



        result = sampler.run_nested()


        pass
