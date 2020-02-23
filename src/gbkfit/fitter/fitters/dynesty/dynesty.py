
import abc
import logging

import dynesty
import numpy as np
import ultranest

import gbkfit.fitter


log = logging.getLogger(__name__)


def residual_fun(pvalues, data, model):
    params = dict(zip(model.get_param_names(), pvalues))
    log.info(params)
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
            print(np.nansum(res))
            chi2 += np.nansum(res)
    print("chi2: ", chi2)
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


class FitterDynestyStaticNestedSampling(FitterDynesty):

    @staticmethod
    def type():
        return 'dynesty.sns'

    @classmethod
    def load(cls, info):
        info.pop('type')
        return cls(**info)

    def dump(self):
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

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

        sampler = ultranest.ReactiveNestedSampler(model.get_param_names(), Likelihood(data, model), Prior(pmins, pmaxs))
        result = sampler.run(min_num_live_points=200)
        sampler.print_results()

        """
        exit()
        sampler = dynesty.NestedSampler(
            residual_fun, ptform_func, ndim, nlive=100, logl_args=logl_args, ptform_args=ptform_args, **self._kwargs)

        result = sampler.run_nested()

        res2 = sampler.results
        import matplotlib.pyplot as plt
        from dynesty import plotting as dyplot

        fig, axes = dyplot.traceplot(res2, truths=np.zeros(ndim),
                                     truth_color='black', show_titles=True,
                                     trace_cmap='viridis', connect=True,
                                     connect_highlight=range(5))


        plt.show()

        print("Result: ", result)
        """


        pass
