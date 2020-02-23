
import abc
import logging

import numpy as np
import scipy.optimize

import gbkfit.fitter


log = logging.getLogger(__name__)


def residual_fun(pvalues, data, model, residual):
    params = dict(zip(model.get_param_names(), pvalues))
    log.info(params)
    outputs = model.evaluate(params, False)
    res_offset = 0
    for dataset, output in zip(data, outputs):
        for name in output:
            dat = dataset[name].data()
            msk = dataset[name].mask()
            err = dataset[name].error()
            mdl = output[name]
            res = msk * (dat - mdl) / err
            res[np.isnan(res)] = 0
            """
            dat_msk = msk
            mdl_msk = np.zeros_like(msk)
            mdl_msk[np.isfinite(mdl)] = 1
            sub_msk = dat_msk - mdl_msk
            fin_msk = np.zeros_like(sub_msk)
            fin_msk[sub_msk == 1] = 1
            res[sub_msk == 1] = np.nanmean(res)
            res[np.isnan(res)] = 0
            """
            """
            import astropy.io.fits as fits
            fits.writeto(f'dat_msk_{name}.fits', dat_msk, overwrite=True)
            fits.writeto(f'mdl_msk_{name}.fits', mdl_msk, overwrite=True)
            fits.writeto(f'sub_msk_{name}.fits', sub_msk, overwrite=True)
            fits.writeto(f'fin_msk_{name}.fits', fin_msk, overwrite=True)
            fits.writeto(f'res_{name}.fits', res, overwrite=True)
            """

            if name != 'mmap0':
                res *= 0.01
            residual[res_offset:res_offset + mdl.size] = res.ravel()
            res_offset += mdl.size
    #exit()
    print("Residual sum: ", np.nansum(residual))
    return np.array(residual.ravel())


class FitterScipy(gbkfit.fitter.Fitter):

    def __init__(self):
        super().__init__()

    def _impl_fit(self, data, model, params):
        result = self._impl_impl_fit(data, model, params)
        return result

    @abc.abstractmethod
    def _impl_impl_fit(self, data, model, params):
        pass


class FitterScipyLeastSquares(FitterScipy):

    @staticmethod
    def type():
        return 'scipy.least_squares'

    @classmethod
    def load(cls, info):
        info.pop('type')
        return cls(**info)

    def dump(self):
        return self._kwargs

    def __init__(self, **kwargs):
        super().__init__()
        x_scale = kwargs.get('x_scale')
        x_scale_is_num = isinstance(x_scale, (int, float))
        if x_scale is not None and not x_scale_is_num and x_scale != 'jac':
            raise RuntimeError(
                "When given as a fitter constructor argument, "
                "x_scale must be None or a number or 'jac'")
        diff_step = kwargs.get('diff_step')
        diff_step_is_num = isinstance(diff_step, (int, float))
        if diff_step is not None and not diff_step_is_num:
            raise RuntimeError(
                "When given as a fitter constructor argument, "
                "diff_step must be None or a number")
        self._kwargs = kwargs.copy()

    def _impl_impl_fit(self, data, model, params):

        kwargs = self._kwargs.copy()
        pvals = []
        pmins = []
        pmaxs = []
        x_scale = kwargs.pop('x_scale')
        x_scales = []
        diff_step = kwargs.pop('diff_step')
        diff_steps = []

        for pname in model.get_param_names():
            pinfo = params[pname]
            if 'x_scale' in pinfo and x_scale == 'jac':
                raise RuntimeError(
                    f"When given as a fitter constructor argument, "
                    f"x_scale can not be also given as a parameter attribute "
                    f"(see {pname} parameter)")
            pvals.append(pinfo['init'])
            pmins.append(pinfo.get('min', -np.nan))
            pmaxs.append(pinfo.get('max', +np.nan))
            x_scales.append(pinfo.get('x_scale', x_scale))
            diff_steps.append(pinfo.get('diff_step', diff_step))

        # Preallocate residual
        residuals_size = 0
        for datamap in data:
            for key, value in datamap.items():
                residuals_size += value.data().size
        residual = np.empty(residuals_size)

        # Perform the fit
        result = scipy.optimize.least_squares(
            residual_fun, pvals, bounds=(pmins, pmaxs), x_scale=x_scales,
            diff_step=diff_steps, **kwargs, args=(data, model, residual))

        return result


class FitterScipyMinimize(FitterScipy):

    @staticmethod
    def type():
        return 'scipy.minimize'

    @classmethod
    def load(cls, info):
        info.pop('type')
        return cls(**info)

    def dump(self):
        return self._kwargs

    def __init__(self, **kwargs):
        super().__init__()
        x_scale = kwargs.get('x_scale')
        x_scale_is_num = isinstance(x_scale, (int, float))
        if x_scale is not None and not x_scale_is_num and x_scale != 'jac':
            raise RuntimeError(
                "When given as a fitter constructor argument, "
                "x_scale must be None or a number or 'jac'")
        diff_step = kwargs.get('diff_step')
        diff_step_is_num = isinstance(diff_step, (int, float))
        if diff_step is not None and not diff_step_is_num:
            raise RuntimeError(
                "When given as a fitter constructor argument, "
                "diff_step must be None or a number")
        self._kwargs = kwargs.copy()

    def _impl_impl_fit(self, data, model, params):
        pass
