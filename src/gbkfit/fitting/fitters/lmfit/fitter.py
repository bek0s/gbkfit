
import abc
import copy

import lmfit
import numpy as np
import numpy.random as random


from gbkfit.fitting.core import Fitter
from .params import FitParamsLMFitLeastSquares, FitParamsLMFitNelderMead

from gbkfit.fitting.result import make_fitter_result


def _residual_params(x, parameters):
    enames = parameters.expressions().enames(free=True, tied=False, fixed=False)
    evalues = x.valuesdict().values()
    eparams = dict(zip(enames, evalues))
    return parameters.expressions().evaluate(eparams)


def _residual_scalar(x, objective, parameters, callback=None):
    params = _residual_params(x, parameters)
    residual = objective.residual_scalar(params)
    print(params)
    return residual


def _residual_vector(x, objective, parameters, callback=None):
    params = _residual_params(x, parameters)
    residuals = objective.residual_vector(params)
    residuals = np.nan_to_num(np.concatenate(residuals, casting='safe'))
    print(params)
    return residuals


class FitterLMFit(Fitter, abc.ABC):

    def dump(self):
        return dict()

    def __init__(self, method, iter_cb, scale_covar, max_nfev, options):
        super().__init__()
        self._method = method
        self._iter_cb = iter_cb
        self._scale_covar = scale_covar
        self._calc_covar = True
        self._nan_policy = 'raise'
        self._max_nfev = max_nfev
        self._options = options

    def _fit_impl(self, objective, parameters):

        # Create lmfit parameters for all free parameters.
        # We also need to transform the parameter names because
        # brackets are not supported by lmfit.
        lmfit_params = lmfit.Parameters()
        for pname, pinfo in parameters.infos().items():
            lmfit_params.add(
                pname.replace('[', '_obracket_').replace(']', '_cbracket_'),
                pinfo.initial_value(), True, pinfo.minimum(), pinfo.maximum())
        # Setup minimiser-specific options
        options = self._setup_minimizer_options(parameters)
        # Run minimisation
        lmfit_result = lmfit.minimize(
            _residual_vector, args=(objective, parameters), params=lmfit_params,
            method=self._method, iter_cb=self._iter_cb,
            scale_covar=self._scale_covar, nan_policy=self._nan_policy,
            calc_covar=self._calc_covar, max_nfev=self._max_nfev, **options)

        #
        solution = dict(mode=list(lmfit_result.params.valuesdict().values()))

        # Extract covariance and std error (if available)
        if hasattr(lmfit_result, 'covar'):
            covar = lmfit_result.covar
            solution.update(covar=covar, std=list(np.sqrt(np.diag(covar))))
        # Extract trivial information (if available)
        extra = dict()
        attrs = [
            'success', 'status', 'message', 'nfev',
            'chisqr',  'redchi', 'aic', 'bic']
        for attr in attrs:
            if hasattr(lmfit_result, attr):
                extra[attr] = getattr(lmfit_result, attr)
        #
        result = make_fitter_result(
            objective, parameters, solutions=solution)

        print(extra)

        return result

    @abc.abstractmethod
    def _setup_minimizer_options(self, parameters):
        pass


class FitterLMFitLeastSquares(FitterLMFit):

    @staticmethod
    def type():
        return 'lmfit.least_squares'

    @staticmethod
    def load_params(info, descs):
        return FitParamsLMFitLeastSquares.load(info, descs)

    def __init__(
            self, iter_cb=None, scale_covar=False, max_nfev=None,
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
            loss='linear', f_scale=1.0, diff_step=None,
            tr_solver=None, tr_options=None,
            jac_sparsity=None, verbose=0):
        super().__init__(
            'least_squares', iter_cb, scale_covar, max_nfev,
            options=dict(
                jac='3-point',
                ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale,
                loss=loss, f_scale=f_scale, diff_step=diff_step,
                tr_solver=tr_solver,
                tr_options=tr_options if tr_options else dict(),
                jac_sparsity=jac_sparsity, verbose=verbose))

    def _setup_minimizer_options(self, parameters):
        options = copy.deepcopy(self._options)
        x_scale = options.pop('x_scale')
        x_scales = []
        diff_step = options.pop('diff_step')
        if diff_step is None:
            diff_step = np.finfo(np.float64).eps ** (1 / 3)
        diff_steps = []
        for pname, pinfo in parameters.infos().items():
            p_x_scale = pinfo.x_scale()
            p_diff_step = pinfo.diff_step()
            if p_x_scale is not None and x_scale == 'jac':
                raise RuntimeError(
                    f"when given as a fitter option, "
                    f"x_scale cannot be also given as a parameter attribute "
                    f"(see {pname} parameter)")
            if p_x_scale is None:
                p_x_scale = x_scale
            if p_diff_step is None:
                p_diff_step = diff_step
            x_scales.append(p_x_scale)
            diff_steps.append(p_diff_step)
        options.update(diff_step=diff_steps, x_scale=x_scales)
        return options


class FitterLMFitNelderMead(FitterLMFit):

    @staticmethod
    def type():
        return 'lmfit.nelder_mead'

    @staticmethod
    def load_params(info, descs):
        return FitParamsLMFitNelderMead.load(info, descs)

    def __init__(
            self, iter_cb=None, scale_covar=False, max_nfev=None,
            tol=None, disp=False, xatol=1e-4, fatol=1e-4, adaptive=False):
        super().__init__(
            'nelder', iter_cb, scale_covar, max_nfev,
            options=dict(
                tol=tol,
                options=dict(
                    disp=disp, return_all=False, xatol=xatol, fatol=fatol,
                    adaptive=adaptive)))

    def _setup_minimizer_options(self, parameters):
        return copy.deepcopy(self._options)


class FitterLMFitSHGO(FitterLMFit):

    @staticmethod
    def type():
        return 'lmfit.shgo'

    @staticmethod
    def load_params(info, descs):
        pass
