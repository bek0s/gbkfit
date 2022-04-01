
import abc

import lmfit
import numpy as np


from gbkfit.fitting.core import FitParam, FitParams, Fitter

from gbkfit.fitting.result import make_fitter_result


def _residual_params(x, parameters):
    enames = parameters.interpreter().enames(free=True, tied=False, fixed=False)
    evalues = x.valuesdict().values()
    eparams = dict(zip(enames, evalues))
    return parameters.interpreter().evaluate(eparams)


def residual_scalar(x, objective, parameters, callback=None):
    params = _residual_params(x, parameters)
    residual = objective.residual_scalar(params)
    print(params)
    return residual


def residual_vector(x, objective, parameters, callback=None):
    params = _residual_params(x, parameters)
    residuals = objective.residual_vector_h(params)
    residuals = np.nan_to_num(np.concatenate(residuals, casting='safe'))
    print(params)
    return residuals


class FitParamLMFit(FitParam, abc.ABC):
    pass


class FitParamsLMFit(FitParams, abc.ABC):
    pass


class FitterLMFit(Fitter, abc.ABC):

    def dump(self):
        return dict()

    def __init__(
            self, method, iter_cb, scale_covar, max_nfev, residual_fun,
            options):
        super().__init__()
        self._method = method
        self._iter_cb = iter_cb
        self._scale_covar = scale_covar
        self._calc_covar = True
        self._nan_policy = 'raise'
        self._max_nfev = max_nfev
        self._residual_fun = residual_fun
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
            self._residual_fun, args=(objective, parameters),
            params=lmfit_params, method=self._method, iter_cb=self._iter_cb,
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
