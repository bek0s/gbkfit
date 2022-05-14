
import abc
import copy

import lmfit
import numpy as np

from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter
from gbkfit.fitting.result import make_fitter_result


__all__ = [
    'FitParamLMFit',
    'FitParamsLMFit',
    'FitterLMFit',
    'residual_scalar',
    'residual_vector'
]


def _make_eparams_dict(x, parameters):
    enames = parameters.enames(fixed=False, tied=False, free=True)
    evalues = x.valuesdict().values()
    return dict(zip(enames, evalues))


def residual_scalar(x, parameters, objective, callback=None):
    eparams = _make_eparams_dict(x, parameters)
    return fitutils.residual_scalar(eparams, parameters, objective, callback)


def residual_vector(x, parameters, objective, callback=None):
    eparams = _make_eparams_dict(x, parameters)
    return fitutils.residual_vector(eparams, parameters, objective, callback)


class FitParamLMFit(FitParam, abc.ABC):
    pass


class FitParamsLMFit(FitParams, abc.ABC):
    pass


class FitterLMFit(Fitter, abc.ABC):

    def __init__(self, residual_func, method, scale_covar, max_nfev, options):
        super().__init__()
        self._residual_func = residual_func
        self._method = method
        self._global_options = dict(
            scale_covar=scale_covar,
            nan_policy='raise',
            calc_covar=True,
            max_nfev=max_nfev)
        self._method_options = copy.deepcopy(options)

    def _fit_impl(self, objective, parameters):
        # Create lmfit parameters for all free parameters.
        # Replace brackets in parameters names
        # because lmfit does not support them.
        lmfit_params = lmfit.Parameters()
        for pname, pinfo in parameters.infos().items():
            lmfit_params.add(
                pname.replace('[', '__obr__').replace(']', '__cbr__'),
                pinfo.initial_value(), True, pinfo.minimum(), pinfo.maximum())
        # Setup minimiser-specific options
        global_options, method_options = self._setup_options(
            parameters,
            copy.deepcopy(self._global_options),
            copy.deepcopy(self._method_options))
        # Run minimisation
        minimizer = lmfit.Minimizer(
            self._residual_func, params=lmfit_params,
            fcn_args=(parameters, objective),
            **global_options, **method_options)
        lmfit_result = minimizer.minimize(method=self._method)
        # Extract the best-fit solution
        solution = dict(mode=list(lmfit_result.params.valuesdict().values()))
        # Extract covariance and std error
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
    def _setup_options(self, parameters, global_options, method_options):
        pass
