
import abc
import copy

import numpy as np
import numpy.random as random
import pygmo as pg

from gbkfit.fitting.fitter import Fitter
from .params import FitParamsLMFitLeastSquares, FitParamsLMFitNelderMead


class FitterLMFit(Fitter, abc.ABC):

    def __init__(self, method, iter_cb, scale_covar, max_nfev):
        super().__init__()
        self._common_attrs = dict(
            method=method,
            iter_cb=iter_cb,
            scale_covar=scale_covar,
            nan_policy='raise',
            calc_covar=True,
            max_nfev=max_nfev)

    def _fit_impl(self, objective, parameters):
        result1 = self._fit_impl2(objective, parameters)
        extra = dict()
        attrs = [
            'success', 'status', 'message', 'nfev',
            'chisqr',  'redchi', 'aic', 'bic', 'covar']
        for attr in attrs:
            if hasattr(result1, attr):
                extra[attr] = getattr(result1, attr)

        print(extra)
        exit()
        return None

    @abc.abstractmethod
    def _fit_impl2(self, objective, parameters):
        pass


class FitterLMFitLeastSquares(FitterLMFit):

    @staticmethod
    def type():
        return 'lmfit.least_squares'

    @staticmethod
    def load_params(info, descs):
        return FitParamsLMFitLeastSquares.load(info, descs)

    def dump(self):
        return {**self._common_kws, **self._method_kws}

    def __init__(
            self, iter_cb=None, scale_covar=False, max_nfev=None,
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
            loss='linear', f_scale=1.0, diff_step=None,
            tr_solver=None, tr_options=None,
            jac_sparsity=None, verbose=0):
        super().__init__('least_squares', iter_cb, scale_covar, max_nfev)
        self._method_kws = dict(
            jac='3-point',
            ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale,
            loss=loss, f_scale=f_scale, diff_step=diff_step,
            tr_solver=tr_solver, tr_options=tr_options if tr_options else {},
            jac_sparsity=jac_sparsity, verbose=verbose)

    def _fit_impl2(self, objective, parameters):
        common_kws = copy.deepcopy(self._common_kws)
        method_kws = copy.deepcopy(self._method_kws)
        x_scale = method_kws.pop('x_scale')
        x_scales = []
        diff_step = method_kws.pop('diff_step')
        if diff_step is None:
            diff_step = np.finfo(np.float64).eps ** (1 / 3)
        diff_steps = []
        parameters_ = lmfit.Parameters()
        for pname, pinfo in parameters.infos().items():
            pname = pname.replace('[', '_obracket_')
            pname = pname.replace(']', '_cbracket_')
            parameters_.add(
                pname, pinfo.initial(), True, pinfo.minimum(), pinfo.maximum())
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
        result = lmfit.minimize(
            _residual_vector, args=(objective, interpreter, kwargs['callback']),
            params=parameters_, x_scale=x_scales, diff_step=diff_steps,
            **common_kws, **method_kws)
        return result


class FitterLMFitNelderMead(FitterLMFit):

    @staticmethod
    def type():
        return 'lmfit.nelder_mead'

    @staticmethod
    def load_params(info, descs):
        return FitParamsLMFitNelderMead.load(info, descs)

    def dump(self):
        return {**self._common_kws, **self._method_kws}

    def __init__(
            self, iter_cb=None, scale_covar=False, max_nfev=None,
            tol=None, disp=False, xatol=1e-4, fatol=1e-4, adaptive=False):
        super().__init__('nelder', iter_cb, scale_covar, max_nfev)
        self._method_kws = dict(
            tol=tol,
            options=dict(
                disp=disp,
                return_all=False,
                xatol=xatol,
                fatol=fatol,
                adaptive=adaptive))

    def _fit_impl2(self, objective, parameters):
        common_kws = copy.deepcopy(self._common_kws)
        method_kws = copy.deepcopy(self._method_kws)
        parameters_ = lmfit.Parameters()
        for pname, pinfo in parameters.infos().items():
            pname = pname.replace('[', '_obracket_')
            pname = pname.replace(']', '_cbracket_')
            parameters_.add(
                pname, pinfo.initial(), True, pinfo.minimum(), pinfo.maximum())
        result = lmfit.minimize(
            _residual_scalar, args=(objective, interpreter),
            params=parameters_, **common_kws, **method_kws)
        return result
