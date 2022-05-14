
import collections.abc
import copy

import numpy as np

from gbkfit.fitting import fitutils
from gbkfit.params import paramutils
from gbkfit.utils import parseutils

from .core import FitParamLMFit, FitParamsLMFit, FitterLMFit, residual_vector


__all__ = [
    'FitParamLMFitLeastSquares',
    'FitParamsLMFitLeastSquares',
    'FitterLMFitLeastSquares'
]


class FitParamLMFitLeastSquares(FitParamLMFit):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_basic_desc(cls, 'param')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='value',
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        info = dict(val=self.initial_value())
        if np.isfinite(self.minimum()):
            info.update(min=self.minimum())
        if np.isfinite(self.maximum()):
            info.update(max=self.maximum())
        if self.x_scale() is not None:
            info.update(x_scale=self.x_scale())
        if self.diff_step() is not None:
            info.update(xdiff_step=self.diff_step())
        return info

    def __init__(
            self, initial_value, minimum=None, maximum=None,
            x_scale=None, diff_step=None):
        super().__init__()
        minimum = -np.inf if minimum is None else minimum
        maximum = +np.inf if maximum is None else maximum
        self._initial_value = initial_value
        self._minimum = minimum
        self._maximum = maximum
        self._x_scale = x_scale
        self._diff_step = diff_step

    def initial_value(self):
        return self._initial_value

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def x_scale(self):
        return self._x_scale

    def diff_step(self):
        return self._diff_step


class FitParamsLMFitLeastSquares(FitParamsLMFit):

    @staticmethod
    def load_param(info):
        return FitParamLMFitLeastSquares.load(info)

    @classmethod
    def load(cls, info, pdescs):
        desc = parseutils.make_basic_desc(cls, 'fit params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        opts = paramutils.load_params_parameters_conversions(
            opts, pdescs, collections.abc.Mapping, cls.load_param)
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return paramutils.dump_params_parameters_conversions(
            self, FitParamLMFitLeastSquares, lambda x: x.dump(), conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        super().__init__(
            pdescs, parameters, conversions, FitParamLMFitLeastSquares)


class FitterLMFitLeastSquares(FitterLMFit):

    @staticmethod
    def type():
        return 'lmfit.least_squares'

    @staticmethod
    def load_params(info, pdescs):
        return FitParamsLMFitLeastSquares.load(info, pdescs)

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        info = dict(type=self.type())
        global_options = copy.deepcopy(self._global_options)
        method_options = copy.deepcopy(self._method_options)
        parseutils.prepare_for_dump(
            global_options,
            remove_nones=True,
            remove_keys=('nan_policy', 'calc_covar'))
        parseutils.prepare_for_dump(
            method_options,
            remove_nones=True,
            remove_keys=('method', 'jac'))
        return info | global_options | method_options

    def __init__(
            self, scale_covar=False, max_nfev=None,
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
            loss='linear', f_scale=1.0, diff_step=None,
            tr_solver=None, tr_options=None,
            jac_sparsity=None, verbose=0):
        super().__init__(
            residual_vector, 'least_squares', scale_covar, max_nfev,
            options=dict(
                method='trf', jac='3-point',
                ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale,
                loss=loss, f_scale=f_scale, diff_step=diff_step,
                tr_solver=tr_solver,
                tr_options=tr_options if tr_options else dict(),
                jac_sparsity=jac_sparsity, verbose=verbose))

    def _setup_options(self, parameters, global_options, method_options):
        x_scale = method_options.pop('x_scale')
        x_scales = []
        diff_step = method_options.pop('diff_step')
        if diff_step is None:
            diff_step = np.finfo(np.float64).eps ** (1 / 3)
        diff_steps = []
        for pname, pinfo in parameters.infos().items():
            p_x_scale = pinfo.x_scale()
            p_diff_step = pinfo.diff_step()
            if p_x_scale is not None and x_scale == 'jac':
                raise RuntimeError(
                    f"problem with parameter '{pname}': "
                    f"x_scale cannot be given as a parameter option when "
                    f"it is also given as a fitter option with value 'jac'")
            if p_x_scale is None:
                p_x_scale = x_scale
            if p_diff_step is None:
                p_diff_step = diff_step
            x_scales.append(p_x_scale)
            diff_steps.append(p_diff_step)
        method_options.update(
            diff_step=diff_steps,
            x_scale=x_scale if x_scale == 'jac' else x_scales)
        return global_options, method_options
