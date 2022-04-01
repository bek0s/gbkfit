
import copy

import numpy as np
import gbkfit.params.utils as paramutils
from gbkfit.utils import parseutils

from .core import FitParamLMFit, FitParamsLMFit, FitterLMFit, residual_vector


from gbkfit.fitting.utils import load_parameters


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
        info.update(x_scale=self.x_scale(), diff_step=self.diff_step())
        return info

    def __init__(
            self, initial_value, minimum=-np.inf, maximum=np.inf,
            x_scale=None, diff_step=None):
        super().__init__()
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

    @classmethod
    def load(cls, info, descs):
        desc = parseutils.make_basic_desc(cls, 'params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        parameters = load_parameters(opts['parameters'], descs, cls.load_param)
        expressions = paramutils.load_expressions(opts.get('expressions'))
        return cls(descs, parameters, expressions)

    @staticmethod
    def load_param(info):
        return FitParamLMFitLeastSquares.load(info)

    def dump(self):
        return dict()

    def __init__(self, descs, parameters, expressions=None):
        super().__init__(
            descs, parameters, expressions, FitParamLMFitLeastSquares)


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
            'least_squares', iter_cb, scale_covar, max_nfev, residual_vector,
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
