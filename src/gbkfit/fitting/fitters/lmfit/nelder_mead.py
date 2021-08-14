
import copy

import numpy as np
import gbkfit.params.utils as paramutils
from gbkfit.utils import parseutils

from .core import FitParamLMFit, FitParamsLMFit, FitterLMFit, residual_scalar, residual_vector

from gbkfit.fitting.utils import parse_parameters


class FitParamLMFitNelderMead(FitParamLMFit):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_basic_desc(cls, 'param')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='val',
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        info = dict(val=self.initial_value())
        if np.isfinite(self.minimum()):
            info.update(min=self.minimum())
        if np.isfinite(self.maximum()):
            info.update(max=self.maximum())
        return info

    def __init__(
            self, initial_value, minimum=-np.inf, maximum=np.inf):
        super().__init__()
        self._initial_value = initial_value
        self._minimum = minimum
        self._maximum = maximum

    def initial_value(self):
        return self._initial_value

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum


class FitParamsLMFitNelderMead(FitParamsLMFit):

    @classmethod
    def load(cls, info, descs):
        desc = parseutils.make_basic_desc(cls, 'params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        parameters = parse_parameters(opts['parameters'], descs, cls.load_param)
        expressions = paramutils.load_expressions(opts.get('expressions'))
        return cls(descs, parameters, expressions)

    @staticmethod
    def load_param(info):
        return FitParamLMFitNelderMead.load(info)

    def dump(self):
        return dict()

    def __init__(self, descs, parameters, expressions=None):
        super().__init__(
            descs, parameters, expressions, FitParamLMFitNelderMead)


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
            'nelder', iter_cb, scale_covar, max_nfev, residual_scalar,
            options=dict(
                tol=tol,
                options=dict(
                    disp=disp, return_all=False, xatol=xatol, fatol=fatol,
                    adaptive=adaptive)))

    def _setup_minimizer_options(self, parameters):
        return copy.deepcopy(self._options)
