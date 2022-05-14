
import collections.abc
import copy

import numpy as np

from gbkfit.fitting import fitutils
from gbkfit.params import paramutils
from gbkfit.utils import parseutils

from .core import FitParamLMFit, FitParamsLMFit, FitterLMFit, residual_scalar


__all__ = [
    'FitParamLMFitNelderMead',
    'FitParamsLMFitNelderMead',
    'FitterLMFitNelderMead'
]


class FitParamLMFitNelderMead(FitParamLMFit):

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
        return info

    def __init__(self, initial_value, minimum=None, maximum=None):
        super().__init__()
        minimum = -np.inf if minimum is None else minimum
        maximum = +np.inf if maximum is None else maximum
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

    @staticmethod
    def load_param(info):
        return FitParamLMFitNelderMead.load(info)

    @classmethod
    def load(cls, info, pdescs):
        desc = parseutils.make_basic_desc(cls, 'params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        opts = paramutils.load_params_parameters_conversions(
            opts, pdescs, collections.abc.Mapping, cls.load_param)
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return paramutils.dump_params_parameters_conversions(
            self, FitParamLMFitNelderMead, lambda x: x.dump(), conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        super().__init__(
            pdescs, parameters, conversions, FitParamLMFitNelderMead)


class FitterLMFitNelderMead(FitterLMFit):

    @staticmethod
    def type():
        return 'lmfit.nelder_mead'

    @staticmethod
    def load_params(info, pdescs):
        return FitParamsLMFitNelderMead.load(info, pdescs)

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
            remove_keys=())
        parseutils.prepare_for_dump(
            method_options['options'],
            remove_nones=True,
            remove_keys=('return_all',))
        return info | global_options | method_options

    def __init__(
            self, scale_covar=False, max_nfev=None,
            tol=None, disp=False, xatol=1e-4, fatol=1e-4, adaptive=False):
        super().__init__(
            residual_scalar, 'nelder', scale_covar, max_nfev,
            options=dict(
                tol=tol,
                options=dict(
                    disp=disp, return_all=False, xatol=xatol, fatol=fatol,
                    adaptive=adaptive)))

    def _setup_options(self, parameters, global_options, method_options):
        return global_options, method_options
