
import abc

import numpy as np

import gbkfit.params.utils as paramutils
from gbkfit.fitting.params import (
    FitParam,
    FitParams,
    make_fitparam_desc,
    make_fitparams_desc)
from gbkfit.utils import parseutils


class FitParamLMFit(FitParam, abc.ABC):
    pass


class FitParamsLMFit(FitParams, abc.ABC):
    pass


class FitParamLMFitLeastSqr(FitParamLMFit):

    @classmethod
    def load(cls, info):
        desc = make_fitparam_desc(cls)
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='val',
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        info = dict(initial_value=self.initial_value())
        if np.isfinite(self.minimum()):
            info.update(minimum=self.minimum())
        if np.isfinite(self.maximum()):
            info.update(maximum=self.maximum())
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


class FitParamLMFitNelderMead(FitParamLMFit):

    @classmethod
    def load(cls, info):
        desc = make_fitparam_desc(cls)
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='init',
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        info = dict(initial_value=self.initial_value())
        if np.isfinite(self.minimum()):
            info.update(minimum=self.minimum())
        if np.isfinite(self.maximum()):
            info.update(maximum=self.maximum())
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


class FitParamsLMFitLeastSquares(FitParamsLMFit):

    @classmethod
    def load(cls, info, descs):
        desc = make_fitparams_desc(cls)
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        infos, exprs = paramutils.parse_param_info(
            opts['parameters'], descs)[4:]
        for key, val in infos.items():
            try:
                infos[key] = FitParamLMFitLeastSqr.load(val)
            except RuntimeError as e:
                raise RuntimeError(
                    f"could not parse information for parameter '{key}'; "
                    f"reason: {str(e)}") from e
        parameters = {**infos, **exprs}
        expressions = paramutils.load_expressions(opts.get('expressions'))
        return cls(descs, parameters, expressions)

    def dump(self):
        return super().dump()

    def __init__(self, descs, parameters, expressions=None):
        super().__init__(descs, parameters, expressions)


class FitParamsLMFitNelderMead(FitParamsLMFit):

    @classmethod
    def load(cls, info, descs):
        desc = make_fitparams_desc(cls)
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        infos, exprs = paramutils.parse_param_info(
            opts['parameters'], descs)[4:]
        for key, val in infos.items():
            try:
                infos[key] = FitParamLMFitNelderMead.load(val)
            except RuntimeError as e:
                raise RuntimeError(
                    f"could not parse information for parameter '{key}'; "
                    f"reason: {str(e)}") from e
        parameters = {**infos, **exprs}
        expressions = paramutils.load_expressions(opts.get('expressions'))
        return cls(descs, parameters, expressions)

    def dump(self):
        return super().dump()

    def __init__(self, descs, parameters, expressions=None):
        super().__init__(descs, parameters, expressions)
