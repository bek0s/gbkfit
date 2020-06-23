
import abc
import copy
import logging
import math

import numpy as np
import scipy.optimize

import gbkfit.fitting.fitter
import gbkfit.fitting.params
from gbkfit.utils import parseutils

import gbkfit.params.utils

log = logging.getLogger(__name__)


class FitParamScipy(gbkfit.fitting.params.FitParam):
    pass


class FitParamScipyLeastSqr(FitParamScipy):

    @classmethod
    def load(cls, info):
        cls_args = parseutils.parse_options(
            info, 'foo', fun=cls.__init__, fun_rename_args={
                'initial': 'init', 'minimum': 'min', 'maximum': 'max'})
        return cls(**cls_args)

    def __init__(
            self, initial, minimum=-math.inf, maximum=math.inf,
            x_scale=None, diff_step=None):
        super().__init__()
        self._initial = initial
        self._minimum = minimum
        self._maximum = maximum
        self._x_scale = x_scale
        self._diff_step = diff_step

    def initial(self):
        return self._initial

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def x_scale(self):
        return self._x_scale

    def diff_step(self):
        return self._diff_step


class FitParamScipyOptimize(FitParamScipy):

    def __init__(
            self, initial, minimum=-math.inf, maximum=math.inf, options=None):
        super().__init__()
        self._initial = initial
        self._minimum = minimum
        self._maximum = maximum
        self._options = copy.deepcopy(options)

    def initial(self):
        return self._initial

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def options(self):
        return self._options


class FitParamsScipy(gbkfit.fitting.params.FitParams):
    def __init__(self, params, descs):
        super().__init__(params, descs)


class FitParamsScipyLeastSqr(FitParamsScipy):

    @classmethod
    def load(cls, info, descs):
        infos, exprs = gbkfit.params.utils.parse_param_info(info, descs)[4:]
        for k, v in infos.items():
            infos[k] = FitParamScipyLeastSqr.load(v)
        return cls({**infos, **exprs}, descs)

    def __init__(self, params, descs):
        super().__init__(params, descs)


class FitParamsScipyOptimize(FitParamsScipy):
    def __init__(self, params, descs):
        super().__init__(params, descs)


class FitterScipy(gbkfit.fitting.fitter.Fitter):

    def __init__(self):
        super().__init__()

    def _fit_impl(self, objective, param_info, param_interp, **kwargs):
        result = self._fit_impl2(objective, param_info, param_interp, **kwargs)
        return result

    @abc.abstractmethod
    def _fit_impl2(self, objective, param_info, param_interp, **kwargs):
        pass

    @staticmethod
    def _residual(x, objective, interpreter):

        eparams = dict(zip(interpreter.get_param_names(), x))

        params = interpreter.evaluate(eparams)

        residual = objective.residual_vector(params)
        print(eparams)
        return np.concatenate(residual)


class FitterScipyLeastSquares(FitterScipy):

    @staticmethod
    def type():
        return 'scipy.least_squares'

    @classmethod
    def load(cls, info):
        desc = f'{cls.__qualname__}(type: {cls.type()})'
        cls_args = parseutils.parse_options(info, desc, fun=cls.__init__)
        return cls(**cls_args)

    @staticmethod
    def load_params(info, desc):
        return FitParamsScipyLeastSqr.load(info, desc)

    def dump(self):
        return self._props

    def __init__(
            self, ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
            loss='linear', f_scale=1.0, diff_step=None,
            tr_solver=None, tr_options=None, max_nfev=None, verbose=0):
        if tr_options is None:
            tr_options = {}
        super().__init__()
        self._props = locals()
        self._props.pop('self')
        self._props.pop('__class__')
        self._props.update(dict(jac='3-point', method='trf'))

    def _fit_impl2(self, objective, param_info, param_interp, **kwargs):
        props = self._props.copy()
        initials = []
        minimums = []
        maximums = []
        x_scale = props.pop('x_scale')
        x_scales = []
        diff_step = props.pop('diff_step')
        if diff_step is None:
            diff_step = np.finfo(np.float64).eps ** (1 / 3)
        diff_steps = []
        for pname, pinfo in param_info.infos().items():
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
            initials.append(pinfo.initial())
            minimums.append(pinfo.minimum())
            maximums.append(pinfo.maximum())
            x_scales.append(p_x_scale)
            diff_steps.append(p_diff_step)
        result = scipy.optimize.least_squares(
            self._residual, initials, bounds=(minimums, maximums),
            x_scale=x_scales, diff_step=diff_steps, **props,
            args=(objective, param_interp))
        return result


class FitterScipyMinimize(FitterScipy):

    @staticmethod
    def type():
        return 'scipy.minimize'

    @classmethod
    def load(cls, info):
        desc = f'{cls.__qualname__}(type: {cls.type()})'
        cls_args = parseutils.parse_options(info, desc, fun=cls.__init__)
        return cls(**cls_args)

    def dump(self):
        return self._kwargs

    def __init__(self, method=None, jac=None, tol=None, options=None):
        super().__init__()
        self._kwargs = locals()
        self._kwargs.pop('self')
        self._kwargs.pop('__class__')

    def _fit_impl2(self, objective, param_info, param_interp, **kwargs):
        scipy.optimize.minimize(self._residual, initials, args=())
