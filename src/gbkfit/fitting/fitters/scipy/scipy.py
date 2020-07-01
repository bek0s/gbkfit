
import abc
import copy
import logging

import numpy as np
import scipy.optimize

import gbkfit.fitting.fitter
import gbkfit.fitting.params
import gbkfit.params.utils
from gbkfit.utils import parseutils

from gbkfit.fitting.result import FitterResult, FitterResultSolution


log = logging.getLogger(__name__)


class FitParamScipy(gbkfit.fitting.params.FitParam, abc.ABC):
    def __init__(self):
        super().__init__()


class FitParamScipyLeastSqr(FitParamScipy):

    @classmethod
    def load(cls, info):
        desc = f'fit parameter (class: {cls.__qualname__})'
        cls_args = parseutils.parse_options(
            info, desc, fun=cls.__init__, fun_rename_args=dict(
                initial='init', minimum='min', maximum='max'))
        return cls(**cls_args)

    def __init__(
            self, initial, minimum=-np.inf, maximum=np.inf,
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


class FitParamScipyMinimize(FitParamScipy):

    @classmethod
    def load(cls, info):
        desc = f'fit parameter (class: {cls.__qualname__})'
        cls_args = parseutils.parse_options(
            info, desc, fun=cls.__init__, fun_rename_args=dict(
                initial='init', minimum='min', maximum='max'))
        return cls(**cls_args)

    def __init__(
            self, initial, minimum=-np.inf, maximum=np.inf, options=None):
        super().__init__()
        if options is None:
            options = dict()
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


class FitParamsScipy(gbkfit.fitting.params.FitParams, abc.ABC):
    def __init__(self, params, descs):
        super().__init__(params, descs)


class FitParamsScipyLeastSqr(FitParamsScipy):

    @classmethod
    def load(cls, info, descs):
        infos, exprs = gbkfit.params.utils.parse_param_info(info, descs)[4:6]
        for k, v in infos.items():
            infos[k] = FitParamScipyLeastSqr.load(v)
        return cls({**infos, **exprs}, descs)

    def __init__(self, params, descs):
        super().__init__(params, descs)


class FitParamsScipyMinimize(FitParamsScipy):

    @classmethod
    def load(cls, info, descs):
        infos, exprs = gbkfit.params.utils.parse_param_info(info, descs)[4:6]
        for k, v in infos.items():
            infos[k] = FitParamScipyMinimize.load(v)
        return cls({**infos, **exprs}, descs)

    def __init__(self, params, descs):
        super().__init__(params, descs)


class FitterScipy(gbkfit.fitting.fitter.Fitter):

    def __init__(self):
        super().__init__()

    def _fit_impl(self, objective, parameters, interpreter, **kwargs):
        result1 = self._fit_impl2(objective, parameters, interpreter, **kwargs)
        extra = dict()
        definite_attrs = ['status', 'message', 'nit', 'nfev']
        optional_attrs = ['jac', 'hess', 'njev', 'nhev']
        for attr in definite_attrs:
            extra[attr] = getattr(result1, attr)
        for attr in optional_attrs:
            if hasattr(result1, attr):
                extra[attr] = getattr(result1, attr)
        result2 = FitterResult(objective, parameters, extra)
        result2.add_mode(mode=result1.x)
        return result2

    @abc.abstractmethod
    def _fit_impl2(self, objective, parameters, interpreter, **kwargs):
        pass


class FitterScipyLeastSquares(FitterScipy):

    @staticmethod
    def type():
        return 'scipy.least_squares'

    @classmethod
    def load(cls, info):
        desc = f'{cls.type()} fitter (class: {cls.__qualname__})'
        cls_args = parseutils.parse_options(info, desc, fun=cls.__init__)
        return cls(**cls_args)

    @staticmethod
    def load_params(info, descs):
        return FitParamsScipyLeastSqr.load(info, descs)

    def dump(self):
        return self._props

    def __init__(
            self, ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
            loss='linear', f_scale=1.0, diff_step=None,
            tr_solver=None, tr_options=None, jac_sparsity=None,
            max_nfev=None, verbose=0):
        if tr_options is None:
            tr_options = dict()
        super().__init__()
        self._props = locals()
        self._props.pop('self')
        self._props.pop('__class__')
        self._props.update(dict(jac='3-point', method='trf'))

    def _fit_impl2(self, objective, parameters, interpreter, **kwargs):
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
            initials.append(pinfo.initial())
            minimums.append(pinfo.minimum())
            maximums.append(pinfo.maximum())
            x_scales.append(p_x_scale)
            diff_steps.append(p_diff_step)
        result = scipy.optimize.least_squares(
            self._residual, initials, bounds=(minimums, maximums),
            x_scale=x_scales, diff_step=diff_steps, **props,
            args=(objective, interpreter))
        return result

    @staticmethod
    def _residual(x, objective, interpreter):
        eparams = dict(zip(interpreter.get_param_names(), x))
        params = interpreter.evaluate(eparams)
        residual = np.concatenate(objective.residual_vector(params))
        print(eparams)
        return residual


class FitterScipyMinimize(FitterScipy):

    @staticmethod
    def type():
        return 'scipy.minimize'

    @classmethod
    def load(cls, info):
        desc = f'{cls.type()} fitter (class: {cls.__qualname__})'
        cls_args = parseutils.parse_options(info, desc, fun=cls.__init__)
        return cls(**cls_args)

    @staticmethod
    def load_params(info, descs):
        return FitParamsScipyMinimize.load(info, descs)

    def dump(self):
        return self._props

    def __init__(self, method=None, jac=None, tol=None, options=None):
        super().__init__()
        if options is None:
            options = dict()
        self._props = locals()
        self._props.pop('self')
        self._props.pop('__class__')
        #self._props.update(dict(jac='3-point', hess='3-point'))

    def _fit_impl2(self, objective, parameters, interpreter, **kwargs):
        props = copy.deepcopy(self._props)
        initials = []
        for pname, pinfo in parameters.infos().items():
            initials.append(pinfo.initial())
        result = scipy.optimize.minimize(
            self._residual, initials, args=(objective, interpreter), **props)
        return result

    @staticmethod
    def _residual(x, objective, interpreter):
        eparams = dict(zip(interpreter.get_param_names(), x))
        params = interpreter.evaluate(eparams)
        residual = objective.residual_scalar(params)
        print(eparams)
        return residual
