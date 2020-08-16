
import abc
import copy
import logging

import numpy as np
import pyswarms.single.general_optimizer
import pyswarms.single.global_best
import pyswarms.single.local_best
from pyswarms.backend.topology import pyramid, random, ring, star, von_neumann

import gbkfit.fitting.fitter
import gbkfit.fitting.params
import gbkfit.fitting.result
import gbkfit.params.utils
from gbkfit.utils import parseutils


log = logging.getLogger(__name__)


_TOPOLOGY_STR2CLS = dict(
    pyramid=pyramid.Pyramid,
    random=random.Random,
    ring=ring.Ring,
    star=star.Star,
    von_neumann=von_neumann.VonNeumann)


_TOPOLOGY_OPTIONS = dict(
    pyramid=['c1', 'c2', 'w'],
    random=['c1', 'c2', 'w', 'k'],
    ring=['c1', 'c2', 'w', 'k', 'p'],
    star=['c1', 'c2', 'w'],
    von_neumann=['c1', 'c2', 'w', 'k', 'p', 'r'])


def _residual_scalar(evalues_list, objective, interpreter):
    #print(evalues_list)
    residuals = []
    for evalues in evalues_list:
        enames = interpreter.get_param_names()
        eparams = dict(zip(enames, evalues))
        params = interpreter.evaluate(eparams)
        residuals.append(objective.residual_scalar(params))
    return residuals


class FitParamPySwarms(gbkfit.fitting.params.FitParam):

    @classmethod
    def load(cls, info):
        desc = f'fit parameter (class: {cls.__qualname__})'
        cls_args = parseutils.parse_options(
            info, desc, fun=cls.__init__, fun_rename_args=dict(
                initial='init', minimum='min', maximum='max'))
        return cls(**cls_args)

    def __init__(self, minimum, maximum, initial=None):
        super().__init__()
        self._initial = initial
        self._minimum = minimum
        self._maximum = maximum

    def initial(self):
        return self._initial

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum


class FitParamsPySwarms(gbkfit.fitting.params.FitParams):

    @classmethod
    def load(cls, info, descs):
        infos, exprs = gbkfit.params.utils.parse_param_info(info, descs)[4:6]
        for k, v in infos.items():
            infos[k] = FitParamPySwarms.load(v)
        return cls({**infos, **exprs}, descs)

    def __init__(self, params, descs):
        super().__init__(params, descs)


class FitterPySwarms(gbkfit.fitting.fitter.Fitter):

    @classmethod
    def load(cls, info):
        desc = f'{cls.type()} fitter (class: {cls.__qualname__})'
        cls_args = parseutils.parse_options(info, desc, fun=cls.__init__)[0]
        return cls(**cls_args)

    @staticmethod
    def load_params(info, descs):
        return FitParamsPySwarms.load(info, descs)

    def dump(self):
        return {**self._common_kws, **self._method_kws}

    def __init__(
            self, n_particles, iters,
            bh_strategy, velocity_clamp, vh_strategy, ftol):
        super().__init__()
        self._common_kws = dict(
            n_particles=n_particles,
            iters=iters,
            bh_strategy=bh_strategy,
            velocity_clamp=velocity_clamp,
            vh_strategy=vh_strategy,
            ftol=ftol)
        self._method_kws = dict()

    def _fit_impl(self, objective, parameters, interpreter):
        result1 = self._fit_impl2(objective, parameters, interpreter)
        print(result1)
        return None

    @abc.abstractmethod
    def _fit_impl2(self, objective, parameters, interpreter):
        pass


class FitterPySwarmsLocal(FitterPySwarms):

    @staticmethod
    def type():
        return 'pyswarms.local'

    def __init__(
            self, n_particles, iters,
            c1, c2, w, k, p, bh_strategy='periodic',
            velocity_clamp=None, vh_strategy='unmodified',
            ftol=-np.inf, static=False):
        super().__init__(
            n_particles, iters,
            bh_strategy, velocity_clamp, vh_strategy, ftol)
        self._method_kws.update(dict(
            options=dict(c1=c1, c2=c2, w=w, k=k, p=p), static=static))

    def _fit_impl2(self, objective, parameters, interpreter):
        common_kws = copy.deepcopy(self._common_kws)
        method_kws = copy.deepcopy(self._method_kws)
        dimensions = len(parameters.infos())
        minimums = []
        maximums = []
        for pname, pinfo in parameters.infos().items():
            minimums.append(pinfo.minimum())
            maximums.append(pinfo.maximum())
        optimizer = pyswarms.single.local_best.LocalBestPSO(
            common_kws['n_particles'],
            dimensions,
            method_kws['options'],
            (np.array(minimums), np.array(maximums)),
            common_kws['bh_strategy'],
            common_kws['velocity_clamp'],
            common_kws['vh_strategy'],
            1.0,
            common_kws['ftol'],
            None,
            method_kws['static'])
        cost, pos = optimizer.optimize(
            _residual_scalar, common_kws['iters'],
            objective=objective, interpreter=interpreter)
        return cost, pos


class FitterPySwarmsGlobal(FitterPySwarms):

    @staticmethod
    def type():
        return 'pyswarms.global'

    def __init__(
            self, n_particles, iters,
            c1, c2, w, bh_strategy='periodic',
            velocity_clamp=None, vh_strategy='unmodified',
            ftol=-np.inf):
        super().__init__(
            n_particles, iters,
            bh_strategy, velocity_clamp, vh_strategy, ftol)
        self._method_kws.update(dict(
            options=dict(c1=c1, c2=c2, w=w)))

    def _fit_impl2(self, objective, parameters, interpreter):
        common_kws = copy.deepcopy(self._common_kws)
        method_kws = copy.deepcopy(self._method_kws)
        dimensions = len(parameters.infos())
        minimums = []
        maximums = []
        for pname, pinfo in parameters.infos().items():
            minimums.append(pinfo.minimum())
            maximums.append(pinfo.maximum())
        optimizer = pyswarms.single.global_best.GlobalBestPSO(
            common_kws['n_particles'],
            dimensions,
            method_kws['options'],
            (np.array(minimums), np.array(maximums)),
            common_kws['bh_strategy'],
            common_kws['velocity_clamp'],
            common_kws['vh_strategy'],
            1.0,
            common_kws['ftol'],
            None)
        cost, pos = optimizer.optimize(
            _residual_scalar, common_kws['iters'],
            objective=objective, interpreter=interpreter)
        return cost, pos


class FitterPySwarmsGeneral(FitterPySwarms):

    @staticmethod
    def type():
        return 'pyswarms.general'

    def __init__(
            self, n_particles, iters, topology,
            c1, c2, w, k=None, p=None, r=None, bh_strategy='periodic',
            velocity_clamp=None, vh_strategy='unmodified',
            ftol=-np.inf, static=False):
        super().__init__(
            n_particles, iters,
            bh_strategy, velocity_clamp, vh_strategy, ftol)
        locals_ = copy.deepcopy(locals())
        options = {o: locals_[o] for o in _TOPOLOGY_OPTIONS[topology]}
        if None in options.values():
            raise RuntimeError(
                f'the following arguments must be defined and not None: '
                f'{str(options)}')
        self._method_kws.update(dict(
            topology=_TOPOLOGY_STR2CLS[topology](static),
            options=options))

    def _fit_impl2(self, objective, parameters, interpreter):
        common_kws = self._common_kws
        method_kws = self._method_kws
        dimensions = len(parameters.infos())
        minimums = []
        maximums = []
        for pname, pinfo in parameters.infos().items():
            minimums.append(pinfo.minimum())
            maximums.append(pinfo.maximum())
        optimizer = pyswarms.single.general_optimizer.GeneralOptimizerPSO(
            common_kws['n_particles'],
            dimensions,
            method_kws['options'],
            method_kws['topology'],
            (np.array(minimums), np.array(maximums)),
            common_kws['bh_strategy'],
            common_kws['velocity_clamp'],
            common_kws['vh_strategy'],
            1.0,
            common_kws['ftol'],
            None)
        cost, pos = optimizer.optimize(
            _residual_scalar, common_kws['iters'],
            objective=objective, interpreter=interpreter)
        return cost, pos
