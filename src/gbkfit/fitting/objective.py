
import abc

import copy

import numpy as np

from gbkfit.utils import iterutils

from gbkfit.utils import parseutils

import gbkfit.params.utils


class Objective(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    def __init__(self):
        pass

    def params(self):
        pass

    def residual(self):
        pass

    def residual_scalar(self):
        pass

    def residual_vector(self):
        pass

    def likelihood(self):
        pass

    def log_likelihood(self):
        pass


class JointObjective(Objective):

    @staticmethod
    def type():
        raise RuntimeError()

    @classmethod
    def load(cls, info):
        raise RuntimeError()

    def dump(self):
        raise RuntimeError()

    def __init__(self, objectives, weights=None):
        super().__init__()
        self._objectives = objectives
        self._weights = weights
        self._residual_ndgrid = None
        self._residual_vector = None

        for objective in self._objectives:
            objective.params()

        self._pdescs, self._pdescs_mappings = gbkfit.params.utils.merge_pdescs(
            [objective.params() for objective in self._objectives])

    def params(self):
        return self._pdescs

    def evaluate(self):
        pass

    def residual_ndgrid(self):
        pass

    def residual_scalar(self):
        pass

    def residual_vector(self):
        pass

    def likelihood(self):
        pass

    def log_likelihood(self):
        pass


class Residual(Objective):

    @classmethod
    def load(cls, info, *args, **kwargs):
        info.update(dict(
            dataset=kwargs['dataset'],
            driver=kwargs['driver'],
            dmodel=kwargs['dmodel'],
            gmodel=kwargs['gmodel']
        ))

        args = parseutils.parse_class_args(cls, info)
        return cls(**args)

    def __init__(self, dataset, driver, dmodel, gmodel):
        super().__init__()
        self._dataset = {}
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._dataset_1d_d = {}
        self._dataset_1d_m = {}
        self._dataset_1d_e = {}
        self._residual_1d = {}

        npixels = 0
        notfound = []
        notcompatible = []

        for oname in dmodel.onames():
            data = dataset.get(oname)
            print(dmodel.size(), dmodel.step(), dmodel.cval())
            if not data:
                notfound.append(oname)
                continue
            if (data.size() != dmodel.size()
                    or data.step() != dmodel.step()
                    or data.cval() != dmodel.cval()):
                notcompatible.append(oname)
                continue
            npixels += data.npix()
            self._dataset[oname] = data

        if notfound:
            raise RuntimeError()

        if notcompatible:
            raise RuntimeError(notcompatible)

        dtype = np.float32

        self._dataset_1d_d = driver.mem_alloc_s(npixels, dtype)
        self._dataset_1d_m = driver.mem_alloc_s(npixels, dtype)
        self._dataset_1d_e = driver.mem_alloc_s(npixels, dtype)
        self._residual_1d = driver.mem_alloc_s(npixels, dtype)

        exit()


    def params(self):
        return self._gmodel.params()

    def evaluate(self, params, out_extra=None):
        return self._dmodel.evaluate(self._driver, self._gmodel, params, out_extra)

    def residual(self, params):

        return self.evaluate(params, None)

        driver = self._driver
        offset = 0

        residual = self._residual_1d

        for mdl in self.evaluate(params, None):

            slice_ = slice(offset, offset + 10)

            dat = self._dataset_1d_d[slice_]
            msk = self._dataset_1d_m[slice_]
            err = self._dataset_1d_e[slice_]
            res = self._residual_1d[slice_]





            pass

        pass

    def residual_scalar(self):
        # return sum(abs(residual))
        pass

    def residual_vector(self):
        # return residual
        pass

    def _evaluate_d(self, params, out_extra=None):
        return self._dmodel.evaluate(self._driver, self._gmodel, params, out_extra)

    def _residual_d(self, params, out_extra=None):
        #

        model = self._evaluate_d(params, out_extra)

        for data_item, model_item in zip(self._data, model):
            pass




        pass



class Likelihood(Residual):

    def __init__(self, dataset, driver, dmodel, gmodel, weights=None):
        super().__init__(dataset, driver, dmodel, gmodel)

    def likelihood(self):
        pass

    def log_likelihood(self):
        pass


class LikelihoodGauss(Likelihood):

    @staticmethod
    def type():
        return 'likelihood_gauss'

    def dump(self):
        return None

    def __init__(self, dataset, driver, dmodel, gmodel, weights=None):
        super().__init__(dataset, driver, dmodel, gmodel, weights)

    def likelihood(self):
        pass


class JointLikelihood(Likelihood):
    pass

"""
fitter = gbkfit.fitter.fitters.dynesty.FitterDynesty()
objective = gbkfit.objective.likelihood.LikelihoodGauss(drivers, dmodels, gmodels, datasets)
objective = gbkfit.objective.likelihood.LikelihoodGauss(drivers, models, datasets)
objective = fitter.default_objective(drivers, models, datasets)
objective = fitter.default_objective(drivers, dmodels, gmodels, datasets)
fitter.fit(objective, params)
"""

parser = parseutils.TypedParser(Objective)
parser.register(Residual)
parser.register(LikelihoodGauss)
