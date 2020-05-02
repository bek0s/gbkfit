
import abc

import copy

import numpy as np

from gbkfit.utils import iterutils

from gbkfit.utils import miscutils, parseutils

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

    def residual_names(self):
        pass

    def residual_list(self, params, out=None, out_extra=None):
        pass

    def residual_vector(self, params, out=None, out_extra=None):
        pass

    def residual_scalar(self, params, out=None, out_extra=None):
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

    def __init__(self, objectives):
        super().__init__()
        #
        self._objectives = iterutils.tuplify(objectives)
        # ...
        self._rnames, self._rnames_mappings = \
            miscutils.merge_lists_and_make_mappings(
                [o.residual_names() for o in self._objectives], 'model')
        # ...
        self._pdescs, self._pdescs_mappings = \
            miscutils.merge_dicts_and_make_mappings(
                [o.params() for o in self._objectives], 'model')

    def params(self):
        return self._pdescs

    def residual_names(self):
        return self._rnames

    def residual_list(self, params, out=None, out_extra=None):

        if out is None:
            out = list()
            for objective in self._objectives:
                pass

        for objective in self._objectives:
            objective.residual_dict(params, out, out_extra)

        return out

    def residual_vector(self, params, out=None, out_extra=None):

        if out is None:
            pass

        for objective in self._objectives:
            objective.residual_vector(params, None, out_extra)

        return out

    def residual_scalar(self, params, out=None, out_extra=None):

        if out is None:
            out = np.empty()

        for objective in self._objectives:
            objective.residual_scalar(params, out, out_extra)

        return out

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
        self._dataset = dataset
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._nitems = 0
        self._names = []
        self._sizes = []
        self._steps = []
        self._npixs = []
        self._dtype = np.float32
        self._d_dataset_d = []
        self._d_dataset_m = []
        self._d_dataset_e = []
        self._d_residual = []
        self._s_residual_sum = None
        # Make sure dataset and dmodel are compatible
        # While doing so, put together some information for convenience.
        for name in dmodel.onames():
            data = dataset.get(name)
            if data is None:
                raise RuntimeError(
                    f"could not find dataset item '{name}' required by dmodel")
            if data.size() != dmodel.size():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible sizes "
                    f"for item '{name}' ({data.size()} != {dmodel.size()})")
            if data.step() != dmodel.step():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible steps "
                    f"for item '{name}' ({data.step()} != {dmodel.step()})")
            if data.cval() != dmodel.cval():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible cvals "
                    f"for item '{name}' ({data.cval()} != {dmodel.cval()})")
            self._names.append(name)
            self._sizes.append(data.size())
            self._steps.append(data.step())
            self._npixs.append(data.npix())
        self._nitems = len(self._names)
        self._names = tuple(self._names)
        self._sizes = tuple(self._sizes)
        self._steps = tuple(self._steps)
        self._npixs = tuple(self._npixs)
        self.prepare()

    def nresiduals(self):
        return self._nitems

    def residual_names(self):
        return self._names

    def residual_sizes(self):
        return self._sizes

    def residual_steps(self):
        return self._steps

    def residual_npixs(self):
        return self._npixs

    def dtype(self):
        return self._dtype

    def params(self):
        return self._gmodel.params()

    def prepare(self):
        driver = self._driver
        for i in range(self._nitems):
            data = self._dataset[self._names[i]]
            shape = self._sizes[i][::-1]
            self._d_dataset_d.append(driver.mem_copy_h2d(data.data()))
            self._d_dataset_m.append(driver.mem_copy_h2d(data.mask()))
            self._d_dataset_e.append(driver.mem_copy_h2d(data.error()))
            self._d_residual.append(driver.mem_alloc_d(shape, self.dtype()))
        self._s_residual_sum = driver.mem_alloc_s((self._nitems,), self.dtype())

    def residual_dict(self, params, out=None, out_extra=None):
        driver = self._driver
        # Allocate host storage, if not provided
        if out is None:
            out = dict()
            for i in range(self._nitems):
                name = self._names[i]
                size = self._sizes[i]
                shape = size[::-1]
                out[name] = driver.mem_alloc_h(shape, self.dtype())
        # Calculate and place residuals on the device arrays
        self._residual_d(params, out_extra)
        # Transfer data from device to host
        for i in range(self._nitems):
            name = self._names[i]
            out[name] = driver.mem_copy_d2h(self._d_residual[i], out[name])
        return out

    def residual_vector(self, params, out=None, out_extra=None):
        driver = self._driver
        # Allocate host storage, if not provided
        if out is None:
            out = driver.mem_alloc_h(sum(self._npixs), self.dtype())
        # Calculate and place residuals on the device arrays
        self._residual_d(params, out_extra)
        # Transfer data from device to host
        ipix = 0
        for i in range(self._nitems):
            npix = self._npixs[i]
            driver.mem_copy_d2h(self._d_residual[i], out[ipix:npix])
            ipix += npix
        return out

    def residual_scalar(self, params, out_extra=None):
        driver = self._driver
        d_residual = self._d_residual
        h_residual_sum = self._s_residual_sum[0]
        d_residual_sum = self._s_residual_sum[1]
        # Calculate and place residuals on the device arrays
        self._residual_d(params, out_extra)
        # Calculate the sum of absolute values and transfer it to host
        for i in range(self._nitems):
            driver.math_abs(d_residual[i], out=d_residual[i])
            driver.math_sum(d_residual[i], out=d_residual_sum[i])
        driver.mem_copy_d2h(d_residual_sum, h_residual_sum)
        return np.sum(h_residual_sum)

    def _residual_d(self, params, out_extra=None):
        driver = self._driver
        gmodel = self._gmodel
        dmodel = self._dmodel
        models = dmodel.evaluate(driver, gmodel, params, out_extra)
        for i in range(self._nitems):
            model = models[self._names[i]]
            data = self._d_dataset_d[i]
            mask = self._d_dataset_m[i]
            error = self._d_dataset_e[i]
            resid = self._d_residual[i]
            resid[:] = mask * (data - model) / error


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
