
from gbkfit.model import ModelGroup
from gbkfit.utils import iterutils

import numpy as np
__all__ = ['Objective']

"""
from gbkfit.utils import iterutils, miscutils
    def evaluate_d(self, params, out_extra=None):
        d_data = []
        for i in range(self.nitems()):
            dmodel = self.dmodels()[i]
            gmodel = self.gmodels()[i]
            driver = self.drivers()[i]
            mapping = self._mappings[i]
            if out_extra is not None:
                out_extra.append({})
            d_data.append(dmodel.evaluate(
                driver, gmodel,
                {param: params[mapping[param]] for param in mapping},
                out_extra[-1] if out_extra is not None else None))
        return d_data
"""

"""
    def evaluate_h(self, params, out_extra=None):
        d_data = self.evaluate_d(params, out_extra)
        h_data = self._h_data
        for i in range(self.nitems()):
            for key in d_data[i]:
                h_data[i][key] = self._drivers[i].mem_copy_d2h(
                    d_data[i][key], h_data[i][key])
        return h_data
"""

"""
dmodels = iterutils.tuplify(dmodels)
        gmodels = iterutils.tuplify(gmodels)
        drivers = iterutils.tuplify(drivers)
        ndmodels = len(dmodels)
        ngmodels = len(gmodels)
        ndrivers = len(drivers)
        self._drivers = drivers
        self._dmodels = dmodels
        self._gmodels = gmodels
        if not (ndmodels == ngmodels == ndrivers):
            raise RuntimeError(
                f"the number of "
                f"gmodels ({ngmodels}), "
                f"dmodels ({ndmodels}), and "
                f"drivers ({ndrivers}) must be equal")
        # ...
        pdescs, mappings = miscutils.merge_dicts_and_make_mappings(
            [g.params() for g in gmodels], 'model')
        self._pdescs = pdescs
        self._mappings = mappings
        # ...
        self._h_data = [{key: None for key in mdl.onames()} for mdl in dmodels]
"""

class Objective:

    def __init__(self, datasets, models):
        datasets = iterutils.tuplify(datasets)
        models = iterutils.tuplify(models)
        if len(datasets) != len(models):
            raise RuntimeError(
                f"the number of datasets and models are not equal "
                f"({len(datasets)} != {len(models)})")
        n = len(models)
        self._datasets = iterutils.make_list((n,), dict(), True)
        self._models = models = ModelGroup(models)
        self._nitems = [None] * n
        self._names = [None] * n
        self._sizes = [None] * n
        self._steps = [None] * n
        self._zeros = [None] * n
        self._npixs = [None] * n
        self._d_dataset_d_vector = [None] * n
        self._d_dataset_m_vector = [None] * n
        self._d_dataset_e_vector = [None] * n
        self._s_residual_vector = [None] * n
        self._s_residual_scalar = [None] * n
        self._s_counts = [None] * n
        self._backends = [None] * n
        self._prepared = False
        for i in range(n):
            dataset = datasets[i]
            dmodel = models[i].dmodel()
            if set(dataset.keys()) != set(dmodel.onames()):
                names_mdl = dmodel.onames()
                names_dat = tuple([n for n in names_mdl if n in dataset])
                raise RuntimeError(
                    f"dataset and dmodel are incompatible "
                    f"for item #{i} "
                    f"({names_mdl} != {names_dat})")
            if dataset.dtype != dmodel.dtype():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible dtypes "
                    f"for item #{i} "
                    f"({dataset.dtype} != {dmodel.dtype()})")
            if dataset.size() != dmodel.size():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible sizes "
                    f"for item #{i} "
                    f"({dataset.size()} != {dmodel.size()})")
            if dataset.step() != dmodel.step():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible steps "
                    f"for item #{i} "
                    f"({dataset.step()} != {dmodel.step()})")
            if dataset.zero() != dmodel.zero():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible zeros "
                    f"for item #{i} "
                    f"({dataset.zero()} != {dmodel.zero()})")
            self._datasets[i] = {name: dataset[name] for name in dmodel.onames()}
            self._nitems[i] = len(dmodel.onames())
            self._names[i] = dmodel.onames()
            self._sizes[i] = dataset.size()
            self._steps[i] = dataset.step()
            self._zeros[i] = dataset.zero()
            self._npixs[i] = dataset.npix()

    def datasets(self):
        return self._datasets

    def models(self):
        return self._models

    def params(self):
        return self._models.pdescs()

    def pdescs(self):
        return self._models.pdescs()

    def prepare(self):

        for i in range(len(self._models)):
            driver = self._models[i].driver()
            dmodel = self._models[i].dmodel()
            nitems = self._nitems[i]
            npixs = self._npixs[i] * nitems
            dtype = dmodel.dtype()

            # Allocate memory and initialise backend
            self._d_dataset_d_vector[i] = driver.mem_alloc_d(npixs, dtype)
            self._d_dataset_m_vector[i] = driver.mem_alloc_d(npixs, dtype)
            self._d_dataset_e_vector[i] = driver.mem_alloc_d(npixs, dtype)
            self._s_residual_vector[i] = driver.mem_alloc_s(npixs, dtype)
            self._s_residual_scalar[i] = driver.mem_alloc_s(1, dtype)
            self._s_counts[i] = driver.mem_alloc_s(3, np.int32)
            self._backends[i] = driver.make_objective(dtype)

            # Copy dataset to the device memory
            ipix = 0
            for j in range(nitems):
                name = self._names[i][j]
                npix = self._npixs[i]
                data = self._datasets[i][name]
                data_d_1d = data.data().copy().ravel().astype(dtype)
                data_m_1d = data.mask().copy().ravel().astype(dtype)
                data_e_1d = data.error().copy().ravel().astype(dtype)
                slice_ = slice(ipix, ipix + npix)
                driver.mem_copy_h2d(
                    data_d_1d, self._d_dataset_d_vector[i][slice_])
                driver.mem_copy_h2d(
                    data_m_1d, self._d_dataset_m_vector[i][slice_])
                driver.mem_copy_h2d(
                    data_e_1d, self._d_dataset_e_vector[i][slice_])
                ipix += npix
        # We are ready to go!
        self._prepared = True

    def residual_nddata(self, params, weighted=True, out_extra=None):
        self._residual_d(params, weighted, out_extra)
        residuals = []
        for i in range(len(self._models)):
            driver = self._models[i].driver()
            nitems = self._nitems[i]
            h_residual_vector = self._s_residual_vector[i][0]
            d_residual_vector = self._s_residual_vector[i][1]
            driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
            ipix = 0
        #   h_residual_nddata = []
            h_residual_nddata = {}
            for j in range(nitems):
                name = self._names[i][j]  # !!!
                size = self._sizes[i]
                npix = self._npixs[i]
                shape = size[::-1]
                slice_ = slice(ipix, ipix + npix)
            #   h_residual_nddata.append(h_residual_vector[slice_].reshape(shape))
                h_residual_nddata[name] = h_residual_vector[slice_].reshape(shape)
                ipix += npix
            residuals.append(h_residual_nddata)
        return residuals

    def residual_vector(self, params, weighted=True, out_extra=None):
        self._residual_d(params, weighted, out_extra)
        residuals = []
        for i in range(len(self._models)):
            driver = self._models[i].driver()
            h_residual_vector = self._s_residual_vector[i][0]
            d_residual_vector = self._s_residual_vector[i][1]
            driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
            residuals.append(h_residual_vector)
        return residuals

    def residual_scalar(self, params, weighed=True, out_extra=None):
        self._residual_d(params, weighed, out_extra)
        residuals = []
        for i in range(len(self._models)):
            driver = self._models[i].driver()
            h_residual_vector = self._s_residual_vector[i][0]
            d_residual_vector = self._s_residual_vector[i][1]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_abs(d_residual_vector, out=d_residual_vector)
            #driver.math_mul(d_residual_vector, d_residual_vector, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            residuals.append(h_residual_scalar[0])
        print('chi2:', residuals)
        print('values:', params)
        return residuals

    def log_likelihood(self, params, out_extra=None):
        return -0.5 * self.residual_scalar(params, out_extra) ** 2

    def _residual_d(self, params, weighted=True, out_extra=None):
        if not self._prepared:
            self.prepare()
        models_out = self._models.evaluate_d(params, out_extra)
        for i, model in enumerate(self._models):
            driver = model.driver()
            backend = self._backends[i]
            counts = self._s_counts[i]
            npix = self._npixs[i]
            ipix = 0
            for name in self._names[i]:
                slice_ = slice(ipix, ipix + npix)
                mdl_d = models_out[i][name]['data'].ravel()
                mdl_m = models_out[i][name]['mask'].ravel()
                dat_d = self._d_dataset_d_vector[i][slice_]
                dat_m = self._d_dataset_m_vector[i][slice_]
                dat_e = self._d_dataset_e_vector[i][slice_]
                res = self._s_residual_vector[i][1][slice_]
                backend.count_pixels(dat_m, mdl_m, mdl_m.size, counts[1])
                driver.mem_copy_d2h(counts[1], counts[0])
                res[:] = (dat_d - mdl_d) / dat_e
                ipix += npix
