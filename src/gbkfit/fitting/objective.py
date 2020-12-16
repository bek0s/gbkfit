
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

        self._models = models = ModelGroup(iterutils.tuplify(models))
        if len(datasets) != models.nmodels():
            raise RuntimeError(
                f"the number of dataset and model are not equal"
                f"({len(datasets)} != {len(models)})")
        n = models.nmodels()
        self._datasets = iterutils.make_list((n,), dict(), True)
        self._nitems = iterutils.make_list((n,), 0, True)
        self._names = iterutils.make_list((n,), list(), True)
        self._sizes = iterutils.make_list((n,), list(), True)
        self._steps = iterutils.make_list((n,), list(), True)
        self._zeros = iterutils.make_list((n,), list(), True)
        self._npixs = iterutils.make_list((n,), list(), True)
        self._d_dataset_d_vector = [None] * n
        self._d_dataset_m_vector = [None] * n
        self._d_dataset_e_vector = [None] * n
        self._s_residual_vector = [None] * n
        self._s_residual_scalar = [None] * n
        for i in range(n):
            dataset = datasets[i]
            dmodel = models.models()[i].dmodel()
            if dataset.dtype != dmodel.dtype():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible dtypes "
                    f"({dataset.dtype} != {dmodel.dtype()})")
            for name in dmodel.onames():
                data = dataset.get(name)
                if data is None:
                    raise RuntimeError(
                        f"could not find dataset "
                        f"for item '{name}' required by dmodel")
                if data.size() != dmodel.size():
                    raise RuntimeError(
                        f"dataset and dmodel have incompatible sizes "
                        f"for item '{name}' ({data.size()} != {dmodel.size()})")
                if data.step() != dmodel.step():
                    raise RuntimeError(
                        f"dataset and dmodel have incompatible steps "
                        f"for item '{name}' ({data.step()} != {dmodel.step()})")
                if data.zero() != dmodel.zero():
                    raise RuntimeError(
                        f"dataset and dmodel have incompatible zeros "
                        f"for item '{name}' ({data.zero()} != {dmodel.zero()})")
                self._nitems[i] += 1
                self._names[i] += [name]
                self._sizes[i] += [data.size()]
                self._steps[i] += [data.step()]
                self._zeros[i] += [data.zero()]
                self._npixs[i] += [data.npix()]
                self._datasets[i][name] = data

        self._backends = [None] * n
        self._s_counts = [None] * n

        self._backend = None
        self._counts = None
        self.prepare()


    def params(self):
        return self._models.pdescs()

    def pdescs(self):
        return self._models.pdescs()

    def datasets(self):
        return self._datasets

    def model(self):
        return self._models

    def prepare(self):
        for i in range(self._models.nmodels()):
            driver = self._models.models()[i].driver()
            dmodel = self._models.models()[i].dmodel()
            nitems = self._nitems[i]
            npixs = self._npixs[i]
            dtype = dmodel.dtype()


            self._backends[i] = driver.make_backend_objective(dtype)
            self._s_counts[i] = driver.mem_alloc_s(3, np.int32)

            # Allocate host and device memory for residuals
            self._s_residual_vector[i] = driver.mem_alloc_s(sum(npixs), dtype)
            self._s_residual_scalar[i] = driver.mem_alloc_s((1,), dtype)
            # Allocate device memory for dataset
            self._d_dataset_d_vector[i] = driver.mem_alloc_d(sum(npixs), dtype)
            self._d_dataset_m_vector[i] = driver.mem_alloc_d(sum(npixs), dtype)
            self._d_dataset_e_vector[i] = driver.mem_alloc_d(sum(npixs), dtype)
            # Copy dataset to the device memory
            ipix = 0
            for j in range(nitems):
                name = self._names[i][j]
                npix = self._npixs[i][j]
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
        # Dataset references no longer needed
        #self._datasets = None

    def residual_nddata(self, params, weighted=True, out_extra=None):
        self._residual_d(params, weighted, out_extra)
        residuals = []
        for i in range(self._models.nmodels()):
            driver = self._models.models()[i].driver()
            nitems = self._nitems[i]
            h_residual_vector = self._s_residual_vector[i][0]
            d_residual_vector = self._s_residual_vector[i][1]
            driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
            ipix = 0
        #   h_residual_nddata = []
            h_residual_nddata = {}
            for j in range(nitems):
                name = self._names[i][j]  # !!!
                size = self._sizes[i][j]
                npix = self._npixs[i][j]
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
        for i in range(self._models.nmodels()):
            driver = self._models.models()[i].driver()
            h_residual_vector = self._s_residual_vector[i][0]
            d_residual_vector = self._s_residual_vector[i][1]
            driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
            residuals.append(h_residual_vector)
        return residuals

    def residual_scalar(self, params, weighed=True, out_extra=None):
        self._residual_d(params, weighed, out_extra)
        residuals = []
        for i in range(self._models.nmodels()):
            driver = self._models.models()[i].driver()
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
        models = self._models.evaluate_d(params, out_extra)
        for i in range(self._models.nmodels()):
            ipix = 0
            for j in range(self._nitems[i]):
                name = self._names[i][j]
                npix = self._npixs[i][j]
                model_d = models[i][name]['data'].ravel()
                model_m = models[i][name]['mask'].ravel()
                slice_ = slice(ipix, ipix + npix)
                dataset_d = self._d_dataset_d_vector[i][slice_]
                dataset_m = self._d_dataset_m_vector[i][slice_]
                dataset_e = self._d_dataset_e_vector[i][slice_]


                resid = self._s_residual_vector[i][1][slice_]


                self._backends[i].count_pixels(dataset_m, model_m, model_m.size, self._s_counts[i][1])

                resid[:] = dataset_m * (dataset_d - model_d) / dataset_e


                """
                if weighted:
                    resid[:] = mask * (data - model) / error
                else:
                    resid[:] = mask * (data - model)
                """
                ipix += npix
