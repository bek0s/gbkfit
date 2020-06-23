
import abc

import numpy as np

from gbkfit.utils import iterutils, miscutils, parseutils


class Objective:

    @classmethod
    def load(cls, info, driver, dataset, dmodel, gmodel):
        info.update(dict(
            driver=driver, dataset=dataset, dmodel=dmodel, gmodel=gmodel))
        cls_args = parseutils.parse_class_args(cls, info)
        return cls(**cls_args)

    def __init__(
            self, dataset, driver, dmodel, gmodel,
            residual=None, likelihood=None):
        super().__init__()
        self._dataset = dataset
        self._driver = driver
        self._dmodel = dmodel
        self._gmodel = gmodel
        self._residual = residual
        self._likelihood = likelihood
        self._nitems = 0
        self._names = []
        self._sizes = []
        self._steps = []
        self._cvals = []
        self._npixs = []
        self._dtype = np.float32

        self._d_dataset_d_vector = None
        self._d_dataset_m_vector = None
        self._d_dataset_e_vector = None
        self._s_residual_vector = None
        self._s_residual_scalar = None

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
            if data.dtype() != dmodel.dtype():
                raise RuntimeError(
                    f"dataset and dmodel have incompatible dtypes "
                )
            self._names.append(name)
            self._sizes.append(data.size())
            self._steps.append(data.step())
            self._cvals.append(data.cval())
            self._npixs.append(data.npix())
        self._nitems = len(self._names)
        self._names = tuple(self._names)
        self._sizes = tuple(self._sizes)
        self._steps = tuple(self._steps)
        self._cvals = tuple(self._cvals)
        self._npixs = tuple(self._npixs)

        self.prepare()

    def residual_count(self):
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
        # Convenience variables
        driver = self._driver
        nitems = self._nitems
        npixel = sum(self._npixs)
        dtype = self._dtype
        # Allocate host and device memory for residuals
        self._s_residual_vector = driver.mem_alloc_s(npixel, dtype)
        self._s_residual_scalar = driver.mem_alloc_s((1,), dtype)
        # Allocate device memory for dataset
        self._d_dataset_d_vector = driver.mem_alloc_d(npixel, dtype)
        self._d_dataset_m_vector = driver.mem_alloc_d(npixel, dtype)
        self._d_dataset_e_vector = driver.mem_alloc_d(npixel, dtype)
        # Copy dataset to the device memory
        ipix = 0
        for i in range(nitems):
            name = self._names[i]
            npix = self._npixs[i]
            data = self._dataset[name]
            data_d_1d = data.data().ravel()
            data_m_1d = data.mask().ravel()
            data_e_1d = data.error().ravel()
            slice_ = slice(ipix, ipix + npix)
            driver.mem_copy_h2d(data_d_1d, self._d_dataset_d_vector[slice_])
            driver.mem_copy_h2d(data_m_1d, self._d_dataset_m_vector[slice_])
            driver.mem_copy_h2d(data_e_1d, self._d_dataset_e_vector[slice_])
            ipix += npix
        # Dataset no longer needed in host memory
        self._dataset = None

    def residual_nddata(self, params, out_extra=None):
        # Convenience variables
        driver = self._driver
        nitems = self._nitems
        h_residual_vector = self._s_residual_vector[0]
        d_residual_vector = self._s_residual_vector[1]
        # Calculate and place residuals on the device arrays
        self._residual_d(params, out_extra)
        # Transfer residuals from device to host
        driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
        # Transform 1d residual to multiple nd residuals
        ipix = 0
        h_residual_nddata = []
        for i in range(nitems):
            size = self._sizes[i]
            npix = self._npixs[i]
            shape = size[::-1]
            slice_ = slice(ipix, ipix + npix)
            h_residual_nddata.append(h_residual_vector[slice_].reshape(shape))
            ipix += npix
        # Return the host memory
        return h_residual_nddata

    def residual_vector(self, params, out_extra=None):
        # Convenience variables
        driver = self._driver
        h_residual_vector = self._s_residual_vector[0]
        d_residual_vector = self._s_residual_vector[1]
        # Calculate and place residuals on the device arrays
        self._residual_d(params, out_extra)
        # Transfer residuals from device to host
        driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
        # Return the host memory
        return h_residual_vector

    def residual_scalar(self, params, out_extra=None):
        # Convenience variables
        driver = self._driver
        d_residual_vector = self._s_residual_vector[1]
        h_residual_scalar = self._s_residual_scalar[0]
        d_residual_scalar = self._s_residual_scalar[1]
        # Calculate and place residuals on the device arrays
        self._residual_d(params, out_extra)
        # Calculate the sum of absolute residuals
        driver.math_abs(d_residual_vector, out=d_residual_vector)
        driver.math_sum(d_residual_vector, out=d_residual_scalar)
        # Transfer residuals from device to host
        driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
        # Return the host memory
        return h_residual_scalar[0]

    def likelihood(self, params, out_extra=None):
        pass

    def log_likelihood(self, params, out_extra=None):
        pass

    def _residual_d(self, params, out_extra=None):
        # Convenience variables
        driver = self._driver
        gmodel = self._gmodel
        dmodel = self._dmodel
        nitems = self._nitems
        # Evaluate model
        models = dmodel.evaluate(driver, gmodel, params, out_extra)
        # Calculate residuals
        ipix = 0
        for i in range(nitems):
            name = self._names[i]
            npix = self._npixs[i]
            model = models[name].ravel()
            slice_ = slice(ipix, ipix + npix)
            data = self._d_dataset_d_vector[slice_]
            mask = self._d_dataset_m_vector[slice_]
            error = self._d_dataset_e_vector[slice_]
            resid = self._s_residual_vector[1][slice_]
            resid[:] = mask * (data - model) / error
            ipix += npix


class ObjectiveGroup:

    def __init__(self, objectives):
        super().__init__()
        self._objectives = iterutils.tuplify(objectives)
        pdescs, pdescs_mappings = miscutils.merge_dicts_and_make_mappings(
            [o.params() for o in objectives], 'model')
        rnames, rnames_mappings = miscutils.merge_lists_and_make_mappings(
            [o.residual_names() for o in objectives], 'model')
        self._pdescs = pdescs
        self._rnames = tuple(rnames)
        self._pdescs_mappings = tuple(pdescs_mappings)
        self._rnames_mappings = tuple(rnames_mappings)
        rsizes = []
        rnpixs = []
        for objective in objectives:
            rsizes.extend(objective.residual_sizes())
            rnpixs.extend(objective.residual_npixs())
        self._rsizes = tuple(rsizes)
        self._rnpixs = tuple(rnpixs)

    def params(self):
        return self._pdescs

    def residual_names(self):
        return self._rnames

    def residual_nddata(self, params, out_extra=None):
        residuals = []
        for objective in self._objectives:
            residuals.append(objective.residual_nddata(params, out_extra))
        return residuals

    def residual_vector(self, params, out_extra=None):
        residuals = []
        for objective in self._objectives:
            residuals.append(objective.residual_vector(params, out_extra))
        return residuals

    def residual_scalar(self, params, out_extra=None):
        residuals = []
        for objective in self._objectives:
            residuals.append(objective.residual_scalar(params, out_extra))
        return sum(residuals)

    def likelihood(self, params, out_extra=None):
        likelihoods = []
        for objective in self._objectives:
            likelihoods.append(objective.likelihood(params, out_extra))
        return sum(likelihoods)

    def log_likelihood(self, params, out_extra=None):
        log_likelihoods = []
        for objective in self._objectives:
            log_likelihoods.append(objective.log_likelihood(params, out_extra))
        return sum(log_likelihoods)


parser = parseutils.SimpleParser(Objective)
