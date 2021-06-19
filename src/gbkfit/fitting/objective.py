
import numbers

import numpy as np

from gbkfit.utils import iterutils, parseutils


__all__ = ['Objective', 'objective_parser']


class Objective(parseutils.BasicParserSupport):

    @classmethod
    def load(cls, info, datasets, models):
        desc = parseutils.make_basic_desc(cls, 'objective')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['datasets', 'models'])
        return cls(datasets, models, **opts)

    def dump(self):
        return dict(
            wd=self._wd,
            wp=self._wp,
            wu=self._wu)

    def __init__(self, datasets, models, wd=False, wp=0.0, wu=1.0):
        datasets = iterutils.tuplify(datasets)
        if len(datasets) != len(models):
            raise RuntimeError(
                f"the number of datasets and models are not equal "
                f"({len(datasets)} != {len(models)})")
        n = len(models)
        self._models = models
        self._datasets = iterutils.make_list(n, dict())
        self._d_dataset_d_vector = iterutils.make_list(n, None)
        self._d_dataset_m_vector = iterutils.make_list(n, None)
        self._d_dataset_e_vector = iterutils.make_list(n, None)
        self._s_residual_vector = iterutils.make_list(n, (None, None))
        self._s_residual_scalar = iterutils.make_list(n, (None, None))
        self._s_counts = iterutils.make_list(n, (None, None))
        self._backends = iterutils.make_list(n, None)
        self._prepared = False
        if not iterutils.is_sequence(wp):
            wp = iterutils.make_tuple(n, wp)
        if not iterutils.is_sequence(wu):
            wu = iterutils.make_tuple(n, wu)
        self._wd = wd
        self._wp = wp
        self._wu = wu
        self._weights_d = iterutils.make_tuple(n, dict())
        self._weights_p = iterutils.make_tuple(n, dict())
        self._weights_u = iterutils.make_tuple(n, dict())
        if len(wp) != n:
            raise RuntimeError(
                f"the length of wp and the number of models are not equal "
                f"({len(wp)} != {n})")
        if len(wu) != n:
            raise RuntimeError(
                f"the length of wu and the number of models are not equal "
                f"({len(wu)} != {n})")
        for i in range(n):
            dataset = datasets[i]
            dmodel = models[i].dmodel()
            names_dat = tuple(dataset.keys())
            names_mdl = tuple(dmodel.onames())
            if set(names_dat) != set(names_mdl):
                raise RuntimeError(
                    f"dataset and dmodel are incompatible "
                    f"for item #{i} "
                    f"({names_dat} != {names_mdl})")
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
            self._datasets[i] = {name: dataset[name] for name in names_mdl}

            for name in names_mdl:
                # ...
                min_ = np.nanmin(dataset[name].data())
                max_ = np.nanmax(dataset[name].data())
                self._weights_d[i][name] = 1 / (max_ - min_) if wd else 1.0
                # ...
                if isinstance(wp[i], type(None)):
                    self._weights_p[i][name] = 0.0
                elif isinstance(wp[i], numbers.Real):
                    self._weights_p[i][name] = wp[i]
                elif isinstance(wp[i], dict):
                    self._weights_p[i][name] = wp[i].get(name, 0.0)
                # ...
                if isinstance(wu[i], type(None)):
                    self._weights_u[i][name] = 1.0
                elif isinstance(wu[i], numbers.Real):
                    self._weights_u[i][name] = wu[i]
                elif isinstance(wu[i], dict):
                    self._weights_u[i][name] = wu[i].get(name, 1.0)

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
            dataset = self._datasets[i]
            driver = self._models[i].driver()
            dmodel = self._models[i].dmodel()
            onames = dmodel.onames()
            nitems = len(onames)
            npix = dmodel.npix()
            nelem = npix * nitems
            dtype = dmodel.dtype()
            # Allocate memory and initialise backend
            self._d_dataset_d_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_m_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_e_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._s_residual_vector[i] = driver.mem_alloc_s(nelem, dtype)
            self._s_residual_scalar[i] = driver.mem_alloc_s(1, dtype)
            self._s_counts[i] = driver.mem_alloc_s(3, np.int32)
            self._backends[i] = driver.backend().make_objective(dtype)
            # Copy dataset to the device memory
            ielem = 0
            for j in range(nitems):
                data = dataset[onames[j]]
                data_d_1d = data.data().copy().ravel().astype(dtype)
                data_m_1d = data.mask().copy().ravel().astype(dtype)
                data_e_1d = data.error().copy().ravel().astype(dtype)
                slice_ = slice(ielem, ielem + npix)
                driver.mem_copy_h2d(
                    data_d_1d, self._d_dataset_d_vector[i][slice_])
                driver.mem_copy_h2d(
                    data_m_1d, self._d_dataset_m_vector[i][slice_])
                driver.mem_copy_h2d(
                    data_e_1d, self._d_dataset_e_vector[i][slice_])
                ielem += npix
        # We are ready to go!
        self._prepared = True

    def residual_nddata(self, params, out_extra=None):
        residuals = []
        self._residual_d(params, out_extra)
        for i, model in enumerate(self._models):
            npix = model.dmodel().npix()
            shape = model.dmodel().size()[::-1]
            names = model.dmodel().onames()
            driver = model.driver()
            h_residual_vector = self._s_residual_vector[i][0]
            d_residual_vector = self._s_residual_vector[i][1]
            driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
            ielem = 0
            residual = dict()
            for name in names:
                slice_ = slice(ielem, ielem + npix)
                residual[name] = h_residual_vector[slice_].reshape(shape)
                ielem += npix
            residuals.append(residual)
        return residuals

    def residual_vector(self, params, out_extra=None):
        residuals = []
        self._residual_d(params, out_extra)
        for i, model in enumerate(self._models):
            driver = model.driver()
            h_residual_vector = self._s_residual_vector[i][0]
            d_residual_vector = self._s_residual_vector[i][1]
            driver.mem_copy_d2h(d_residual_vector, h_residual_vector)
            residuals.append(h_residual_vector)
        return residuals

    def residual_scalar(self, params, out_extra=None):
        residuals = []
        self._residual_d(params, out_extra)
        for i, model in enumerate(self._models):
            driver = model.driver()
            d_residual_vector = self._s_residual_vector[i][1]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_abs(d_residual_vector, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            residuals.append(h_residual_scalar[0])
        return residuals

    def log_likelihood(self, params, out_extra=None):
        likelihoods = []
        self._residual_d(params, out_extra)
        for i, model in enumerate(self._models):
            driver = model.driver()
            d_residual_vector = self._s_residual_vector[i][1]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_pow(d_residual_vector, 2.0, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            likelihoods.append(-0.5 * h_residual_scalar[0] ** 2.0)
        return likelihoods

    def _residual_d(self, params, out_extra=None):
        if not self._prepared:
            self.prepare()
        models_data = self._models.evaluate_d(params, out_extra)
        for i, model in enumerate(self._models):
            npix = model.dmodel().npix()
            names = model.dmodel().onames()
            driver = model.driver()
            counts = self._s_counts[i]
            backend = self._backends[i]
            ielem = 0
            for name in names:
                # Grub references to model and data
                slice_ = slice(ielem, ielem + npix)
                mdl_d = models_data[i][name]['data'].ravel()
                mdl_m = models_data[i][name]['mask'].ravel()
                dat_d = self._d_dataset_d_vector[i][slice_]
                dat_m = self._d_dataset_m_vector[i][slice_]
                dat_e = self._d_dataset_e_vector[i][slice_]
                res = self._s_residual_vector[i][1][slice_]
                # Calculate weighting
                weight = 1.0
                weight *= self._weights_d[i][name]
                weight *= self._weights_u[i][name]
                if self._weights_p[i][name] != 0:
                    driver.mem_fill(counts[1], 0)
                    backend.count_pixels(dat_m, mdl_m, mdl_m.size, counts[1])
                    driver.mem_copy_d2h(counts[1], counts[0])
                    weight *= counts[0][1]**self._weights_p[i][name]
                # Calculate weighted residual
                res[:] = weight * (dat_d - mdl_d) / dat_e
                ielem += npix


objective_parser = parseutils.BasicParser(Objective)
