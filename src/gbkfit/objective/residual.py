
import numbers
import time

import numpy as np

from gbkfit.utils import iterutils, parseutils, timeutils
from .model import ObjectiveModel


class ObjectiveResidual(ObjectiveModel):

    @classmethod
    def load(cls, info, datasets, drivers, dmodels, gmodels):
        desc = parseutils.make_basic_desc(cls, ' goodness objective')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=[
                'datasets', 'drivers', 'dmodels', 'gmodels'])
        return cls(datasets, drivers, dmodels, gmodels, **opts)

    def dump(self):
        return dict(
            wd=self._wd,
            wp=self._wp,
            wu=self._wu)

    def __init__(
            self, datasets, drivers, dmodels, gmodels,
            wd=False, wp=0.0, wu=1.0):
        super().__init__(drivers, dmodels, gmodels)
        self._datasets = datasets = iterutils.tuplify(datasets)
        if len(datasets) != len(dmodels):
            raise RuntimeError(
                f"the number of datasets and dmodels must be equal "
                f"({len(datasets)} != {len(dmodels)})")
        n = self.nitems()
        self._d_dataset_d_vector = iterutils.make_list(n, None)
        self._d_dataset_m_vector = iterutils.make_list(n, None)
        self._d_dataset_e_vector = iterutils.make_list(n, None)
        self._d_residual_vector = iterutils.make_list(n, None)
        self._h_residual_vector = iterutils.make_list(n, None)
        self._d_residual_nddata = iterutils.make_list(n, dict())
        self._h_residual_nddata = iterutils.make_list(n, dict())
        self._s_counts = iterutils.make_list(n, (None, None))
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
                f"the length of wp and the number of datasets are not equal "
                f"({len(wp)} != {n})")
        if len(wu) != n:
            raise RuntimeError(
                f"the length of wu and the number of datasets are not equal "
                f"({len(wu)} != {n})")
        for i in range(n):
            dataset = datasets[i]
            dmodel = dmodels[i]
            names_dat = tuple(dataset.keys())
            names_mdl = tuple(dmodel.keys())
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
            for name in names_mdl:
                min_ = np.nanmin(dataset[name].data())
                max_ = np.nanmax(dataset[name].data())
                self._weights_d[i][name] = 1 / (max_ - min_) if wd else 1.0
                if isinstance(wp[i], type(None)):
                    self._weights_p[i][name] = 0.0
                elif isinstance(wp[i], numbers.Real):
                    self._weights_p[i][name] = wp[i]
                elif isinstance(wp[i], dict):
                    self._weights_p[i][name] = wp[i].get(name, 0.0)
                if isinstance(wu[i], type(None)):
                    self._weights_u[i][name] = 1.0
                elif isinstance(wu[i], numbers.Real):
                    self._weights_u[i][name] = wu[i]
                elif isinstance(wu[i], dict):
                    self._weights_u[i][name] = wu[i].get(name, 1.0)
        self._backends = iterutils.make_list(n, None)
        self._prepared = False

    def datasets(self):
        return self._datasets

    def prepare(self):
        for i in range(self.nitems()):
            dataset = self.datasets()[i]
            driver = self.drivers()[i]
            dmodel = self.dmodels()[i]
            onames = dmodel.onames()
            shape = dmodel.size()[::-1]
            dtype = dmodel.dtype()
            npix = dmodel.npix()
            nelem = npix * len(onames)
            # Allocate memory and initialise backend
            self._d_dataset_d_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_m_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_e_vector[i] = driver.mem_alloc_d(nelem, dtype)
            (self._h_residual_vector[i],
             self._d_residual_vector[i]) = driver.mem_alloc_s(nelem, dtype)
            for j, name in enumerate(onames):
                slice_ = slice(j * npix, (j + 1) * npix)
                # Copy dataset to the device memory
                data = dataset[name]
                data_d_1d = data.data().copy().ravel().astype(dtype)
                data_m_1d = data.mask().copy().ravel().astype(dtype)
                data_e_1d = data.error().copy().ravel().astype(dtype)
                driver.mem_copy_h2d(
                    data_d_1d, self._d_dataset_d_vector[i][slice_])
                driver.mem_copy_h2d(
                    data_m_1d, self._d_dataset_m_vector[i][slice_])
                driver.mem_copy_h2d(
                    data_e_1d, self._d_dataset_e_vector[i][slice_])
                # Create nd data views to the actual memory
                self._d_residual_nddata[i][name] = self._d_residual_vector[i][
                    slice_].reshape(shape)
                self._h_residual_nddata[i][name] = self._h_residual_vector[i][
                    slice_].reshape(shape)
            self._backends[i] = driver.backend().make_objective(dmodel.dtype())
        self._prepared = True

    def residual_vector_d(
            self, params, weighted=True, out_extra=None, out_extra_model=None):
        self._residual_d(params, weighted, out_extra, out_extra_model)
        return self._d_residual_vector

    def residual_vector_h(
            self, params, weighted=True, out_extra=None, out_extra_model=None):
        self._residual_h(params, weighted, out_extra, out_extra_model)
        return self._h_residual_vector

    def residual_nddata_d(
            self, params, weighted=True, out_extra=None, out_extra_model=None):
        self._residual_d(params, weighted, out_extra, out_extra_model)
        return self._d_residual_nddata

    def residual_nddata_h(
            self, params, weighted=True, out_extra=None, out_extra_model=None):
        self._residual_h(params, weighted, out_extra, out_extra_model)
        return self._h_residual_nddata

    def _residual_d(
            self, params, weighted=True, out_extra=None, out_extra_model=None):
        if not self._prepared:
            self.prepare()
        t1 = time.time_ns()
        t = timeutils.SimpleTimer('res_eval')
        t.start()
        model_data = self.model_d(params, out_extra_model)
        for i, dmodel in enumerate(self.dmodels()):
            npix = dmodel.npix()
            for j, name in enumerate(dmodel.onames()):
                # Grab references to model and data
                slice_ = slice(j * npix, (j + 1) * npix)
                mdl_d = model_data[i][name]['d'].ravel()
                mdl_m = model_data[i][name]['m'].ravel()
                dat_d = self._d_dataset_d_vector[i][slice_]
                dat_m = self._d_dataset_m_vector[i][slice_]
                dat_e = self._d_dataset_e_vector[i][slice_]
                res = self._d_residual_vector[i][slice_]
                # Calculate weights
                # TODO
                # Calculate residual
                res[:] = dat_m * mdl_m * (dat_d - mdl_d) / dat_e
        t.stop()
        t2 = time.time_ns()
        self.time_stats_samples(False)['res_eval'].append(t2 - t1)

    def _residual_h(
            self, params, weighted=True, out_extra=None, out_extra_model=None):
        self._residual_d(params, weighted, out_extra, out_extra_model)
        t1 = time.time_ns()
        t = timeutils.SimpleTimer('res_d2h')
        t.start()
        for i, driver in enumerate(self.drivers()):
            d_data = self._d_residual_vector[i]
            h_data = self._h_residual_vector[i]
            driver.mem_copy_d2h(d_data, h_data)
        t.stop()
        t2 = time.time_ns()
        self.time_stats_samples(False)['res_d2h'].append(t2 - t1)


residual_objective_parser = parseutils.BasicParser(ObjectiveResidual)
