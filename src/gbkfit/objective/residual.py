
import time

from gbkfit.utils import iterutils
from .evaluate import ObjectiveEvaluate
from . import _detail


class ObjectiveResidual(ObjectiveEvaluate):

    def __init__(self, datasets, drivers, dmodels, gmodels):
        super().__init__(drivers, dmodels, gmodels)
        datasets = iterutils.tuplify(datasets)
        n = self.nitems()
        if len(datasets) != n:
            raise RuntimeError(
                f"the number of datasets and dmodels must be equal "
                f"({len(datasets)} != {n})")
        self._datasets = iterutils.make_list(n, dict())
        self._d_dataset_d_vector = iterutils.make_list(n, None)
        self._d_dataset_m_vector = iterutils.make_list(n, None)
        self._d_dataset_e_vector = iterutils.make_list(n, None)
        self._d_residual_vector = iterutils.make_list(n, None)
        self._h_residual_vector = iterutils.make_list(n, None)
        self._d_residual_nddata = iterutils.make_list(n, dict())
        self._h_residual_nddata = iterutils.make_list(n, dict())
        for i in range(n):
            dataset = datasets[i]
            dmodel = self.dmodels()[i]
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
        self._times_res_eval = []
        self._times_res_d2h = []
        self._prepared = False

    def datasets(self):
        return self._datasets

    def time_stats(self):
        time_stats = super().time_stats()
        return time_stats | _detail.time_stats(dict(
            res_eval=self._times_res_eval,
            res_d2h=self._times_res_d2h))

    def time_stats_reset(self):
        super().time_stats_reset()
        self._times_res_eval.clear()
        self._times_res_d2h.clear()

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
            self._d_residual_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._h_residual_vector[i] = driver.mem_alloc_h(nelem, dtype)
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
        self._prepared = True

    def residual_vector_d_data(self):
        assert self._prepared
        return self._d_residual_vector

    def residual_vector_h_data(self):
        assert self._prepared
        return self._h_residual_vector

    def residual_nddata_d_data(self):
        assert self._prepared
        return self._d_residual_nddata

    def residual_nddata_h_data(self):
        assert self._prepared
        return self._h_residual_nddata

    def residual_vector_d(self, params, out_extra=None):
        self._evaluate_residual_d(params, out_extra)
        return self._d_residual_vector

    def residual_vector_h(self, params, out_extra=None):
        self._evaluate_residual_h(params, out_extra)
        return self._h_residual_vector

    def residual_nddata_d(self, params, out_extra=None):
        self._evaluate_residual_d(params, out_extra)
        return self._d_residual_nddata

    def residual_nddata_h(self, params, out_extra=None):
        self._evaluate_residual_h(params, out_extra)
        return self._h_residual_nddata

    def _evaluate_residual_d(self, params, out_extra=None):
        if not self._prepared:
            self.prepare()
        models_data = self.evaluate_d(params, out_extra)
        t1 = time.time_ns()
        for i, dmodel in enumerate(self.dmodels()):
            npix = dmodel.npix()
            for j, name in enumerate(dmodel.onames()):
                # Grab references to model and data
                slice_ = slice(j * npix, (j + 1) * npix)
                mdl_d = models_data[i][name]['d'].ravel()
                mdl_m = models_data[i][name]['m'].ravel()
                dat_d = self._d_dataset_d_vector[i][slice_]
                dat_m = self._d_dataset_m_vector[i][slice_]
                dat_e = self._d_dataset_e_vector[i][slice_]
                res = self._d_residual_vector[i][slice_]
                # Calculate residual
                res[:] = dat_m * mdl_m * (dat_d - mdl_d) / dat_e
        t2 = time.time_ns()
        self._times_res_eval.append(t2 - t1)

    def _evaluate_residual_h(self, params, out_extra=None):
        self._evaluate_residual_d(params, out_extra)
        t1 = time.time_ns()
        for i, driver in enumerate(self.drivers()):
            d_data = self._d_residual_vector
            h_data = self._h_residual_vector
            driver.mem_copy_d2h(d_data, h_data)
        t2 = time.time_ns()
        self._times_res_d2h.append(t2 - t1)
