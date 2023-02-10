
from collections.abc import Mapping, Sequence

from gbkfit.utils import iterutils, parseutils, timeutils


class Objective:

    @classmethod
    def load(cls, info, datasets, model):
        desc = parseutils.make_basic_desc(cls, 'objective')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=[
                'datasets', 'model'])
        return cls(datasets, model, **opts)

    def dump(self):
        return dict(
            wp=self._wp,
            wu=self._wu)

    def __init__(
            self,
            datasets,
            model,
            wp:
            int | float |
            Sequence[int | float] |
            Sequence[Mapping[str, int | float]] = 0.0,
            wu:
            int | float |
            Sequence[int | float] |
            Sequence[Mapping[str, int | float]] = 1.0
    ):
        self._datasets = datasets = iterutils.tuplify(datasets)
        self._model = model
        n = self.nitems()
        if len(datasets) != n:
            raise RuntimeError(
                f"the number of datasets and models must be equal "
                f"({len(datasets)} != {n})")
        # These lists hold n x dataset data in 1d arrays
        self._d_dataset_d_vector = iterutils.make_list(n, None)
        self._d_dataset_m_vector = iterutils.make_list(n, None)
        self._d_dataset_e_vector = iterutils.make_list(n, None)
        # These lists hold n x residuals in 1d arrays
        self._d_residual_vector = iterutils.make_list(n, None)
        self._h_residual_vector = iterutils.make_list(n, None)
        # These lists hold n x residuals in nd arrays
        self._d_residual_nddata = iterutils.make_list(n, {})
        self._h_residual_nddata = iterutils.make_list(n, {})
        self._s_residual_scalar = iterutils.make_list(n, (None, None))
        # These lists hold n x 1d arrays of size 3
        self._d_pixel_counts = iterutils.make_list(n, None)
        self._h_pixel_counts = iterutils.make_list(n, None)
        # If scalars are given for weights,
        # use the same scalar value for all models.
        if isinstance(wp, (int, float)):
            wp = iterutils.make_tuple(n, wp)
        if isinstance(wu, (int, float)):
            wu = iterutils.make_tuple(n, wu)
        self._wp = wp
        self._wu = wu
        self._weights_p = iterutils.make_tuple(n, {})
        self._weights_u = iterutils.make_tuple(n, {})
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
            dmodel = self.model().dmodels()[i]
            keys_dat = tuple(dataset.keys())
            keys_mdl = tuple(dmodel.keys())
            if set(keys_dat) != set(keys_mdl):
                raise RuntimeError(
                    f"dataset and dmodel are incompatible "
                    f"for item #{i} "
                    f"({keys_dat} != {keys_mdl})")
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
            for key in keys_mdl:
                # wp
                if isinstance(wp[i], (int, float)):
                    self._weights_p[i][key] = wp[i]
                elif isinstance(wp[i], Mapping):
                    self._weights_p[i][key] = wp[i].get(key, 0.0)  # noqa
                # wu
                if isinstance(wu[i], (int, float)):
                    self._weights_u[i][key] = wu[i]
                elif isinstance(wu[i], Mapping):
                    self._weights_u[i][key] = wu[i].get(key, 1.0)  # noqa
        # One backend for each driver
        self._backends = iterutils.make_list(n, None)
        self._prepared = False

    def nitems(self):
        return self._model.nitems()

    def model(self):
        return self._model

    def datasets(self):
        return self._datasets

    def pdescs(self):
        return self.model().pdescs()

    def prepare(self):
        for i in range(self.nitems()):
            dataset = self.datasets()[i]
            driver = self.model().drivers()[i]
            dmodel = self.model().dmodels()[i]
            keys = dmodel.keys()
            shape = dmodel.size()[::-1]
            dtype = dmodel.dtype()
            npix = dmodel.npix()
            nelem = npix * len(keys)
            # Allocate memory as 1d arrays
            self._d_dataset_d_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_m_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_e_vector[i] = driver.mem_alloc_d(nelem, dtype)
            (self._h_residual_vector[i],
             self._d_residual_vector[i]) = driver.mem_alloc_s(nelem, dtype)
            self._s_residual_scalar[i] = driver.mem_alloc_s(1, dtype)
            for j, key in enumerate(keys):
                slice_ = slice(j * npix, (j + 1) * npix)
                # Copy dataset to the device memory
                data = dataset[key]
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
                self._d_residual_nddata[i][key] = \
                    self._d_residual_vector[i][slice_].reshape(shape)
                self._h_residual_nddata[i][key] = \
                    self._h_residual_vector[i][slice_].reshape(shape)
            # One backend for each driver
            self._backends[i] = driver.backends().objective(dmodel.dtype())
        self._prepared = True

    def residual_scalar(self, params, out_extra=None):
        t = timeutils.SimpleTimer('gds_scalar').start()
        d_residual_vectors = self.residual_vector_d(params, out_extra)
        residuals = []
        for i, driver in enumerate(self.model().drivers()):
            d_residual_vector = d_residual_vectors[i]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_abs(d_residual_vector, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            residuals.append(h_residual_scalar[0])
        t.stop()
        return residuals

    def log_likelihood(self, params, out_extra=None):
        t = timeutils.SimpleTimer('objective_loglike').start()
        d_residual_vectors = self.residual_vector_d(params, out_extra)
        log_likelihoods = []
        for i, driver in enumerate(self.model().drivers()):
            # TODO
            d_residual_vector = d_residual_vectors[i]
            h_residual_scalar = self._s_residual_scalar[i][0]
            d_residual_scalar = self._s_residual_scalar[i][1]
            driver.math_mul(d_residual_vector, d_residual_vector, out=d_residual_vector)
            driver.math_sum(d_residual_vector, out=d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            log_likelihoods.append(-0.5*h_residual_scalar[0])
        t.stop()
        return log_likelihoods

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
        t = timeutils.SimpleTimer('residual_eval').start()
        model_data = self._model.model_d(params, out_extra_model)
        for i, dmodel in enumerate(self._model.dmodels()):
            npix = dmodel.npix()
            for j, name in enumerate(dmodel.keys()):
                # Grab references to model and data
                slice_ = slice(j * npix, (j + 1) * npix)
                mdl_d = model_data[i][name]['d'].ravel()
                # mdl_m = model_data[i][name]['m'].ravel()
                dat_d = self._d_dataset_d_vector[i][slice_]
                dat_m = self._d_dataset_m_vector[i][slice_]
                dat_e = self._d_dataset_e_vector[i][slice_]
                res = self._d_residual_vector[i][slice_]
                # Calculate weights
                # TODO
                # Calculate residual
                # res[:] = dat_m * mdl_m * (dat_d - mdl_d) / dat_e
                res[:] = dat_m * (dat_d - mdl_d) / dat_e
        t.stop()

    def _residual_h(
            self, params, weighted=True, out_extra=None, out_extra_model=None):
        self._residual_d(params, weighted, out_extra, out_extra_model)
        t = timeutils.SimpleTimer('residual_d2h').start()
        for i, driver in enumerate(self._model.drivers()):
            d_data = self._d_residual_vector[i]
            h_data = self._h_residual_vector[i]
            driver.mem_copy_d2h(d_data, h_data)
        t.stop()


objective_parser = parseutils.BasicParser(Objective)
