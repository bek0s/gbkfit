
from abc import ABC
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.model import ModelGroup
from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils, parseutils, timeutils


class Objective(parseutils.BasicSerializable, ABC):

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'Objective':
        datasets = kwargs.get('datasets')
        models = kwargs.get('models')
        desc = parseutils.make_basic_desc(cls, 'objective')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['datasets', 'models'])
        return cls(datasets, models, **opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            wu=self._wu)

    def __init__(
            self,
            datasets: Dataset | Sequence[Dataset],
            models: ModelGroup,
            wu:
            Real |
            Mapping[str, Real] |
            Sequence[Real] |
            Sequence[Mapping[str, Real]] = 1.0
    ):
        self._datasets = datasets = iterutils.tuplify(datasets)
        self._models = models
        n = models.nmodels()
        if len(datasets) != n:
            raise RuntimeError(
                f"the number of datasets and models are not equal "
                f"({len(datasets)} != {n})")
        # These lists hold n x dataset data in 1d arrays
        self._d_dataset_d_vector = iterutils.make_list(n, None)
        self._d_dataset_m_vector = iterutils.make_list(n, None)
        self._d_dataset_e_vector = iterutils.make_list(n, None)
        # These lists hold n x dataset data in nd arrays.
        # These are views to the above, just for convenience.
        self._d_dataset_d_nddata = iterutils.make_list(n, {})
        self._d_dataset_m_nddata = iterutils.make_list(n, {})
        self._d_dataset_e_nddata = iterutils.make_list(n, {})
        # These lists hold n x residual data in 1d arrays
        self._h_residual_vector = iterutils.make_list(n, None)
        self._d_residual_vector = iterutils.make_list(n, None)
        # These lists hold n x residual data in nd arrays
        # These are views to the above, just for convenience.
        self._h_residual_nddata = iterutils.make_list(n, {})
        self._d_residual_nddata = iterutils.make_list(n, {})
        # These lists hold n x 1d arrays of size 1
        self._h_residual_scalar = iterutils.make_list(n, None)
        self._d_residual_scalar = iterutils.make_list(n, None)
        # If we have one weight (or one dict of weights) but
        # multiple datasets, replicate the value multiple times.
        if isinstance(wu, (Real, Mapping)):
            wu = iterutils.make_tuple(n, wu)
        if len(datasets) != len(wu):
            raise RuntimeError(
                f"the number of datasets and the length of wu are not equal "
                f"({len(datasets)} != {len(wu)})")
        # We will use these weight variables when dumping this object.
        # This is in both a compact and an ambiguity-free form.
        self._wu = wu
        # These variables will contain the fully-expanded weights,
        # and they are most convenient to work with.
        self._weights_u = iterutils.make_tuple(n, {})
        for i in range(n):
            dataset = datasets[i]
            dmodel = self.models().models()[i].dmodel()
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
            # Expand weights fully
            for key in keys_mdl:
                if isinstance(wu[i], Real):
                    self._weights_u[i][key] = wu[i]
                elif isinstance(wu[i], Mapping):
                    self._weights_u[i][key] = wu[i].get(key, 1.0)
        # One backend for each driver
        self._backends = iterutils.make_list(n, None)
        # This class is lazily initialized
        self._prepared = False

    def nitems(self) -> int:
        return self._models.nmodels()

    def datasets(self) -> tuple[Dataset, ...]:
        return self._datasets

    def models(self) -> ModelGroup:
        return self._models

    def pdescs(self) -> dict[str, ParamDesc]:
        return self.models().pdescs()

    def prepare(self) -> None:
        for i in range(self.nitems()):
            dataset = self.datasets()[i]
            driver = self.models().models()[i].driver()
            dmodel = self.models().models()[i].dmodel()
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
            (self._h_residual_scalar[i],
             self._d_residual_scalar[i]) = driver.mem_alloc_s(1, dtype)
            # Populate allocated arrays and create views
            for j, key in enumerate(keys):
                data = dataset[key]
                slice_ = slice(j * npix, (j + 1) * npix)
                # Copy measurement data to the internal 1d array,
                # and create nd array view
                driver.mem_copy_h2d(
                    data.data().ravel().astype(dtype),
                    self._d_dataset_d_vector[i][slice_])
                self._d_dataset_d_nddata[i][key] = \
                    self._d_dataset_d_vector[i][slice_].reshape(shape)
                # Copy mask data to the internal 1d array,
                # and create nd array view
                if data.mask() is not None:
                    driver.mem_copy_h2d(
                        data.mask().ravel().astype(dtype),
                        self._d_dataset_m_vector[i][slice_])
                    self._d_dataset_m_nddata[i][key] = \
                        self._d_dataset_m_vector[i][slice_].reshape(shape)
                # Copy uncertainty data to the internal 1d array,
                # and create nd array view
                if data.error() is not None:
                    driver.mem_copy_h2d(
                        data.error().ravel().astype(dtype),
                        self._d_dataset_e_vector[i][slice_])
                    self._d_dataset_e_nddata[i][key] = \
                        self._d_dataset_e_vector[i][slice_].reshape(shape)
                # Create nd array views to the residuals
                self._h_residual_nddata[i][key] = \
                    self._h_residual_vector[i][slice_].reshape(shape)
                self._d_residual_nddata[i][key] = \
                    self._d_residual_vector[i][slice_].reshape(shape)
            # One backend for each driver
            self._backends[i] = driver.backends().objective(dmodel.dtype())
        self._prepared = True

    def residual_scalar(
            self,
            params: dict[str, Real | np.ndarray],
            squared: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[Real]:
        self._update_residual_d(params, True, out_extra)
        t = timeutils.SimpleTimer('objective_residual_sum_eval').start()
        residuals = []
        for i in range(self.nitems()):
            driver = self.models().models()[i].driver()
            backend = self._backends[i]
            d_residual_vector = self._d_residual_vector[i]
            h_residual_scalar = self._h_residual_scalar[i]
            d_residual_scalar = self._d_residual_scalar[i]
            backend.residual_sum(d_residual_vector, squared, d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            residuals.append(h_residual_scalar[0])
        t.stop()
        return residuals

    def log_likelihood(
            self,
            params: dict[str, Real | np.ndarray],
            out_extra: dict[str, Any] | None = None
    ) -> list[float]:
        self._update_residual_d(params, True, out_extra)
        t = timeutils.SimpleTimer('objective_log_likelihood_eval').start()
        log_likelihoods = []
        for i in range(self.nitems()):
            driver = self.models().models()[i].driver()
            backend = self._backends[i]
            d_residual_vector = self._d_residual_vector[i]
            h_residual_scalar = self._h_residual_scalar[i]
            d_residual_scalar = self._d_residual_scalar[i]
            backend.residual_sum(d_residual_vector, True, d_residual_scalar)
            driver.mem_copy_d2h(d_residual_scalar, h_residual_scalar)
            log_likelihoods.append(-0.5 * h_residual_scalar[0])
        t.stop()
        return log_likelihoods

    def residual_vector_h(
            self,
            params: dict[str, Real | np.ndarray],
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[np.ndarray]:
        self._update_residual_h(params, weighted, out_extra)
        return self._h_residual_vector

    def residual_vector_d(
            self,
            params: dict[str, Real | np.ndarray],
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[np.ndarray]:
        self._update_residual_d(params, weighted, out_extra)
        return self._d_residual_vector

    def residual_nddata_h(
            self,
            params: dict[str, Real | np.ndarray],
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, np.ndarray]]:
        self._update_residual_h(params, weighted, out_extra)
        return self._h_residual_nddata

    def residual_nddata_d(
            self,
            params: dict[str, Real | np.ndarray],
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, np.ndarray]]:
        self._update_residual_d(params, weighted, out_extra)
        return self._d_residual_nddata

    def _update_residual_h(
            self,
            params: dict[str, Real | np.ndarray],
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ):
        self._update_residual_d(params, weighted, out_extra)
        t = timeutils.SimpleTimer('objective_residual_d2h').start()
        for i in range(self.nitems()):
            driver = self.models().models()[i].driver()
            d_data = self._d_residual_vector[i]
            h_data = self._h_residual_vector[i]
            driver.mem_copy_d2h(d_data, h_data)
        t.stop()

    def _update_residual_d(
            self,
            params: dict[str, Real | np.ndarray],
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> None:
        if not self._prepared:
            self.prepare()
        # Evaluate model
        out_extra_model = {} if out_extra is not None else None
        model_data = self.models().model_d(params, out_extra_model)
        # Evaluate residuals
        t = timeutils.SimpleTimer('objective_residual_eval').start()
        for i in range(self.nitems()):
            dmodel = self.models().models()[i].dmodel()
            backend = self._backends[i]
            for j, key in enumerate(dmodel.keys()):
                weights = self._weights_u[i][key] if weighted else 1.0
                residual = self._d_residual_nddata[i][key]
                observed_d = self._d_dataset_d_nddata[i][key]
                observed_m = self._d_dataset_m_nddata[i].get(key, None)
                observed_e = self._d_dataset_e_nddata[i].get(key, None)
                expected_d = model_data[i][key]['d']
                expected_m = model_data[i][key]['m']
                expected_w = model_data[i][key]['w']
                backend.residual(
                    observed_d, observed_e, observed_m,
                    expected_d, expected_w, expected_m,
                    weights, residual)
        t.stop()


objective_parser = parseutils.BasicParser(Objective)
