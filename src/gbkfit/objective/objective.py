
from numbers import Real
from typing import Any

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.observation import ObservationGroup
from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils, timeutils


class Objective:
    """
    The comparison of the models of a group of observations with their
    data, each under its likelihood (which is Gaussian, with weights for
    its data items). Each observation checks that its data are of the
    form of its model (see Observation).
    """

    def __init__(self, group: ObservationGroup):
        n = group.nobservations()
        observations = group.observations()
        missing = [i for i, obs in enumerate(observations) if obs.data() is None]
        if missing:
            raise RuntimeError(
                f"every observation of an objective needs data; these have "
                f"none: {missing}")
        self._group = group
        self._datasets = tuple(obs.data() for obs in observations)
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
        # The weight of each data item, from the likelihood
        self._weights_u = tuple(
            {key: obs.likelihood().weight(key)
             for key in obs.observable().keys()}
            for obs in observations)
        # One backend for each driver
        self._backends = iterutils.make_list(n, None)
        # This class is lazily initialized
        self._prepared = False

    def nitems(self) -> int:
        return self._group.nobservations()

    def datasets(self) -> tuple[Dataset, ...]:
        return self._datasets

    def group(self) -> ObservationGroup:
        return self._group

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._group.pdescs()

    def constants(self) -> dict[str, Any]:
        return self._group.constants()

    def prepare(self) -> None:
        for i in range(self.nitems()):
            dataset = self.datasets()[i]
            observation = self._group.observations()[i]
            driver = observation.driver()
            keys = observation.observable().keys()
            dtype = observation.dtype()
            # The data items, one after the other (they have one shape)
            shape = dataset.shape()
            npix = int(np.prod(shape))
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
            self._backends[i] = driver.native_class(
                'Objective', dtype)()
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
            driver = self._group.observations()[i].driver()
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
            driver = self._group.observations()[i].driver()
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
            driver = self._group.observations()[i].driver()
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
        model_data = self._group.model_d(params, out_extra_model)
        # Evaluate residuals
        t = timeutils.SimpleTimer('objective_residual_eval').start()
        for i in range(self.nitems()):
            observable = self._group.observations()[i].observable()
            backend = self._backends[i]
            for j, key in enumerate(observable.keys()):
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

