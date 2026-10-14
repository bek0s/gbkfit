import math
from typing import Any

import numpy as np
import scipy.special

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray
from gbkfit.observation import ObservationGroup
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Objective'
]


class Objective:
    """
    The comparison of the models of a group of observations with their
    data, each under its likelihood.

    The likelihood is Gaussian. The residual of each value is
    (model - data) / error, times the weights of the model (if any) and
    of the data item (from the likelihood of its observation). Values
    the data mask have a residual of 0 and do not count. Where the data
    have a value but the model none (NaN, e.g. the velocity of a spectrum
    without flux), the residual is the missing_residual of the
    likelihood (times the weight of the item): a model must not lower the
    chi-square by having no values. Each observation checks that its
    data are of the form of its model (see Observation).

    A likelihood with free amplitudes (see LikelihoodGaussian) scales
    each spectrum of the model by its best amplitude before the
    residuals, and its log-likelihood has them integrated out; they are
    nuisance parameters (see nnuisance), and an extra output,
    '{prefix}amplitudes' (see ObservationGroup.extra_prefix).

    The methods for fitters return totals over all observations and new
    arrays (chi2, log_likelihood, residuals). The others return a value
    or the arrays of each observation, which are overwritten by the next
    evaluation.

    Parameters
    ----------
    group : ObservationGroup
        The observations, each with data.

    Raises
    ------
    ConfigError
        If an observation has no data.
    """

    def __init__(self, group: ObservationGroup):
        n = group.nobservations()
        observations = group.observations()
        if missing := [
                i for i, obs in enumerate(observations) if obs.data() is None]:
            raise ConfigError(
                f"every observation of an objective needs data; these have "
                f"none: {missing}")
        self._group = group
        self._datasets = tuple(obs.data() for obs in observations)
        # The data items of each observation, one after the other, in 1d
        # arrays: the data (d), masks (m) and errors (e)
        self._d_dataset_d_vector = iterutils.make_list(n, None)
        self._d_dataset_m_vector = iterutils.make_list(n, None)
        self._d_dataset_e_vector = iterutils.make_list(n, None)
        # Views of the above, as the nd arrays of each item
        self._d_dataset_d_nddata = iterutils.make_list(n, {})
        self._d_dataset_m_nddata = iterutils.make_list(n, {})
        self._d_dataset_e_nddata = iterutils.make_list(n, {})
        # The residuals of each observation, in 1d arrays on the host (h)
        # and the device (d), and their views as the nd arrays of each item
        self._h_residual_vector = iterutils.make_list(n, None)
        self._d_residual_vector = iterutils.make_list(n, None)
        self._h_residual_nddata = iterutils.make_list(n, {})
        self._d_residual_nddata = iterutils.make_list(n, {})
        # The sum of the residuals of each observation (1d, size 1)
        self._h_residual_scalar = iterutils.make_list(n, None)
        self._d_residual_scalar = iterutils.make_list(n, None)
        # The weight of each data item, and the residual of the values the
        # model does not have, from the likelihood
        self._weights_u = tuple(
            {key: obs.likelihood().weight(key)
             for key in obs.observable().keys()}
            for obs in observations)
        # (none where the model cannot lack values: NaN would show one)
        missing = [obs.likelihood().missing_residual() for obs in observations]
        self._missing_residuals = tuple(
            np.nan if residual is None else residual for residual in missing)
        # The data item of the spectra of free amplitudes of each
        # observation, if any; the sums of each spectrum (A, B and F, see
        # LikelihoodGaussian) on the host (h) and the device (d), and the
        # amplitudes, which scale the spectra of the model
        self._amplitude_items = tuple(
            obs.observable().spectra_item()
            if obs.likelihood().amplitudes() is not None else None
            for obs in observations)
        self._h_amplitude_sums = iterutils.make_list(n, None)
        self._d_amplitude_sums = iterutils.make_list(n, None)
        self._h_amplitudes = iterutils.make_list(n, None)
        self._d_amplitudes = iterutils.make_list(n, None)
        # The native objective of each observation, for its driver
        self._backends = iterutils.make_list(n, None)
        # The memory is allocated on the first evaluation (see prepare)
        self._prepared = False

    def nobservations(self) -> int:
        """Return the number of observations."""
        return self._group.nobservations()

    def datasets(self) -> tuple[Dataset, ...]:
        """Return the data of each observation."""
        return self._datasets

    def group(self) -> ObservationGroup:
        """Return the observations."""
        return self._group

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the parameters of the models, by name (prefixed)."""
        return self._group.pdescs()

    def constants(self) -> dict[str, Any]:
        """Return the constants of the models, by name (prefixed)."""
        return self._group.constants()

    def prepare(self) -> None:
        """
        Allocate the memory of the residuals, and copy the data to the
        device of each observation. The first evaluation calls it.
        """
        for i in range(self.nobservations()):
            dataset = self._datasets[i]
            observation = self._group.observations()[i]
            driver = observation.driver()
            keys = observation.observable().keys()
            dtype = observation.dtype()
            # The data items, one after the other (they have one shape)
            shape = dataset.shape()
            npix = int(np.prod(shape))
            nelem = npix * len(keys)
            self._d_dataset_d_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_m_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_e_vector[i] = driver.mem_alloc_d(nelem, dtype)
            (self._h_residual_vector[i],
             self._d_residual_vector[i]) = driver.mem_alloc_s(nelem, dtype)
            # (the sums are in double precision, of millions of terms)
            (self._h_residual_scalar[i],
             self._d_residual_scalar[i]) = driver.mem_alloc_s(1, np.float64)
            for j, key in enumerate(keys):
                data = dataset[key]
                slice_ = slice(j * npix, (j + 1) * npix)
                driver.mem_copy_h2d(
                    data.data().ravel().astype(dtype),
                    self._d_dataset_d_vector[i][slice_])
                self._d_dataset_d_nddata[i][key] = \
                    self._d_dataset_d_vector[i][slice_].reshape(shape)
                if data.mask() is not None:
                    driver.mem_copy_h2d(
                        data.mask().ravel().astype(dtype),
                        self._d_dataset_m_vector[i][slice_])
                    self._d_dataset_m_nddata[i][key] = \
                        self._d_dataset_m_vector[i][slice_].reshape(shape)
                if data.error() is not None:
                    driver.mem_copy_h2d(
                        data.error().ravel().astype(dtype),
                        self._d_dataset_e_vector[i][slice_])
                    self._d_dataset_e_nddata[i][key] = \
                        self._d_dataset_e_vector[i][slice_].reshape(shape)
                self._h_residual_nddata[i][key] = \
                    self._h_residual_vector[i][slice_].reshape(shape)
                self._d_residual_nddata[i][key] = \
                    self._d_residual_vector[i][slice_].reshape(shape)
            self._backends[i] = driver.native_class('Objective', dtype)()
            if self._amplitude_items[i] is not None:
                spectra = shape[1:]
                sums = [driver.mem_alloc_s(spectra, np.float64)
                        for _ in range(3)]
                self._h_amplitude_sums[i] = [h for h, _ in sums]
                self._d_amplitude_sums[i] = [d for _, d in sums]
                (self._h_amplitudes[i],
                 self._d_amplitudes[i]) = driver.mem_alloc_s(spectra, dtype)
        self._prepared = True

    def chi2(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> float:
        """
        Return the chi-square of all observations: the sum of their squared
        weighted residuals.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : dict, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        float
            The chi-square.
        """
        return float(sum(self.residual_sums(params, True, out_extra)))

    def log_likelihood(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> float:
        """
        Return the log-likelihood of all observations: -chi2 / 2, and
        with free amplitudes, their integral (see LikelihoodGaussian). The
        constant of the Gaussian normalisation (which depends only on the
        errors of the data) is left out.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : dict, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        float
            The log-likelihood.
        """
        return self.chi2_and_log_likelihood(params, out_extra)[1]

    def chi2_and_log_likelihood(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> tuple[float, float]:
        """
        Return the chi-square and the log-likelihood of all observations,
        of one evaluation (see chi2 and log_likelihood).

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : dict, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        tuple of float
            The chi-square and the log-likelihood.
        """
        chi2 = self.chi2(params, out_extra)
        return chi2, -0.5 * chi2 + self._amplitude_terms()

    def residuals(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> np.ndarray:
        """
        Return the weighted residuals of all observations, one after the
        other, in a new array (in float64; see residual_vector_h for each
        observation).

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : dict, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        np.ndarray
            The residuals (a new array on every call: solvers that keep
            the residuals of earlier calls, e.g. scipy's least_squares,
            need one).
        """
        vectors = self.residual_vector_h(params, True, out_extra)
        return np.concatenate(vectors, dtype=np.float64)

    def nnuisance(self) -> int:
        """
        Return the number of nuisance parameters that the objective solves
        itself: the free amplitude of each spectrum with values in the
        data (see LikelihoodGaussian). The degrees of freedom of a fit
        exclude them.
        """
        count = 0
        for dataset, key in zip(self._datasets, self._amplitude_items):
            if key is not None:
                mask = dataset[key].mask()
                count += int(np.count_nonzero(np.any(mask != 0, axis=0)))
        return count

    def nresiduals(self) -> int:
        """
        Return the number of residuals that count: those of the values the
        data do not mask (the values the model does not have count too,
        with the missing_residual of their likelihood).
        """
        return sum(
            int(np.count_nonzero(dataset[key].mask()))
            for dataset, obs in zip(
                self._datasets, self._group.observations())
            for key in obs.observable().keys())

    def residual_sums(
            self,
            params: ParamValues,
            squared: bool = True,
            out_extra: dict[str, Any] | None = None
    ) -> list[float]:
        """
        Return the sum of the weighted residuals of each observation.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        squared : bool, optional
            Whether to sum the squared residuals (the chi-square), or their
            absolute values.
        out_extra : dict, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        list of float
            The sum of each observation.
        """
        self._update_residual_d(params, True, out_extra)
        sums = []
        with self._group.timers().measure('objective_residual_sum_eval'):
            for i in range(self.nobservations()):
                driver = self._group.observations()[i].driver()
                self._backends[i].residual_sum(
                    self._d_residual_vector[i], squared,
                    self._d_residual_scalar[i])
                driver.mem_copy_d2h(
                    self._d_residual_scalar[i], self._h_residual_scalar[i])
                sums.append(float(self._h_residual_scalar[i][0]))
        return sums

    def residual_vector_h(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[np.ndarray]:
        """
        Return the residuals of each observation, on the host: its data
        items one after the other, in one array that the next evaluation
        overwrites.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        weighted : bool
            Whether to apply the weights of the data items (from the
            likelihoods).
        out_extra : dict, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        list of np.ndarray
            The residuals of each observation.
        """
        self._update_residual_h(params, weighted, out_extra)
        return self._h_residual_vector

    def residual_vector_d(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[DeviceArray]:
        """
        Return the residuals of each observation, on its device (see
        residual_vector_h).
        """
        self._update_residual_d(params, weighted, out_extra)
        return self._d_residual_vector

    def residual_nddata_h(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, np.ndarray]]:
        """
        Return the residuals of each data item of each observation, on the
        host, as arrays of the shape of the data, which the next
        evaluation overwrites (see residual_vector_h).
        """
        self._update_residual_h(params, weighted, out_extra)
        return self._h_residual_nddata

    def residual_nddata_d(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, DeviceArray]]:
        """
        Return the residuals of each data item of each observation, on its
        device (see residual_nddata_h).
        """
        self._update_residual_d(params, weighted, out_extra)
        return self._d_residual_nddata

    def _update_residual_h(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> None:
        """Evaluate the residuals, and copy them to the host."""
        self._update_residual_d(params, weighted, out_extra)
        with self._group.timers().measure('objective_residual_d2h'):
            for i in range(self.nobservations()):
                driver = self._group.observations()[i].driver()
                driver.mem_copy_d2h(
                    self._d_residual_vector[i], self._h_residual_vector[i])

    def _update_residual_d(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> None:
        """Evaluate the models and the residuals."""
        if not self._prepared:
            self.prepare()
        model_data = self._group.model_d(params, out_extra)
        with self._group.timers().measure('objective_residual_eval'):
            for i in range(self.nobservations()):
                if self._amplitude_items[i] is not None:
                    self._scale_spectra(i, model_data[i], out_extra)
                observable = self._group.observations()[i].observable()
                for key in observable.keys():
                    weight = self._weights_u[i][key] if weighted else 1.0
                    self._backends[i].residual(
                        self._d_dataset_d_nddata[i][key],
                        self._d_dataset_e_nddata[i].get(key),
                        self._d_dataset_m_nddata[i].get(key),
                        model_data[i][key]['d'],
                        model_data[i][key]['w'],
                        weight,
                        self._missing_residuals[i],
                        self._d_residual_nddata[i][key])

    def _scale_spectra(
            self,
            i: int,
            model_data: dict[str, dict[str, DeviceArray | None]],
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Scale each spectrum of the model of observation i by its best free
        amplitude (see LikelihoodGaussian), and keep its sums for the
        log-likelihood; with extras, the amplitudes go to them
        ('{prefix}amplitudes', of the shape of the spectra but their first
        axis).
        """
        key = self._amplitude_items[i]
        driver = self._group.observations()[i].driver()
        model = model_data[key]
        self._backends[i].amplitude_sums(
            self._d_dataset_d_nddata[i][key],
            self._d_dataset_e_nddata[i].get(key),
            self._d_dataset_m_nddata[i].get(key),
            model['d'], model['w'], *self._d_amplitude_sums[i])
        for h, d in zip(self._h_amplitude_sums[i], self._d_amplitude_sums[i]):
            driver.mem_copy_d2h(d, h)
        a, b, _ = self._h_amplitude_sums[i]
        amplitudes = self._h_amplitudes[i]
        amplitudes[...] = 0
        np.divide(np.maximum(b, 0), a, out=amplitudes, where=a > 0)
        driver.mem_copy_h2d(amplitudes, self._d_amplitudes[i])
        driver.math_mul(model['d'], self._d_amplitudes[i], out=model['d'])
        if out_extra is not None:
            out_extra[f'{self._group.extra_prefix(i)}amplitudes'] = \
                amplitudes.copy()

    def _amplitude_terms(self) -> float:
        """
        Return the sum of the terms of the log-likelihood of the free
        amplitudes of the last evaluation, beyond -chi2 / 2 (see
        LikelihoodGaussian): log(F sqrt(2 pi / A) Phi(B / sqrt(A))) + (B^2
        / 2A if B < 0), of each spectrum where the model has light (F > 0
        and A > 0), with the sums of the weights of the likelihood.
        """
        total = 0.0
        for i, key in enumerate(self._amplitude_items):
            if key is None:
                continue
            a, b, f = self._h_amplitude_sums[i]
            weight = self._weights_u[i][key] ** 2
            lit = (a > 0) & (f > 0)
            a, b, f = weight * a[lit], weight * b[lit], f[lit]
            x = b / np.sqrt(a)
            total += float(np.sum(
                np.log(f) + 0.5 * np.log(2 * math.pi / a)
                + scipy.special.log_ndtr(x)
                + np.where(b < 0, 0.5 * x ** 2, 0.0)))
        return total
