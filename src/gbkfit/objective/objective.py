from collections.abc import Sequence
from typing import Any, NamedTuple, Self

import numpy as np
import scipy.special

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.observation import ObservationGroup
from gbkfit.observation.likelihood import LikelihoodUser, Whitening, whitening
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Objective'
]


class _Noise(NamedTuple):
    """
    The fitted noise of a data item (see LikelihoodGaussian): the names of
    the parameters of its scale and of its jitter (None for those not
    fitted), and the smallest scale (1 if the scale is not fitted).
    """
    scale: str | None
    jitter: str | None
    scale_minimum: float


class _Whitener:
    """
    The whitening of the residuals of a data item (see Whitening) on the
    device of a driver: the indices of its pixels and its matrices (that
    of the sky transposed), which copies share (see shared), and the
    residuals of its pixels and their products, of its own.
    """

    def __init__(
            self,
            driver: Driver,
            pixels: DeviceArray,
            spatial: DeviceArray | None,
            spectral: DeviceArray | None,
            nchannels: int,
            dtype: np.dtype
    ):
        self._driver = driver
        self._pixels = pixels
        self._spatial = spatial
        self._spectral = spectral
        self._dtype = dtype
        # (the channels by the pixels; one channel for an image)
        size = (nchannels, pixels.size)
        self._values = driver.mem_alloc_d(size, dtype)
        self._products = driver.mem_alloc_d(size, dtype)

    @classmethod
    def on_device(
            cls,
            driver: Driver,
            whitening: Whitening,
            shape: tuple[int, ...],
            dtype: np.dtype
    ) -> Self:
        """
        Return the whitening on the device of a driver, for residuals of a
        shape and a type.
        """
        spatial = None if whitening.spatial is None \
            else driver.mem_copy_h2d(np.ascontiguousarray(
                whitening.spatial.T, dtype))
        spectral = None if whitening.spectral is None \
            else driver.mem_copy_h2d(whitening.spectral.astype(dtype))
        return cls(
            driver, driver.mem_copy_h2d(whitening.pixels), spatial, spectral,
            int(np.prod(shape[:-2])), dtype)

    def shared(self) -> Self:
        """
        Return a copy that shares its pixels and matrices, with residuals
        of its own.
        """
        return type(self)(
            self._driver, self._pixels, self._spatial, self._spectral,
            self._values.shape[0], self._dtype)

    def __call__(self, residual: DeviceArray) -> None:
        """Whiten the residuals of the item (of its shape) in place."""
        driver = self._driver
        values, products = self._values, self._products
        flat = residual.reshape(values.shape[0], -1)
        driver.mem_gather(flat, self._pixels, out=values)
        if self._spatial is not None:
            driver.math_matmul(values, self._spatial, out=products)
            values, products = products, values
        if self._spectral is not None:
            driver.math_matmul(self._spectral, values, out=products)
            values, products = products, values
        driver.mem_scatter(values, self._pixels, flat)


class Objective:
    """
    The comparison of the models of a group of observations with their
    data, each under its likelihood.

    The likelihood is Gaussian (but a user likelihood, see below). The
    residual of each value is (model - data) / error, times the weights
    of the model (if any) and of the data item (from the likelihood of
    its observation). Values the data mask have a residual of 0 and do
    not count. Where the data have a value but the model none (NaN, e.g.
    the velocity of a spectrum without flux), the residual is the
    missing_residual of the likelihood (times the weight of the item): a
    model must not lower the chi-square by having no values. A data item
    with an upper limit (see LikelihoodGaussian) gives its masked values
    the residual of a non-detection instead. Each observation checks that
    its data are of the form of its model (see Observation).

    A likelihood with amplitudes of spectra (see LikelihoodGaussian)
    scales each spectrum of the model by its best amplitude before the
    residuals, whose prior adds a residual of each spectrum after those of
    the data; its log-likelihood has them integrated out. They are
    nuisance parameters (see nnuisance), and an extra output,
    '{prefix}amplitudes' (see ObservationGroup.extra_prefix).

    A likelihood with a fitted noise (see LikelihoodGaussian) divides the
    residuals of its data items by their noise in place of their errors,
    and adds the noise residual of each of those items after all the
    other residuals of the observations (see residuals and noise_chi2).

    The weighted residuals of a data item whose noise is correlated are
    whitened (see LikelihoodGaussian): the sum of their squares is its
    chi-square. The objective works out their whitening on its first
    evaluation (see prepare); its copies with other data share it (see
    with_data).

    An observation of a user likelihood (see LikelihoodUser) has the
    residuals its function gives for each data item, on the host, times
    the weights of the model, of the data mask and of the item, as above;
    or of a function of the log-likelihood, residuals of 0 and the
    chi-square -2 log L (see residual_sums), which has no residuals for
    least squares (see residuals).

    Its memory on the devices (and the host) is allocated on the first
    evaluation, and released by close, or when Python collects it. An
    objective is a context manager that closes on exit: fitters can keep
    it alive after a fit (e.g. lmfit, in reference cycles), with its
    memory, until Python collects it.

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
        # The upper limit of each data item where it is masked, and its
        # noise (NaN for none; see LikelihoodGaussian)
        self._upper_limits = tuple(
            {key: obs.likelihood().upper_limit(key) or (np.nan, np.nan)
             for key in obs.observable().keys()}
            for obs in observations)
        # The data item of the spectra of amplitudes of each observation,
        # if any, and the standard deviation of their prior
        self._amplitude_items = tuple(
            obs.observable().spectra_item()
            if obs.likelihood().amplitudes() is not None else None
            for obs in observations)
        self._amplitude_sigmas = tuple(
            obs.likelihood().amplitude_sigma() for obs in observations)
        # The data items of each observation whose noise is fitted, by
        # name, and their noise residuals, in float64 (each sums a term of
        # every value of its item: float32 would round it off)
        self._noise = tuple(
            {key: _Noise(*obs.likelihood().noise_params(key),
                         noise.scale_minimum if noise.scale else 1.0)
             for key, noise in obs.likelihood().fitted_noise().items()}
            for obs in observations)
        self._noise_residuals = [
            np.zeros(len(noise)) for noise in self._noise]
        # The user likelihood of each observation, if any, and the
        # chi-square of those of the log-likelihood (see LikelihoodUser)
        self._users = tuple(
            obs.likelihood()
            if isinstance(obs.likelihood(), LikelihoodUser) else None
            for obs in observations)
        self._user_chi2 = [0.0] * n
        # The whitening to share with copies of it of other data, of each
        # data item of each observation, by name (see with_data)
        self._shared = iterutils.make_list(n, {})
        self._reset_memory()

    def _reset_memory(self) -> None:
        """
        Drop its memory (see prepare and close): the next evaluation
        allocates it again.
        """
        n = self.nobservations()
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
        # The sums of each spectrum of amplitudes (A and B, see
        # LikelihoodGaussian) on the host (h) and the device (d), the
        # amplitudes, which scale the spectra of the model, and the
        # residuals of their prior (views of the end of the residuals of
        # the observation)
        self._h_amplitude_sums = iterutils.make_list(n, None)
        self._d_amplitude_sums = iterutils.make_list(n, None)
        self._h_amplitudes = iterutils.make_list(n, None)
        self._d_amplitudes = iterutils.make_list(n, None)
        self._h_prior_residuals = iterutils.make_list(n, None)
        self._d_prior_residuals = iterutils.make_list(n, None)
        # The squares of the noise residuals of the data items whose noise
        # is fitted, on the host (h) and the device (d)
        self._h_noise_sums = iterutils.make_list(n, None)
        self._d_noise_sums = iterutils.make_list(n, None)
        # The whitening of the residuals of the data items of each
        # observation whose noise is correlated, by name, and the numbers
        # of independent values they are worth (see Whitening)
        self._whiteners = iterutils.make_list(n, {})
        self._nindependent = iterutils.make_list(n, {})
        # The share of its values that each data item whose noise is fitted
        # counts in its noise residual: 1, but the independent values they
        # are worth over their number for correlated noise (see
        # LikelihoodGaussian)
        self._noise_shares = [np.ones(len(noise)) for noise in self._noise]
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

    def with_driver(self, driver: Driver) -> 'Objective':
        """
        Return a copy of it with all its work on another driver: its
        models (see ObservationGroup.with_driver) and its residuals. The
        copy shares the data; its memory is its own. Copies on several
        devices evaluate several points at once.
        """
        return Objective(self._group.with_driver(driver))

    def with_data(self, datasets: Sequence[Dataset]) -> 'Objective':
        """
        Return a copy of it with other data of its observations (e.g. those
        of a bootstrap; see ObservationGroup.with_data). The copy shares
        the whitening of the residuals of the data items whose noise is
        correlated as before, with the same masks, if it is worked out
        (see prepare): e.g. the refits of a bootstrap work it out once.
        """
        objective = Objective(self._group.with_data(datasets))
        for i, dataset in enumerate(objective.datasets()):
            before = self._datasets[i]
            for key, whitener in self._whiteners[i].items():
                item, old = dataset[key], before[key]
                if item.spatial_correlation() is old.spatial_correlation() \
                        and item.spectral_correlation() \
                        is old.spectral_correlation() \
                        and np.array_equal(item.mask(), old.mask()):
                    objective._shared[i][key] = (
                        whitener, self._nindependent[i][key])
        return objective

    def close(self) -> None:
        """
        Release its memory on the devices and the host now, rather than
        when Python collects it (e.g. hundreds of megabytes of the whitening
        of correlated noise; see the class). The next evaluation allocates
        it again.
        """
        self._reset_memory()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the parameters of the models, by name (prefixed)."""
        return self._group.pdescs()

    def constants(self) -> dict[str, Any]:
        """Return the constants of the models, by name (prefixed)."""
        return self._group.constants()

    def prepare(self) -> None:
        """
        Allocate the memory of the residuals, copy the data to the device of
        each observation, and work out the whitening of the residuals of
        data items whose noise is correlated (once, for every evaluation;
        e.g. about 4 s for 7000 pixels), unless shared (see with_data). The
        first evaluation calls it; it does nothing once done (until close).
        """
        if self._prepared:
            return
        for i in range(self.nobservations()):
            dataset = self._datasets[i]
            observation = self._group.observations()[i]
            driver = observation.driver()
            keys = observation.observable().keys()
            dtype = observation.dtype()
            # The data items, one after the other (they have one shape);
            # the residuals, and after them those of the prior of the
            # amplitudes of spectra, if any
            shape = dataset.shape()
            npix = int(np.prod(shape))
            nelem = npix * len(keys)
            nprior = self._nprior(i)
            self._d_dataset_d_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_m_vector[i] = driver.mem_alloc_d(nelem, dtype)
            self._d_dataset_e_vector[i] = driver.mem_alloc_d(nelem, dtype)
            (self._h_residual_vector[i],
             self._d_residual_vector[i]) = driver.mem_alloc_s(
                nelem + nprior, dtype)
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
                if key in self._shared[i]:
                    whitener, nindependent = self._shared[i][key]
                    self._whiteners[i][key] = whitener.shared()
                    self._nindependent[i][key] = nindependent
                elif self._users[i] is None and (
                        correlation := dataset.noise_correlation(key)) \
                        is not None:
                    whitening_ = whitening(
                        key, data.mask(), correlation,
                        observation.likelihood().white_floor())
                    self._whiteners[i][key] = _Whitener.on_device(
                        driver, whitening_, shape, dtype)
                    self._nindependent[i][key] = whitening_.nindependent
            self._backends[i] = driver.native_class('Objective', dtype)()
            if self._amplitude_items[i] is not None:
                spectra = shape[1:]
                sums = [driver.mem_alloc_s(spectra, np.float64)
                        for _ in range(2)]
                self._h_amplitude_sums[i] = [h for h, _ in sums]
                self._d_amplitude_sums[i] = [d for _, d in sums]
                (self._h_amplitudes[i],
                 self._d_amplitudes[i]) = driver.mem_alloc_s(spectra, dtype)
                self._h_prior_residuals[i] = self._h_residual_vector[i][nelem:]
                self._d_prior_residuals[i] = self._d_residual_vector[i][nelem:]
            if self._noise[i]:
                (self._h_noise_sums[i],
                 self._d_noise_sums[i]) = driver.mem_alloc_s(
                    len(self._noise[i]), np.float64)
            for j, key in enumerate(self._noise[i]):
                if key in self._nindependent[i]:
                    self._noise_shares[i][j] = self._nindependent[i][key] \
                        / np.count_nonzero(dataset[key].mask())
        self._prepared = True

    def chi2(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> float:
        """
        Return the chi-square of all observations: the sum of their squared
        weighted residuals (with those of the priors of amplitudes and
        the noise residuals, if any; see residuals).

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
        part of the Gaussian normalisation that depends only on the errors
        of the data is left out (with a fitted noise, the rest is in the
        chi-square, as its noise residuals).

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
        observation), and after them the noise residuals of each
        observation (see LikelihoodGaussian), if any.

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
        self._check_residuals()
        vectors = self.residual_vector_h(params, True, out_extra)
        return np.concatenate(
            [*vectors, *self._noise_residuals], dtype=np.float64)

    def noise_chi2(self) -> float:
        """
        Return the sum of the squared noise residuals of all observations
        (see LikelihoodGaussian), of the last evaluation: the part of the
        chi-square that is not of the data (e.g. for a reduced
        chi-square); 0 without a fitted noise.
        """
        return float(sum(
            np.sum(residuals ** 2) for residuals in self._noise_residuals))

    def amplitudes(self) -> list[np.ndarray | None]:
        """
        Return the free amplitudes of the spectra of each observation (see
        LikelihoodGaussian), of the last evaluation, as new arrays of the
        shape of the spectra but their first axis; None for the
        observations without them, or before an evaluation.
        """
        return [
            None if amplitudes is None else amplitudes.copy()
            for amplitudes in self._h_amplitudes]

    def nnuisance(self) -> int:
        """
        Return the number of nuisance parameters that the objective solves
        itself: the amplitude of each spectrum (see LikelihoodGaussian).
        The degrees of freedom of a fit exclude them.
        """
        return sum(self._nprior(i) for i in range(self.nobservations()))

    def nresiduals(self) -> int:
        """
        Return the number of residuals that count: those of the values the
        data do not mask (the values the model does not have count too,
        with the missing_residual of their likelihood), all values of the
        data items with upper limits, and that of the prior of each
        amplitude of a spectrum, if any; not the noise residuals, which are
        not of the data (see LikelihoodGaussian).
        """
        return sum(
            int(dataset[key].mask().size) if not np.isnan(limits[key][0])
            else int(np.count_nonzero(dataset[key].mask()))
            for dataset, obs, limits in zip(
                self._datasets, self._group.observations(),
                self._upper_limits)
            for key in obs.observable().keys()) + self.nnuisance()

    def nindependent(self) -> float:
        """
        Return the number of independent residuals that the residuals are
        worth (see nresiduals): the chi-square that the noise of the data
        gives them on average, one for each that counts, but those of the
        data items whose noise is correlated, which are worth fewer (see
        Whitening). The degrees of freedom of a fit are of them. It works
        out the whitening of the residuals, if not yet (see prepare).
        """
        self.prepare()
        total = float(self.nresiduals())
        for dataset, nindependent in zip(self._datasets, self._nindependent):
            for key, n in nindependent.items():
                total += n - np.count_nonzero(dataset[key].mask())
        return total

    def _nprior(self, i: int) -> int:
        """
        Return the number of amplitudes of spectra of observation i, each
        with the residual of its prior (0 without them).
        """
        key = self._amplitude_items[i]
        if key is None:
            return 0
        return int(np.prod(self._datasets[i][key].shape()[1:]))

    def residual_sums(
            self,
            params: ParamValues,
            squared: bool = True,
            out_extra: dict[str, Any] | None = None
    ) -> list[float]:
        """
        Return the sum of the weighted residuals of each observation, its
        noise residuals included (see residuals).

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
        if not squared:
            self._check_residuals()
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
                noise = self._noise_residuals[i]
                sums.append(float(self._h_residual_scalar[i][0]) + float(
                    np.sum(noise ** 2) if squared else np.sum(np.abs(noise)))
                    + self._user_chi2[i])
        return sums

    def _check_residuals(self) -> None:
        """
        Raise ConfigError if an observation has a user likelihood of the
        log-likelihood, which has no residuals (see LikelihoodUser).
        """
        for i, user in enumerate(self._users):
            if user is not None and user.form() == 'log_likelihood':
                raise ConfigError(
                    f"the likelihood of observation {i} is a function of "
                    f"the log-likelihood, which has no residuals (e.g. for "
                    f"least squares); give the residuals instead "
                    f"(form 'residuals'), or use another fitter")

    def residual_vector_h(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: dict[str, Any] | None = None
    ) -> list[np.ndarray]:
        """
        Return the residuals of each observation, on the host: its data
        items one after the other, in one array that the next evaluation
        overwrites (without its noise residuals; see residuals).

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
        self.prepare()
        model_data = self._group.model_d(params, out_extra)
        with self._group.timers().measure('objective_residual_eval'):
            for i in range(self.nobservations()):
                if self._users[i] is not None:
                    self._update_user_residuals(
                        i, model_data[i], params, weighted)
                    continue
                if self._amplitude_items[i] is not None:
                    self._scale_spectra(i, model_data[i], params, out_extra)
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
                        *self._upper_limits[i][key],
                        *self._noise_values(i, key, params),
                        self._d_residual_nddata[i][key])
                    if weighted and key in self._whiteners[i]:
                        self._whiteners[i][key](
                            self._d_residual_nddata[i][key])
                if self._noise[i]:
                    self._update_noise_residuals(
                        i, model_data[i], params, weighted)

    def _update_user_residuals(
            self,
            i: int,
            model_data: dict[str, dict[str, DeviceArray | None]],
            params: ParamValues,
            weighted: bool
    ) -> None:
        """
        Evaluate the residuals of observation i, of a user likelihood (see
        LikelihoodUser), on the host, and copy them to its device: those
        its function gives each data item, times the weights of the model,
        of the data mask and of the item (if weighted); or of a function
        of the log-likelihood, residuals of 0, and its chi-square.
        """
        likelihood = self._users[i]
        driver = self._group.observations()[i].driver()
        self._user_chi2[i] = 0.0
        for key in self._group.observations()[i].observable().keys():
            item = self._datasets[i][key]
            mask = item.mask()
            model = np.asarray(
                driver.mem_copy_d2h(model_data[key]['d']), dtype=np.float64)
            result = likelihood.evaluate(
                key, item.data(), model, item.error(), mask, params)
            residuals = self._h_residual_nddata[i][key]
            if likelihood.form() == 'log_likelihood':
                self._user_chi2[i] -= 2 * result
                residuals[...] = 0
                continue
            if (weights := model_data[key]['w']) is not None:
                result = result * driver.mem_copy_d2h(weights)
            if mask is not None:
                result = np.where(mask != 0, result * mask, 0)
            weight = self._weights_u[i][key] if weighted else 1.0
            residuals[...] = result * weight
        driver.mem_copy_h2d(
            self._h_residual_vector[i], self._d_residual_vector[i])

    def _noise_values(
            self,
            i: int,
            key: str,
            params: ParamValues
    ) -> tuple[float, float]:
        """
        Return the scale and the jitter of the noise of data item key of
        observation i (see LikelihoodGaussian): 1 and 0 for those that are
        not fitted.
        """
        noise = self._noise[i].get(key)
        if noise is None:
            return 1.0, 0.0
        scale = 1.0 if noise.scale is None else float(params[noise.scale])
        jitter = 0.0 if noise.jitter is None else float(params[noise.jitter])
        return scale, jitter

    def _update_noise_residuals(
            self,
            i: int,
            model_data: dict[str, dict[str, DeviceArray | None]],
            params: ParamValues,
            weighted: bool
    ) -> None:
        """
        Evaluate the noise residuals of observation i (see
        LikelihoodGaussian), with the weights of its data items if
        weighted (as its residuals).
        """
        driver = self._group.observations()[i].driver()
        for j, (key, noise) in enumerate(self._noise[i].items()):
            weight = self._weights_u[i][key] if weighted else 1.0
            self._backends[i].noise_sum(
                self._d_dataset_e_nddata[i].get(key),
                self._d_dataset_m_nddata[i].get(key),
                model_data[key]['d'],
                model_data[key]['w'],
                weight,
                *self._noise_values(i, key, params),
                noise.scale_minimum,
                self._d_noise_sums[i][j:j + 1])
        driver.mem_copy_d2h(self._d_noise_sums[i], self._h_noise_sums[i])
        # (each term is at least 0, but rounding can leave a sum of terms
        # of 0, at the smallest scale and no jitter, a hair below it)
        self._noise_residuals[i][...] = np.sqrt(
            np.maximum(self._h_noise_sums[i], 0) * self._noise_shares[i])

    def _scale_spectra(
            self,
            i: int,
            model_data: dict[str, dict[str, DeviceArray | None]],
            params: ParamValues,
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Scale each spectrum of the model of observation i by its best
        amplitude (see LikelihoodGaussian), write the residuals of their
        prior, and keep their sums for the log-likelihood; with extras, the
        amplitudes go to them ('{prefix}amplitudes', of the shape of the
        spectra but their first axis).
        """
        key = self._amplitude_items[i]
        driver = self._group.observations()[i].driver()
        model = model_data[key]
        self._backends[i].amplitude_sums(
            self._d_dataset_d_nddata[i][key],
            self._d_dataset_e_nddata[i].get(key),
            self._d_dataset_m_nddata[i].get(key),
            model['d'], model['w'], *self._noise_values(i, key, params),
            *self._d_amplitude_sums[i])
        for h, d in zip(self._h_amplitude_sums[i], self._d_amplitude_sums[i]):
            driver.mem_copy_d2h(d, h)
        # The best amplitude: the minimum of the chi-square of the spectrum
        # (its sums A and B times the weight of the item, squared) and that
        # of the prior, (a - 1)^2 / s^2
        a, b = self._h_amplitude_sums[i]
        weight = self._weights_u[i][key] ** 2
        precision = 1 / self._amplitude_sigmas[i] ** 2
        amplitudes = self._h_amplitudes[i]
        amplitudes[...] = np.maximum(weight * b + precision, 0) \
            / (weight * a + precision)
        driver.mem_copy_h2d(amplitudes, self._d_amplitudes[i])
        driver.math_mul(model['d'], self._d_amplitudes[i], out=model['d'])
        # (the prior as residuals, (a - 1) / s, after those of the data)
        self._h_prior_residuals[i][...] = \
            (amplitudes.ravel() - 1) / self._amplitude_sigmas[i]
        driver.mem_copy_h2d(
            self._h_prior_residuals[i], self._d_prior_residuals[i])
        if out_extra is not None:
            out_extra[f'{self._group.extra_prefix(i)}amplitudes'] = \
                amplitudes.copy()

    def _amplitude_terms(self) -> float:
        """
        Return the sum of the terms of the log-likelihood of the amplitudes
        of spectra of the last evaluation, beyond -chi2 / 2 (whose
        chi-square has the residuals of their prior; see
        LikelihoodGaussian): for each spectrum, log(Phi(B' / sqrt(A')) /
        (s sqrt(A') Phi(1 / s))) + (B'^2 / 2A' if B' < 0), with A' = w A +
        1/s^2 and B' = w B + 1/s^2 (w the weight of the item, squared). A
        spectrum that does not count in the chi-square (A = 0) has none.
        """
        total = 0.0
        for i, key in enumerate(self._amplitude_items):
            if key is None:
                continue
            a, b = self._h_amplitude_sums[i]
            sigma = self._amplitude_sigmas[i]
            weight = self._weights_u[i][key] ** 2
            a, b = weight * a + 1 / sigma ** 2, weight * b + 1 / sigma ** 2
            x = b / np.sqrt(a)
            total += float(np.sum(
                np.where(b < 0, 0.5 * x ** 2, 0.0)
                + scipy.special.log_ndtr(x) - np.log(sigma * np.sqrt(a))
                - scipy.special.log_ndtr(1 / sigma)))
        return total
