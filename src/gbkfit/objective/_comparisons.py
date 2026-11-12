"""
The comparisons of the models of observations with their data, each of
the kind of its likelihood (see comparison_of): their residuals and
chi-square.
"""

import abc
import math
from collections.abc import Collection, Mapping, Sequence
from typing import NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.observation import (
    Likelihood, LikelihoodGauss, LikelihoodUser, LinearPieces, ModelData,
    Observation, UpperLimit, channel_mask)
from gbkfit.params import ParamValues
from ._whitening import Whitener, whitening


__all__ = [
    'Comparison',
    'ComparisonGauss',
    'ComparisonUser',
    'comparison_of'
]


# The upper limit of a data item without one, and its noise, as the
# native residual takes them (see ComparisonGauss.evaluate)
_NO_UPPER_LIMIT = (np.nan, np.nan)


class Comparison(abc.ABC):
    """
    The comparison of the model of an observation with its data, under its
    likelihood, on the driver of its model (see Objective): the
    residual of each value of its data items, one item after the other,
    and the terms of the chi-square beyond them (e.g. the noise residuals
    of a fitted noise).

    The model it compares can have scales (the solved values of a flux
    scale of the observation; see Objective): factors of the values of
    data items, which it applies to a copy of the model of its own.

    Its weighted residuals can be linear in the model, W (m - d) of a
    linear map W (see linear_form), and pointwise, of each value alone
    (see pointwise_form): elements of the parameters that the model is
    linear in can then be solved (see Objective). It gives the weighted
    pieces of the model, W p (see weighted_pieces), and of a solved scale
    its sums (see scale_sums).

    Its memory on the device (and the host) is allocated by prepare and
    released by close. The methods of the residuals return arrays that the
    next evaluation overwrites.

    Parameters
    ----------
    observation : Observation
        The observation, with data.
    driver : Driver
        The driver of its model.
    """

    def __init__(self, observation: Observation, driver: Driver):
        self._observation = observation
        self._driver = driver
        self._dataset = observation.data()
        # The number of values of each data item that the data do not mask
        # (over which the pixel weights of the model are normalised; see
        # _pixel_weight)
        self._nunmasked = {
            key: int(np.count_nonzero(self._dataset[key].mask()))
            for key in observation.keys()}
        self._reset_memory()

    def _reset_memory(self) -> None:
        """Drop its memory (see prepare and close)."""
        # The residuals, one data item after the other, in 1d arrays on the
        # host (h) and the device (d), and their views as the nd arrays of
        # each item
        self._h_residual_vector: np.ndarray | None = None
        self._d_residual_vector: DeviceArray | None = None
        self._h_residual_nddata: dict[str, np.ndarray] = {}
        self._d_residual_nddata: dict[str, DeviceArray] = {}
        # The masks of the data items on the device, by name
        self._d_mask: dict[str, DeviceArray] = {}
        # The sum of the squared residuals (1d, size 1; in double
        # precision, of millions of terms)
        self._h_residual_sum: np.ndarray | None = None
        self._d_residual_sum: DeviceArray | None = None
        # The scaled model of each data item that has scales, by name
        self._d_scaled_models: dict[str, DeviceArray] = {}
        # The native objective, for the driver of the observation
        self._backend = None
        self._prepared = False

    def observation(self) -> Observation:
        """Return the observation."""
        return self._observation

    def nresiduals(self) -> int:
        """Return the number of residuals that count (see counted)."""
        return int(np.count_nonzero(self.counted()))

    def counted(self) -> np.ndarray:
        """
        Return which of its residuals count, one bool per value of its data
        items, one item after the other (as its residuals): those of the
        values the data do not mask.
        """
        return np.concatenate([
            self._dataset[key].mask().ravel() != 0
            for key in self._observation.keys()])

    def nindependent(self) -> float:
        """
        Return the number of independent residuals that its residuals are
        worth (see Objective.nindependent): nresiduals here. It needs to be
        prepared.
        """
        return float(self.nresiduals())

    def ncensored(self) -> int:
        """
        Return the number of its residuals of censored values (see
        censored_chi2): none here.
        """
        return 0

    def censored_chi2(self) -> float:
        """
        Return the part of its chi-square of censored values, of the last
        evaluation: values below a limit (e.g. upper limits, see
        LikelihoodGauss), whose residuals are of the probability of
        being below it, about 0 where the model is well below it, rather
        than of a deviation; none here.
        """
        return 0.0

    def gives_residuals(self) -> bool:
        """
        Check whether it gives residuals (e.g. for least squares), or only
        its chi-square (see ComparisonUser).
        """
        return True

    def share(self, other: 'Comparison') -> None:
        """
        Take what it can share with a comparison of the same observation
        with other data, before it is prepared (see Objective.with_data):
        nothing here.
        """

    def prepare(self) -> None:
        """
        Allocate its memory and copy what it needs to the device; it does
        nothing once done (until close).
        """
        if self._prepared:
            return
        driver = self._driver
        dtype = self._observation.dtype()
        shape = self._dataset.shape()
        keys = self._observation.keys()
        (self._h_residual_vector,
         self._d_residual_vector) = driver.mem_alloc_s(
            math.prod(shape) * len(keys), dtype)
        (self._h_residual_sum,
         self._d_residual_sum) = driver.mem_alloc_s(1, np.float64)
        self._h_residual_nddata = _item_views(
            self._h_residual_vector, keys, shape)
        self._d_residual_nddata = _item_views(
            self._d_residual_vector, keys, shape)
        for key in keys:
            self._d_mask[key] = driver.mem_copy_h2d(
                np.asarray(self._dataset[key].mask(), dtype))
        self._backend = driver.native_class('Objective', dtype)()
        self._prepare(driver, shape, dtype)
        self._prepared = True

    def _prepare(
            self, driver: Driver, shape: tuple[int, ...], dtype: np.dtype
    ) -> None:
        """Allocate the memory of its kind (see prepare); none here."""

    def close(self) -> None:
        """Release its memory; prepare allocates it again."""
        self._reset_memory()

    @abc.abstractmethod
    def evaluate(
            self,
            model: ModelData,
            params: ParamValues,
            scales: Mapping[str, DeviceArray]
    ) -> None:
        """
        Evaluate its residuals, on the device, and the terms of the
        chi-square beyond them. It needs to be prepared.

        Parameters
        ----------
        model : Mapping
            The model of the observation (see GroupPlan.model_d).
        params : ParamValues
            The parameters, prefixed (see Objective.pdescs).
        scales : Mapping
            The scales of data items, by name (none for the others): arrays
            on the device that broadcast to the item.
        """

    def linear_form(self, keys: Collection[str]) -> str | None:
        """
        Return why its weighted residuals of the given data items are not
        W (m - d), linear in the model m by a map W of their own, or None
        if they are (e.g. amplitudes of the light are solved from the
        weighted pieces of the model, see weighted_pieces): here, they are
        not known to be.
        """
        return (
            f"the likelihood of the observation, "
            f"{self._observation.likelihood().type()}, is not known to be "
            f"linear in the model")

    def pointwise_form(self, keys: Collection[str]) -> str | None:
        """
        Return why its weighted residual of each value of the given data
        items is not c (m - d), of the model m of that value alone (W of
        linear_form diagonal), or None if it is (e.g. solved scales are
        solved from the sums of each spectrum, see scale_sums, and the
        weights of orbits from c, see pointwise_factors): here, the reason
        of linear_form.
        """
        return self.linear_form(keys)

    def pointwise_factors(
            self, keys: Collection[str], params: ParamValues
    ) -> dict[str, np.ndarray]:
        """
        Return the factor c of the weighted residual c (m - d) of each
        value of the given data items (see pointwise_form), without the
        pixel weights of the model, which multiply it (see Weights). Only
        comparisons whose weighted residuals are pointwise give them.

        Parameters
        ----------
        keys : Collection of str
            The names of the data items.
        params : ParamValues
            The parameters, prefixed (see Objective.pdescs).

        Returns
        -------
        dict
            The factors of each item, by name: new arrays of its shape, in
            float64.
        """
        raise NotImplementedError

    def weighted_pieces(
            self,
            pieces: LinearPieces,
            params: ParamValues,
            out: DeviceArray
    ) -> None:
        """
        Write its weighted pieces of the model (see
        GroupPlan.amplitude_pieces) to the rows of out, one for each piece,
        as its residuals: W p, of the map W of its weighted residuals (see
        linear_form). Only comparisons whose weighted residuals are linear
        in the model give them. It overwrites its residuals.

        Parameters
        ----------
        pieces : LinearPieces
            The pieces of the model of the observation.
        params : ParamValues
            The parameters, prefixed (see Objective.pdescs).
        out : DeviceArray
            The rows (float64), of the number of pieces by that of its
            residuals.
        """
        raise NotImplementedError

    def scale_sums(
            self,
            model: ModelData,
            params: ParamValues,
            key: str,
            axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the sums that the best scale of the model of a data item
        takes along an axis (see Solve): at each point of the other axes,
        A = sum w m^2 and B = sum w m d of the values along it, m the
        model, d the data and w the weight of the value in the chi-square.
        Only comparisons whose weighted residuals are pointwise give them
        (see pointwise_form). It overwrites the residuals of the item.

        Parameters
        ----------
        model : ModelData
            The model of the observation, with the scale at 1.
        params : ParamValues
            The parameters, prefixed (see Objective.pdescs).
        key : str
            The name of the data item.
        axis : int
            The axis of the item along which the values share a scale.

        Returns
        -------
        A, B : np.ndarray
            The sums: new arrays of the shape of the item, of 1 along the
            axis.

        Raises
        ------
        NotImplementedError
            If its weighted residuals are not pointwise.
        """
        raise NotImplementedError

    def noise_residuals(self) -> np.ndarray:
        """
        Return its noise residuals (see LikelihoodGauss), of the last
        evaluation: none here.
        """
        return np.zeros(0)

    def noise_constant(self) -> float:
        """
        Return the constant that its noise residuals add to -2 ln L (see
        LikelihoodGauss): none here.
        """
        return 0.0

    def chi2(self) -> float:
        """
        Return its chi-square, of the last evaluation: the sum of its
        squared residuals and of the terms beyond them.
        """
        self._backend.residual_sum(
            self._d_residual_vector, True, self._d_residual_sum)
        self._driver.mem_copy_d2h(
            self._d_residual_sum, self._h_residual_sum)
        return float(self._h_residual_sum[0]) + self._terms_chi2()

    def _terms_chi2(self) -> float:
        """
        Return the part of its chi-square beyond its residuals, of the
        last evaluation.
        """
        return float(np.sum(self.noise_residuals() ** 2))

    def copy_to_host(self) -> None:
        """Copy the residuals of the last evaluation to the host."""
        self._driver.mem_copy_d2h(
            self._d_residual_vector, self._h_residual_vector)

    def residual_vector_h(self) -> np.ndarray:
        """
        Return its residuals on the host (see copy_to_host), its data
        items one after the other.
        """
        return self._h_residual_vector

    def residual_vector_d(self) -> DeviceArray:
        """Return its residuals on the device (see residual_vector_h)."""
        return self._d_residual_vector

    def residual_nddata_h(self) -> dict[str, np.ndarray]:
        """
        Return the residuals of each data item on the host (see
        copy_to_host), of its shape.
        """
        return self._h_residual_nddata

    def residual_nddata_d(self) -> dict[str, DeviceArray]:
        """
        Return the residuals of each data item on the device (see
        residual_nddata_h).
        """
        return self._d_residual_nddata

    def _scaled_model(
            self,
            model: ModelData,
            key: str,
            scales: Mapping[str, DeviceArray]
    ) -> DeviceArray:
        """
        Return the model of data item key times its scales, if any (in a
        copy of its own, allocated on first use), or as it is.
        """
        if key not in scales:
            return model[key].data
        driver = self._driver
        if key not in self._d_scaled_models:
            self._d_scaled_models[key] = driver.mem_alloc_d(
                self._dataset.shape(), self._observation.dtype())
        return driver.math_mul(
            model[key].data, scales[key], out=self._d_scaled_models[key])

    def _pixel_weight(self, model: ModelData, key: str) -> float:
        """
        Return the factor of the weights of the model of data item key (its
        pixel weights, see Weights) that makes their mean square over the
        values the data do not mask 1; 1 without weights. It overwrites
        the residuals of the item (its scratch), so its callers write them
        after it.
        """
        # Pixel weights follow the free geometry of their disk, and the
        # chi-square expected at the truth is the sum of the squares of the
        # weights of the values compared. Unnormalised, a fit lowers it by
        # turning the geometry so that the values fall where the weights
        # are low: in a Monte Carlo of velocity fields, the inclination ran
        # 1 to 4 degrees high, and to the end of its range with cos^2
        # weights. Normalised, only the shape of the weights counts, and
        # the bias is gone. (A choice over weights fixed at a given
        # geometry, which remove it as well but stop following the disk.)
        weights = model[key].weights
        if weights is None:
            return 1.0
        driver = self._driver
        flat = weights.reshape(-1)
        # (the weights where the data do not mask them, in the residuals of
        # the item, which take the type of the model: no array of its size
        # allocated at each evaluation)
        unmasked = driver.math_mul(
            flat, self._d_mask[key].reshape(-1),
            out=self._d_residual_nddata[key].reshape(-1))
        total = float(driver.math_matmul(unmasked, flat))
        return float(np.sqrt(self._nunmasked[key] / total)) \
            if total > 0 else 1.0


class _NoiseParams(NamedTuple):
    """
    The parameters of the fitted noise of a data item (see
    LikelihoodGauss): the names of those of its scale and of its jitter
    (None for those not fitted), and the smallest scale (1 if the scale is
    not fitted).
    """
    scale: str | None
    jitter: str | None
    scale_min: float


class ComparisonGauss(Comparison):
    """
    The comparison of a Gaussian likelihood (see LikelihoodGauss).

    The residual of each value is (model - data) / error, times the
    weights of the model (if any; their pixel weights, normalised to a
    mean square of 1 over the values the data do not mask, see
    _pixel_weight) and of the data item. Values the data
    mask have a residual of 0 and do not count. Where the data have a
    value but the model none (NaN, e.g. the velocity of a spectrum without
    flux), the residual is the missing_residual of the likelihood (times
    the weight of the item): a model must not lower the chi-square by
    having no values. A data item with an upper limit gives its masked
    values the residual of a non-detection instead (within its footprint,
    if it has one; see Data).

    With a fitted noise, the residuals of its data items are over their
    noise in place of their errors, and each such item has a noise
    residual (see noise_residuals). The weighted residuals of a data item
    whose noise is correlated are whitened: the sum of their squares is
    its chi-square; prepare works out their whitening, which copies with
    other data can share (see share).

    Parameters
    ----------
    observation : Observation
        The observation, with data, of a Gaussian likelihood.
    driver : Driver
        The driver of its model.
    """

    def __init__(self, observation: Observation, driver: Driver):
        super().__init__(observation, driver)
        likelihood = observation.likelihood()
        keys = observation.keys()
        # The weight of each data item, and the residual of the values the
        # model does not have: NaN where the model cannot have missing
        # values, so that a missing one makes a chi-square that is not a
        # number, of a point fitters cannot evaluate (a model that
        # failed), even with a missing_residual
        self._weights = {key: likelihood.weight(key) for key in keys}
        missing = likelihood.missing_residual()
        self._missing_residual = missing \
            if missing is not None \
            and observation.model_can_have_missing_values() \
            else np.nan
        # The upper limit of the data items that have one, where they are
        # masked, by name, and the footprints of those that have both
        self._upper_limits: dict[str, UpperLimit] = {
            key: limit for key in keys
            if (limit := likelihood.upper_limit(key)) is not None}
        self._footprints = {
            key: footprint for key in self._upper_limits
            if (footprint := self._dataset[key].footprint()) is not None}
        # The parameters of the noise of the data items whose noise is
        # fitted, by name, and their noise residuals, in float64 (each sums
        # a term of every value of its item: float32 would round it off)
        self._noise_params = {
            key: _NoiseParams(
                *likelihood.noise_params(key),
                noise.scale_min if noise.scale else 1.0)
            for key, noise in likelihood.fitted_noise().items()}
        self._noise_residuals = np.zeros(len(self._noise_params))
        self._white_floor = likelihood.white_floor()

    def _reset_memory(self) -> None:
        super()._reset_memory()
        # The data and errors of the data items on the device, by name
        # (errors only of the items that have them)
        self._d_data: dict[str, DeviceArray] = {}
        self._d_error: dict[str, DeviceArray] = {}
        # The footprints of upper limits, by name
        self._d_footprints: dict[str, DeviceArray] = {}
        # Data of 0 of each item, by name, allocated on first use: the
        # weighted residual of a piece of the model p is W p (see
        # weighted_pieces)
        self._d_zeros: dict[str, DeviceArray] = {}
        # The sums of each item and axis of scales (A and B; see
        # scale_sums), allocated on first use
        self._h_scale_sums: dict[tuple[str, int], list[np.ndarray]] = {}
        self._d_scale_sums: dict[tuple[str, int], list[DeviceArray]] = {}
        # The squares of the noise residuals, on the host (h) and the
        # device (d)
        self._h_noise_sums: np.ndarray | None = None
        self._d_noise_sums: DeviceArray | None = None
        # The whitening of the residuals of the data items whose noise is
        # correlated, by name, and the numbers of independent values they
        # are worth (see Whitening)
        self._whiteners: dict[str, Whitener] = {}
        self._nindependent: dict[str, float] = {}
        # The whitening to take from a copy of it of other data, of each
        # data item, by name (see share), until prepared
        self._shared: dict[str, tuple[Whitener, float]] = {}
        # The share of its values that each data item whose noise is fitted
        # counts in its noise residual: 1, but the independent values they
        # are worth over their number for correlated noise (see
        # LikelihoodGauss)
        self._noise_shares: np.ndarray | None = None

    def counted(self) -> np.ndarray:
        """
        Return which of its residuals count (see Comparison.counted): those
        of the values the data do not mask (the values the model does not
        have count too, with the missing_residual of the likelihood), and
        all values of the data items with upper limits (within their
        footprints, if any).
        """
        counted = []
        for key in self._observation.keys():
            mask = self._dataset[key].mask()
            if key in self._footprints:
                counted.append(self._footprints[key].ravel() != 0)
            elif key in self._upper_limits:
                counted.append(np.ones(mask.size, dtype=bool))
            else:
                counted.append(mask.ravel() != 0)
        return np.concatenate(counted)

    def ncensored(self) -> int:
        """
        Return the number of its residuals of censored values: the values
        the data mask of the data items with upper limits (within their
        footprints, if any).
        """
        return sum(
            int(np.count_nonzero(censored))
            for censored in map(self._censored, self._observation.keys())
            if censored is not None)

    def censored_chi2(self) -> float:
        total = 0.0
        for key in self._observation.keys():
            censored = self._censored(key)
            if censored is not None:
                residuals = self._driver.mem_copy_d2h(
                    self._d_residual_nddata[key])
                total += float(np.sum(
                    residuals[censored].astype(np.float64) ** 2))
        return total

    def _censored(self, key: str) -> np.ndarray | None:
        """
        Return which values of data item key are censored (see ncensored),
        or None if it has no upper limit.
        """
        if key not in self._upper_limits:
            return None
        censored = self._dataset[key].mask() == 0
        if key in self._footprints:
            censored &= self._footprints[key] != 0
        return censored

    def nindependent(self) -> float:
        """
        Return the number of independent residuals that its residuals are
        worth: nresiduals, but fewer for the data items whose noise is
        correlated (see Whitening). It needs to be prepared.
        """
        total = float(self.nresiduals())
        for key, n in self._nindependent.items():
            total += n - self._nunmasked[key]
        return total

    def share(self, other: Comparison) -> None:
        """
        Take the whitening of the residuals of the data items whose noise
        is correlated from a comparison of the same observation with other
        data, if worked out (see prepare), for those of the same
        correlation and mask.
        """
        if not isinstance(other, ComparisonGauss):
            return
        for key, whitener in other._whiteners.items():
            item, old = self._dataset[key], other._dataset[key]
            if item.spatial_correlation() is old.spatial_correlation() \
                    and item.spectral_correlation() \
                    is old.spectral_correlation() \
                    and np.array_equal(item.mask(), old.mask()):
                self._shared[key] = (whitener, other._nindependent[key])

    def _prepare(
            self, driver: Driver, shape: tuple[int, ...], dtype: np.dtype
    ) -> None:
        """
        Copy the data to the device, and work out the whitening of the
        residuals of data items whose noise is correlated (once, for every
        evaluation; e.g. about 4 s for 7000 pixels), unless shared.
        """
        for key in self._observation.keys():
            data = self._dataset[key]
            self._d_data[key] = driver.mem_copy_h2d(
                np.asarray(data.data(), dtype))
            if data.error() is not None:
                self._d_error[key] = driver.mem_copy_h2d(
                    np.asarray(data.error(), dtype))
            if key in self._footprints:
                self._d_footprints[key] = driver.mem_copy_h2d(
                    self._footprints[key].astype(dtype))
            if key in self._shared:
                # (dropped once taken: the whitener of the other holds
                # residuals of its own, which this one does not need)
                whitener, nindependent = self._shared.pop(key)
                self._whiteners[key] = whitener.shared()
                self._nindependent[key] = nindependent
            elif (correlation := self._dataset.noise_correlation(key)) \
                    is not None:
                mask = channel_mask(self._dataset, key)
                whitening_ = whitening(
                    key, mask, correlation, self._white_floor)
                self._whiteners[key] = Whitener.on_device(
                    driver, whitening_, mask.shape[0], dtype)
                self._nindependent[key] = whitening_.nindependent
        if self._noise_params:
            (self._h_noise_sums,
             self._d_noise_sums) = driver.mem_alloc_s(
                len(self._noise_params), np.float64)
        # The terms of the noise residual of an item whose noise is
        # correlated count by the share n_ind / n of the n values that
        # count, n_ind the independent values they are worth (see
        # Whitening): their sum is n_ind log f^2 for a scale f. With all n
        # (the Gaussian of C exactly), the best scale would be f^2 = chi2 /
        # n, too small where the noise of the data has less variance than
        # the floor gives C (e.g. beam noise alone, which has none from
        # pixel to pixel): on 20 mocks of NGC 1313 of LVHIS, f = 0.47 of
        # the true noise, and errors half the scatter of the fits; counting
        # n_ind, f = 1.00 and errors of the scatter. Hence n_ind, rather
        # than the exact Gaussian.
        self._noise_shares = np.ones(len(self._noise_params))
        for j, key in enumerate(self._noise_params):
            if key in self._nindependent:
                self._noise_shares[j] = \
                    self._nindependent[key] / self._nunmasked[key]

    def evaluate(
            self,
            model: ModelData,
            params: ParamValues,
            scales: Mapping[str, DeviceArray]
    ) -> None:
        driver = self._driver
        scaled_models = {
            key: self._scaled_model(model, key, scales)
            for key in self._observation.keys()}
        # The weight of each data item, times the normalisation of the
        # pixel weights of its model, which its noise residual takes too
        weights = {}
        for key in self._observation.keys():
            weights[key] = self._weights[key] * self._pixel_weight(model, key)
            residual = self._d_residual_nddata[key]
            self._backend.residual(
                self._d_data[key],
                self._d_error.get(key),
                self._d_mask[key],
                scaled_models[key],
                model[key].weights,
                weights[key],
                self._missing_residual,
                *self._upper_limits.get(key, _NO_UPPER_LIMIT),
                *self._noise_values(key, params),
                residual)
            if key in self._d_footprints:
                # (the limits count nothing outside the footprint)
                driver.math_mul(
                    residual, self._d_footprints[key], out=residual)
            if key in self._whiteners:
                self._whiteners[key](residual)
        if self._noise_params:
            self._update_noise_residuals(
                model, scaled_models, weights, params)

    def linear_form(self, keys: Collection[str]) -> str | None:
        """
        Return why its weighted residuals of the given data items are not
        linear in the model (see Comparison.linear_form), or None if they
        are: a data item with an upper limit, whose residuals of the
        values below it are not. (The residual of a value missing from
        the model is a constant: the models of observables linear in the
        light have every value, see Observable.linear_in_light.)
        """
        for key in keys:
            if key in self._upper_limits:
                return (
                    f"'{key}' has an upper limit, whose residuals are not "
                    f"linear in the model")
        return None

    def pointwise_form(self, keys: Collection[str]) -> str | None:
        """
        Return why its weighted residual of each value of the given data
        items is not of the model of that value alone (see
        Comparison.pointwise_form), or None if it is: that of linear_form,
        or a data item with correlated noise, whose whitened residuals mix
        the values.
        """
        if (reason := self.linear_form(keys)) is not None:
            return reason
        for key in keys:
            if self._dataset.noise_correlation(key) is not None:
                return (
                    f"'{key}' has correlated noise, whose whitened residuals "
                    f"mix the values")
        return None

    def pointwise_factors(
            self, keys: Collection[str], params: ParamValues
    ) -> dict[str, np.ndarray]:
        """
        Return the factor c of the weighted residual c (m - d) of each
        value of the given data items (see Comparison.pointwise_factors):
        the weight of the item times the mask, over the noise of the value
        (see LikelihoodGauss.noise); 0 where the data mask it.
        """
        likelihood = self._observation.likelihood()
        factors = {}
        for key in keys:
            item = self._dataset[key]
            mask = np.asarray(item.mask(), np.float64)
            error = None if item.error() is None \
                else np.asarray(item.error(), np.float64)
            # (the errors are NaN where the data mask the values)
            factors[key] = np.where(
                mask != 0,
                self._weights[key] * mask
                / likelihood.noise(key, error, params), 0)
        return factors

    def weighted_pieces(
            self,
            pieces: LinearPieces,
            params: ParamValues,
            out: DeviceArray
    ) -> None:
        # (W p is the residual of p of data 0, without upper limits or a
        # residual of values missing from the model, which W leaves
        # out)
        driver = self._driver
        keys = self._observation.keys()
        if not self._d_zeros:
            self._alloc_zeros()
        weights = {
            key: self._weights[key] * self._pixel_weight(pieces.base, key)
            for key in keys}
        noise_values = {
            key: self._noise_values(key, params) for key in keys}
        for k, row in enumerate(out):
            for key in keys:
                residual = self._d_residual_nddata[key]
                self._backend.residual(
                    self._d_zeros[key],
                    self._d_error.get(key),
                    self._d_mask[key],
                    pieces.pieces[key][k],
                    pieces.base[key].weights,
                    weights[key],
                    np.nan, *_NO_UPPER_LIMIT,
                    *noise_values[key],
                    residual)
                if key in self._whiteners:
                    self._whiteners[key](residual)
            driver.mem_copy_d2d(self._d_residual_vector, row)

    def _alloc_zeros(self) -> None:
        """Allocate the data of 0 of each item (see weighted_pieces)."""
        driver = self._driver
        shape = self._dataset.shape()
        keys = self._observation.keys()
        zeros = driver.mem_alloc_d(
            math.prod(shape) * len(keys), self._observation.dtype())
        driver.mem_fill(zeros, 0)
        self._d_zeros = _item_views(zeros, keys, shape)

    def scale_sums(
            self,
            model: ModelData,
            params: ParamValues,
            key: str,
            axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the sums that the best scale of the model of a data item
        takes along an axis (see Comparison.scale_sums), w the weight of
        the value in the chi-square, (weight / noise)^2 of the weights of
        the model, of the data mask and of the item, and of the noise of
        the value.
        """
        driver = self._driver
        if (key, axis) not in self._d_scale_sums:
            shape = tuple(
                1 if k == axis else n
                for k, n in enumerate(self._dataset.shape()))
            sums = [driver.mem_alloc_s(shape, np.float64) for _ in range(2)]
            self._h_scale_sums[key, axis] = [h for h, _ in sums]
            self._d_scale_sums[key, axis] = [d for _, d in sums]
        h_sums = self._h_scale_sums[key, axis]
        d_sums = self._d_scale_sums[key, axis]
        self._backend.amplitude_sums(
            self._d_data[key], self._d_error.get(key), self._d_mask[key],
            model[key].data, model[key].weights,
            *self._noise_values(key, params), axis, *d_sums)
        for h, d in zip(h_sums, d_sums):
            driver.mem_copy_d2h(d, h)
        weight = (self._weights[key] * self._pixel_weight(model, key)) ** 2
        return weight * h_sums[0], weight * h_sums[1]

    def noise_constant(self) -> float:
        # (the noise residual of a fitted scale is sum c^2 ln(sigma^2 /
        # (f_ref e)^2), f_ref = min(f_min, 1), for least squares never
        # negative, and those of the values missing from the model c^2
        # ln(1 / f_ref^2): its part sum c^2 ln(f_ref^2), with sum c^2 over
        # the values compared (the pixel weights normalised to their
        # number; fewer independent values of correlated noise), is a
        # constant of scale_min alone, which -2 ln L and the AICp must not
        # hold)
        return float(sum(
            self._weights[key] ** 2 * self._nunmasked[key] * share
            * 2 * np.log(min(noise.scale_min, 1.0))
            for (key, noise), share in zip(
                self._noise_params.items(), self._noise_shares)))

    def noise_residuals(self) -> np.ndarray:
        """
        Return its noise residuals (see LikelihoodGauss), of the last
        evaluation: one for each data item whose noise is fitted (none
        without); not a copy.
        """
        return self._noise_residuals

    def _noise_values(
            self, key: str, params: ParamValues
    ) -> tuple[float, float]:
        """
        Return the scale and the jitter of the noise of data item key: 1
        and 0 for those that are not fitted.
        """
        noise = self._noise_params.get(key)
        if noise is None:
            return 1.0, 0.0
        scale = 1.0 if noise.scale is None else float(params[noise.scale])
        jitter = 0.0 if noise.jitter is None else float(params[noise.jitter])
        return scale, jitter

    def _update_noise_residuals(
            self,
            model: ModelData,
            scaled_models: Mapping[str, DeviceArray],
            weights: Mapping[str, float],
            params: ParamValues
    ) -> None:
        """
        Evaluate the noise residuals, of the scaled models of the data
        items, with the weight of each item that its residuals take (see
        evaluate).
        """
        # The noise residual of an item is sqrt(sum c^2 log(sigma^2 /
        # (f_ref e)^2)), f_ref the minimum of the scale, at most 1 (1 if
        # the scale is not fitted): its part of -2 log L, sum c^2 log
        # sigma^2, less sum c^2 log (f_ref e)^2, which does not depend on
        # the noise, so that the sum is never negative, as a residual
        # needs (the scale is at least its minimum, the jitter at least 0;
        # see noise_constant). A value missing from the model costs
        # missing_residual, which does not take the noise, and adds the
        # term of the noise of its error, c^2 log(1 / f_ref^2), which the
        # noise does not change: taking its noise would bias the scale low
        # (by about half the share of such values), and adding nothing
        # would make a missing value cheaper by the term (9.2 at a minimum
        # scale of 0.01, against a missing_residual of 3 costing 9).
        driver = self._driver
        for j, (key, noise) in enumerate(self._noise_params.items()):
            self._backend.noise_sum(
                self._d_error.get(key),
                self._d_mask[key],
                scaled_models[key],
                model[key].weights,
                weights[key],
                *self._noise_values(key, params),
                noise.scale_min,
                self._d_noise_sums[j:j + 1])
        driver.mem_copy_d2h(self._d_noise_sums, self._h_noise_sums)
        # (each term is at least 0, but rounding can leave a sum of terms
        # of 0, at the smallest scale and no jitter, a hair below it)
        self._noise_residuals[...] = np.sqrt(
            np.maximum(self._h_noise_sums, 0) * self._noise_shares)


class ComparisonUser(Comparison):
    """
    The comparison of a user likelihood (see LikelihoodUser), on the host:
    the residuals its function gives for each data item, times the
    weights of the model and of the data mask; or, of a function of the
    log-likelihood, residuals of 0 and the chi-square -2 log L, which
    gives no residuals for least squares (see gives_residuals).

    Parameters
    ----------
    observation : Observation
        The observation, with data, of a user likelihood.
    driver : Driver
        The driver of its model.
    """

    def __init__(self, observation: Observation, driver: Driver):
        super().__init__(observation, driver)
        self._likelihood: LikelihoodUser = observation.likelihood()
        # The data, errors and mask of each data item that its function
        # takes, by name: read-only views, so that it cannot change the
        # dataset (e.g. by 'data -= 1')
        self._arrays = {
            key: tuple(map(_read_only, (
                self._dataset[key].data(), self._dataset[key].error(),
                self._dataset[key].mask())))
            for key in observation.keys()}
        # The chi-square of a function of the log-likelihood
        self._chi2 = 0.0

    def gives_residuals(self) -> bool:
        return self._likelihood.gives_residuals()

    def evaluate(
            self,
            model: ModelData,
            params: ParamValues,
            scales: Mapping[str, DeviceArray]
    ) -> None:
        driver = self._driver
        self._chi2 = 0.0
        for key in self._observation.keys():
            data, error, mask = self._arrays[key]
            values = np.asarray(
                driver.mem_copy_d2h(self._scaled_model(model, key, scales)),
                dtype=np.float64)
            result = self._likelihood.evaluate(
                key, data, values, error, mask, params)
            residuals = self._h_residual_nddata[key]
            if self._likelihood.form() == 'log_likelihood':
                self._chi2 -= 2 * result
                residuals[...] = 0
                continue
            if (weights := model[key].weights) is not None:
                # (normalised as those of every comparison, before the
                # residuals of the item are written: see _pixel_weight)
                result = result * self._pixel_weight(model, key) \
                    * np.asarray(driver.mem_copy_d2h(weights), float)
            residuals[...] = np.where(mask != 0, result * mask, 0)
        driver.mem_copy_h2d(self._h_residual_vector, self._d_residual_vector)

    def _terms_chi2(self) -> float:
        return self._chi2


# The comparison of each kind of likelihood (see comparison_of)
_COMPARISONS: dict[type[Likelihood], type[Comparison]] = {
    LikelihoodGauss: ComparisonGauss,
    LikelihoodUser: ComparisonUser
}


def comparison_of(observation: Observation, driver: Driver) -> Comparison:
    """
    Return the comparison of an observation with its data, of the kind of
    its likelihood (that of the nearest base class of the likelihood that
    has one), on the driver of its model.

    Parameters
    ----------
    observation : Observation
        The observation, with data.
    driver : Driver
        The driver of its model.

    Returns
    -------
    Comparison
        The comparison.

    Raises
    ------
    ValueError
        If no comparison is of its kind of likelihood.
    """
    likelihood = observation.likelihood()
    for kind in type(likelihood).__mro__:
        if kind in _COMPARISONS:
            return _COMPARISONS[kind](observation, driver)
    raise ValueError(
        f"no comparison is of the likelihood {likelihood.type()!r}")


def _read_only(array: np.ndarray | None) -> np.ndarray | None:
    """Return a read-only view of an array, or None for None."""
    if array is None:
        return None
    view = array.view()
    view.flags.writeable = False
    return view


def _item_views(
        vector: DeviceArray, keys: Sequence[str], shape: tuple[int, ...]
) -> dict[str, DeviceArray]:
    """
    Return the views of a 1d array of the values of data items of a
    shape, one item after the other, as arrays of the shape, by name.
    """
    size = math.prod(shape)
    return {
        key: vector[j * size:(j + 1) * size].reshape(shape)
        for j, key in enumerate(keys)}
