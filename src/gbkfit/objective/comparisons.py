import abc
from collections.abc import Collection, Mapping
from typing import NamedTuple, Self

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.observation import LinearPieces, Observation
from gbkfit.observation.likelihood import (
    LikelihoodGaussian, LikelihoodUser, Whitening, whitening)
from gbkfit.params import ParamValues
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Comparison',
    'ComparisonGaussian',
    'ComparisonUser',
    'comparison'
]


# The model of an observation: the parts of each data item (see
# Observation), by name
_Model = Mapping[str, Mapping[str, DeviceArray | None]]


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
    pieces of the model, W p (see weigh), and of a solved scale its sums
    (see scale_sums).

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
        # The sum of the squared residuals (1d, size 1; in double
        # precision, of millions of terms)
        self._h_residual_sum: np.ndarray | None = None
        self._d_residual_sum: DeviceArray | None = None
        # The scaled model of each data item that has scales, by name
        self._d_scaled: dict[str, DeviceArray] = {}
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
        pass

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
        npix = int(np.prod(shape))
        keys = self._observation.keys()
        (self._h_residual_vector,
         self._d_residual_vector) = driver.mem_alloc_s(npix * len(keys), dtype)
        (self._h_residual_sum,
         self._d_residual_sum) = driver.mem_alloc_s(1, np.float64)
        for j, key in enumerate(keys):
            slice_ = slice(j * npix, (j + 1) * npix)
            self._h_residual_nddata[key] = \
                self._h_residual_vector[slice_].reshape(shape)
            self._d_residual_nddata[key] = \
                self._d_residual_vector[slice_].reshape(shape)
        self._backend = driver.native_class('Objective', dtype)()
        self._prepare(driver, shape, dtype)
        self._prepared = True

    def _prepare(
            self, driver: Driver, shape: tuple[int, ...], dtype: np.dtype
    ) -> None:
        """Allocate the memory of its kind (see prepare); none here."""
        pass

    def close(self) -> None:
        """Release its memory; prepare allocates it again."""
        self._reset_memory()

    @abc.abstractmethod
    def evaluate(
            self,
            model: _Model,
            params: ParamValues,
            weighted: bool,
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
        weighted : bool
            Whether to apply the weights of the data items (from the
            likelihood).
        scales : Mapping
            The scales of data items, by name (none for the others): arrays
            on the device that broadcast to the item.
        """
        pass

    def linear_form(self, keys: Collection[str]) -> str | None:
        """
        Return why its weighted residuals of the given data items are not
        W (m - d), linear in the model m by a map W of their own, or None
        if they are (e.g. amplitudes of the light are solved from the
        weighted pieces of the model, see weigh): here, they are not known
        to be.
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
        weights of orbits from c, the residual of the model d + 1): here,
        the reason of linear_form.
        """
        return self.linear_form(keys)

    def weigh(
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
            model: _Model,
            params: ParamValues,
            key: str,
            axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the sums that the best scale of the model of a data item
        takes along an axis (see Solve): at each point of the other axes,
        A = sum w m^2 and B = sum w m d of the values along it, m the
        model, d the data and w the weight of the value in the chi-square;
        new arrays of the shape of the item, of 1 along the axis. Only
        comparisons whose weighted residuals are pointwise give them (see
        pointwise_form).
        """
        raise NotImplementedError

    def noise_residuals(self) -> np.ndarray:
        """
        Return its noise residuals (see LikelihoodGaussian), of the last
        evaluation: none here.
        """
        return np.zeros(0)

    def noise_constant(self) -> float:
        """
        Return the constant that its noise residuals add to -2 ln L (see
        LikelihoodGaussian): none here.
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

    def _scaled(
            self,
            model: _Model,
            key: str,
            scales: Mapping[str, DeviceArray]
    ) -> DeviceArray:
        """
        Return the model of data item key times its scales, if any (in a
        copy of its own, allocated on first use), or as it is.
        """
        if key not in scales:
            return model[key]['d']
        driver = self._driver
        if key not in self._d_scaled:
            self._d_scaled[key] = driver.mem_alloc_d(
                self._dataset.shape(), self._observation.dtype())
        return driver.math_mul(
            model[key]['d'], scales[key], out=self._d_scaled[key])


class _Noise(NamedTuple):
    """
    The fitted noise of a data item (see LikelihoodGaussian): the names of
    the parameters of its scale and of its jitter (None for those not
    fitted), and the smallest scale (1 if the scale is not fitted).
    """
    scale: str | None
    jitter: str | None
    scale_min: float


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


class ComparisonGaussian(Comparison):
    """
    The comparison of a Gaussian likelihood (see LikelihoodGaussian).

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
    if any).

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

    Raises
    ------
    ConfigError
        If the footprint of an upper limit is not of the shape of its item
        or of its last axes, or the item has values outside it.
    """

    def __init__(self, observation: Observation, driver: Driver):
        super().__init__(observation, driver)
        likelihood = observation.likelihood()
        keys = observation.keys()
        # The weight of each data item, and the residual of the values the
        # model does not have (none where the model cannot lack values:
        # NaN would show one)
        self._weights = {key: likelihood.weight(key) for key in keys}
        # The number of values of each data item that the data do not mask
        # (over which the weights of the model are normalised)
        self._nunmasked = {
            key: int(np.count_nonzero(self._dataset[key].mask()))
            for key in keys}
        missing = likelihood.missing_residual()
        self._missing_residual = np.nan if missing is None else missing
        # The upper limit of each data item where it is masked, and its
        # noise (NaN for none), and the footprints of those that have one
        self._upper_limits = {
            key: likelihood.upper_limit(key) or (np.nan, np.nan)
            for key in keys}
        self._footprints = {
            key: _footprint(key, likelihood.footprint(key), self._dataset)
            for key in keys if likelihood.footprint(key) is not None}
        # The data items whose noise is fitted, by name, and their noise
        # residuals, in float64 (each sums a term of every value of its
        # item: float32 would round it off)
        self._noise = {
            key: _Noise(*likelihood.noise_params(key),
                        noise.scale_min if noise.scale else 1.0)
            for key, noise in likelihood.fitted_noise().items()}
        self._noise_residuals = np.zeros(len(self._noise))
        self._white_floor = likelihood.white_floor()
        # The whitening to take from a copy of it of other data, of each
        # data item, by name (see share)
        self._shared: dict[str, tuple[_Whitener, float]] = {}

    def _reset_memory(self) -> None:
        super()._reset_memory()
        # The data items, one after the other, in 1d arrays: the data (d),
        # masks (m) and errors (e), and their views as the nd arrays of
        # each item
        self._d_data_vector: DeviceArray | None = None
        self._d_mask_vector: DeviceArray | None = None
        self._d_error_vector: DeviceArray | None = None
        self._d_data: dict[str, DeviceArray] = {}
        self._d_mask: dict[str, DeviceArray] = {}
        self._d_error: dict[str, DeviceArray] = {}
        # The footprints of upper limits, by name
        self._d_footprints: dict[str, DeviceArray] = {}
        # Data of 0 of each item, by name, allocated on first use: the
        # weighted residual of a piece of the model p is W p (see weigh)
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
        self._whiteners: dict[str, _Whitener] = {}
        self._nindependent: dict[str, float] = {}
        # The share of its values that each data item whose noise is fitted
        # counts in its noise residual: 1, but the independent values they
        # are worth over their number for correlated noise (see
        # LikelihoodGaussian)
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
            elif not np.isnan(self._upper_limits[key][0]):
                counted.append(np.ones(mask.size, dtype=bool))
            else:
                counted.append(mask.ravel() != 0)
        return np.concatenate(counted)

    def nindependent(self) -> float:
        """
        Return the number of independent residuals that its residuals are
        worth: nresiduals, but fewer for the data items whose noise is
        correlated (see Whitening). It needs to be prepared.
        """
        total = float(self.nresiduals())
        for key, n in self._nindependent.items():
            total += n - np.count_nonzero(self._dataset[key].mask())
        return total

    def share(self, other: Comparison) -> None:
        """
        Take the whitening of the residuals of the data items whose noise
        is correlated from a comparison of the same observation with other
        data, if worked out (see prepare), for those of the same
        correlation and mask.
        """
        if not isinstance(other, ComparisonGaussian):
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
        keys = self._observation.keys()
        npix = int(np.prod(shape))
        nelem = npix * len(keys)
        self._d_data_vector = driver.mem_alloc_d(nelem, dtype)
        self._d_mask_vector = driver.mem_alloc_d(nelem, dtype)
        self._d_error_vector = driver.mem_alloc_d(nelem, dtype)
        for j, key in enumerate(keys):
            data = self._dataset[key]
            slice_ = slice(j * npix, (j + 1) * npix)
            self._d_data[key] = _copied(
                driver, data.data(), self._d_data_vector[slice_], shape)
            if data.mask() is not None:
                self._d_mask[key] = _copied(
                    driver, data.mask(), self._d_mask_vector[slice_], shape)
            if data.error() is not None:
                self._d_error[key] = _copied(
                    driver, data.error(), self._d_error_vector[slice_], shape)
            if key in self._footprints:
                self._d_footprints[key] = driver.mem_copy_h2d(
                    self._footprints[key].astype(dtype))
            if key in self._shared:
                whitener, nindependent = self._shared[key]
                self._whiteners[key] = whitener.shared()
                self._nindependent[key] = nindependent
            elif (correlation := self._dataset.noise_correlation(key)) \
                    is not None:
                whitening_ = whitening(
                    key, data.mask(), correlation, self._white_floor)
                self._whiteners[key] = _Whitener.on_device(
                    driver, whitening_, shape, dtype)
                self._nindependent[key] = whitening_.nindependent
        if self._noise:
            (self._h_noise_sums,
             self._d_noise_sums) = driver.mem_alloc_s(
                len(self._noise), np.float64)
        self._noise_shares = np.ones(len(self._noise))
        for j, key in enumerate(self._noise):
            if key in self._nindependent:
                self._noise_shares[j] = self._nindependent[key] \
                    / np.count_nonzero(self._dataset[key].mask())

    def evaluate(
            self,
            model: _Model,
            params: ParamValues,
            weighted: bool,
            scales: Mapping[str, DeviceArray]
    ) -> None:
        driver = self._driver
        scaled = {
            key: self._scaled(model, key, scales)
            for key in self._observation.keys()}
        for key in self._observation.keys():
            weight = (self._weights[key] if weighted else 1.0) \
                * self._pixel_weight(model, key)
            residual = self._d_residual_nddata[key]
            self._backend.residual(
                self._d_data[key],
                self._d_error.get(key),
                self._d_mask.get(key),
                scaled[key],
                model[key]['w'],
                weight,
                self._missing_residual,
                *self._upper_limits[key],
                *self._noise_values(key, params),
                residual)
            if key in self._d_footprints:
                # (the limits count nothing outside the footprint)
                driver.math_mul(
                    residual, self._d_footprints[key], out=residual)
            if weighted and key in self._whiteners:
                self._whiteners[key](residual)
        if self._noise:
            self._update_noise_residuals(model, scaled, params, weighted)

    def linear_form(self, keys: Collection[str]) -> str | None:
        """
        Return why its weighted residuals of the given data items are not
        linear in the model (see Comparison.linear_form), or None if they
        are: a data item with an upper limit, whose residuals of the
        values below it are not. (The residual of a value the model lacks
        is a constant: the models of observables linear in the light have
        every value, see Observable.linear_in_light.)
        """
        for key in keys:
            if not np.isnan(self._upper_limits[key][0]):
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

    def weigh(
            self,
            pieces: LinearPieces,
            params: ParamValues,
            out: DeviceArray
    ) -> None:
        # (W p is the residual of p of data 0, without upper limits or a
        # residual of values the model lacks, which W leaves out)
        driver = self._driver
        keys = self._observation.keys()
        if not self._d_zeros:
            self._alloc_zeros()
        weights = {
            key: self._weights[key] * self._pixel_weight(pieces.base, key)
            for key in keys}
        noise = {key: self._noise_values(key, params) for key in keys}
        for k, row in enumerate(out):
            for key in keys:
                residual = self._d_residual_nddata[key]
                self._backend.residual(
                    self._d_zeros[key],
                    self._d_error.get(key),
                    self._d_mask.get(key),
                    pieces.pieces[key][k],
                    pieces.base[key]['w'],
                    weights[key],
                    np.nan, np.nan, np.nan,
                    *noise[key],
                    residual)
                if key in self._whiteners:
                    self._whiteners[key](residual)
            driver.mem_copy_d2d(self._d_residual_vector, row)

    def _alloc_zeros(self) -> None:
        """Allocate the data of 0 of each item (see weigh)."""
        driver = self._driver
        shape = self._dataset.shape()
        npix = int(np.prod(shape))
        keys = self._observation.keys()
        zeros = driver.mem_alloc_d(npix * len(keys), self._observation.dtype())
        driver.mem_fill(zeros, 0)
        for j, key in enumerate(keys):
            self._d_zeros[key] = zeros[j * npix:(j + 1) * npix].reshape(shape)

    def scale_sums(
            self,
            model: _Model,
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
            self._d_data[key], self._d_error.get(key), self._d_mask.get(key),
            model[key]['d'], model[key]['w'],
            *self._noise_values(key, params), axis, *d_sums)
        for h, d in zip(h_sums, d_sums):
            driver.mem_copy_d2h(d, h)
        weight = (self._weights[key] * self._pixel_weight(model, key)) ** 2
        return weight * h_sums[0], weight * h_sums[1]

    def _pixel_weight(self, model: _Model, key: str) -> float:
        """
        Return the factor of the weights of the model of data item key (its
        pixel weights, see Weights) that makes their mean square over the
        values the data do not mask 1; 1 without weights.
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
        weights = model[key]['w']
        if weights is None:
            return 1.0
        driver = self._driver
        flat = weights.reshape(-1)
        mask = self._d_mask.get(key)
        unmasked = flat if mask is None \
            else driver.math_mul(flat, mask.reshape(-1))
        total = float(driver.math_matmul(unmasked, flat))
        return float(np.sqrt(self._nunmasked[key] / total)) \
            if total > 0 else 1.0

    def noise_constant(self) -> float:
        # (the noise residual of a fitted scale is sum c^2 ln(sigma^2 /
        # (f_min e)^2), for least squares never negative: its part sum c^2
        # ln(f_min^2), with sum c^2 over the values compared (the pixel
        # weights normalised to their number; fewer independent values of
        # correlated noise), is a constant of scale_min alone, which
        # -2 ln L and the AICp must not hold)
        return float(sum(
            self._weights[key] ** 2 * self._nunmasked[key] * share
            * 2 * np.log(noise.scale_min)
            for (key, noise), share in zip(
                self._noise.items(), self._noise_shares)))

    def noise_residuals(self) -> np.ndarray:
        """
        Return its noise residuals (see LikelihoodGaussian), of the last
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
        noise = self._noise.get(key)
        if noise is None:
            return 1.0, 0.0
        scale = 1.0 if noise.scale is None else float(params[noise.scale])
        jitter = 0.0 if noise.jitter is None else float(params[noise.jitter])
        return scale, jitter

    def _update_noise_residuals(
            self,
            model: _Model,
            scaled: Mapping[str, DeviceArray],
            params: ParamValues,
            weighted: bool
    ) -> None:
        """
        Evaluate the noise residuals, with the weights of the data items
        if weighted (as the residuals).
        """
        driver = self._driver
        for j, (key, noise) in enumerate(self._noise.items()):
            weight = (self._weights[key] if weighted else 1.0) \
                * self._pixel_weight(model, key)
            self._backend.noise_sum(
                self._d_error.get(key),
                self._d_mask.get(key),
                scaled[key],
                model[key]['w'],
                weight,
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
        # The chi-square of a function of the log-likelihood
        self._chi2 = 0.0

    def gives_residuals(self) -> bool:
        return self._likelihood.form() == 'residuals'

    def evaluate(
            self,
            model: _Model,
            params: ParamValues,
            weighted: bool,
            scales: Mapping[str, DeviceArray]
    ) -> None:
        driver = self._driver
        self._chi2 = 0.0
        for key in self._observation.keys():
            item = self._dataset[key]
            mask = item.mask()
            values = np.asarray(
                driver.mem_copy_d2h(self._scaled(model, key, scales)),
                dtype=np.float64)
            result = self._likelihood.evaluate(
                key, item.data(), values, item.error(), mask, params)
            residuals = self._h_residual_nddata[key]
            if self._likelihood.form() == 'log_likelihood':
                self._chi2 -= 2 * result
                residuals[...] = 0
                continue
            if (weights := model[key]['w']) is not None:
                # (normalised to a mean square of 1 over the values the
                # data do not mask, as ComparisonGaussian._pixel_weight)
                weights = np.asarray(driver.mem_copy_d2h(weights), float)
                unmasked = weights if mask is None else weights[mask != 0]
                total = float(np.sum(unmasked ** 2))
                if total > 0:
                    weights = weights * np.sqrt(unmasked.size / total)
                result = result * weights
            if mask is not None:
                result = np.where(mask != 0, result * mask, 0)
            residuals[...] = result
        driver.mem_copy_h2d(self._h_residual_vector, self._d_residual_vector)

    def _terms_chi2(self) -> float:
        return self._chi2


def comparison(observation: Observation, driver: Driver) -> Comparison:
    """
    Return the comparison of an observation with its data, of the kind of
    its likelihood, on the driver of its model.
    """
    likelihood = observation.likelihood()
    for kind, cls in _COMPARISONS:
        if isinstance(likelihood, kind):
            return cls(observation, driver)
    raise TypeError(
        f"no comparison of the likelihood {type(likelihood).__name__}")


# The comparison of each kind of likelihood
_COMPARISONS = (
    (LikelihoodGaussian, ComparisonGaussian),
    (LikelihoodUser, ComparisonUser))


def _copied(
        driver: Driver,
        values: np.ndarray,
        out: DeviceArray,
        shape: tuple[int, ...]
) -> DeviceArray:
    """
    Copy the values of a data item to a part of a 1d array on the device,
    in its type, and return its view of their shape.
    """
    driver.mem_copy_h2d(values.ravel().astype(out.dtype), out)
    return out.reshape(shape)


def _footprint(
        key: str, footprint: np.ndarray, dataset: Dataset
) -> np.ndarray:
    """
    Return the footprint of the upper limit of a data item, of the shape
    of the item (see LikelihoodGaussian).

    Raises
    ------
    ConfigError
        If it is not of the shape of the item or of its last axes, or the
        item has values outside it.
    """
    data = dataset[key]
    try:
        footprint = np.broadcast_to(footprint, data.shape())
    except ValueError:
        raise ConfigError(
            f"the footprint of the upper limit of '{key}' is of the "
            f"shape {footprint.shape}, not of its data, {data.shape()}, "
            f"or of their last axes") from None
    if outside := int(np.count_nonzero(
            (data.mask() != 0) & (footprint == 0))):
        raise ConfigError(
            f"the data of '{key}' have {outside} values outside the "
            f"footprint of their upper limit; mask them")
    return footprint
