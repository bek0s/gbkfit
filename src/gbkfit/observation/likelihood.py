import abc
import copy
import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any, Literal, Self

import numpy as np

from gbkfit.dataset import NoiseCorrelation
from gbkfit.math.correlation import (
    correlation_matrix, periodic_correlation, whitening_matrix)
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamValues
from gbkfit.utils import fitsutils, funcutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'MAX_CORRELATED_PIXELS',
    'FittedNoise',
    'Likelihood',
    'LikelihoodGaussian',
    'LikelihoodUser',
    'Whitening',
    'correlated_pixels',
    'likelihood_parser',
    'whitening'
]


# The most pixels of the sky over which the noise of a data item can be
# correlated (see LikelihoodGaussian): its dense correlation matrix takes
# 8 n^2 bytes (3.2 GB)
MAX_CORRELATED_PIXELS = 20000


@dataclasses.dataclass(frozen=True)
class FittedNoise:
    """
    How the noise of a data item is fitted (see LikelihoodGaussian):
    whether its errors are times a scale, of at least scale_minimum, and
    whether a jitter adds to them in quadrature.
    """
    scale: bool
    scale_minimum: float | None
    jitter: bool


class Likelihood(parseutils.TypedSerializable, abc.ABC):
    """
    How the model of an observation is compared with its data: the noise
    model of the data. It can have parameters of its own, whose names
    start with its prefix (see with_prefix).
    """

    # (the start of the names of its parameters)
    _prefix: str = ''

    def with_prefix(self, prefix: str) -> Self:
        """
        Return a copy of it whose parameters have names that start with a
        prefix (e.g. 'muse_', for 'muse_noise_scale_moment1').
        """
        likelihood = copy.copy(self)
        likelihood._prefix = prefix
        return likelihood

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the parameters of the likelihood, by name."""
        return {}

    @abc.abstractmethod
    def fitted_noise(self) -> dict[str, FittedNoise]:
        """
        Return how the noise of data items is fitted, by name, for those
        whose noise is fitted (see LikelihoodGaussian).
        """
        pass

    def noise_params(self, key: str) -> tuple[str | None, str | None]:
        """
        Return the names of the parameters of the scale and of the jitter
        of the noise of the data item of the given name (see
        LikelihoodGaussian), None for those that are not fitted.
        """
        noise = self.fitted_noise().get(key)
        if noise is None:
            return None, None
        return (
            f'{self._prefix}noise_scale_{key}' if noise.scale else None,
            f'{self._prefix}noise_jitter_{key}' if noise.jitter else None)

    def noise(
            self,
            key: str,
            error: np.ndarray | None,
            params: ParamValues
    ) -> np.ndarray:
        """
        Return the noise of the values of the data item of the given name,
        of its errors (1 where it has none), at the values of the
        parameters: the errors, or with a fitted noise (see
        LikelihoodGaussian), sqrt((f e)^2 + j^2).
        """
        error = np.ones(()) if error is None else error
        scale, jitter = self.noise_params(key)
        return np.hypot(
            (1.0 if scale is None else float(params[scale])) * error,
            0.0 if jitter is None else float(params[jitter]))

    def weight(self, key: str) -> float:
        """
        Return the weight of the residuals of the data item of the given
        name (see LikelihoodGaussian); 1 here.
        """
        return 1.0

    def named_items(self) -> frozenset[str]:
        """
        Return the names of the data items that its options name (e.g.
        their weights, upper limits or fitted noise; see
        LikelihoodGaussian); none here.
        """
        return frozenset()

    @abc.abstractmethod
    def missing_residual(self) -> float | None:
        """
        Return the residual of a value that the model does not have where
        the data have one, if given (see LikelihoodGaussian).
        """
        pass

    def takes_lacking_values(self) -> bool:
        """
        Check whether it takes the values that the model lacks where the
        data have some (e.g. the velocity of a spectrum below the
        flux_floor of a moment map): with a missing_residual here.
        """
        return self.missing_residual() is not None

    @abc.abstractmethod
    def upper_limit(self, key: str) -> tuple[float, float] | None:
        """
        Return the detection limit below which the data item of the given
        name is where it is masked, and the noise there, if given (see
        LikelihoodGaussian).
        """
        pass

    def footprint(self, key: str) -> np.ndarray | None:
        """
        Return where the data item of the given name was observed, for its
        upper limit (see LikelihoodGaussian): 1 inside, 0 outside; None
        for all of it (here).
        """
        return None

    @abc.abstractmethod
    def white_floor(self) -> float:
        """
        Return the share of the variance of correlated noise taken as white
        (see LikelihoodGaussian).
        """
        pass


class LikelihoodGaussian(Likelihood):
    """
    Independent Gaussian noise with the errors of the data: the weighted
    chi-squared.

    Where the data have a value but the model none (e.g. the velocity of
    a spectrum whose flux is below the flux_floor of a moment map), there
    is no residual to compute: the residual is then missing_residual.
    Counting such values as 0 would let a fit lower its chi-square by
    making its model lack values where it fits the data badly (e.g. by
    making the model too faint there); a fixed residual stops that.

    The value is a choice, with no right answer, so it has no default: an
    observation whose model can lack values needs it (see
    Observable.model_can_lack_values; e.g. moment maps of order 1 and
    above). It is in errors, or in the units of the data where the data
    have no errors (e.g. km/s for velocities). A larger value guards
    better against a model that lacks values, but weighs more where the
    data are odd; a missed value costs as much as a miss of
    missing_residual errors (e.g. 3). A better guard is to fit moment 0
    (the flux, which the model always has) with the higher moments: where
    the model is too faint, its flux then counts with its own residual.

    The values the data mask count nothing, unless their data item has an
    upper limit (upper_limits): then every masked value is a non-detection
    (e.g. pixels clipped below a threshold, or blanked as empty), below
    the limit L, with noise e (the error of a value there; the data keep
    no errors where they are masked), both in the units of the data, and
    the model m pays where it is above it: the chi-square of the value is
    -2 log Phi((L - m) / e), Phi the normal cdf, the log-likelihood of
    censored data (about 0 for a model below the limit, about ((m - L) /
    e)^2 well above it). Without limits a model could put light where the
    data are blank for free (e.g. a disk too large). The limits are for
    values of light (brightness, spectra, moment 0), not for velocities,
    of which a blank says nothing. An upper limit can have a footprint:
    where the instrument looked (e.g. the field of an IFU, whose corners
    are blank but were never observed), 1 inside and 0 outside, of the
    shape of the data or of their last axes (e.g. an image of the field
    of a cube, the same in every channel); the masked values outside it
    count nothing, and the data must have none there.

    With noise, the noise of data items is fitted: the errors e of an item
    (1 where the data have none) are times a scale f, and a jitter j adds
    to them in quadrature, so that each value has the noise sigma =
    sqrt((f e)^2 + j^2) in place of its error. Each is a parameter of the
    observation, '{name}_noise_scale_{item}' and
    '{name}_noise_jitter_{item}' (the observation needs a name; see
    Observation). The scale is
    for errors that are off by a factor (e.g. too small, from noise
    correlated within the beam), and without errors it is the noise
    itself, in the unit of the data; the jitter, in the unit of the data,
    is for scatter that the model cannot make (e.g. turbulence in a
    velocity map, or clumps). Both at once need errors that differ between
    values: with equal errors only (f e)^2 + j^2 counts.

    A fitted noise makes the normalisation of the Gaussian count: each
    value adds c^2 log sigma^2 to -2 log L, c the factor of its residual
    (the weight of the item, times those of the model and of the data
    mask), which the residuals lack. The objective adds it as one more
    residual for each such item, after those of the data, its noise
    residual: sqrt(sum c^2 log(sigma^2 / (f_min e)^2)), f_min the minimum
    of the scale (1 if the scale is not fitted), so that the sum is never
    negative (the scale is at least f_min, the jitter at least 0); the
    part it leaves out, c^2 log (f_min e)^2, does not depend on the noise.
    Least squares then fit the noise too, and exactly, as do the other
    fitters. Masked values with an upper limit take the fitted noise of
    their error, but add nothing (a probability has no normalisation);
    values the model lacks cost missing_residual, which does not take the
    noise, and add nothing either (counting them would bias the scale low,
    by about half their share of the values). The chi-square has the
    noise residuals; the reduced chi-square of a fit does not (see
    Objective.noise_chi2).

    The residuals of a data item whose noise is correlated (see Data:
    e.g. by the beam of a radio image) are whitened. The noise of its
    values has the covariance S C S, S the diagonal of their noise (their
    errors, or sigma: the jitter is correlated as the noise) over their
    weights, and C the correlation of the noise of the data: its
    chi-square is r^T C^-1 r, of its residuals r (as above), the sum of
    the squares of the whitened residuals L^-1 r, L the Cholesky factor
    of C (C = L L^T). They are those of the values that the data do not
    mask, the values the model lacks among them (with missing_residual).
    C has a white floor f, (1 - f) C + f I, a share f of the variance
    without correlation, which keeps it positive definite where the
    shape of the correlation alone makes it singular (e.g. a Gaussian
    beam on pixels much smaller than it); the noise of a cube correlated
    on the sky and along its spectral axis has a floor on each, C =
    C_xy' (x) C_z'. The determinant of C is the same at every parameter,
    and left out of the log-likelihood. A fitted noise adds its noise
    residual as above, of the terms of its values times the share
    n_ind / n of the n values that count, n_ind the independent values
    they are worth (see Whitening): its sum is n_ind log f^2 for a scale
    f. With all n (the Gaussian of C exactly), the best scale would be
    f^2 = chi2 / n, too small where the noise of the data has less
    variance than the floor gives C (e.g. beam noise alone, which has
    none from pixel to pixel): on 20 mocks of NGC 1313 of LVHIS, f = 0.47
    of the true noise, and errors half the scatter of the fits; counting
    n_ind, f = 1.00 and errors of the scatter (2026-10-21; the user chose
    this, rather than the exact Gaussian). The mask of a cube must keep
    whole spectra (the same pixels in every channel), so that C is that
    product, and the pixels must be at most MAX_CORRELATED_PIXELS (C is a
    dense matrix).
    Correlated noise does not go with upper limits (the probability of
    correlated non-detections has no closed form), or with solved flux
    scales of the item (see ObservationSky; the best scale of each
    spectrum would depend on the others). Observations of correlated noise
    are in float64 by default (see ObservationSky). Solved flux scales do
    not go with upper limits of their items either: they are solved
    without the terms of the limits.

    Parameters
    ----------
    weights : float or Mapping, optional
        The weight of all the data items, or of each, by name (1 for the
        others).
    missing_residual : float, optional
        The residual of a value the model does not have (see above).
    upper_limits : Mapping, optional
        The upper limit of the masked values of data items, by name (none
        for the others): a mapping of its 'limit' and its 'error' (the
        noise of the non-detections), and optionally its 'footprint' (a
        FITS file, relative to the configuration: a filename, or a
        mapping of it, 'file', and its 'hdu'); see above. None by default.
    noise : Mapping, optional
        The fitted noise of data items, by name (none for the others): a
        mapping of 'scale' (whether the errors are times a fitted scale),
        'scale_minimum' (the smallest scale, finite and positive; required
        with the scale, and only with it) and 'jitter' (whether a fitted
        jitter adds to them), at least one of the two fitted; see above.
        None by default.
    white_floor : float, optional
        The share of the variance of correlated noise taken as white, in
        [0, 1) (see above); 0.01 by default.

    Raises
    ------
    ConfigError
        If missing_residual is negative or not finite, an upper limit is
        not a finite limit and a finite, positive error (and a footprint
        that can be read, of values 0 and 1), the noise of an item is not
        as above, or the white floor is not in [0, 1).
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, Any]:
        limits = {} if self._upper_limits is None else dict(
            upper_limits=self._upper_limits)
        noise = {} if not self._noise else dict(noise={
            key: {name: value
                  for name, value in dataclasses.asdict(item).items()
                  if value is not None}
            for key, item in self._noise.items()})
        return dict(
            type=self.type(), weights=self._weights,
            missing_residual=self._missing_residual) \
            | limits | noise \
            | dict(white_floor=self._white_floor)

    def __init__(
            self,
            weights: float | Mapping[str, float] = 1.0,
            missing_residual: float | None = None,
            upper_limits: Mapping[str, Mapping[str, float]] | None = None,
            noise: Mapping[str, Mapping[str, Any]] | None = None,
            white_floor: float = 0.01
    ):
        if missing_residual is not None and not (
                math.isfinite(missing_residual) and missing_residual >= 0):
            raise ConfigError(
                f"option 'missing_residual' must be finite and not "
                f"negative; it is {missing_residual}")
        self._weights = weights if isinstance(weights, Real) \
            else dict(weights)
        self._missing_residual = None if missing_residual is None \
            else float(missing_residual)
        self._upper_limits = None if upper_limits is None else {
            key: _check_upper_limit(key, limit)
            for key, limit in upper_limits.items()}
        self._footprints = {
            key: _read_footprint(key, limit['footprint'])
            for key, limit in (self._upper_limits or {}).items()
            if 'footprint' in limit}
        self._noise = {
            key: _check_noise(key, info)
            for key, info in (noise or {}).items()}
        if not 0 <= white_floor < 1:
            raise ConfigError(
                f"option 'white_floor' must be in [0, 1); it is {white_floor}")
        self._white_floor = float(white_floor)

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return its parameters, by name: the scale and the jitter of the
        fitted noise of data items (see the class and with_prefix).
        """
        pdescs = {}
        for key, noise in self._noise.items():
            scale, jitter = self.noise_params(key)
            if scale is not None:
                pdescs[scale] = ParamScalarDesc(
                    desc=f"the scale of the errors of {key}",
                    minimum=noise.scale_minimum)
            if jitter is not None:
                pdescs[jitter] = ParamScalarDesc(
                    desc=f"the jitter of {key}, which adds to its errors in "
                         f"quadrature (in its unit)",
                    minimum=0)
        return pdescs

    def weight(self, key: str) -> float:
        """Return the weight of the data item of the given name."""
        if isinstance(self._weights, Real):
            return self._weights
        return self._weights.get(key, 1.0)

    def named_items(self) -> frozenset[str]:
        weights = () if isinstance(self._weights, Real) else self._weights
        return frozenset(
            [*weights, *(self._upper_limits or ()), *self._noise])

    def missing_residual(self) -> float | None:
        return self._missing_residual

    def upper_limit(self, key: str) -> tuple[float, float] | None:
        if self._upper_limits is None or key not in self._upper_limits:
            return None
        limit = self._upper_limits[key]
        return limit['limit'], limit['error']

    def footprint(self, key: str) -> np.ndarray | None:
        return self._footprints.get(key)

    def fitted_noise(self) -> dict[str, FittedNoise]:
        return dict(self._noise)

    def white_floor(self) -> float:
        return self._white_floor


@dataclasses.dataclass(frozen=True)
class Whitening:
    """
    The whitening of the residuals of a data item whose noise is
    correlated (see LikelihoodGaussian and whitening). Its values are on
    pixels of the sky (the last two axes of its arrays) and, for a cube,
    on channels (its first axis). The residuals of the pixels it keeps,
    as a matrix R of the channels by those pixels (one row for an image),
    are whitened into spectral R spatial^T, with the whitening of each
    axis whose noise is correlated (None for the others). Noise of exactly
    the correlation C of the data (without the white floor) gives them
    the chi-square tr(C C'^-1) on average, nindependent: the number of
    independent values they are worth (e.g. one in 4.5 of the pixels of
    the moment maps of NGC 1313 of LVHIS, of a beam of 6.6 by 5.4 pixels,
    with a floor of 0.01).
    """
    pixels: np.ndarray
    spatial: np.ndarray | None
    spectral: np.ndarray | None
    nindependent: float


def correlated_pixels(key: str, mask: np.ndarray) -> np.ndarray:
    """
    Return the flat indices of the pixels of the sky that a data item
    whose noise is correlated keeps (see Whitening), of its mask; raise
    ConfigError unless it keeps the same pixels in every channel, and at
    most MAX_CORRELATED_PIXELS.
    """
    channels = mask.reshape(-1, *mask.shape[-2:])
    if np.any(channels != channels[0]):
        raise ConfigError(
            f"the noise of '{key}' is correlated, so its mask must keep "
            f"whole spectra (the same pixels in every channel, e.g. not a "
            f"mask of the emission); it does not. The option "
            f"mask_whole_spectra of the data masks every spectrum with a "
            f"masked channel")
    pixels = np.flatnonzero(channels[0])
    if pixels.size > MAX_CORRELATED_PIXELS:
        raise ConfigError(
            f"the noise of '{key}' is correlated over {pixels.size} pixels; "
            f"at most {MAX_CORRELATED_PIXELS} can be (mask more of them)")
    return pixels


def whitening(
        key: str,
        mask: np.ndarray,
        correlation: NoiseCorrelation,
        white_floor: float
) -> Whitening:
    """
    Return the whitening of the residuals of a data item whose noise is
    correlated (see LikelihoodGaussian), of its mask and of the
    correlation of its noise, with a white floor.

    Raises
    ------
    ConfigError
        If its mask is not one of correlated noise (see correlated_pixels),
        or its correlation is not positive definite with the white floor.
    """
    pixels = correlated_pixels(key, mask)
    nchannels = mask.shape[0] if mask.ndim == 3 else 1
    spatial = spectral = None
    if correlation.spatial is not None:
        points = np.column_stack(np.unravel_index(pixels, mask.shape[-2:]))
        spatial = _whitening(
            key, correlation.spatial, points, white_floor)
    if correlation.spectral is not None:
        points = np.arange(nchannels)[:, None]
        spectral = _whitening(
            key, correlation.spectral, points, white_floor)
    nindependent = _nindependent(spatial, pixels.size, white_floor) \
        * _nindependent(spectral, nchannels, white_floor)
    return Whitening(pixels, spatial, spectral, nindependent)


def _nindependent(
        whitening: np.ndarray | None, size: int, white_floor: float
) -> float:
    """
    Return the number of independent values that the values of an axis
    of a size are worth, of their whitening (see Whitening; all of them
    without one): tr(C C'^-1) of their correlation C, with C' = (1 - f) C
    + f I, which is (size - f tr(C'^-1)) / (1 - f), and tr(C'^-1) the sum
    of the squares of the whitening W (C'^-1 = W^T W).
    """
    if whitening is None:
        return size
    return (size - white_floor * np.sum(whitening ** 2)) / (1 - white_floor)


def _whitening(
        key: str,
        shape: np.ndarray,
        points: np.ndarray,
        white_floor: float
) -> np.ndarray:
    """
    Return the whitening of the values at points of a grid (see
    whitening_matrix), of the shape of the correlation of the noise of
    data item key on the separations of the grid (see NoiseCorrelation),
    with a white floor; raise ConfigError if it is not positive definite.
    """
    matrix = correlation_matrix(
        periodic_correlation(shape, shape.shape), points)
    try:
        return whitening_matrix(matrix, white_floor)
    except np.linalg.LinAlgError:
        raise ConfigError(
            f"the correlation of the noise of '{key}' is singular with the "
            f"white floor {white_floor} (see LikelihoodGaussian): raise it"
        ) from None


def _check_noise(key: str, info: Mapping[str, Any]) -> FittedNoise:
    """
    Return the fitted noise of a data item from its options (see
    LikelihoodGaussian), or raise ConfigError if they are not valid.
    """
    if not isinstance(info, Mapping) \
            or not set(info) <= {'scale', 'scale_minimum', 'jitter'}:
        raise ConfigError(
            f"noise of '{key}': give 'scale' and its 'scale_minimum', and "
            f"'jitter'; it is {info!r}")
    scale, jitter = info.get('scale', False), info.get('jitter', False)
    if not isinstance(scale, bool) or not isinstance(jitter, bool):
        raise ConfigError(
            f"noise of '{key}': 'scale' and 'jitter' are true or false; "
            f"they are {scale!r} and {jitter!r}")
    if not (scale or jitter):
        raise ConfigError(
            f"noise of '{key}': fit its scale, its jitter, or both")
    minimum = info.get('scale_minimum')
    if scale != (minimum is not None):
        raise ConfigError(
            f"noise of '{key}': 'scale_minimum' (the smallest scale of its "
            f"errors, e.g. 0.01) goes with 'scale', and only with it")
    if minimum is not None:
        minimum = float(minimum)
        if not (math.isfinite(minimum) and minimum > 0):
            raise ConfigError(
                f"noise of '{key}': 'scale_minimum' must be finite and "
                f"positive; it is {minimum}")
    return FittedNoise(scale=scale, scale_minimum=minimum, jitter=jitter)


def _check_upper_limit(
        key: str,
        limit: Mapping[str, float]
) -> dict[str, float]:
    """
    Return an upper limit (a mapping of its limit and error, and its
    footprint if given), or raise ConfigError unless both are finite and
    the error is positive.
    """
    if not isinstance(limit, Mapping) or not (
            {'limit', 'error'} <= set(limit)
            <= {'limit', 'error', 'footprint'}):
        raise ConfigError(
            f"upper limit of '{key}': give its 'limit' and its 'error' (the "
            f"noise of the non-detections), and optionally its "
            f"'footprint'; it is {limit!r}")
    value, error = float(limit['limit']), float(limit['error'])
    if not (math.isfinite(value) and math.isfinite(error) and error > 0):
        raise ConfigError(
            f"upper limit of '{key}': its limit must be finite and its "
            f"error finite and positive; they are {value} and {error}")
    footprint = {} if 'footprint' not in limit \
        else dict(footprint=limit['footprint'])
    return dict(limit=value, error=error) | footprint


def _read_footprint(key: str, file: str | Mapping[str, Any]) -> np.ndarray:
    """
    Return the footprint of the upper limit of a data item, read from its
    FITS file (see LikelihoodGaussian), or raise ConfigError unless its
    values are 0 and 1.
    """
    with parseutils.config_path('upper_limits', key, 'footprint'):
        footprint = fitsutils.read_data(*parseutils.parse_file(file))[0]
    if not np.all(np.isin(footprint, (0, 1))):
        raise ConfigError(
            f"upper limit of '{key}': its footprint must be 1 where the "
            f"data were observed and 0 elsewhere")
    return footprint.astype(float)


class LikelihoodUser(Likelihood):
    """
    A likelihood given by a Python function (as BayesianBBarolo's
    log_like), which the objective calls on the host at each evaluation,
    for each data item of the observation: func(item, data, model, error,
    mask, params), with item the name of the item (e.g. 'moment1'), data
    and model arrays of its shape (the model NaN where it lacks a value),
    error its errors and mask its mask (None without; values where it is
    0 do not count), and params the values of the parameters of the
    likelihood (a dict, by their names in it). It returns, by its form:

    - 'residuals': the residual r of each value (an array of the shape of
      the data), of the log-likelihood -r^2 / 2 (e.g. (model - data) /
      error for Gaussian noise; sign(x) sqrt(2 log(1 + x^2)) of x = (model
      - data) / s for Cauchy noise of the scale s), which the objective
      takes as those of LikelihoodGaussian (times the weights of the
      model and of the data mask); every fitter takes them;
    - 'log_likelihood': the log-likelihood of the item (a number, of the
      values its mask keeps); the chi-square of the item is -2 log L, and
      least squares do not take it (they need residuals).

    It has none of the options of LikelihoodGaussian (fitted noise, upper
    limits, missing_residual; correlated noise is not whitened), and
    solves no flux scales (see ObservationSky): its function does what it
    needs, and takes the values that the model lacks.

    Parameters
    ----------
    file : str
        The Python file that defines the function (relative to the
        configuration).
    func : str
        The name of the function.
    form : {'residuals', 'log_likelihood'}
        What the function returns.
    params : Sequence of str, optional
        The names of the parameters of the likelihood, which its
        observation names (see with_prefix).

    Raises
    ------
    ConfigError
        If the function cannot be loaded, the form is not valid, or the
        names of the parameters are not valid or repeat.
    """

    @staticmethod
    def type() -> str:
        return 'user'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), file=self._file, func=self._func_name,
            form=self._form, params=list(self._params))

    def __init__(
            self,
            file: str,
            func: str,
            form: Literal['residuals', 'log_likelihood'],
            params: Sequence[str] = ()
    ):
        if form not in ('residuals', 'log_likelihood'):
            raise ConfigError(
                f"form must be 'residuals' or 'log_likelihood'; it is "
                f"{form!r}")
        for param in params:
            parseutils.check_name(param)
        names = list(params)
        if repeated := sorted({p for p in names if names.count(p) > 1}):
            raise ConfigError(f"the parameters repeat: {repeated}")
        try:
            self._func: Callable[..., Any] = \
                funcutils.load_function_from_file(
                    parseutils.resolve_path(file), func)
        except RuntimeError as e:
            raise ConfigError(str(e)) from e
        self._file = file
        self._func_name = func
        self._form = form
        self._params = tuple(params)

    def form(self) -> str:
        """Return what its function returns: its form."""
        return self._form

    def pdescs(self) -> dict[str, ParamDesc]:
        return {f'{self._prefix}{name}': ParamScalarDesc()
                for name in self._params}

    def fitted_noise(self) -> dict[str, FittedNoise]:
        return {}

    def missing_residual(self) -> float | None:
        return None

    def takes_lacking_values(self) -> bool:
        return True

    def upper_limit(self, key: str) -> tuple[float, float] | None:
        return None

    def white_floor(self) -> float:
        return 0.0

    def evaluate(
            self,
            key: str,
            data: np.ndarray,
            model: np.ndarray,
            error: np.ndarray | None,
            mask: np.ndarray | None,
            params: ParamValues
    ) -> np.ndarray | float:
        """
        Return what its function returns for a data item, at the values
        of the parameters of the observation (prefixed): the residuals of
        its values, or its log-likelihood, by its form.

        Raises
        ------
        ValueError
            If the function returns residuals not of the shape of the data,
            or not one log-likelihood.
        """
        values = {name: params[f'{self._prefix}{name}']
                  for name in self._params}
        result = self._func(key, data, model, error, mask, values)
        if self._form == 'residuals':
            residuals = np.asarray(result, dtype=float)
            if residuals.shape != data.shape:
                raise ValueError(
                    f"the function '{self._func_name}' of the likelihood "
                    f"returned residuals of shape {residuals.shape} for the "
                    f"data item '{key}' of shape {data.shape}")
            return residuals
        if np.ndim(result) != 0:
            raise ValueError(
                f"the function '{self._func_name}' of the likelihood "
                f"returned {np.shape(result)} values for the data item "
                f"'{key}': a log-likelihood is one number")
        return float(result)


likelihood_parser = parseutils.TypedParser(
    Likelihood, [LikelihoodGaussian, LikelihoodUser])
