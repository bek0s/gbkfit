import abc
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from gbkfit.params import ParamDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Likelihood',
    'LikelihoodGaussian',
    'likelihood_parser'
]


class Likelihood(parseutils.TypedSerializable, abc.ABC):
    """
    How the model of an observation is compared with its data: the noise
    model of the data. It can have parameters of its own.
    """

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the parameters of the likelihood, by name."""
        return {}

    @abc.abstractmethod
    def missing_residual(self) -> float | None:
        """
        Return the residual of a value that the model does not have where
        the data have one, if given (see LikelihoodGaussian).
        """
        pass

    @abc.abstractmethod
    def upper_limit(self, key: str) -> tuple[float, float] | None:
        """
        Return the detection limit below which the data item of the given
        name is where it is masked, and the noise there, if given (see
        LikelihoodGaussian).
        """
        pass

    @abc.abstractmethod
    def amplitudes(self) -> str | None:
        """
        Return the amplitudes of the spectra of the model, if any (see
        LikelihoodGaussian).
        """
        pass

    @abc.abstractmethod
    def amplitude_sigma(self) -> float | None:
        """
        Return the standard deviation of the prior of the amplitudes, if
        any (see LikelihoodGaussian).
        """
        pass


class LikelihoodGaussian(Likelihood):
    """
    Independent Gaussian noise with the errors of the data: the weighted
    chi-squared.

    Where the data have a value but the model none (e.g. the velocity of
    a spectrum whose flux is below the mask_cutoff of a moment map), there
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
    of which a blank says nothing.

    With amplitudes 'per_spectrum' (spectra only, e.g. those of
    pixel_spectra), each spectrum of the model is times an amplitude of
    its own, a >= 0: the fit then takes the shape of each spectrum from
    the model (where its lines are, how wide; with beam smearing, also how
    the light of the model weighs its velocities), and its brightness
    mostly from the data, which suits a model whose light misses
    structure of the data (e.g. the clumps of a clumpy disk). The fitters
    never see the amplitudes: the objective works them out for each
    spectrum, with exact formulas, never putting the flux of the data into
    the model. They have a prior: a is about 1, a Gaussian of standard
    deviation amplitude_sigma (s) around 1, cut at a = 0; that is, each
    spectrum may be brighter or fainter than the model by a factor of
    about 1 +- s. With the sums of a spectrum A = sum w m^2, B = sum w m d
    and C = sum w d^2 over its values (m the model, d the data, w the
    weight of each value in the chi-square, (weight / error)^2), and A' =
    A + 1/s^2, B' = B + 1/s^2:

    - for optimizers, the objective takes the best amplitude, a =
      max(0, B' / A'), the minimum of the chi-square of the spectrum
      plus that of the prior, (a - 1)^2 / s^2, which is a residual too:
      (a - 1) / s, after those of the data;
    - for samplers, the log-likelihood has the amplitudes integrated out
      under their prior: for each spectrum, the log of the integral over
      a >= 0 of exp(-chi2(a) / 2) times the prior, which is -chi2 / 2 at
      the best amplitude + log(Phi(B' / sqrt(A')) / (s sqrt(A') Phi(1 /
      s))) + (B'^2 / 2A' if B' < 0), Phi the normal cdf.

    Each amplitude counts as a parameter in the degrees of freedom, and
    its prior as a residual: they cancel.

    Why this prior (the user chose it, 2026-10-15; revise here): the
    log-likelihood of samplers needs one. A flat prior on a makes the
    integral over a grow as the model gets fainter (half the light,
    twice the amplitudes, twice the integral): samplers would prefer
    galaxies of no light. A flat prior on the flux of each spectrum (a
    times the sum of the model there) removes that, but favours model
    lines whose flux the data pin down loosely (broad lines) where the
    spectra are faint: in 4000 random spectra of 61 channels of 10 km/s
    (noise 1), the log-evidence of a 60 km/s model line over a 30 km/s
    one was +0.35 on noise alone and +0.22 on a faint line truly 30 km/s
    wide (flux 3); with this prior (s = 0.5), +0.03 and -0.09. Over
    thousands of faint spectra (e.g. of clumpy disks at z ~ 1.5) the flat
    prior would pull the dispersion up. This prior is proper (the
    evidence is a normal one), and keeps the brightness of the model
    meaningful: it is the mean brightness, about which the spectra
    scatter. Its cost is s, a statement of how much the spectra may
    differ from the model: a very large s is a flat prior on a again,
    with its bias. (s could become a parameter of the likelihood, fitted
    with the others; it is a fixed option for now.)

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
        noise of the non-detections); see above. None by default.
    amplitudes : str, optional
        'per_spectrum' for an amplitude for each spectrum of the model
        (see above); none by default.
    amplitude_sigma : float, optional
        The standard deviation of the prior of the amplitudes around 1
        (see above); required with them.

    Raises
    ------
    ConfigError
        If missing_residual is negative or not finite, an upper limit is
        not a finite limit and a finite, positive error, amplitudes is not
        'per_spectrum', or amplitude_sigma is missing with it, given
        without it, or not positive.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, Any]:
        amplitudes = {} if self._amplitudes is None else dict(
            amplitudes=self._amplitudes,
            amplitude_sigma=self._amplitude_sigma)
        limits = {} if self._upper_limits is None else dict(
            upper_limits=self._upper_limits)
        return dict(
            type=self.type(), weights=self._weights,
            missing_residual=self._missing_residual) | limits | amplitudes

    def __init__(
            self,
            weights: float | Mapping[str, float] = 1.0,
            missing_residual: float | None = None,
            upper_limits: Mapping[str, Mapping[str, float]] | None = None,
            amplitudes: str | None = None,
            amplitude_sigma: float | None = None
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
        if amplitudes not in (None, 'per_spectrum'):
            raise ConfigError(
                f"option 'amplitudes' must be 'per_spectrum' (or none); it "
                f"is {amplitudes!r}")
        if (amplitudes is None) != (amplitude_sigma is None):
            raise ConfigError(
                "options 'amplitudes' and 'amplitude_sigma' (the spread of "
                "the amplitudes around 1, e.g. 0.5) go together")
        if amplitude_sigma is not None and not (
                math.isfinite(amplitude_sigma) and amplitude_sigma > 0):
            raise ConfigError(
                f"option 'amplitude_sigma' must be finite and positive; it "
                f"is {amplitude_sigma}")
        self._amplitudes = amplitudes
        self._amplitude_sigma = None if amplitude_sigma is None \
            else float(amplitude_sigma)

    def weight(self, key: str) -> float:
        """Return the weight of the data item of the given name."""
        if isinstance(self._weights, Real):
            return self._weights
        return self._weights.get(key, 1.0)

    def missing_residual(self) -> float | None:
        return self._missing_residual

    def upper_limit(self, key: str) -> tuple[float, float] | None:
        if self._upper_limits is None or key not in self._upper_limits:
            return None
        limit = self._upper_limits[key]
        return limit['limit'], limit['error']

    def amplitudes(self) -> str | None:
        return self._amplitudes

    def amplitude_sigma(self) -> float | None:
        return self._amplitude_sigma


def _check_upper_limit(
        key: str,
        limit: Mapping[str, float]
) -> dict[str, float]:
    """
    Return an upper limit (a mapping of its limit and error), or raise
    ConfigError unless both are finite and the error is positive.
    """
    if not isinstance(limit, Mapping) or set(limit) != {'limit', 'error'}:
        raise ConfigError(
            f"upper limit of '{key}': give its 'limit' and its 'error' (the "
            f"noise of the non-detections); it is {limit!r}")
    value, error = float(limit['limit']), float(limit['error'])
    if not (math.isfinite(value) and math.isfinite(error) and error > 0):
        raise ConfigError(
            f"upper limit of '{key}': its limit must be finite and its "
            f"error finite and positive; they are {value} and {error}")
    return dict(limit=value, error=error)


likelihood_parser = parseutils.TypedParser(Likelihood, [LikelihoodGaussian])
