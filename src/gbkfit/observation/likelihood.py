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
    def amplitudes(self) -> str | None:
        """
        Return the free amplitudes of the model, if any (see
        LikelihoodGaussian).
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

    With amplitudes 'per_spectrum' (spectra only, e.g. those of
    pixel_spectra), each spectrum of the model is times a free amplitude
    of its own, a >= 0: the fit then takes the shape of each spectrum
    from the model (where its lines are, how wide; with beam smearing,
    also how the light of the model weighs its velocities), and its
    brightness from the data, which suits a model whose light misses
    structure of the data (e.g. the clumps of a clumpy disk). The
    amplitudes are nuisance parameters, which the objective solves
    exactly, never the flux of the data put into the model:

    - the residuals and the chi-square (of optimizers) are those of the
      best amplitude of each spectrum, a = max(0, B / A), with A = sum w
      m^2 and B = sum w m d over its values (m the model, d the data, w
      the weight of each value in the chi-square, (weight / error)^2);
    - the log-likelihood (of samplers) has them integrated out, under a
      flat prior on the flux of each spectrum, a F >= 0 (F = sum m, the
      flux of the model there), of unit width: the log of the integral
      of exp(-chi2(a) / 2) d(a F), which is -chi2(a_best) / 2 + log(F
      sqrt(2 pi / A) Phi(B / sqrt(A))) + (B^2 / 2A if B < 0), Phi the
      normal cdf. (A flat prior on a itself would favour models of
      little light: their amplitudes span more.) Spectra where the
      model has no light (F <= 0) have no term: their amplitude is 0.

    Each spectrum with values in the data counts as a parameter in the
    degrees of freedom. The amplitudes take the brightness of the model
    as a whole: its overall scale (e.g. the amplitude of its density) is
    then free of the data, and must be fixed. The prior has no natural
    width: the evidence of a fit with free amplitudes compares only with
    that of fits with them, of the same data.

    Parameters
    ----------
    weights : float or Mapping, optional
        The weight of all the data items, or of each, by name (1 for the
        others).
    missing_residual : float, optional
        The residual of a value the model does not have (see above).
    amplitudes : str, optional
        'per_spectrum' for a free amplitude for each spectrum of the
        model (see above); none by default.

    Raises
    ------
    ConfigError
        If missing_residual is negative or not finite, or amplitudes is
        not 'per_spectrum'.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, Any]:
        amplitudes = {} if self._amplitudes is None \
            else dict(amplitudes=self._amplitudes)
        return dict(
            type=self.type(), weights=self._weights,
            missing_residual=self._missing_residual) | amplitudes

    def __init__(
            self,
            weights: float | Mapping[str, float] = 1.0,
            missing_residual: float | None = None,
            amplitudes: str | None = None
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
        if amplitudes not in (None, 'per_spectrum'):
            raise ConfigError(
                f"option 'amplitudes' must be 'per_spectrum' (or none); it "
                f"is {amplitudes!r}")
        self._amplitudes = amplitudes

    def weight(self, key: str) -> float:
        """Return the weight of the data item of the given name."""
        if isinstance(self._weights, Real):
            return self._weights
        return self._weights.get(key, 1.0)

    def missing_residual(self) -> float | None:
        return self._missing_residual

    def amplitudes(self) -> str | None:
        return self._amplitudes


likelihood_parser = parseutils.TypedParser(Likelihood, [LikelihoodGaussian])
