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


class LikelihoodGaussian(Likelihood):
    """
    Independent Gaussian noise with the errors of the data: the weighted
    chi-squared.

    Where the data have a value but the model none (e.g. the velocity of
    a spectrum without flux), the residual is 'missing' (in errors): a
    model must not lower the chi-square by having no values.

    Parameters
    ----------
    weights : float or Mapping, optional
        The weight of all the data items, or of each, by name (1 for the
        others).
    missing : float, optional
        The residual of a value the model does not have, in errors (by
        default, a miss of 3 errors).

    Raises
    ------
    ConfigError
        If missing is negative or not finite.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), weights=self._weights, missing=self._missing)

    def __init__(
            self,
            weights: float | Mapping[str, float] = 1.0,
            missing: float = 3.0
    ):
        if not (math.isfinite(missing) and missing >= 0):
            raise ConfigError(
                f"option 'missing' must be finite and not negative; it is "
                f"{missing}")
        self._weights = weights if isinstance(weights, Real) \
            else dict(weights)
        self._missing = float(missing)

    def weight(self, key: str) -> float:
        """Return the weight of the data item of the given name."""
        if isinstance(self._weights, Real):
            return self._weights
        return self._weights.get(key, 1.0)

    def missing(self) -> float:
        """Return the residual of a value the model does not have."""
        return self._missing


likelihood_parser = parseutils.TypedParser(Likelihood, [LikelihoodGaussian])
