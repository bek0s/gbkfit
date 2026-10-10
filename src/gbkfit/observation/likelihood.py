import abc
from collections.abc import Mapping
from numbers import Real
from typing import Any

from gbkfit.params import ParamDesc
from gbkfit.utils import parseutils


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

    Parameters
    ----------
    weights : float or Mapping, optional
        The weight of all the data items, or of each, by name (1 for the
        others).
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), weights=self._weights)

    def __init__(self, weights: float | Mapping[str, float] = 1.0):
        self._weights = weights if isinstance(weights, Real) \
            else dict(weights)

    def weight(self, key: str) -> float:
        """Return the weight of the data item of the given name."""
        if isinstance(self._weights, Real):
            return self._weights
        return self._weights.get(key, 1.0)


likelihood_parser = parseutils.TypedParser(Likelihood, [LikelihoodGaussian])
