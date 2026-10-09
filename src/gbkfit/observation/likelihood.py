import abc
from collections.abc import Mapping
from numbers import Real

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

    def pdescs(self):
        """The parameters of the likelihood, by name."""
        return {}


class LikelihoodGaussian(Likelihood):
    """
    Independent Gaussian noise with the errors of the data: the weighted
    chi-squared, with weights for all data items or for each (by name).
    """

    @staticmethod
    def type():
        return 'gaussian'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'likelihood')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self):
        return dict(type=self.type(), weights=self._weights)

    def __init__(self, weights: Real | Mapping[str, Real] = 1.0):
        self._weights = weights if isinstance(weights, Real) \
            else dict(weights)

    def weight(self, key):
        """The weight of the data item of the given name."""
        if isinstance(self._weights, Real):
            return self._weights
        return self._weights.get(key, 1.0)


likelihood_parser = parseutils.TypedParser(Likelihood, [LikelihoodGaussian])
