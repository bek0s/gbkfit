"""
The options of the free elements for the fitters: each fitter loads the
free property of each element (see ParamSpace.free_properties) into an
object of its class of them.
"""

import abc
import math
from collections.abc import Mapping
from typing import Any, TypeAlias

from gbkfit.fitting.priors import Prior, PriorUniform, prior_parser
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitParam',
    'FitParamBounded',
    'FitParamClass',
    'FitParamPrior'
]


def _check_value(value: float, minimum: float, maximum: float) -> None:
    """Check that a value is finite and in a range."""
    if not math.isfinite(value):
        raise ConfigError(f"option 'value' must be finite; it is {value}")
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"option 'value' ({value}) must be in the range [{minimum}, "
            f"{maximum}]")


class FitParam(parseutils.Serializable, abc.ABC):
    """
    The options of a free element for a fitter.

    The config names the minimum and maximum 'min' and 'max'.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'FitParam':
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__,
            rename_params=dict(minimum='min', maximum='max')))

    @abc.abstractmethod
    def minimum(self) -> float:
        """Return the lowest value the fitter may try (may be -inf)."""

    @abc.abstractmethod
    def maximum(self) -> float:
        """Return the highest value the fitter may try (may be inf)."""

    def prior(self) -> Prior | None:
        """Return the prior, if the fitter uses priors."""
        return None

    def scale(self) -> float:
        """
        Return the typical scale of the value, for the steps of finite
        differences near 0: 1% of the width of the bounds, or 1 if
        unbounded.
        """
        width = self.maximum() - self.minimum()
        return 0.01 * width if math.isfinite(width) else 1.0


# A class of FitParam (an alias, for the classes with a type() method,
# which hides the builtin in their bodies)
FitParamClass: TypeAlias = type[FitParam]


class FitParamBounded(FitParam):
    """
    The options of a free element for an optimizer: its initial value and
    its bounds.

    Parameters
    ----------
    value : float
        The initial value.
    minimum, maximum : float, optional
        The bounds (config: min, max; by default none).

    Raises
    ------
    ConfigError
        If the bounds are empty, or the value is not finite or not in
        them.
    """

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(value=self._value)
        if math.isfinite(self._minimum):
            info['min'] = self._minimum
        if math.isfinite(self._maximum):
            info['max'] = self._maximum
        return info

    def __init__(
            self,
            value: float,
            minimum: float = -math.inf,
            maximum: float = math.inf
    ):
        if not minimum < maximum:
            raise ConfigError(
                f"option 'min' ({minimum}) must be less than option 'max' "
                f"({maximum})")
        _check_value(value, minimum, maximum)
        self._value = float(value)
        self._minimum = float(minimum)
        self._maximum = float(maximum)

    def value(self) -> float:
        """Return the initial value."""
        return self._value

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum


class FitParamPrior(FitParam):
    """
    The options of a free element for a sampler: its prior, and an
    optional value (e.g. where to start).

    In a config, min and max without a prior give a uniform prior.

    Parameters
    ----------
    prior : Prior
        The prior.
    value : float, optional
        A value in the support of the prior.

    Raises
    ------
    ConfigError
        If the value is not finite or not in the support of the prior.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'FitParamPrior':
        info = dict(info)
        has_range = 'min' in info or 'max' in info
        if info.get('prior') is not None and has_range:
            raise ConfigError(
                "give either option 'prior', or options 'min' and 'max' "
                "(a uniform prior), not both")
        if isinstance(info.get('prior'), Mapping):
            info['prior'] = parseutils.load_option(
                prior_parser.load, info, 'prior')
        elif has_range:
            if 'min' not in info or 'max' not in info:
                raise ConfigError(
                    "a uniform prior needs both options 'min' and 'max'")
            info['prior'] = PriorUniform(info.pop('min'), info.pop('max'))
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(prior=prior_parser.dump(self._prior))
        if self._value is not None:
            info['value'] = self._value
        return info

    def __init__(self, prior: Prior, value: float | None = None):
        if value is not None:
            _check_value(value, prior.minimum(), prior.maximum())
        self._prior = prior
        self._value = None if value is None else float(value)

    def value(self) -> float | None:
        """Return the value, if any."""
        return self._value

    def minimum(self) -> float:
        return self._prior.minimum()

    def maximum(self) -> float:
        return self._prior.maximum()

    def prior(self) -> Prior:
        return self._prior

    def scale(self) -> float:
        """
        Return the typical scale of the value: 1% of the width of the
        support of the prior, or of its 16-84% range if unbounded.
        """
        width = self.maximum() - self.minimum()
        return 0.01 * (width if math.isfinite(width) else self._prior.width())
