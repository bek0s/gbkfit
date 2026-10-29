"""
The options of the free elements for each kind of fitter: what a kind of
fitter needs of the properties of an element (see FreeElement), and
what it works out from them. Each fitter reads the free elements into
its class of them (see Fitter.load_fit_params).
"""

import abc
import copy
import math
import typing
from collections.abc import Mapping
from typing import Any, ClassVar, Self, TypeAlias

import numpy as np

from gbkfit.params import FreeElement, Prior
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitParam',
    'FitParamBounded',
    'FitParamClass',
    'FitParamLeastSquares',
    'FitParamPrior',
    'FitParamWalkers'
]


def _check_value(value: float, minimum: float, maximum: float) -> None:
    """Check that a value is finite and in a range."""
    if not math.isfinite(value):
        raise ConfigError(f"option 'value' must be finite; it is {value}")
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"option 'value' ({value}) must be in the range [{minimum}, "
            f"{maximum}]")


def _value(element: FreeElement) -> float:
    """
    Return the starting value of a free element for an optimizer; raise
    ConfigError if it has none.
    """
    if (value := element.value()) is None:
        raise ConfigError(
            "an optimizer needs option 'value' (where it starts)")
    return value


def _scale(minimum: float, maximum: float, width: float) -> float:
    """
    Return the typical scale of a value in a range (see FitParam.scale):
    the smallest magnitude it can take, if the range excludes 0 (so that
    the steps are relative to the value, even in ranges of decades), or
    else 1% of a width (1 if infinite).
    """
    if minimum > 0 or maximum < 0:
        return min(abs(minimum), abs(maximum))
    return 0.01 * width if math.isfinite(width) else 1.0


class FitParam(abc.ABC):
    """The options of a free element for a fitter."""

    # The properties of the free elements that it reads (see FreeElement):
    # each fitter reads those of its class, and ignores the others
    properties: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abc.abstractmethod
    def of(
            cls, element: FreeElement, defaults: Mapping[str, Any]
    ) -> Self:
        """
        Return the options of a free element for the fitters of this
        class.

        Parameters
        ----------
        element : FreeElement
            The element.
        defaults : Mapping
            The tuning of the fitter, for the elements without their own
            (e.g. {'sigma': 5}; see Fitter.load_fit_params).

        Raises
        ------
        ConfigError
            If the element lacks what the fitters need (e.g. a value).
        """

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
        Return the typical scale of the value, the floor of the steps of
        finite differences (see finite_difference_steps): the smallest
        magnitude the value can take, if its bounds exclude 0, or else 1%
        of the width of the bounds (1 if unbounded).
        """
        return _scale(
            self.minimum(), self.maximum(), self.maximum() - self.minimum())


# A class of FitParam (an alias, for the classes with a type() method,
# which hides the builtin in their bodies)
FitParamClass: TypeAlias = type[FitParam]


class FitParamBounded(FitParam):
    """
    The options of a free element for an optimizer: its initial value, its
    bounds, and optionally a prior (then the optimizer finds the maximum a
    posteriori: see FitProblem.cost). The options of each optimizer are of
    this class, or of a subclass with more options (e.g. FitParamCMAES),
    so that a fitter of several starts can run any optimizer (see
    with_value and FitterMultiStart).

    Parameters
    ----------
    value : float
        The initial value.
    minimum, maximum : float, optional
        The bounds (config: min, max; by default none). With a prior, the
        optimizer tries values in both the bounds and its support.
    prior : Prior, optional
        The prior.

    Raises
    ------
    ConfigError
        If the bounds (with the support of the prior) are empty, or the
        value is not finite or not in them.
    """

    properties = ('value', 'min', 'max', 'prior')

    @classmethod
    def of(
            cls, element: FreeElement, defaults: Mapping[str, Any]
    ) -> Self:
        return cls(
            _value(element), element.minimum(), element.maximum(),
            element.prior())

    def __init__(
            self,
            value: float,
            minimum: float = -math.inf,
            maximum: float = math.inf,
            prior: Prior | None = None
    ):
        self._prior = prior
        if prior is not None:
            low, high = prior.support()
            minimum = max(minimum, low)
            maximum = min(maximum, high)
        if not minimum < maximum:
            raise ConfigError(
                f"the bounds, with the support of the prior, must not be "
                f"empty; they are [{minimum}, {maximum}]")
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

    def with_value(self, value: float) -> Self:
        """
        Return these options (of this class, with all its other options)
        with another initial value.

        Raises
        ------
        ConfigError
            If the value is not finite or not within the bounds.
        """
        _check_value(value, self._minimum, self._maximum)
        result = copy.copy(self)
        result._value = float(value)
        return result

    def prior(self) -> Prior | None:
        return self._prior


class FitParamLeastSquares(FitParamBounded):
    """
    The options of a free element for a least-squares optimizer (see
    FitParamBounded): a prior must have a residual (see Prior.residual;
    e.g. Gaussian or uniform, not log-uniform).

    Raises
    ------
    ConfigError
        As FitParamBounded, or if the prior has no residual.
    """

    def __init__(
            self,
            value: float,
            minimum: float = -math.inf,
            maximum: float = math.inf,
            prior: Prior | None = None
    ):
        if prior is not None and not prior.has_residual():
            raise ConfigError(
                f"least squares cannot take a prior of type "
                f"'{prior.type()}' (it has no residual); use a Gaussian or "
                f"a uniform prior, or another optimizer")
        super().__init__(value, minimum, maximum, prior)


class FitParamPrior(FitParam):
    """
    The options of a free element for a sampler: its prior, and an
    optional value (e.g. where to start).

    Without a prior, an element has the uniform prior of its bounds
    (see FreeElement.proper_prior).

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

    properties = ('prior', 'min', 'max', 'value')

    @classmethod
    def of(
            cls, element: FreeElement, defaults: Mapping[str, Any]
    ) -> Self:
        return cls(element.proper_prior(), element.value())

    def __init__(self, prior: Prior, value: float | None = None):
        if value is not None:
            _check_value(value, *prior.support())
        self._prior = prior
        self._value = None if value is None else float(value)

    def value(self) -> float | None:
        """Return the value, if any."""
        return self._value

    def minimum(self) -> float:
        return self._prior.support()[0]

    def maximum(self) -> float:
        return self._prior.support()[1]

    def prior(self) -> Prior:
        return self._prior

    def scale(self) -> float:
        """
        Return the typical scale of the value: the smallest magnitude it
        can take, if the support of the prior excludes 0, or else 1% of
        the width of the support (of the 16-84% range, if unbounded).
        """
        width = self.maximum() - self.minimum()
        return _scale(
            self.minimum(), self.maximum(),
            width if math.isfinite(width) else self._prior.width())


class FitParamWalkers(FitParamPrior):
    """
    The options of a free element for an ensemble of walkers (emcee,
    zeus): its prior, and where the walkers start.

    The walkers start uniformly in a box of the given width around the
    value, if a width is given, or else from the prior.

    Parameters
    ----------
    prior : Prior
        The prior.
    value : float, optional
        The centre of the box of the walkers.
    width : float, optional
        The width of the box of the walkers (needs a value; by default
        that of the fitter, if any, see FitterEmcee).

    Raises
    ------
    ConfigError
        If the value is not in the support of the prior, or a width is
        given without a value, or is not positive.
    """

    properties = FitParamPrior.properties + ('width',)

    @classmethod
    def of(
            cls, element: FreeElement, defaults: Mapping[str, Any]
    ) -> Self:
        width = element.width()
        return cls(
            element.proper_prior(), element.value(),
            defaults.get('width') if width is None else width)

    def __init__(
            self,
            prior: Prior,
            value: float | None = None,
            width: float | None = None
    ):
        super().__init__(prior, value)
        if width is not None and value is None:
            raise ConfigError("option 'width' needs option 'value'")
        if width is not None and not (math.isfinite(width) and width > 0):
            raise ConfigError(
                f"option 'width' must be finite and positive; it is {width}")
        self._width = width

    def width(self) -> float | None:
        """Return the width of the box of the walkers, if any."""
        return self._width

    def sample_start(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Return the starting values of the walkers."""
        if self._width is None:
            return self.prior().sample(rng, size)
        value = typing.cast(float, self.value())
        lo = max(value - self._width / 2, self.minimum())
        hi = min(value + self._width / 2, self.maximum())
        return rng.uniform(lo, hi, size)
