"""
What sets the elements of the parameters of a space (see ParamSpace): a
number, or an element of this module. A Python caller gives them, and
the parsing of the properties of a configuration makes them (see
ParamSpace.from_properties).
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from gbkfit.config import parse_options_for_callable
from .pdescs import ParamUnit
from .priors import Prior, PriorTruncated, PriorUniform, prior_parser
from .solves import Solve


__all__ = [
    'DerivedValue',
    'Element',
    'Elements',
    'FreeElement',
    'Solved',
    'Tied',
    'Transformed'
]


class FreeElement:
    """
    A free element, and its properties: the same for every fitter.

    A fitter tries values within the bounds and the support of the
    prior; a prior with bounds is cut to them (see PriorTruncated), and
    without a prior an element is flat within its bounds (optimizers add
    no prior to the cost; samplers take the uniform prior of the bounds,
    see proper_prior). The tuning of some fitters is here too: each
    fitter reads its own and ignores the others, so that the properties
    suit every fitter.

    Parameters
    ----------
    value : float, optional
        The starting value (optimizers need it; walkers start around
        it, see width).
    min, max : float, optional
        The bounds (by default none).
    prior : Prior, optional
        The prior.
    sigma : float, optional
        The width of the first search of CMA-ES.
    width : float, optional
        The width of the box around the value where the walkers of emcee
        and zeus start.
    n : int, optional
        The number of values of a grid, evenly spaced from the minimum to
        the maximum.

    Raises
    ------
    ValueError
        If the bounds, with the support of the prior, are empty, the
        value is not finite or not within them, sigma or width is not
        finite and positive, or n is not an integer of at least 2.
    """

    @classmethod
    def load(cls, info: Mapping[str, Any]) -> 'FreeElement':
        return cls(**parse_options_for_callable(
            dict(info), cls.__init__))

    def dump(self) -> dict[str, Any]:
        """
        Return the properties as a configuration gives them: those given
        (the prior as it was given, not cut to the bounds).
        """
        info: dict[str, Any] = {}
        for name, value in self._given.items():
            if value is None or (name in ('min', 'max') and math.isinf(value)):
                continue
            info[name] = prior_parser.dump(value) if name == 'prior' \
                else value
        return info

    def __init__(
            self,
            value: float | None = None,
            min: float = -math.inf,
            max: float = math.inf,
            prior: Prior | None = None,
            sigma: float | None = None,
            width: float | None = None,
            n: int | None = None
    ):
        if not min < max:
            raise ValueError(
                f"the bounds must not be empty; they are [{min}, {max}]")
        self._given = dict(
            value=value, min=min, max=max, prior=prior, sigma=sigma,
            width=width, n=n)
        minimum, maximum = min, max
        if prior is not None:
            low, high = prior.support()
            if minimum > low or maximum < high:
                prior = PriorTruncated(prior, minimum, maximum)
            minimum, maximum = prior.support()
        if value is not None:
            if not math.isfinite(value):
                raise ValueError(
                    f"option 'value' must be finite; it is {value}")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"option 'value' ({value}) must be in the range "
                    f"[{minimum}, {maximum}]")
        for name, option in (('sigma', sigma), ('width', width)):
            if option is not None and not (
                    math.isfinite(option) and option > 0):
                raise ValueError(
                    f"option '{name}' must be finite and positive; it is "
                    f"{option}")
        if n is not None and (
                isinstance(n, bool) or not isinstance(n, int) or n < 2):
            raise ValueError(
                f"option 'n' must be an integer of at least 2 (a fixed "
                f"value is not free); it is {n!r}")
        self._value = None if value is None else float(value)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._prior = prior
        self._sigma = sigma
        self._width = width
        self._n = n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FreeElement) and self.dump() == other.dump()

    def __repr__(self) -> str:
        options = ', '.join(f'{k}={v!r}' for k, v in self.dump().items())
        return f"FreeElement({options})"

    def with_value(self, value: float | None) -> 'FreeElement':
        """Return a copy of it that starts at another value."""
        return FreeElement(**self._given | dict(value=value))

    def value(self) -> float | None:
        """Return the starting value, if any."""
        return self._value

    def minimum(self) -> float:
        """
        Return the lowest value the fitters may try: the lower bound, or
        that of the support of the prior, whichever is higher.
        """
        return self._minimum

    def maximum(self) -> float:
        """
        Return the highest value the fitters may try: the upper bound, or
        that of the support of the prior, whichever is lower.
        """
        return self._maximum

    def prior(self) -> Prior | None:
        """Return the prior (cut to the bounds), if any."""
        return self._prior

    def proper_prior(self) -> Prior:
        """
        Return the prior, or else the uniform prior of the bounds (of the
        samplers, which need a proper one).

        Raises
        ------
        ValueError
            If there is no prior, and a bound is missing.
        """
        if self._prior is not None:
            return self._prior
        if not (math.isfinite(self._minimum) and math.isfinite(self._maximum)):
            raise ValueError(
                "a sampler needs a prior, or both options 'min' and 'max' "
                "(a uniform prior)")
        return PriorUniform(self._minimum, self._maximum)

    def sigma(self) -> float | None:
        """Return the width of the first search of CMA-ES, if any."""
        return self._sigma

    def width(self) -> float | None:
        """Return the width of the box of the starts of walkers, if any."""
        return self._width

    def n(self) -> int | None:
        """Return the number of values of a grid, if any."""
        return self._n


@dataclasses.dataclass(frozen=True)
class Tied:
    """
    Elements tied to an expression of other parameters and constants
    (e.g. Tied('vmax * 0.9'); see ParamSpace for what it can use).
    """
    expr: str


@dataclasses.dataclass(frozen=True)
class Solved:
    """
    Elements solved at each evaluation of an objective, from the data,
    by a solve (see Solve), in the valid range of the parameter: only
    those of parameters that the model is linear in can be (the objective
    says which).
    """
    solve: Solve


@dataclasses.dataclass(frozen=True)
class Transformed:
    """Elements that the transforms function of the space sets."""


@dataclasses.dataclass(frozen=True)
class DerivedValue:
    """
    A derived value (see ParamSpace): its expression, and the unit of its
    values (an astropy unit, or a label as those of parameters; see
    ParamDesc), if any.
    """
    expr: str
    unit: ParamUnit | str | None = None


# What sets some elements of a parameter: a number fixes them
Element: TypeAlias = float | Tied | FreeElement | Solved | Transformed

# What sets the elements of the parameters, by key (e.g. 'v[1:]'): one
# element for all those of the key, or one for each
Elements: TypeAlias = Mapping[str, Element | Sequence[Element]]
