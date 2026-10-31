
"""
Interpolators of the values y at the points x (increasing) of nodes, by
the name of their kind (see INTERPOLATORS).
"""

import abc

import numpy as np
import numpy.typing as npt
import scipy.interpolate


class Interpolator(abc.ABC):
    """An interpolator of the values y at the points x."""

    @staticmethod
    @abc.abstractmethod
    def type() -> str:
        """Return the name of its kind."""

    @abc.abstractmethod
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        pass

    @abc.abstractmethod
    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        """Return the interpolated values at the points x."""


class InterpolatorLinear(Interpolator):

    @staticmethod
    def type() -> str:
        return 'linear'

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        self._interp = scipy.interpolate.interp1d(x, y, 'linear')

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        return self._interp(x)


class InterpolatorAkima(Interpolator):

    @staticmethod
    def type() -> str:
        return 'akima'

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        self._interp = scipy.interpolate.Akima1DInterpolator(x, y)

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        return self._interp(x)


class InterpolatorPCHIP(Interpolator):

    @staticmethod
    def type() -> str:
        return 'pchip'

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        self._interp = scipy.interpolate.PchipInterpolator(x, y)

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        return self._interp(x)


# The interpolators, by the name of their kind
INTERPOLATORS: dict[str, type[Interpolator]] = {
    cls.type(): cls
    for cls in (InterpolatorLinear, InterpolatorAkima, InterpolatorPCHIP)}
