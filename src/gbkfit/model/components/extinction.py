"""
Extinction laws: the optical depth of dust at a wavelength, relative to
that at V (A(lambda) / A(V)), which scales the opacity of dust (see
DustOpacity) at the wavelength of each line or band.
"""

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'ExtinctionCCM',
    'ExtinctionCalzetti',
    'ExtinctionLaw',
    'ExtinctionTable',
    'extinction_law_parser'
]


class ExtinctionLaw(parseutils.TypedSerializable, abc.ABC):
    """
    An extinction law: A(lambda) / A(V), the optical depth of dust at a
    wavelength relative to that at V. Two laws are equal if they are of
    the same type and options.
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExtinctionLaw) and self.dump() == other.dump()

    def __hash__(self) -> int:
        return hash(repr(sorted(self.dump().items())))

    @abc.abstractmethod
    def range(self) -> tuple[float, float]:
        """Return the wavelengths (um) where it is defined."""

    @abc.abstractmethod
    def _ratio(self, wavelength: float) -> float:
        """Return A(lambda) / A(V) at a wavelength (um) in its range."""

    def ratio(self, wavelength: float) -> float:
        """
        Return A(lambda) / A(V) at a rest wavelength.

        Parameters
        ----------
        wavelength : float
            The wavelength (um, in vacuum).

        Returns
        -------
        float
            The ratio.

        Raises
        ------
        ConfigError
            If the wavelength is outside its range.
        """
        low, high = self.range()
        if not low <= wavelength <= high:
            raise ConfigError(
                f"the extinction law {self.type()!r} is defined from {low} "
                f"to {high} um; a wavelength is {wavelength:.6g} um")
        return float(self._ratio(wavelength))


class ExtinctionCCM(ExtinctionLaw):
    """
    The extinction law of the Milky Way of Cardelli, Clayton & Mathis
    (1989, ApJ 345, 245), of the given R_V, from 0.1 to 3.33 um.

    Parameters
    ----------
    rv : float, optional
        The ratio of total to selective extinction, A(V) / E(B - V).

    Raises
    ------
    ConfigError
        If rv is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'ccm'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), rv=self._rv)

    def __init__(self, rv: float = 3.1):
        if not rv > 0:
            raise ConfigError(f"rv must be positive; it is {rv}")
        self._rv = rv

    def range(self) -> tuple[float, float]:
        return 0.1, 1 / 0.3

    def _ratio(self, wavelength: float) -> float:
        x = 1 / wavelength
        if x < 1.1:
            # Infrared
            a = 0.574 * x ** 1.61
            b = -0.527 * x ** 1.61
        elif x < 3.3:
            # Optical and near infrared
            y = x - 1.82
            a = np.polyval([
                0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447,
                0.17699, 1], y)
            b = np.polyval([
                -2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305,
                1.41338, 0], y)
        elif x < 8:
            # Ultraviolet
            fa = fb = 0.0
            if x >= 5.9:
                fa = -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3
                fb = 0.2130 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3
            a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) + fa
            b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + fb
        else:
            # Far ultraviolet
            a = np.polyval([-0.070, 0.137, -0.628, -1.073], x - 8)
            b = np.polyval([0.374, -0.420, 4.257, 13.670], x - 8)
        return a + b / self._rv


class ExtinctionCalzetti(ExtinctionLaw):
    """
    The attenuation law of starburst galaxies of Calzetti et al. (2000,
    ApJ 533, 682), k(lambda) / R_V, of the given R_V, from 0.12 to 2.2
    um.

    Parameters
    ----------
    rv : float, optional
        The ratio of total to selective attenuation, A(V) / E(B - V).

    Raises
    ------
    ConfigError
        If rv is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'calzetti'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), rv=self._rv)

    def __init__(self, rv: float = 4.05):
        if not rv > 0:
            raise ConfigError(f"rv must be positive; it is {rv}")
        self._rv = rv

    def range(self) -> tuple[float, float]:
        return 0.12, 2.2

    def _ratio(self, wavelength: float) -> float:
        x = 1 / wavelength
        if wavelength < 0.63:
            k = 2.659 * np.polyval([0.011, -0.198, 1.509, -2.156], x)
        else:
            k = 2.659 * (-1.857 + 1.040 * x)
        return (k + self._rv) / self._rv


class ExtinctionTable(ExtinctionLaw):
    """
    An extinction law of the user: A(lambda) / A(V) at wavelengths,
    interpolated linearly between them.

    Parameters
    ----------
    wavelengths : Sequence of float
        The wavelengths (um, increasing).
    ratios : Sequence of float
        A(lambda) / A(V) at each.

    Raises
    ------
    ConfigError
        If there are fewer than two wavelengths, they do not increase,
        or there is not a ratio for each.
    """

    @staticmethod
    def type() -> str:
        return 'table'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), wavelengths=list(self._wavelengths),
            ratios=list(self._ratios))

    def __init__(self, wavelengths: Sequence[float], ratios: Sequence[float]):
        wavelengths = tuple(float(w) for w in wavelengths)
        ratios = tuple(float(r) for r in ratios)
        if len(wavelengths) < 2 or len(ratios) != len(wavelengths):
            raise ConfigError(
                "an extinction table needs at least two wavelengths and a "
                "ratio for each")
        if not all(a < b for a, b in zip(wavelengths, wavelengths[1:])):
            raise ConfigError(
                f"the wavelengths of an extinction table must increase; "
                f"they are {list(wavelengths)}")
        self._wavelengths = wavelengths
        self._ratios = ratios

    def range(self) -> tuple[float, float]:
        return self._wavelengths[0], self._wavelengths[-1]

    def _ratio(self, wavelength: float) -> float:
        return np.interp(wavelength, self._wavelengths, self._ratios)


extinction_law_parser = parseutils.TypedParser(ExtinctionLaw, [
    ExtinctionCCM,
    ExtinctionCalzetti,
    ExtinctionTable])
