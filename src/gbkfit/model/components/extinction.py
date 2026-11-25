"""
Extinction laws: the optical depth of dust at a wavelength, relative to
that at V (A(lambda) / A(V)), which scales the opacity of dust (see
DustOpacity) at the wavelength of each line or band.
"""

import abc
from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from annotated_types import Ge, Gt, Len

from gbkfit.config import (
    TypedParser, TypedSerializable, Unit, check_options)


__all__ = [
    'ExtinctionCCM',
    'ExtinctionCalzetti',
    'ExtinctionLaw',
    'ExtinctionTable',
    'extinction_law_parser'
]


class ExtinctionLaw(TypedSerializable, abc.ABC):
    """
    An extinction law: A(lambda) / A(V), the optical depth of dust at a
    wavelength relative to that at V. Two laws are equal if they are of
    the same type and options.
    """

    config_titles = dict(rv='R(V)')

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExtinctionLaw) and self.dump() == other.dump()

    def __hash__(self) -> int:
        # (equal laws are of the same type; __eq__ tells laws of a type
        # apart)
        return hash(self.type())

    @abc.abstractmethod
    def range(self) -> tuple[float, float]:
        """Return the wavelengths (um) where it is defined."""

    @abc.abstractmethod
    def _ratio(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Return A(lambda) / A(V) at each of the wavelengths (um) of an
        array, all in its range.
        """

    def ratio(self, wavelength: npt.ArrayLike) -> np.ndarray:
        """
        Return A(lambda) / A(V) at rest wavelengths: one, or an array of
        them at once (e.g. the channels of a cube).

        Parameters
        ----------
        wavelength : array_like
            The wavelengths (um, in vacuum).

        Returns
        -------
        np.ndarray
            The ratio at each, of the shape of wavelength (0-d for one).

        Raises
        ------
        ValueError
            If a wavelength is outside its range (or not a number).
        """
        wavelength = np.asarray(wavelength, dtype=float)
        low, high = self.range()
        outside = ~((wavelength >= low) & (wavelength <= high))
        if outside.any():
            raise ValueError(
                f"the extinction law {self.type()!r} is defined from {low} "
                f"to {high} um; a wavelength is "
                f"{wavelength[outside].flat[0]:.6g} um")
        return np.asarray(self._ratio(wavelength), dtype=float)


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
    ValueError
        If rv is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'ccm'

    @staticmethod
    def type_title() -> str:
        return 'Milky Way (Cardelli et al.)'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), rv=self._rv)

    def __init__(self, rv: Annotated[float, Gt(0)] = 3.1):
        check_options(ExtinctionCCM.__init__, locals())
        self._rv = float(rv)

    def range(self) -> tuple[float, float]:
        return 0.1, 1 / 0.3

    def _ratio(self, wavelength: np.ndarray) -> np.ndarray:
        x = 1 / wavelength
        a = np.empty_like(x)
        b = np.empty_like(x)
        # Infrared
        on = x < 1.1
        a[on] = 0.574 * x[on] ** 1.61
        b[on] = -0.527 * x[on] ** 1.61
        # Optical and near infrared
        on = (x >= 1.1) & (x < 3.3)
        y = x[on] - 1.82
        a[on] = np.polyval([
            0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447,
            0.17699, 1], y)
        b[on] = np.polyval([
            -2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305,
            1.41338, 0], y)
        # Ultraviolet, with the terms beyond x = 5.9
        on = (x >= 3.3) & (x < 8)
        u = x[on]
        far = np.maximum(u - 5.9, 0)
        a[on] = 1.752 - 0.316 * u - 0.104 / ((u - 4.67) ** 2 + 0.341) \
            + (-0.04473 * far ** 2 - 0.009779 * far ** 3)
        b[on] = -3.090 + 1.825 * u + 1.206 / ((u - 4.62) ** 2 + 0.263) \
            + (0.2130 * far ** 2 + 0.1207 * far ** 3)
        # Far ultraviolet
        on = x >= 8
        a[on] = np.polyval([-0.070, 0.137, -0.628, -1.073], x[on] - 8)
        b[on] = np.polyval([0.374, -0.420, 4.257, 13.670], x[on] - 8)
        return a + b / self._rv


class ExtinctionCalzetti(ExtinctionLaw):
    """
    The attenuation law of starburst galaxies (Calzetti et al. 2000, ApJ
    533, 682), k(lambda) / R_V, of the given R_V, from 0.12 to 2.2 um.

    Parameters
    ----------
    rv : float, optional
        The ratio of total to selective attenuation, A(V) / E(B - V).

    Raises
    ------
    ValueError
        If rv is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'calzetti'

    @staticmethod
    def type_title() -> str:
        return 'Starburst (Calzetti et al.)'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), rv=self._rv)

    def __init__(self, rv: Annotated[float, Gt(0)] = 4.05):
        check_options(ExtinctionCalzetti.__init__, locals())
        self._rv = float(rv)

    def range(self) -> tuple[float, float]:
        return 0.12, 2.2

    def _ratio(self, wavelength: np.ndarray) -> np.ndarray:
        x = 1 / wavelength
        k = np.where(
            wavelength < 0.63,
            2.659 * np.polyval([0.011, -0.198, 1.509, -2.156], x),
            2.659 * (-1.857 + 1.040 * x))
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
    ValueError
        If there are fewer than two wavelengths, they do not increase,
        or there is not a ratio for each.
    """

    config_titles = dict(wavelengths='Wavelengths', ratios='Ratios')
    config_start = dict(
        wavelengths=[0.36, 0.44, 0.55, 0.65, 0.8],
        ratios=[1.53, 1.32, 1.0, 0.82, 0.6])

    @staticmethod
    def type() -> str:
        return 'table'

    @staticmethod
    def type_title() -> str:
        return 'Custom table'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), wavelengths=list(self._wavelengths),
            ratios=list(self._ratios))

    def __init__(
            self,
            wavelengths: Annotated[
                Sequence[Annotated[float, Gt(0)]], Len(2), Unit('um')],
            ratios: Sequence[Annotated[float, Ge(0)]]
    ):
        check_options(ExtinctionTable.__init__, locals())
        wavelengths = tuple(float(w) for w in wavelengths)
        ratios = tuple(float(r) for r in ratios)
        if len(ratios) != len(wavelengths):
            raise ValueError(
                f"an extinction table needs a ratio for each of its "
                f"{len(wavelengths)} wavelengths; it has {len(ratios)}")
        if not all(a < b for a, b in zip(wavelengths, wavelengths[1:])):
            raise ValueError(
                f"the wavelengths of an extinction table must increase; "
                f"they are {list(wavelengths)}")
        self._wavelengths = wavelengths
        self._ratios = ratios

    def range(self) -> tuple[float, float]:
        return self._wavelengths[0], self._wavelengths[-1]

    def _ratio(self, wavelength: np.ndarray) -> np.ndarray:
        return np.interp(wavelength, self._wavelengths, self._ratios)


extinction_law_parser = TypedParser(ExtinctionLaw, [
    ExtinctionCCM,
    ExtinctionCalzetti,
    ExtinctionTable], family='extinction_law')
