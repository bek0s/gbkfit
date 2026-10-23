"""
The traces of slitless spectra (see SlitlessSpectra): where the light of
each wavelength lands on the detector, moved by (dx, dy) pixels from the
place of its undispersed light.
"""

import abc
from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt

from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Trace',
    'TraceGrismConf',
    'TracePolynomial',
    'trace_parser'
]


# The Newton iterations of the parameter t of a wavelength of a GRISMCONF
# trace: at most this many, to this tolerance
_NEWTON_ITERATIONS = 50
_NEWTON_TOLERANCE = 1e-12


class Trace(parseutils.TypedSerializable, abc.ABC):
    """
    The trace of a disperser: the offsets (dx, dy; pixels of the detector)
    of the light of each wavelength from the place of its undispersed
    light.
    """

    @abc.abstractmethod
    def offsets(
            self,
            wavelength: npt.ArrayLike,
            x: npt.ArrayLike,
            y: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the offsets (dx, dy; pixels) of the wavelengths (m, in the
        medium of the spectral axis) at the places (x, y) of their
        undispersed light (pixels of the image, from its first pixel),
        broadcast together.
        """


class TracePolynomial(Trace):
    """
    A trace of polynomials of the wavelength, the same at every place:

        dx(w) = sum_n dx[n] ((w - wave) / 1 um)^n

    (pixels) and likewise dy.

    Parameters
    ----------
    wave : str or Quantity
        The wavelength the trace is expanded about.
    dx : Sequence of float
        The coefficients along x.
    dy : Sequence of float, optional
        The coefficients along y; 0 by default.

    Raises
    ------
    ConfigError
        If wave is not a wavelength, or dx or dy has no coefficients.
    """

    @staticmethod
    def type() -> str:
        return 'polynomial'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), wave=str(self._wave),
                    dx=list(self._dx), dy=list(self._dy))

    def __init__(
            self,
            wave: str | astropy.units.Quantity,
            dx: Sequence[float],
            dy: Sequence[float] = (0.0,)
    ):
        quantity = gridutils.make_rest(wave)
        if quantity.unit.physical_type != 'length':
            raise ConfigError(
                f"the wave of a trace must be a wavelength; it is {wave!r}")
        if not len(dx) or not len(dy):
            raise ConfigError("the dx and dy of a trace need coefficients")
        self._wave = quantity.to('m')
        self._dx = tuple(float(c) for c in dx)
        self._dy = tuple(float(c) for c in dy)

    def offsets(
            self,
            wavelength: npt.ArrayLike,
            x: npt.ArrayLike,
            y: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        um = (np.asarray(wavelength, dtype=float) - self._wave.value) * 1e6
        um, _, _ = np.broadcast_arrays(um, x, y)
        return npp.polyval(um, self._dx), npp.polyval(um, self._dy)


class TraceGrismConf(Trace):
    """
    A trace of the GRISMCONF form of JWST (Pirzkal & Ryan 2016; of the
    configurations of NIRCam and NIRISS, and of the JWST pipeline): the
    wavelength (um) and the offsets as polynomials of a parameter t,

        w(t) = sum_k L_k t^k,  dx(t) = sum_k X_k t^k,  dy(t) = sum_k Y_k t^k

    each coefficient a polynomial of the place (x0, y0) of the undispersed
    light on the detector (pixels), of the coefficients in the order of
    GRISMCONF: 1, x0, y0, x0^2, x0 y0, y0^2, x0^3, ... The t of a
    wavelength solves w(t) = w (by Newton iterations from t = 0.5, to
    1e-12); the wavelengths are in vacuum.

    Parameters
    ----------
    displ, dispx, dispy : Sequence of Sequence of float
        The coefficients L_k, X_k and Y_k, by power of t (each a sequence
        of the coefficients of its polynomial of the place).
    origin : Sequence of float, optional
        The place on the detector (x0, y0; pixels) of the first pixel of
        the image (e.g. of a cutout); (0, 0) by default.

    Raises
    ------
    ConfigError
        If a polynomial has no terms, or a polynomial of the place has a
        number of coefficients that is not (n + 1)(n + 2) / 2.
    """

    @staticmethod
    def type() -> str:
        return 'grismconf'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            displ=[list(c) for c in self._displ],
            dispx=[list(c) for c in self._dispx],
            dispy=[list(c) for c in self._dispy],
            origin=list(self._origin))

    def __init__(
            self,
            displ: Sequence[Sequence[float]],
            dispx: Sequence[Sequence[float]],
            dispy: Sequence[Sequence[float]],
            origin: Sequence[float] = (0.0, 0.0)
    ):
        polynomials = dict(displ=displ, dispx=dispx, dispy=dispy)
        for name, terms in polynomials.items():
            if not len(terms):
                raise ConfigError(f"{name} needs coefficients")
            for coefficients in terms:
                _field_degree(len(coefficients), name)
        if len(origin) != 2:
            raise ConfigError("origin must be (x0, y0)")
        self._displ, self._dispx, self._dispy = (
            tuple(tuple(float(c) for c in coefficients)
                  for coefficients in terms)
            for terms in (displ, dispx, dispy))
        self._origin = (float(origin[0]), float(origin[1]))

    def offsets(
            self,
            wavelength: npt.ArrayLike,
            x: npt.ArrayLike,
            y: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        um, x, y = np.broadcast_arrays(
            np.asarray(wavelength, dtype=float) * 1e6,
            np.asarray(x, dtype=float) + self._origin[0],
            np.asarray(y, dtype=float) + self._origin[1])
        lam, dlam = (_field_polynomials(self._displ, x, y, derivative)
                     for derivative in (False, True))
        t = np.full(um.shape, 0.5)
        for _ in range(_NEWTON_ITERATIONS):
            step = (npp.polyval(t, lam, tensor=False) - um) \
                / npp.polyval(t, dlam, tensor=False)
            t = t - step
            if np.all(np.abs(step) < _NEWTON_TOLERANCE):
                break
        else:
            raise ValueError(
                "the parameter t of the wavelengths of a GRISMCONF trace "
                "did not converge")
        return (npp.polyval(t, _field_polynomials(self._dispx, x, y),
                            tensor=False),
                npp.polyval(t, _field_polynomials(self._dispy, x, y),
                            tensor=False))


def _field_degree(ncoefficients: int, name: str) -> int:
    """
    Return the degree n of a polynomial of the place of ncoefficients =
    (n + 1)(n + 2) / 2 (of a polynomial named name), or raise ConfigError.
    """
    degree = int(round((np.sqrt(8 * ncoefficients + 1) - 3) / 2))
    if (degree + 1) * (degree + 2) // 2 != ncoefficients:
        raise ConfigError(
            f"the polynomials of the place of {name} need (n + 1)(n + 2) / 2 "
            f"coefficients; one has {ncoefficients}")
    return degree


def _field_polynomials(
        terms: Sequence[Sequence[float]],
        x: np.ndarray,
        y: np.ndarray,
        derivative: bool = False
) -> np.ndarray:
    """
    Return the coefficients of a polynomial of t at the places (x, y): the
    values of its polynomials of the place (one per power of t, first
    axis); of its derivative in t, if asked.
    """
    values = []
    for coefficients in terms:
        degree = _field_degree(len(coefficients), '')
        powers = [(i - j, j) for i in range(degree + 1) for j in range(i + 1)]
        values.append(sum(c * x ** px * y ** py
                          for c, (px, py) in zip(coefficients, powers)))
    values = np.array(values)
    if not derivative:
        return values
    if len(values) == 1:
        return np.zeros_like(values)
    powers = np.arange(1, len(values)).reshape((-1,) + (1,) * x.ndim)
    return values[1:] * powers


trace_parser = parseutils.TypedParser(Trace, [
    TracePolynomial,
    TraceGrismConf])
