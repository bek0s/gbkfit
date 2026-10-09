import abc
from numbers import Real

import numpy as np
import scipy.special

from gbkfit.psflsf import check_scale
from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'PrimaryBeam',
    'PrimaryBeamAiry',
    'PrimaryBeamGauss',
    'primary_beam_parser'
]


# The argument of the Airy pattern (2 J1(u) / u)^2 at its half maximum
_AIRY_HALF_MAXIMUM = 1.616339948310703


class PrimaryBeam(parseutils.TypedSerializable, abc.ABC):
    """
    The response of a telescope to the sky, which attenuates its light
    before the PSF (e.g. the primary beam of a radio interferometer): 1 at
    its centre, the pointing (x and y in arcsec, like xpos and ypos).
    """

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'primary beam')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self):
        return dict(type=self.type(), fwhm=self._fwhm, x=self._x, y=self._y)

    def __init__(self, fwhm: Real, x: Real = 0, y: Real = 0):
        """fwhm is the full width at half maximum (arcsec)."""
        check_scale('fwhm', fwhm)
        self._fwhm = fwhm
        self._x = x
        self._y = y

    def response(self, grid: fitsutils.Grid) -> np.ndarray:
        """The response at the pixels of the x and y axes of a grid."""
        x, y = fitsutils.sky_positions(grid)
        radius = np.hypot(x - self._x, y - self._y)
        return self._response_impl(radius / self._fwhm)

    @abc.abstractmethod
    def _response_impl(self, radius: np.ndarray) -> np.ndarray:
        """The response at the given radii (in units of the fwhm)."""
        pass


class PrimaryBeamGauss(PrimaryBeam):
    """A Gaussian response."""

    @staticmethod
    def type():
        return 'gauss'

    def _response_impl(self, radius):
        return np.exp(-4 * np.log(2) * radius ** 2)


class PrimaryBeamAiry(PrimaryBeam):
    """The response of a uniformly illuminated dish: an Airy pattern."""

    @staticmethod
    def type():
        return 'airy'

    def _response_impl(self, radius):
        u = 2 * _AIRY_HALF_MAXIMUM * radius
        u_safe = np.where(u > 0, u, 1)
        return np.where(u > 0, (2 * scipy.special.j1(u_safe) / u_safe) ** 2, 1)


primary_beam_parser = parseutils.TypedParser(PrimaryBeam, [
    PrimaryBeamAiry,
    PrimaryBeamGauss])
