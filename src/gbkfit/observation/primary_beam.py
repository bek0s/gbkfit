import abc
import os.path
from numbers import Real
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.special

from gbkfit.psflsf import check_scale
from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'PrimaryBeam',
    'PrimaryBeamAiry',
    'PrimaryBeamGauss',
    'PrimaryBeamImage',
    'PrimaryBeamRadial',
    'primary_beam_parser'
]


# The argument of the Airy pattern (2 J1(u) / u)^2 at its half maximum
_AIRY_HALF_MAXIMUM = 1.616339948310703


class PrimaryBeam(parseutils.TypedSerializable, abc.ABC):
    """
    The response of a telescope to the sky, which attenuates its light
    before the PSF (e.g. the primary beam of a radio interferometer).
    """

    @abc.abstractmethod
    def response(self, grid: fitsutils.Grid) -> np.ndarray:
        """The response at the pixels of the x and y axes of a grid."""
        pass


class PrimaryBeamRadial(PrimaryBeam, abc.ABC):
    """
    A response that depends on the distance from its centre, the pointing
    (x and y in arcsec, like xpos and ypos), where it is 1, in units of
    its full width at half maximum (fwhm, arcsec).
    """

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'primary beam')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self):
        return dict(type=self.type(), fwhm=self._fwhm, x=self._x, y=self._y)

    def __init__(self, fwhm: Real, x: Real = 0, y: Real = 0):
        check_scale('fwhm', fwhm)
        self._fwhm = fwhm
        self._x = x
        self._y = y

    def response(self, grid):
        x, y = fitsutils.sky_positions(grid)
        radius = np.hypot(x - self._x, y - self._y)
        return self._response_impl(radius / self._fwhm)

    @abc.abstractmethod
    def _response_impl(self, radius: np.ndarray) -> np.ndarray:
        """The response at the given radii (in units of the fwhm)."""
        pass


class PrimaryBeamGauss(PrimaryBeamRadial):
    """A Gaussian response."""

    @staticmethod
    def type():
        return 'gauss'

    def _response_impl(self, radius):
        return np.exp(-4 * np.log(2) * radius ** 2)


class PrimaryBeamAiry(PrimaryBeamRadial):
    """The response of a uniformly illuminated dish: an Airy pattern."""

    @staticmethod
    def type():
        return 'airy'

    def _response_impl(self, radius):
        u = 2 * _AIRY_HALF_MAXIMUM * radius
        u_safe = np.where(u > 0, u, 1)
        return np.where(u > 0, (2 * scipy.special.j1(u_safe) / u_safe) ** 2, 1)


class PrimaryBeamImage(PrimaryBeam):
    """
    A response given as an image with its world coordinates (e.g. the
    primary beam that an imaging package writes with the data), sampled
    at the pixels (bilinearly), and 0 beyond the image.
    """

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info, prefix=''):
        desc = parseutils.make_typed_desc(cls, 'primary beam')
        file = info.get('file')
        if isinstance(file, str):
            file = dict(file=file)
        with parseutils.config_path('file'):
            options = parseutils.parse_options(
                file or {}, 'primary beam file', required={'file'},
                optional={'hdu'})
            data, coords = fitsutils.read_data(
                prefix + options['file'], options.get('hdu', 0))
        info = dict(info) | dict(
            data=data, step=coords.step, rpix=coords.rpix, rval=coords.rval,
            rota=coords.rota)
        info.pop('file')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        filename = f'{prefix}primary_beam.fits'
        coords = self._grid.coords
        fitsutils.write_data(filename, self._data, coords, None, overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename))

    def __init__(
            self,
            data: np.ndarray,
            step: Any = None,
            rpix: Any = None,
            rval: Any = None,
            rota: Real | None = None
    ):
        """
        The image is on the grid of the given world coordinates (see
        fitsutils.make_grid); its pixels that are not finite are 0.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise RuntimeError(
                f"the primary beam must be an image; it has {data.ndim} axes")
        self._data = np.where(np.isfinite(data), data, 0)
        self._grid = fitsutils.make_grid(data.shape[::-1], step, rpix, rval, rota)

    def response(self, grid):
        x, y = fitsutils.sky_positions(grid)
        matrix, offset = fitsutils.sky_to_pixel(self._grid)
        pixel_x = matrix[0, 0] * x + matrix[0, 1] * y + offset[0]
        pixel_y = matrix[1, 0] * x + matrix[1, 1] * y + offset[1]
        return scipy.ndimage.map_coordinates(
            self._data, [pixel_y, pixel_x], order=1, mode='constant', cval=0)


primary_beam_parser = parseutils.TypedParser(PrimaryBeam, [
    PrimaryBeamAiry,
    PrimaryBeamGauss,
    PrimaryBeamImage])
