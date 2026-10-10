"""
Primary beams: how a telescope attenuates the light of the sky away from
where it points (e.g. the primary beam of a radio interferometer).
"""

import abc
import os.path
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.special

from gbkfit.utils import fitsutils, gridutils, parseutils
from ._detail import check_scale


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
    before the PSF.
    """

    @abc.abstractmethod
    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the response to its configuration.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data (e.g. an
            image), if any.
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
        """

    @abc.abstractmethod
    def response(self, grid: gridutils.Grid) -> np.ndarray:
        """
        Return the response at the pixels of a grid.

        Parameters
        ----------
        grid : gridutils.Grid
            The grid; only its x and y axes are used.

        Returns
        -------
        ndarray
            The response, of shape (ny, nx).
        """


class PrimaryBeamRadial(PrimaryBeam, abc.ABC):
    """
    A response that depends on the distance from its centre, the pointing,
    where it is 1.

    Parameters
    ----------
    fwhm : float
        The full width at half maximum (arcsec), the unit of the distance.
    x, y : float, optional
        The pointing (arcsec, like xpos and ypos).
    """

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        # A radial response has no data to write
        return dict(type=self.type(), fwhm=self._fwhm, x=self._x, y=self._y)

    def __init__(self, fwhm: float, x: float = 0, y: float = 0):
        check_scale('fwhm', fwhm)
        self._fwhm = fwhm
        self._x = x
        self._y = y

    def response(self, grid: gridutils.Grid) -> np.ndarray:
        x, y = gridutils.sky_positions(grid)
        radius = np.hypot(x - self._x, y - self._y)
        return self._response_impl(radius / self._fwhm)

    @abc.abstractmethod
    def _response_impl(self, radius: np.ndarray) -> np.ndarray:
        """The response at the given radii (in units of the fwhm)."""
        pass


class PrimaryBeamGauss(PrimaryBeamRadial):
    """A Gaussian response."""

    @staticmethod
    def type() -> str:
        return 'gauss'

    def _response_impl(self, radius: np.ndarray) -> np.ndarray:
        return np.exp(-4 * np.log(2) * radius ** 2)


class PrimaryBeamAiry(PrimaryBeamRadial):
    """The response of a uniformly illuminated dish: an Airy pattern."""

    @staticmethod
    def type() -> str:
        return 'airy'

    def _response_impl(self, radius: np.ndarray) -> np.ndarray:
        u = 2 * _AIRY_HALF_MAXIMUM * radius
        u_safe = np.where(u > 0, u, 1)
        return np.where(u > 0, (2 * scipy.special.j1(u_safe) / u_safe) ** 2, 1)


class PrimaryBeamImage(PrimaryBeam):
    """
    A response given as an image with its world coordinates (e.g. the
    primary beam that an imaging package writes with the data), placed by
    its RA and Dec, sampled at the pixels (bilinearly), and 0 beyond the
    image.

    Its configuration has its file (a filename, or a dict with the
    filename and the HDU); the world coordinates come from the header,
    unless given.

    Parameters
    ----------
    data : ndarray
        The image; its pixels that are not finite are 0.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the image (see gridutils.make_grid).
    rota : float, optional
        The rotation of the image on the sky (see gridutils.Coords).
    """

    @staticmethod
    def type() -> str:
        return 'image'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'PrimaryBeamImage':
        with parseutils.config_path('file'):
            file, hdu = parseutils.parse_file(
                info.get('file') or {})
            # (the world coordinates given are those of the image, which
            # the others come from)
            data, coords = fitsutils.read_data(
                prefix + file, hdu, info.get('rpix'), info.get('rval'))
        info = dict(info) | dict(
            data=data, rpix=coords.rpix, rval=coords.rval,
            step=coords.step if info.get('step') is None else info['step'],
            rota=coords.rota if info.get('rota') is None else info['rota'])
        info.pop('file')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}primary_beam.fits'
        coords = self._grid.coords
        fitsutils.write_data(filename, self._data, coords, None, overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename))

    def __init__(
            self,
            data: np.ndarray,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise parseutils.ConfigError(
                f"the primary beam must be an image; it has {data.ndim} axes")
        self._data = np.where(np.isfinite(data), data, 0)
        self._grid = gridutils.make_grid(
            data.shape[::-1], step, rpix, rval, rota)

    def response(self, grid: gridutils.Grid) -> np.ndarray:
        gridutils.check_overlap(grid, self._grid, "the primary beam image")
        pixel_x, pixel_y = gridutils.pixels_on(grid, self._grid)
        return scipy.ndimage.map_coordinates(
            self._data, [pixel_y, pixel_x], order=1, mode='constant', cval=0)


primary_beam_parser = parseutils.TypedParser(PrimaryBeam, [
    PrimaryBeamAiry,
    PrimaryBeamGauss,
    PrimaryBeamImage])
