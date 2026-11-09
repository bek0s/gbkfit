"""
Primary beams: how a telescope attenuates the light of the sky away from
where it points (e.g. the primary beam of a radio interferometer).
"""

import abc
import os.path
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import astropy.units as u
import numpy as np
import scipy.ndimage
import scipy.special
from annotated_types import Gt

from gbkfit.config import (
    DataFile, TypedParser, Unit, check_options, split_data_file)
from gbkfit.grid import (
    Grid, VelocityConvention, check_overlap, make_grid, pixels_on,
    sky_positions)
from gbkfit.io import header_shape, read_data, read_header, write_data
from . import _detail, varying


__all__ = [
    'PrimaryBeam',
    'PrimaryBeamAiry',
    'PrimaryBeamGauss',
    'PrimaryBeamImage',
    'primary_beam_parser'
]


# The argument of the Airy pattern (2 J1(u) / u)^2 at its half maximum
_AIRY_HALF_MAXIMUM_ARG = 1.616339948310703

# The world coordinates of an image (see make_grid): its reference pixel,
# the RA and Dec there (degrees), and its rotation on the sky (degrees)
_Rpix: TypeAlias = float | Sequence[float]
_Rval: TypeAlias = Annotated[float | Sequence[float], Unit('deg')]
_Rota: TypeAlias = Annotated[float, Unit('deg')]


class PrimaryBeam(_detail.Response):
    """
    The response of a telescope to the sky, which attenuates its light
    before the PSF: the same at all frequencies, or of each (see varies
    and at).
    """

    def response(self, grid: Grid) -> np.ndarray:
        """
        Return the response at the pixels of a grid.

        Parameters
        ----------
        grid : Grid
            The grid; only its x and y axes are used.

        Returns
        -------
        ndarray
            The response, of shape (ny, nx).

        Raises
        ------
        RuntimeError
            If the beam varies along the spectral axis (it is that at each
            channel; see at).
        """
        self._check_not_varying()
        return self._response_impl(grid)

    @abc.abstractmethod
    def _response_impl(self, grid: Grid) -> np.ndarray:
        """The response at the pixels of a grid (see response)."""


class _RadialPrimaryBeam(PrimaryBeam):
    """
    A response that depends on the distance from its centre, the pointing,
    where it is 1; of a width that is the same at all frequencies, or
    inversely proportional to the frequency (that of a dish, lambda / D).

    Parameters
    ----------
    fwhm : float
        The full width at half maximum (arcsec), the unit of the distance.
    x, y : float, optional
        The pointing (arcsec, like xpos and ypos).
    frequency : float, optional
        The frequency of the fwhm (Hz), for a width fwhm frequency / nu at
        the frequency nu of each channel; without it, the width is fwhm
        at all frequencies.

    Raises
    ------
    ValueError
        If the fwhm or the frequency is not positive.
    """

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        # A radial response has no data to write
        return dict(
            type=self.type(), fwhm=self._fwhm, x=self._x, y=self._y,
            frequency=self._frequency)

    def __init__(
            self,
            fwhm: Annotated[float, Gt(0), Unit('arcsec')],
            x: Annotated[float, Unit('arcsec')] = 0,
            y: Annotated[float, Unit('arcsec')] = 0,
            frequency: Annotated[float, Gt(0), Unit('Hz')] | None = None
    ):
        check_options(_RadialPrimaryBeam.__init__, locals())
        self._fwhm = fwhm
        self._x = x
        self._y = y
        self._frequency = frequency

    def varies(self) -> bool:
        return self._frequency is not None

    def _at_impl(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None
    ) -> list[PrimaryBeam]:
        # (of a width inversely proportional to the frequency)
        return [
            type(self)(self._fwhm * self._frequency / frequency, self._x,
                       self._y)
            for frequency in varying.to_frequencies(points, convention)]

    def _response_impl(self, grid: Grid) -> np.ndarray:
        x, y = sky_positions(grid)
        radius = np.hypot(x - self._x, y - self._y)
        return self._profile(radius / self._fwhm)

    @abc.abstractmethod
    def _profile(self, radius: np.ndarray) -> np.ndarray:
        """The response at the given radii (in units of the fwhm)."""


class PrimaryBeamGauss(_RadialPrimaryBeam):
    """A Gaussian response."""

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

    def _profile(self, radius: np.ndarray) -> np.ndarray:
        return np.exp(-4 * np.log(2) * radius ** 2)


class PrimaryBeamAiry(_RadialPrimaryBeam):
    """The response of a uniformly illuminated dish: an Airy pattern."""

    @staticmethod
    def type() -> str:
        return 'airy'

    @staticmethod
    def type_title() -> str:
        return 'Airy pattern'

    def _profile(self, radius: np.ndarray) -> np.ndarray:
        arg = 2 * _AIRY_HALF_MAXIMUM_ARG * radius
        arg_safe = np.where(arg > 0, arg, 1)
        return np.where(
            arg > 0, (2 * scipy.special.j1(arg_safe) / arg_safe) ** 2, 1)


class PrimaryBeamImage(PrimaryBeam):
    """
    A response given as an image with its world coordinates (e.g. the
    primary beam that an imaging package writes with the data). It is
    placed by its RA and Dec, sampled at the pixels (bilinearly), and 0
    beyond the image: the same at all frequencies (not supported yet: a
    cube of an image for each channel, e.g. a CASA .pb cube).

    Configurations give its file (see from_file).

    Parameters
    ----------
    data : ndarray
        The image; its pixels that are not finite are 0.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the image (see make_grid).
    rota : float, optional
        The rotation of the image on the sky (see Coords).
    """

    # TODO: a cube of an image for each channel (e.g. a CASA .pb cube),
    # which varies (see varies and at), as the radial beams do

    config_constructors = ('from_file',)

    @staticmethod
    def type() -> str:
        return 'image'

    @staticmethod
    def type_title() -> str:
        return 'Image'

    @classmethod
    def from_file(
            cls,
            file: DataFile,
            step: _detail.Step | None = None,
            rpix: _Rpix | None = None,
            rval: _Rval | None = None,
            rota: _Rota | None = None
    ) -> 'PrimaryBeamImage':
        """
        Read a primary beam image, and its world coordinates, from a file.

        Parameters
        ----------
        file : DataFile
            The file of the image, and where it is in it (see read_image).
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the image (see make_grid);
            by default, those of the header. Either rpix or rval can be
            given, and the other comes from the header (see
            read_data).
        rota : float, optional
            The rotation of the image on the sky (see Coords); by
            default, that of the header.

        Returns
        -------
        PrimaryBeamImage
            The primary beam.

        Raises
        ------
        ValueError
            If the file has no image (e.g. a cube of the beam of each
            channel), or its header has invalid world coordinates (see
            read_data).
        """
        check_options(PrimaryBeamImage.from_file, locals())
        filename = split_data_file(file)[0]
        axes = sum(n > 1 for n in header_shape(read_header(file)))
        if axes > 2:
            raise ValueError(
                f"{filename}: a primary beam is an image; it has {axes} axes "
                f"of more than one pixel (a cube of the beam of each channel, "
                f"e.g. a CASA .pb cube, is not supported yet: give the image "
                f"of one channel)")
        data, coords = read_data(file, rpix=rpix, rval=rval)
        return cls(
            data, coords.step if step is None else step, coords.rpix,
            coords.rval, coords.rota if rota is None else rota)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}primary_beam.fits'
        coords = self._grid.coords
        write_data(filename, self._data, coords, overwrite=overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename))

    def __init__(
            self,
            data: np.ndarray,
            step: _detail.Step | None = None,
            rpix: _Rpix | None = None,
            rval: _Rval | None = None,
            rota: _Rota | None = None
    ):
        check_options(PrimaryBeamImage.__init__, locals())
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(
                f"the primary beam must be an image; it has {data.ndim} axes")
        self._data = np.where(np.isfinite(data), data, 0)
        self._grid = make_grid(
            data.shape[::-1], step, rpix, rval, rota)

    def _response_impl(self, grid: Grid) -> np.ndarray:
        check_overlap(grid, self._grid, "the primary beam image")
        pixel_x, pixel_y = pixels_on(grid, self._grid)
        return scipy.ndimage.map_coordinates(
            self._data, [pixel_y, pixel_x], order=1, mode='constant', cval=0)


primary_beam_parser = TypedParser(PrimaryBeam, [
    PrimaryBeamAiry,
    PrimaryBeamGauss,
    PrimaryBeamImage])
