"""
The noise of mock observations: what they add to the data a model gives
where the instrument records them (see ObservationGroup.mock).
"""

import abc
import os
from typing import Any, Self

import astropy.io.fits
import astropy.units as u
import numpy as np
import scipy.fft

from gbkfit.instrument import LSF, PSF, lsf_parser, psf_parser
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Noise',
    'NoiseGauss',
    'noise_parser'
]


# The spatial correlation of the noise that the PSF of the instrument gives
INSTRUMENT = 'instrument'


class Noise(parseutils.TypedSerializable, abc.ABC):
    """
    The noise of the data of an observation: random values added to the
    data where the instrument records them, on their grid.
    """

    @abc.abstractmethod
    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the noise to its configuration.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data (e.g. an
            error map), if any.
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
    def sigma(self, grid: gridutils.Grid) -> np.ndarray:
        """
        Return the dispersion of the noise of each value of data on a grid.

        Parameters
        ----------
        grid : Grid
            The grid of the data.

        Returns
        -------
        ndarray
            The dispersion, of the shape of the data (the reversed size of
            the grid).

        Raises
        ------
        ConfigError
            If the noise does not suit data of that shape.
        """

    @abc.abstractmethod
    def draw(
            self,
            rng: np.random.Generator,
            grid: gridutils.Grid,
            psf: PSF | None
    ) -> np.ndarray:
        """
        Draw the noise of data on a grid.

        Parameters
        ----------
        rng : Generator
            The random numbers.
        grid : Grid
            The grid of the data.
        psf : PSF, optional
            The PSF of the instrument that records the data.

        Returns
        -------
        ndarray
            The noise, of the shape of the data (see sigma).

        Raises
        ------
        ConfigError
            If the noise does not suit the data or the instrument.
        """


class NoiseGauss(Noise):
    """
    Gaussian noise of mean 0, of the dispersion rms at every value or of
    an error map; independent from value to value, unless correlated on
    the sky or along the spectral axis.

    The correlation of the noise of two values at the separation (dx, dy)
    on the sky and dz along the spectral axis is C_xy(dx, dy) C_z(dz):
    the shapes of the PSF spatial_correlation and of the LSF
    spectral_correlation, divided by their values at their centres. The
    PSF of the instrument ('instrument') is the correlation of the noise
    of a radio interferometer, which the beam correlates; a correlation
    measured on data (e.g. in a region without light) can be given as an
    image (PSFImage). A shape that is not that of a correlation, whose
    Fourier transform is negative somewhere (e.g. a box), takes the
    nearest one: its symmetric part, without the negative parts of its
    transform. A PSF that varies along the spectral axis is the spatial
    correlation of each channel.

    Its configuration has the error map as a FITS file (a filename, or a
    dict with the filename and the HDU).

    Parameters
    ----------
    rms : float, optional
        The dispersion of every value.
    error : ndarray, optional
        The dispersion of each value: an array that broadcasts to the
        shape of the data (e.g. one value per pixel, ny by nx, for all the
        channels of a cube); values whose error is not finite are
        missing (NaN).
    spatial_correlation : PSF or str, optional
        The correlation on the sky: a PSF, or 'instrument' (the PSF of the
        instrument).
    spectral_correlation : LSF, optional
        The correlation along the spectral axis; it must not vary.

    Raises
    ------
    ConfigError
        If not exactly one of rms and error is given, rms is not
        positive, the error map has negative values, the spatial
        correlation is a str other than 'instrument', or the spectral one
        varies.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        info = dict(info)
        file = parseutils.load_option(parseutils.parse_file, info, 'error')
        if file is not None:
            with parseutils.config_path('error'):
                info['error'] = fitsutils.read_data(*file)[0]
        if isinstance(info.get('spatial_correlation'), dict):
            parseutils.load_option_and_update_info(
                psf_parser, info, 'spatial_correlation')
        parseutils.load_option_and_update_info(
            lsf_parser, info, 'spectral_correlation')
        return super().load(info)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict(type=self.type())
        if self._rms is not None:
            info.update(rms=self._rms)
        else:
            filename = f'{prefix}noise_error.fits'
            astropy.io.fits.writeto(filename, self._error, overwrite=overwrite)
            info.update(
                error=filename if dump_path else os.path.basename(filename))
        kwargs = dict(prefix=prefix, dump_path=dump_path, overwrite=overwrite)
        spatial = self._spatial_correlation
        if spatial is not None:
            info.update(spatial_correlation=spatial if spatial == INSTRUMENT
                        else psf_parser.dump(spatial, **kwargs))
        if self._spectral_correlation is not None:
            info.update(spectral_correlation=lsf_parser.dump(
                self._spectral_correlation, **kwargs))
        return info

    def __init__(
            self,
            rms: float | None = None,
            error: np.ndarray | None = None,
            spatial_correlation: PSF | str | None = None,
            spectral_correlation: LSF | None = None
    ):
        if (rms is None) == (error is None):
            raise ConfigError("give one of rms and error")
        if rms is not None and not rms > 0:
            raise ConfigError(f"rms must be positive; it is {rms}")
        if error is not None:
            error = np.asarray(error, dtype=float)
            if np.any(error < 0):
                raise ConfigError("the error map has negative values")
        if isinstance(spatial_correlation, str) \
                and spatial_correlation != INSTRUMENT:
            raise ConfigError(
                f"spatial_correlation is a PSF or '{INSTRUMENT}'; it is "
                f"{spatial_correlation!r}")
        if spectral_correlation is not None and spectral_correlation.varies():
            raise ConfigError("the spectral correlation must not vary")
        self._rms = rms
        self._error = error
        self._spatial_correlation = spatial_correlation
        self._spectral_correlation = spectral_correlation

    def rms(self) -> float | None:
        """Return the dispersion of every value, if it has one."""
        return self._rms

    def error(self) -> np.ndarray | None:
        """Return the error map, if it has one."""
        return self._error

    def spatial_correlation(self) -> PSF | str | None:
        """Return the correlation on the sky, if any."""
        return self._spatial_correlation

    def spectral_correlation(self) -> LSF | None:
        """Return the correlation along the spectral axis, if any."""
        return self._spectral_correlation

    def sigma(self, grid: gridutils.Grid) -> np.ndarray:
        shape = grid.size[::-1]
        if self._rms is not None:
            return np.full(shape, self._rms)
        try:
            return np.broadcast_to(self._error, shape).copy()
        except ValueError:
            raise ConfigError(
                f"the error map, of shape {self._error.shape}, does not "
                f"broadcast to the data, of shape {shape}") from None

    def draw(
            self,
            rng: np.random.Generator,
            grid: gridutils.Grid,
            psf: PSF | None
    ) -> np.ndarray:
        sigma = self.sigma(grid)
        spatial = self._spatial_kernels(grid, psf)
        spectral = self._spectral_kernel(grid)
        # White noise of unit variance on a periodic grid larger than the
        # data by the kernels, which their correlations do not wrap around
        # (see _correlated)
        shape = sigma.shape
        padded = list(shape)
        if spectral is not None:
            padded[0] += spectral.shape[-1] - 1
        if spatial is not None:
            padded[-2] += spatial.shape[-2] - 1
            padded[-1] += spatial.shape[-1] - 1
        noise = rng.standard_normal(padded)
        # Correlated along the spectral axis, then on the sky
        if spectral is not None:
            noise = np.moveaxis(_correlated(
                np.moveaxis(noise, 0, -1), spectral, 1), -1, 0)[:shape[0]]
        if spatial is not None:
            noise = _correlated(noise, spatial, 2)
        return noise[tuple(slice(0, n) for n in shape)] * sigma

    def _spatial_kernels(
            self, grid: gridutils.Grid, psf: PSF | None
    ) -> np.ndarray | None:
        """
        Return the spatial correlation on the pixels of a grid: one image
        for all the channels, or one for each if it varies (nz, ny, nx).
        """
        spatial = self._spatial_correlation
        if spatial == INSTRUMENT:
            if psf is None:
                raise ConfigError(
                    f"the spatial correlation of the noise is "
                    f"'{INSTRUMENT}', but the instrument has no PSF")
            spatial = psf
        if spatial is None:
            return None
        step, rota = grid.coords.step[:2], grid.coords.rota
        if not spatial.varies():
            return spatial.asarray(step, rota=rota)
        if grid.spectral_axis is None:
            raise ConfigError(
                "the spatial correlation of the noise varies along the "
                "spectral axis, but the data have none")
        psfs = spatial.at(_velocities(grid) * u.km / u.s, grid.coords.rest)
        size = tuple(np.max([p.size(step) for p in psfs], axis=0).tolist())
        return np.stack([p.asarray(step, size, rota=rota) for p in psfs])

    def _spectral_kernel(self, grid: gridutils.Grid) -> np.ndarray | None:
        """Return the spectral correlation on the channels of a grid."""
        spectral = self._spectral_correlation
        if spectral is None:
            return None
        if grid.spectral_axis is None:
            raise ConfigError(
                "the noise has a spectral correlation, but the data have no "
                "spectral axis")
        return spectral.asarray(grid.coords.step[2])


def _velocities(grid: gridutils.Grid) -> np.ndarray:
    """Return the velocities of the channels of a grid (km/s)."""
    step = grid.coords.step[2]
    return grid.zero()[2] + step * np.arange(grid.size[2])


def _correlated(
        white: np.ndarray, correlation: np.ndarray, naxes: int
) -> np.ndarray:
    """
    Return white noise of unit variance filtered along its last naxes
    axes into noise of unit variance and the given correlation: its array
    along those axes (odd sizes, centred; e.g. a PSF image), one for all
    the other axes or one for each (e.g. for each channel).

    The noise is periodic, so its first m - k + 1 values along each axis
    (m its size, k that of the correlation) have the correlation (divided
    by its value at the centre) exactly. The filter is the
    square root of its power spectrum: that of its symmetric part, without
    negative parts (none for a correlation).
    """
    shape = white.shape[-naxes:]
    axes = tuple(range(-naxes, 0))
    size = correlation.shape[-naxes:]
    # The correlation on the periodic grid of the noise, centred on 0
    kernel = np.zeros(correlation.shape[:-naxes] + shape)
    kernel[(..., *(slice(0, n) for n in size))] = correlation
    kernel = np.roll(kernel, [-(n // 2) for n in size], axis=axes)
    power = np.maximum(scipy.fft.rfftn(kernel, axes=axes).real, 0)
    # The variance of the noise it gives: its value at 0
    variance = scipy.fft.irfftn(power, shape, axes=axes)[
        (..., *(0,) * naxes)]
    gain = np.sqrt(power / variance[(..., *(None,) * naxes)])
    return scipy.fft.irfftn(
        scipy.fft.rfftn(white, axes=axes) * gain, shape, axes=axes)


noise_parser = parseutils.TypedParser(Noise, [NoiseGauss])
