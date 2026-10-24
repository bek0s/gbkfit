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
from gbkfit.math.correlation import periodic_correlation
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Noise',
    'NoiseGauss',
    'NoisePoisson',
    'NoiseStudentT',
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

    def correlations(self) -> tuple[PSF | str | None, LSF | None]:
        """
        Return its correlation on the sky (a PSF, or 'instrument': the PSF
        of the instrument) and along the spectral axis, which the mock
        data of it declare (see ObservationGroup.mock); none for noise
        independent from value to value.
        """
        return None, None

    @abc.abstractmethod
    def sigma(self, grid: gridutils.Grid, signal: np.ndarray) -> np.ndarray:
        """
        Return the dispersion of the noise of each value of data on a grid.

        Parameters
        ----------
        grid : Grid
            The grid of the data.
        signal : ndarray
            The data without noise, which the noise is added to (of the
            shape of the data); noise that does not depend on them ignores
            them.

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
            psf: PSF | None,
            signal: np.ndarray
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
        signal : ndarray
            The data without noise, which the noise is added to (see
            sigma).

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
        info = _load_error(info)
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
        info = dict(type=self.type()) | _dump_dispersion(
            self._rms, self._error, prefix, dump_path, overwrite)
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
        error = _check_dispersion(rms, error)
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

    def correlations(self) -> tuple[PSF | str | None, LSF | None]:
        return self._spatial_correlation, self._spectral_correlation

    def sigma(self, grid: gridutils.Grid, signal: np.ndarray) -> np.ndarray:
        return _dispersion(self._rms, self._error, grid)

    def draw(
            self,
            rng: np.random.Generator,
            grid: gridutils.Grid,
            psf: PSF | None,
            signal: np.ndarray
    ) -> np.ndarray:
        sigma = self.sigma(grid, signal)
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
        sky_axes = len(grid.size) - (grid.spectral_axis is not None)
        if spatial is not None and sky_axes < 2:
            raise ConfigError(
                "the noise has a spatial correlation, but the data are not "
                "on the sky (e.g. visibilities)")
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
        psfs = spatial.at(
            _velocities(grid) * u.km / u.s,
            gridutils.VelocityConvention.of(grid.coords))
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
        return spectral.asarray(abs(grid.coords.step[grid.spectral_axis]))


class NoiseStudentT(Noise):
    """
    Student-t noise of mean 0 and of dof degrees of freedom (above 2), of
    the dispersion rms at every value or of an error map (as NoiseGauss),
    independent from value to value: heavier tails than Gaussian noise
    (outliers), e.g. to test how fits take them; Gaussian as dof grows.
    Its values are t draws times sigma sqrt((dof - 2) / dof), whose
    dispersion is sigma.

    Its configuration has the error map as a FITS file (see NoiseGauss).

    Parameters
    ----------
    dof : float
        The degrees of freedom.
    rms : float, optional
        The dispersion of every value.
    error : ndarray, optional
        The dispersion of each value (see NoiseGauss).

    Raises
    ------
    ConfigError
        If dof is not above 2, not exactly one of rms and error is given,
        rms is not positive, or the error map has negative values.
    """

    @staticmethod
    def type() -> str:
        return 'student_t'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        return super().load(_load_error(info))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type(), dof=self._dof) | _dump_dispersion(
            self._rms, self._error, prefix, dump_path, overwrite)

    def __init__(
            self,
            dof: float,
            rms: float | None = None,
            error: np.ndarray | None = None
    ):
        if not dof > 2:
            raise ConfigError(
                f"dof must be above 2 (a finite dispersion); it is {dof}")
        self._dof = dof
        self._rms = rms
        self._error = _check_dispersion(rms, error)

    def dof(self) -> float:
        """Return the degrees of freedom."""
        return self._dof

    def sigma(self, grid: gridutils.Grid, signal: np.ndarray) -> np.ndarray:
        return _dispersion(self._rms, self._error, grid)

    def draw(
            self,
            rng: np.random.Generator,
            grid: gridutils.Grid,
            psf: PSF | None,
            signal: np.ndarray
    ) -> np.ndarray:
        sigma = self.sigma(grid, signal)
        t = rng.standard_t(self._dof, sigma.shape)
        return t * sigma * np.sqrt((self._dof - 2) / self._dof)


class NoisePoisson(Noise):
    """
    Poisson noise of data of counts: the data become counts / gain, the
    counts Poisson draws of the signal (the data without noise) times the
    gain, of the dispersion sqrt(signal gain) / gain (a signal below 0,
    e.g. of the ringing of a convolution, counts as 0). Independent from
    value to value.

    Parameters
    ----------
    gain : float, optional
        The counts per unit of the data (e.g. of the photons per unit of
        flux).

    Raises
    ------
    ConfigError
        If the gain is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'poisson'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type(), gain=self._gain)

    def __init__(self, gain: float = 1.0):
        if not gain > 0:
            raise ConfigError(f"gain must be positive; it is {gain}")
        self._gain = gain

    def gain(self) -> float:
        """Return the counts per unit of the data."""
        return self._gain

    def sigma(self, grid: gridutils.Grid, signal: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(signal, 0) * self._gain) / self._gain

    def draw(
            self,
            rng: np.random.Generator,
            grid: gridutils.Grid,
            psf: PSF | None,
            signal: np.ndarray
    ) -> np.ndarray:
        counts = rng.poisson(np.maximum(signal, 0) * self._gain)
        return counts / self._gain - signal


def _load_error(info: dict[str, Any]) -> dict[str, Any]:
    """Return the options of a noise with its error map read (if any)."""
    info = dict(info)
    file = parseutils.load_option(parseutils.parse_file, info, 'error')
    if file is not None:
        with parseutils.config_path('error'):
            info['error'] = fitsutils.read_data(*file)[0]
    return info


def _dump_dispersion(
        rms: float | None,
        error: np.ndarray | None,
        prefix: str,
        dump_path: bool,
        overwrite: bool
) -> dict[str, Any]:
    """
    Return the options of the dispersion of a noise: rms, or its error map
    written to a file.
    """
    if rms is not None:
        return dict(rms=rms)
    filename = f'{prefix}noise_error.fits'
    astropy.io.fits.writeto(filename, error, overwrite=overwrite)
    return dict(error=filename if dump_path else os.path.basename(filename))


def _check_dispersion(
        rms: float | None, error: np.ndarray | None) -> np.ndarray | None:
    """
    Return the error map (as an array, if any) of a noise of one of rms
    and error.

    Raises
    ------
    ConfigError
        If not exactly one is given, rms is not positive, or the error map
        has negative values.
    """
    if (rms is None) == (error is None):
        raise ConfigError("give one of rms and error")
    if rms is not None and not rms > 0:
        raise ConfigError(f"rms must be positive; it is {rms}")
    if error is not None:
        error = np.asarray(error, dtype=float)
        if np.any(error < 0):
            raise ConfigError("the error map has negative values")
    return error


def _dispersion(
        rms: float | None,
        error: np.ndarray | None,
        grid: gridutils.Grid
) -> np.ndarray:
    """
    Return the dispersion of each value of data on a grid, of rms or of
    the error map.

    Raises
    ------
    ConfigError
        If the error map does not broadcast to the data.
    """
    shape = grid.size[::-1]
    if rms is not None:
        return np.full(shape, rms)
    try:
        return np.broadcast_to(error, shape).copy()
    except ValueError:
        raise ConfigError(
            f"the error map, of shape {error.shape}, does not broadcast to "
            f"the data, of shape {shape}") from None


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
    by its value at the centre) exactly. The filter is the square root of
    the power spectrum of the nearest correlation on the periodic grid of
    the noise (see periodic_correlation).
    """
    shape = white.shape[-naxes:]
    axes = tuple(range(-naxes, 0))
    # (the power spectrum of a correlation is not negative, but rounding
    # can leave values a hair below 0)
    power = np.maximum(scipy.fft.rfftn(
        periodic_correlation(correlation, shape), axes=axes).real, 0)
    return scipy.fft.irfftn(
        scipy.fft.rfftn(white, axes=axes) * np.sqrt(power), shape, axes=axes)


noise_parser = parseutils.TypedParser(
    Noise, [NoiseGauss, NoiseStudentT, NoisePoisson])
