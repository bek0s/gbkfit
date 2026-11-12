"""
The noise of mock observations: what they add to the data a model gives
where the instrument records them (see draw_mocks).
"""

import abc
import os
from typing import Annotated, Any, Literal, Self, TypeAlias

import astropy.units as u
import numpy as np
import numpy.typing as npt
import scipy.fft
from annotated_types import Ge, Gt

from gbkfit.config import (
    Choices, DataFile, TypedParser, TypedSerializable, check_options,
    split_data_file)
from gbkfit.grid import (
    Grid, VelocityConvention, frame_to_pixels, sky_positions)
from gbkfit.instrument import LSF, PSF, Channel, lsf_parser, psf_parser
from gbkfit.io import (
    has_wcs, parse_wcs, read_data, read_header, write_data, write_image)
from gbkfit.math.correlation import periodic_correlation


__all__ = [
    'Noise',
    'NoiseGauss',
    'NoiseOfDispersion',
    'NoisePoisson',
    'NoiseStudentT',
    'noise_parser'
]


# The spatial correlation of the noise that the PSF of the instrument
# gives, and its option
INSTRUMENT = 'instrument'
_Instrument: TypeAlias = Annotated[
    Literal[INSTRUMENT], Choices({INSTRUMENT: 'PSF of the instrument'})]

# How far (in pixels) the world coordinates of an error map may put its
# pixels from those of the data it is drawn on
_GRID_TOLERANCE = 1e-3


class Noise(TypedSerializable, abc.ABC):
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
        data of it declare (see draw_mocks); none for noise
        independent from value to value.
        """
        return None, None

    @abc.abstractmethod
    def sigma(self, grid: Grid, signal: np.ndarray) -> np.ndarray:
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
        ValueError
            If the noise does not suit data of that shape.
        """

    @abc.abstractmethod
    def draw(
            self,
            rng: np.random.Generator,
            grid: Grid,
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
        ValueError
            If the noise does not suit the data or the instrument.
        """


class NoiseOfDispersion(Noise):
    """
    Noise of a given dispersion: rms at every value, or an error map of
    the dispersion of each value. Its kinds draw it from their
    distributions (e.g. Gaussian, Student-t).

    An error map read from a file (see from_file of its kinds) keeps the
    world coordinates of its file, if they place it on the sky, and is
    drawn only on data whose pixels are its own there (see sigma); an
    error map given as an array, on any data that it broadcasts to.

    Parameters
    ----------
    rms : float, optional
        The dispersion of every value.
    error : array_like, optional
        The dispersion of each value: an array that broadcasts to the
        shape of the data (e.g. one value per pixel, ny by nx, for all the
        channels of a cube); values whose error is not finite are
        missing (NaN).

    Raises
    ------
    ValueError
        If not exactly one of rms and error is given, or the error map has
        negative values.
    """

    def __init__(
            self,
            rms: float | None = None,
            error: npt.ArrayLike | None = None
    ):
        if (rms is None) == (error is None):
            raise ValueError("give one of rms and error")
        if error is not None:
            error = np.asarray(error, dtype=float)
            if np.any(error < 0):
                raise ValueError("the error map has negative values")
        self._rms = rms
        self._error = error
        # (of an error map read from a file; see _of_error_file)
        self._error_grid: Grid | None = None

    def rms(self) -> float | None:
        """Return the dispersion of every value, if it has one."""
        return self._rms

    def error(self) -> np.ndarray | None:
        """Return the error map, if it has one."""
        return self._error

    def sigma(self, grid: Grid, signal: np.ndarray) -> np.ndarray:
        """
        Return the dispersion of each value of data on a grid, of rms or of
        the error map (see Noise.sigma); raise ValueError if the error map
        does not broadcast to the data, or the world coordinates of its
        file do not put its pixels on those of the data (see the class).
        """
        shape = grid.size[::-1]
        if self._rms is not None:
            return np.full(shape, self._rms)
        try:
            sigma = np.broadcast_to(self._error, shape).copy()
        except ValueError:
            raise ValueError(
                f"the error map, of shape {self._error.shape}, does not "
                f"broadcast to the data, of shape {shape}") from None
        if self._error_grid is not None:
            _check_same_sky(self._error_grid, grid)
        return sigma

    def _dump_dispersion(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        """
        Return the options of its dispersion: rms, or its error map
        written to a file.
        """
        if self._rms is not None:
            return dict(rms=self._rms)
        filename = f'{prefix}noise_error.fits'
        grid = self._error_grid
        if grid is None:
            write_image(filename, self._error, overwrite=overwrite)
        else:
            # (with the world coordinates of its file, which loading it
            # checks again)
            write_data(
                filename, self._error, grid.coords, grid.spectral_axis,
                overwrite=overwrite)
        return dict(
            error=filename if dump_path else os.path.basename(filename))

    @classmethod
    def _of_error_file(
            cls, file: DataFile, *args: Any, **options: Any
    ) -> Self:
        """
        Return a noise of the error map of a file (see _read_error) and of
        the other options given (see from_file of its kinds), which keeps
        the grid of the map on the sky, if its file gives one (see the
        class).
        """
        error, grid = _read_error(file)
        noise = cls(*args, error=error, **options)
        noise._error_grid = grid
        return noise


class NoiseGauss(NoiseOfDispersion):
    """
    Gaussian noise of mean 0, of the dispersion rms at every value or of
    an error map. It is independent from value to value, unless
    correlated on the sky or along the spectral axis.

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

    Configurations give the error map by its FITS file (see from_file).

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
        The correlation along the spectral axis (see Data), drawn at the
        middle channel of the data (see LSF.on); it must not vary.

    Raises
    ------
    ValueError
        If not exactly one of rms and error is given, the error map has
        negative values, or the spectral correlation varies.
    """

    config_constructors = ('__init__', 'from_file')

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict(type=self.type()) | self._dump_dispersion(
            prefix, dump_path, overwrite)
        kwargs = dict(prefix=prefix, dump_path=dump_path, overwrite=overwrite)
        spatial = self._spatial_correlation
        if spatial is not None:
            info.update(spatial_correlation=spatial if spatial == INSTRUMENT
                        else psf_parser.dump(spatial, **kwargs))
        if self._spectral_correlation is not None:
            info.update(spectral_correlation=lsf_parser.dump(
                self._spectral_correlation, **kwargs))
        return info

    @classmethod
    def from_file(
            cls,
            error: DataFile,
            spatial_correlation: PSF | _Instrument | None = None,
            spectral_correlation: LSF | None = None
    ) -> 'NoiseGauss':
        """
        Make a noise of an error map read from a FITS file, in the
        orientation of the model (see read_data): by its world
        coordinates, or without them as the file has it, which must then
        be that of the model (e.g. as gbkfit.prep writes data). World
        coordinates of the sky must put its pixels on those of the data
        it is drawn on (see NoiseOfDispersion).

        Parameters
        ----------
        error : str or tuple
            The file of the error map: a filename, or a filename and the
            HDU.
        spatial_correlation, spectral_correlation : optional
            As those of NoiseGauss.

        Returns
        -------
        NoiseGauss
            The noise.
        """
        check_options(NoiseGauss.from_file, locals())
        return cls._of_error_file(
            error, spatial_correlation=spatial_correlation,
            spectral_correlation=spectral_correlation)

    def __init__(
            self,
            rms: Annotated[float, Gt(0)] | None = None,
            error: npt.ArrayLike | None = None,
            spatial_correlation: PSF | _Instrument | None = None,
            spectral_correlation: LSF | None = None
    ):
        check_options(NoiseGauss.__init__, locals())
        super().__init__(rms, error)
        if spectral_correlation is not None and spectral_correlation.varies():
            raise ValueError("the spectral correlation must not vary")
        self._spatial_correlation = spatial_correlation
        self._spectral_correlation = spectral_correlation

    def spatial_correlation(self) -> PSF | str | None:
        """Return the correlation on the sky, if any."""
        return self._spatial_correlation

    def spectral_correlation(self) -> LSF | None:
        """Return the correlation along the spectral axis, if any."""
        return self._spectral_correlation

    def correlations(self) -> tuple[PSF | str | None, LSF | None]:
        return self._spatial_correlation, self._spectral_correlation

    def draw(
            self,
            rng: np.random.Generator,
            grid: Grid,
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
            self, grid: Grid, psf: PSF | None
    ) -> np.ndarray | None:
        """
        Return the spatial correlation on the pixels of a grid: one image
        for all the channels, or one for each if it varies (nz, ny, nx).
        """
        spatial = self._spatial_correlation
        sky_axes = len(grid.size) - (grid.spectral_axis is not None)
        if spatial is not None and sky_axes < 2:
            raise ValueError(
                "the noise has a spatial correlation, but the data are not "
                "on the sky (e.g. visibilities)")
        if spatial == INSTRUMENT:
            if psf is None:
                raise ValueError(
                    f"the spatial correlation of the noise is "
                    f"'{INSTRUMENT}', but the instrument has no PSF")
            spatial = psf
        if spatial is None:
            return None
        step, rota = grid.coords.step[:2], grid.coords.rota
        if not spatial.varies():
            return spatial.asarray(step, rota=rota)
        if grid.spectral_axis is None:
            raise ValueError(
                "the spatial correlation of the noise varies along the "
                "spectral axis, but the data have none")
        velocities = _velocities(grid)
        convention = VelocityConvention.of(grid.coords)
        spatial.check_covers(
            velocities, convention, "the spatial correlation of the noise")
        psfs = spatial.at(velocities * u.km / u.s, convention)
        size = tuple(np.max([p.size(step) for p in psfs], axis=0).tolist())
        return np.stack([p.asarray(step, size, rota=rota) for p in psfs])

    def _spectral_kernel(self, grid: Grid) -> np.ndarray | None:
        """Return the spectral correlation on the channels of a grid."""
        spectral = self._spectral_correlation
        if spectral is None:
            return None
        if grid.spectral_axis is None:
            raise ValueError(
                "the noise has a spectral correlation, but the data have no "
                "spectral axis")
        channel = Channel.middle(grid)
        try:
            return spectral.asarray(channel.width, channel)
        except RuntimeError as e:
            raise RuntimeError(
                f"the spectral correlation of the noise is drawn on the "
                f"channels of the data, as they are: {e}") from None


class NoiseStudentT(NoiseOfDispersion):
    """
    Student-t noise: heavier tails than Gaussian noise (outliers), e.g.
    to test how fits take them. It is of mean 0 and of dof degrees of
    freedom (above 2), of the dispersion rms at every value or of an
    error map (as NoiseGauss), independent from value to value; Gaussian
    as dof grows. Its values are t draws times sigma sqrt((dof - 2) /
    dof), whose dispersion is sigma.

    Configurations give the error map by its FITS file (see from_file).

    Parameters
    ----------
    dof : float
        The degrees of freedom.
    rms : float, optional
        The dispersion of every value.
    error : ndarray, optional
        The dispersion of each value (see NoiseOfDispersion).

    Raises
    ------
    ValueError
        If not exactly one of rms and error is given, or the error map has
        negative values.
    """

    config_constructors = ('__init__', 'from_file')

    @staticmethod
    def type() -> str:
        return 'student_t'

    @staticmethod
    def type_title() -> str:
        return 'Student-t'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type(), dof=self._dof) \
            | self._dump_dispersion(prefix, dump_path, overwrite)

    @classmethod
    def from_file(
            cls,
            dof: Annotated[float, Gt(2)],
            error: DataFile
    ) -> 'NoiseStudentT':
        """
        Make a noise of an error map read from a FITS file (as
        NoiseGauss.from_file).

        Parameters
        ----------
        dof : float
            The degrees of freedom.
        error : str or tuple
            The file of the error map: a filename, or a filename and the
            HDU.

        Returns
        -------
        NoiseStudentT
            The noise.
        """
        check_options(NoiseStudentT.from_file, locals())
        return cls._of_error_file(error, dof)

    def __init__(
            self,
            dof: Annotated[float, Gt(2)],
            rms: Annotated[float, Gt(0)] | None = None,
            error: npt.ArrayLike | None = None
    ):
        check_options(NoiseStudentT.__init__, locals())
        super().__init__(rms, error)
        self._dof = dof

    def dof(self) -> float:
        """Return the degrees of freedom."""
        return self._dof

    def draw(
            self,
            rng: np.random.Generator,
            grid: Grid,
            psf: PSF | None,
            signal: np.ndarray
    ) -> np.ndarray:
        sigma = self.sigma(grid, signal)
        t = rng.standard_t(self._dof, sigma.shape)
        return t * sigma * np.sqrt((self._dof - 2) / self._dof)


class NoisePoisson(Noise):
    """
    Poisson noise of data of counts, independent from value to value, of
    the signal (the data without noise) over a background: the light that
    was taken out of the data (e.g. that of the sky), whose counts are
    noisy too.

    The data become counts / gain - background, the counts Poisson draws
    of (signal + background) gain, of the dispersion sqrt((signal +
    background) gain) / gain (a signal below -background, e.g. of the
    ringing of a convolution, counts as none). Without a background, the
    data without signal have no noise, and so errors of 0, which leave
    them out of fits (see Data).

    Parameters
    ----------
    gain : float, optional
        The counts per unit of the data (e.g. of the photons per unit of
        flux).
    background : float, optional
        The level of the background, in the unit of the data; 0 by
        default.
    """

    @staticmethod
    def type() -> str:
        return 'poisson'

    @staticmethod
    def type_title() -> str:
        return 'Poisson'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict(type=self.type(), gain=self._gain)
        if self._background:
            info['background'] = self._background
        return info

    def __init__(
            self,
            gain: Annotated[float, Gt(0)] = 1.0,
            background: Annotated[float, Ge(0)] = 0.0
    ):
        check_options(NoisePoisson.__init__, locals())
        self._gain = gain
        self._background = background

    def gain(self) -> float:
        """Return the counts per unit of the data."""
        return self._gain

    def background(self) -> float:
        """Return the level of the background, in the unit of the data."""
        return self._background

    def sigma(self, grid: Grid, signal: np.ndarray) -> np.ndarray:
        return np.sqrt(self._expected_counts(signal)) / self._gain

    def draw(
            self,
            rng: np.random.Generator,
            grid: Grid,
            psf: PSF | None,
            signal: np.ndarray
    ) -> np.ndarray:
        counts = rng.poisson(self._expected_counts(signal))
        return counts / self._gain - self._background - signal

    def _expected_counts(self, signal: np.ndarray) -> np.ndarray:
        """
        Return the expected counts of each value of data without noise:
        those of the signal and the background, none below 0.
        """
        return np.maximum(signal + self._background, 0) * self._gain


def _read_error(file: DataFile) -> tuple[np.ndarray, Grid | None]:
    """
    Return the error map of a noise, read from its FITS file in the
    orientation of the model (see read_data), and its grid if the world
    coordinates of the file place it on the sky (of celestial axes), or
    else None.
    """
    error, coords = read_data(file)
    header = read_header(file)
    if error.ndim < 2 or not has_wcs(header):
        return error, None
    wcs = parse_wcs(split_data_file(file)[0], header)
    if not wcs.has_celestial:
        return error, None
    # (the orientation of the model puts a spectral axis last, and drops
    # it if it is of one pixel)
    spectral_axis = 2 if error.ndim == 3 and wcs.wcs.spec >= 0 else None
    return error, Grid(error.shape[::-1], coords, spectral_axis)


def _check_same_sky(error_grid: Grid, grid: Grid) -> None:
    """
    Raise ValueError unless the grid of an error map on the sky (see
    _read_error) puts its pixels on those of the data of a grid, by their
    world coordinates (the step, rpix, rval and rota of x and y): the
    first and the last pixels of the data at those of the map, within
    _GRID_TOLERANCE.
    """
    if len(grid.size) - (grid.spectral_axis is not None) < 2:
        raise ValueError(
            "the error map is an image of the sky, by the world "
            "coordinates of its file, but the data are not on the sky "
            "(e.g. visibilities)")
    sky = grid.spatial()
    x, y = sky_positions(sky)
    ends = np.ix_([0, -1], [0, -1])
    pixel_x, pixel_y = frame_to_pixels(
        x[ends], y[ends], sky.coords.rval, error_grid.spatial())
    size_x, size_y = sky.size
    expected = np.meshgrid([0, size_x - 1], [0, size_y - 1])
    if not np.allclose(
            [pixel_x, pixel_y], expected, rtol=0, atol=_GRID_TOLERANCE):
        first = (float(pixel_x[0, 0]), float(pixel_y[0, 0]))
        last = (float(pixel_x[-1, -1]), float(pixel_y[-1, -1]))
        raise ValueError(
            f"the error map is not on the pixels of the data: by their world "
            f"coordinates, the first and the last pixels of the data are at "
            f"the pixels ({first[0]:.6g}, {first[1]:.6g}) and "
            f"({last[0]:.6g}, {last[1]:.6g}) of the map, not (0, 0) and "
            f"({size_x - 1}, {size_y - 1}); check the world coordinates "
            f"(step, rpix, rval and rota) of the file of the map against "
            f"those of the data")


def _velocities(grid: Grid) -> np.ndarray:
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


noise_parser = TypedParser(
    Noise, [NoiseGauss, NoiseStudentT, NoisePoisson], family='noise')
