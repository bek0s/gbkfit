"""
Line spread functions (LSFs): how the instrument spreads the light of a
single velocity along the spectral axis.

Widths are in km/s of the light: c dlambda / lambda at the wavelength of
the light (c / R for a resolving power R), as instruments give their
resolution. A spectral axis whose velocities refer to a rest far from
the light (e.g. the rest wavelength of a line of a galaxy at redshift z)
spans other km/s of its own per km/s of the light (1 + z on an axis of
optical velocities; see VelocityConvention.stretch), and the
LSF is stretched to it (see LSF.stretched): once, at the middle of the
data, or at each channel if it varies; the Hanning smoothing, of the
channels of the axis, is not (see LSFHanning). The options of the
analytic LSFs can vary along the spectral axis (see varying): an LSF
whose width changes across the data (e.g. of a constant resolving power
over a wide band) is one that varies.
"""

import abc
import os
from collections.abc import Sequence
from typing import Annotated, Any

import astropy.units as u
import astropy.wcs
import numpy as np
import scipy.ndimage
import scipy.special
from annotated_types import Gt

from gbkfit.config import (
    DataFile, TypedParser, Unit, Varying, check_options, split_data_file)
from gbkfit.grid import Coords, VelocityConvention
from gbkfit.io import parse_wcs, read_image, write_data, write_image
import gbkfit.math
from . import _detail, varying
from ._detail import MIN_EXTENT, WING_FLUX


__all__ = [
    'LSF',
    'LSFConvolution',
    'LSFGGauss',
    'LSFGauss',
    'LSFHanning',
    'LSFImage',
    'LSFImages',
    'LSFLorentz',
    'LSFMoffat',
    'LSFPoint',
    'LSFSum',
    'LSFVarying',
    'lsf_parser'
]


# The unit of the widths of the LSFs
_KMS = u.km / u.s


class LSF(_detail.SpreadFunction):
    """
    A line spread function, drawn on arrays of channels.
    """

    @staticmethod
    def _varying_class() -> type['LSFVarying']:
        return LSFVarying

    @abc.abstractmethod
    def stretched(self, factor: float) -> 'LSF':
        """
        Return the LSF on a spectral axis that spans factor km/s per km/s
        of the light (see VelocityConvention.stretch): its
        widths times factor.

        Parameters
        ----------
        factor : float
            The km/s of the axis per km/s of the light.

        Returns
        -------
        LSF
            The LSF on the axis.
        """

    def size(self, step: float) -> int:
        """
        Return the size of the array that holds the LSF.

        Parameters
        ----------
        step : float
            The width of the channels (km/s).

        Returns
        -------
        int
            The number of channels, odd.

        Raises
        ------
        RuntimeError
            If the wings of the LSF are too heavy for a finite array.
        """
        size = self._size_impl(step)
        _detail.check_finite_size(self, size)
        return int(gbkfit.math.roundu_odd(size))

    def asarray(
            self,
            step: float,
            size: int | None = None,
            offset: int = 0
    ) -> np.ndarray:
        """
        Return the LSF on an array of channels, normalised to sum to 1.

        Parameters
        ----------
        step : float
            The width of the channels (km/s).
        size : int, optional
            The number of channels; by default, that of size.
        offset : int, optional
            The offset of the centre of the LSF from the channel
            size // 2. size + offset must be odd.

        Returns
        -------
        ndarray
            The LSF.

        Raises
        ------
        RuntimeError
            If size + offset is even.
        """
        if size is None:
            size = self.size(step)
        if gbkfit.math.is_even(size + offset):
            raise RuntimeError(
                f"invalid LSF size: (size + offset) = "
                f"({size} + {offset} = {size + offset}), "
                f"but it must be odd")
        return self._asarray_impl(step, size, offset)

    @abc.abstractmethod
    def _size_impl(self, step: float) -> float:
        """The size of the array that holds the LSF, before rounding."""

    @abc.abstractmethod
    def _asarray_impl(
            self,
            step: float,
            size: int,
            offset: int
    ) -> np.ndarray:
        """The LSF on an array (see asarray)."""


def _create_grid_1d(size: int, step: float, offset: int) -> np.ndarray:
    """The velocities of the channels of an array from its centre."""
    center = size // 2 + offset
    return (np.array(range(size)) - center) * step


class _AnalyticLSF(LSF):
    """
    An LSF of a symmetric profile of the velocity, drawn out to its
    extent on each side (see MIN_EXTENT).
    """

    @abc.abstractmethod
    def _extent(self) -> float:
        """The half-width out to which it is drawn (see MIN_EXTENT)."""

    @abc.abstractmethod
    def _profile(self, v: np.ndarray) -> np.ndarray:
        """Its profile at the velocities v (km/s), to any factor."""

    def _size_impl(self, step: float) -> float:
        return 2 * self._extent() / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = self._profile(z)
        data[np.abs(z) > self._extent()] = 0
        return data / np.sum(data)


class LSFPoint(LSF):
    """
    No spreading: the light stays in its channel.
    """

    @staticmethod
    def type() -> str:
        return 'point'

    @staticmethod
    def type_title() -> str:
        return 'None (a point)'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type())

    def stretched(self, factor: float) -> 'LSFPoint':
        return self

    def _size_impl(self, step: float) -> float:
        return 1

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        data = np.zeros(size)
        data[size // 2 + offset] = 1
        return data


class LSFGauss(_AnalyticLSF):
    """
    A Gaussian, by its sigma.

    Its profile is exp(-v^2 / 2 sigma^2).

    Parameters
    ----------
    sigma : float
        The standard deviation (km/s of the light).
    """

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
        return dict(
            type=self.type(),
            sigma=self._sigma)

    def __init__(
            self, sigma: Annotated[float, Gt(0), Unit('km/s'), Varying()]
    ):
        check_options(LSFGauss.__init__, locals())
        self._sigma = sigma

    def stretched(self, factor: float) -> 'LSFGauss':
        return LSFGauss(self._sigma * factor)

    def _extent(self) -> float:
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a Gaussian within |z| < w is erf(w / sqrt(2) sigma).
        """
        return max(
            MIN_EXTENT * self._sigma,
            self._sigma * np.sqrt(2) * scipy.special.erfinv(1 - WING_FLUX))

    def _profile(self, v: np.ndarray) -> np.ndarray:
        return gbkfit.math.gauss_1d_fun(v, 1, 0, self._sigma)


class LSFGGauss(_AnalyticLSF):
    """
    A Gaussian with a shape parameter beta: flat-topped above 2, peaked
    below 2, exponential at 1.

    Its profile is exp(-(|v| / alpha)^beta).

    Parameters
    ----------
    alpha : float
        The scale (km/s of the light).
    beta : float
        The shape, > 0: 2 is a Gaussian, and 1 an exponential.
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def type_title() -> str:
        return 'Generalised Gaussian'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta)

    def __init__(
            self,
            alpha: Annotated[float, Gt(0), Unit('km/s'), Varying()],
            beta: Annotated[float, Gt(0), Varying()]
    ):
        check_options(LSFGGauss.__init__, locals())
        self._alpha = alpha
        self._beta = beta

    def stretched(self, factor: float) -> 'LSFGGauss':
        return LSFGGauss(self._alpha * factor, self._beta)

    def _extent(self) -> float:
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a ggauss within |z| < w is P(1 / beta, (w / alpha)^beta),
        P the regularized lower incomplete gamma function.
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * scipy.special.gammaincinv(
                1 / self._beta, 1 - WING_FLUX) ** (1 / self._beta))

    def _profile(self, v: np.ndarray) -> np.ndarray:
        return gbkfit.math.ggauss_1d_fun(v, 1, 0, self._alpha, self._beta)


class LSFLorentz(_AnalyticLSF):
    """
    Broad wings, as of a Fabry-Perot interferometer (whose profile about
    each of its peaks is nearly a Lorentzian).

    Its profile is gamma^2 / (v^2 + gamma^2).

    Parameters
    ----------
    gamma : float
        The scale (km/s of the light): the half width at half maximum.
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def type_title() -> str:
        return 'Lorentzian'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            gamma=self._gamma)

    def __init__(
            self, gamma: Annotated[float, Gt(0), Unit('km/s'), Varying()]
    ):
        check_options(LSFLorentz.__init__, locals())
        self._gamma = gamma

    def stretched(self, factor: float) -> 'LSFLorentz':
        return LSFLorentz(self._gamma * factor)

    def _extent(self) -> float:
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a Lorentzian within |z| < w is 2 / pi arctan(w / gamma).
        """
        return max(
            MIN_EXTENT * self._gamma,
            self._gamma * np.tan(np.pi / 2 * (1 - WING_FLUX)))

    def _profile(self, v: np.ndarray) -> np.ndarray:
        return gbkfit.math.lorentz_1d_fun(v, 1, 0, self._gamma)


class LSFMoffat(_AnalyticLSF):
    """
    A core with power-law wings.

    Its profile is (1 + (v / alpha)^2)^(-beta).

    Parameters
    ----------
    alpha : float
        The scale of the core (km/s of the light).
    beta : float
        The slope of the wings, > 0.5 (for a finite flux).
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def type_title() -> str:
        return 'Moffat'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta)

    def __init__(
            self,
            alpha: Annotated[float, Gt(0), Unit('km/s'), Varying()],
            beta: Annotated[float, Gt(0.5), Varying()]
    ):
        check_options(LSFMoffat.__init__, locals())
        self._alpha = alpha
        self._beta = beta

    def stretched(self, factor: float) -> 'LSFMoffat':
        return LSFMoffat(self._alpha * factor, self._beta)

    def _extent(self) -> float:
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a Moffat within |z| < w is I_u(1/2, beta - 1/2), I the
        regularized incomplete beta function and u = w^2 / (alpha^2 + w^2).
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * np.sqrt(1 / (1 - scipy.special.betaincinv(
                0.5, self._beta - 0.5, 1 - WING_FLUX)) - 1))

    def _profile(self, v: np.ndarray) -> np.ndarray:
        return gbkfit.math.moffat_1d_fun(v, 1, 0, self._alpha, self._beta)


class LSFImage(LSF):
    """
    A measured LSF, from a 1D image.

    It is centred on the centre of the image, and resampled to the
    channels it is drawn on (with splines of order 5).

    Configurations give its file (see from_file).

    Parameters
    ----------
    data : ndarray
        The image, finite.
    step : float, optional
        The width of the channels of the image (km/s of the light).
    """

    config_constructors = ('from_file',)

    @staticmethod
    def type() -> str:
        return 'image'

    @staticmethod
    def type_title() -> str:
        return 'Image (measured)'

    @classmethod
    def from_file(
            cls,
            file: DataFile,
            step: Annotated[float, Gt(0), Unit('km/s')] | None = None
    ) -> 'LSFImage':
        """
        Read an LSF image (a profile) from a file, with its axis as it
        is: the profile is reversed where the axis runs against the
        velocities of the light (velocities or wavelengths that decrease
        along it, or frequencies that increase).

        Parameters
        ----------
        file : DataFile
            The file of the image, and where it is in it (see read_image).
        step : float, optional
            The width of the channels of the image (km/s of the light); by
            default, that of its axis, if it is in a velocity unit.

        Returns
        -------
        LSFImage
            The LSF.

        Raises
        ------
        ValueError
            If the file has no 1D image, its header has invalid world
            coordinates, or step is not given and the axis is not in a
            velocity unit.
        """
        check_options(LSFImage.from_file, locals())
        filename = split_data_file(file)[0]
        data, wcs = _read_profiles(file)
        if data.ndim != 1:
            raise ValueError(
                f"{filename}: an LSF profile is a 1D image; it has "
                f"{data.ndim} axes")
        return cls(data, _channel_width(filename, wcs) if step is None
                   else step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}lsf.fits'
        rpix = self._data.size / 2 - 0.5
        coords = Coords((self._step,), (rpix,), (0.0,), 0.0)
        write_data(filename, self._data, coords, 0, overwrite=overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            step=self._step)

    def __init__(
            self,
            data: np.ndarray,
            step: Annotated[float, Gt(0), Unit('km/s')] = 1.0
    ):
        check_options(LSFImage.__init__, locals())
        data = np.squeeze(data)  # Remove singleton dimensions
        if data.ndim != 1:
            raise ValueError(
                f"expected a 1D LSF image, but got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError(
                "non-finite pixels found in the supplied LSF image")
        self._data = data
        self._step = step

    def stretched(self, factor: float) -> 'LSFImage':
        return LSFImage(self._data, self._step * factor)

    def _size_impl(self, step: float) -> float:
        return (self._step / step) * self._data.shape[0]

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        scale = step / self._step
        # The centre of the image, and the centre of the LSF in the array,
        # where the analytic LSFs put it
        old_center = self._data.shape[0] / 2 - 0.5
        new_center = size // 2 + offset
        x = np.arange(size)
        nx = (x - new_center) * scale + old_center
        data = scipy.ndimage.map_coordinates(
            self._data, [nx], order=5, mode='grid-constant')
        return data / np.sum(data)


class LSFSum(LSF):
    """
    A sum of LSFs (e.g. a double Gaussian), each with its fraction of the
    light.

    Parameters
    ----------
    lsfs : Sequence of LSF
        The LSFs.
    weights : Sequence of float
        The weight of each LSF, positive; they are normalised to sum to 1.
    """

    @staticmethod
    def type() -> str:
        return 'sum'

    @staticmethod
    def type_title() -> str:
        return 'Sum'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            lsfs=_detail.dump_terms(
                lsf_parser, self._lsfs, prefix, dump_path, overwrite),
            weights=self._weights.tolist())

    def __init__(self, lsfs: Sequence[LSF], weights: Sequence[float]):
        check_options(LSFSum.__init__, locals())
        self._lsfs = tuple(lsfs)
        self._weights = _detail.sum_weights(
            weights, len(self._lsfs), "a sum of LSFs")

    def stretched(self, factor: float) -> 'LSFSum':
        return LSFSum(
            [lsf.stretched(factor) for lsf in self._lsfs], self._weights)

    def varies(self) -> bool:
        return any(lsf.varies() for lsf in self._lsfs)

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> 'LSF | list[LSF]':
        if not self.varies():
            return super().at(points, convention)
        return varying.per_point(points, lambda points_: [
            LSFSum(lsfs, self._weights) for lsfs in
            _detail.terms_at(self._lsfs, points_, convention)])

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        return _detail.terms_velocity_range(self._lsfs, convention)

    def _size_impl(self, step: float) -> float:
        return max(lsf._size_impl(step) for lsf in self._lsfs)

    def _asarray_impl(self, step: float, size: int, offset: int) -> np.ndarray:
        # Each term holds its fraction of the light on the same grid
        return sum(
            weight * lsf.asarray(step, size, offset)
            for weight, lsf in zip(self._weights, self._lsfs))


class LSFConvolution(LSF):
    """
    The convolution of LSFs: e.g. the LSF of an instrument and the
    Hanning smoothing of the channels of the data (hanning).

    Parameters
    ----------
    lsfs : Sequence of LSF
        The LSFs, at least one.
    """

    @staticmethod
    def type() -> str:
        return 'convolution'

    @staticmethod
    def type_title() -> str:
        return 'Convolution'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            lsfs=_detail.dump_terms(
                lsf_parser, self._lsfs, prefix, dump_path, overwrite))

    def __init__(self, lsfs: Sequence[LSF]):
        check_options(LSFConvolution.__init__, locals())
        if not lsfs:
            raise ValueError("a convolution of LSFs needs at least one LSF")
        self._lsfs = tuple(lsfs)

    def stretched(self, factor: float) -> 'LSFConvolution':
        return LSFConvolution([lsf.stretched(factor) for lsf in self._lsfs])

    def varies(self) -> bool:
        return any(lsf.varies() for lsf in self._lsfs)

    def at(
            self,
            points: u.Quantity,
            convention: VelocityConvention | None = None
    ) -> 'LSF | list[LSF]':
        if not self.varies():
            return super().at(points, convention)
        return varying.per_point(points, lambda points_: [
            LSFConvolution(lsfs) for lsfs in
            _detail.terms_at(self._lsfs, points_, convention)])

    def velocity_range(
            self, convention: VelocityConvention | None = None
    ) -> tuple[float, float]:
        return _detail.terms_velocity_range(self._lsfs, convention)

    def _size_impl(self, step: float) -> float:
        # The sum of the (odd) sizes of the terms, less one for each
        # convolution
        sizes = [lsf.size(step) for lsf in self._lsfs]
        return sum(sizes) - len(sizes) + 1

    def _asarray_impl(self, step: float, size: int, offset: int) -> np.ndarray:
        data = np.ones(1)
        for lsf in self._lsfs:
            data = np.convolve(data, lsf.asarray(step))
        return _detail.embed(data / data.sum(), (size,), (offset,))


class LSFHanning(LSF):
    """
    The smoothing of channels by a Hanning window, as of radio
    spectrometers.

    1/4, 1/2 and 1/4 of the light of each channel go to the channel
    before, the channel itself and the channel after.

    Parameters
    ----------
    width : float
        The width of the smoothed channels (km/s of the spectral axis, not
        of the light: it is not stretched); a multiple of the width of the
        channels it is drawn on (e.g. those of the data, with any
        oversampling).
    """

    @staticmethod
    def type() -> str:
        return 'hanning'

    @staticmethod
    def type_title() -> str:
        return 'Hanning smoothing'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type(), width=self._width)

    def __init__(self, width: Annotated[float, Gt(0), Unit('km/s')]):
        check_options(LSFHanning.__init__, locals())
        self._width = width

    def stretched(self, factor: float) -> 'LSFHanning':
        # (the channels of the axis are its own)
        return self

    def _steps(self, step: float) -> int:
        """The number of steps in a channel width."""
        steps = self._width / step
        if abs(steps - round(steps)) > 1e-6 * steps:
            raise RuntimeError(
                f"the channel width of the Hanning smoothing ({self._width}) "
                f"must be a multiple of the step it is drawn with ({step})")
        return round(steps)

    def _size_impl(self, step: float) -> float:
        return 2 * self._steps(step) + 1

    def _asarray_impl(self, step: float, size: int, offset: int) -> np.ndarray:
        steps = self._steps(step)
        data = np.zeros(2 * steps + 1)
        data[[0, steps, 2 * steps]] = 0.25, 0.5, 0.25
        return _detail.embed(data, (size,), (offset,))


def _read_profiles(file: DataFile) -> tuple[np.ndarray, astropy.wcs.WCS]:
    """
    The LSF profile, or profiles (one per row of an image: nz, nk), of a
    file, along its first axis, and its world coordinates (its axes as
    they are, unlike those of data: the axis of the profiles may be
    spectral, and so that of the rows). The profiles run along the
    velocities of the light: they are reversed where their axis runs
    against them (velocities or wavelengths that decrease along it, or
    frequencies that increase).
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    data = np.asarray(data, dtype=float)
    wcs = parse_wcs(filename, header)
    unit = u.Unit(wcs.wcs.cunit[0])
    scale = wcs.pixel_scale_matrix[0, 0]
    if unit.is_equivalent(_KMS) or unit.is_equivalent(u.m):
        against = scale < 0
    else:
        # (frequencies, energies and wave numbers fall as velocities rise)
        against = scale > 0 and unit.is_equivalent(u.Hz, u.spectral())
    return (data[..., ::-1] if against else data), wcs


def _channel_width(source: str, wcs: astropy.wcs.WCS) -> float:
    """
    The width of the channels of LSF profiles (km/s), from the first axis
    of their world coordinates (see _read_profiles).

    Raise ValueError if it is not in a velocity unit.
    """
    unit = u.Unit(wcs.wcs.cunit[0])
    if not unit.is_equivalent(_KMS):
        raise ValueError(
            f"{source}: the axis of the profiles is not in a velocity unit; "
            f"give the width of their channels (step, km/s)")
    return abs(float(wcs.pixel_scale_matrix[0, 0])) * unit.to(_KMS)


class LSFImages(_detail.SpreadFunctionImages, LSF):
    """
    A measured LSF that changes with wavelength, from an image.

    It is given as profiles at points of the spectral axis (e.g. measured
    from sky lines or arc lamps). At each channel it is the image LSF (see
    LSFImage) of the profiles of the two nearest points, interpolated
    linearly; beyond the points, that of the nearest profile.

    Configurations give its file, an image with one profile per row (see
    from_file).

    Parameters
    ----------
    data : ndarray
        The profiles (nz, nk), finite.
    points : Quantity
        The point of each profile: wavelengths, frequencies or velocities.
    step : float
        The width of the channels of the profiles (km/s of the light).
    """

    config_constructors = ('from_file',)

    @staticmethod
    def type() -> str:
        return 'images'

    @staticmethod
    def type_title() -> str:
        return 'Image per wavelength (measured)'

    @classmethod
    def from_file(
            cls,
            file: DataFile,
            points: u.Quantity | None = None,
            step: Annotated[float, Gt(0), Unit('km/s')] | None = None
    ) -> 'LSFImages':
        """
        Read LSF profiles from an image, one per row (nz, nk), with its
        axes as they are: the profiles are reversed where the axis of the
        columns runs against the velocities of the light (see
        LSFImage.from_file).

        Parameters
        ----------
        file : DataFile
            The file of the image, and where it is in it (see read_image).
        points : Quantity, optional
            The point of each profile: wavelengths, frequencies or
            velocities; by default, those of the axis of the rows, if it
            is spectral.
        step : float, optional
            The width of the channels of the profiles (km/s of the light);
            by default, that of the axis of the columns, if it is in a
            velocity unit.

        Returns
        -------
        LSFImages
            The LSF.

        Raises
        ------
        ValueError
            If the file has no 2D image, its header has invalid world
            coordinates, or the points or step are neither given nor in
            its header.
        """
        check_options(LSFImages.from_file, locals())
        filename = split_data_file(file)[0]
        data, wcs = _read_profiles(file)
        if data.ndim != 2:
            raise ValueError(
                f"{filename}: LSF profiles must be an image, with one profile "
                f"per row; it has {data.ndim} axes")
        if points is None:
            points = _detail.spectral_points(wcs.sub([2]), data.shape[0])
            if points is None:
                raise ValueError(
                    f"{filename}: the rows have no spectral axis; give the "
                    f"points of the profiles (points)")
        if step is None:
            step = _channel_width(filename, wcs)
        return cls(data, points, step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}lsf_images.fits'
        write_image(filename, self._data, overwrite=overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            points=self._dump_points(),
            step=self._step)

    def __init__(
            self,
            data: np.ndarray,
            points: u.Quantity,
            step: Annotated[float, Gt(0), Unit('km/s')]
    ):
        check_options(LSFImages.__init__, locals())
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or not np.all(np.isfinite(data)):
            raise ValueError(
                f"LSF profiles must be finite, of shape (nz, nk); their "
                f"shape is {data.shape}")
        super().__init__(data, points, "LSF profiles")
        self._step = step

    def _image(self, data: np.ndarray) -> LSFImage:
        return LSFImage(data, self._step)

    def stretched(self, factor: float) -> 'LSFImages':
        raise RuntimeError(
            "LSF profiles are stretched at each channel (see at)")


class LSFVarying(_detail.VaryingSpreadFunction, LSF):
    """
    An LSF of a type whose options vary along the spectral axis: an LSF of
    the type at each channel (see at). It is not drawn itself.
    It is made by the varying of the class of the type (e.g.
    LSFGauss.varying).
    """

    def stretched(self, factor: float) -> 'LSFVarying':
        raise RuntimeError(
            "an LSF that varies along the spectral axis is stretched at each "
            "channel (see at)")


lsf_parser = TypedParser(LSF, [
    LSFPoint,
    LSFGauss,
    LSFGGauss,
    LSFLorentz,
    LSFMoffat,
    LSFImage,
    LSFSum,
    LSFConvolution,
    LSFHanning,
    LSFImages])
