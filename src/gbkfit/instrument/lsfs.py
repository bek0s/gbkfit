"""
Line spread functions (LSFs): how the instrument spreads the light of a
single velocity along the spectral axis.

Their widths are widths of the light, each with its unit, as instruments
give their resolution: in a unit of wavelength (e.g. Angstrom) or of
frequency (e.g. kHz), in km/s of the light, c dw / w at its wavelength w
(c / R for a resolving power R), or in pixels (pix), channels of the data
(the pixels of an instrument only where the data keep its sampling). An
LSF is drawn at the channels of the spectral axis of the data (see
LSF.on), where its widths become km/s of the axis, by the convention of
its velocities (see Channel.span): on an axis of optical velocities, a
width in wavelength spans the same km/s of the axis everywhere, and one
in km/s of the light 1 + v / c km/s of the axis (more, at a high
redshift); on an axis of radio velocities, a width in frequency spans
the same km/s everywhere. Widths in pixels, and the Hanning smoothing
(see LSFHanning), are of the channels of the data themselves.

The analytic LSFs are given by their scale (e.g. sigma) or by their FWHM
(see from_fwhm), and their options can vary along the spectral axis (see
varying).
"""

import abc
import dataclasses
import os
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import astropy.io.fits
import astropy.units as u
import astropy.wcs
import numpy as np
import scipy.ndimage
import scipy.special
from annotated_types import Gt

import gbkfit.math
import gbkfit.units
from gbkfit.config import (
    DataFile, ScalarQuantity, TypedParser, UnitKinds, Varying,
    check_options, split_data_file)
from gbkfit.grid import Grid, VelocityConvention
from gbkfit.io import parse_wcs, read_image, write_image
from . import _detail
from ._detail import MIN_EXTENT, WING_FLUX


__all__ = [
    'LSF',
    'Channel',
    'ChannelLSF',
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


# The unit of velocities
_KMS = u.km / u.s

# A width of the light (see Channel.span): a Quantity in a unit of
# wavelength or of frequency, in km/s of the light, or in pixels (channels
# of the data); in a configuration, its text (e.g. '2.5 Angstrom')
_Width: TypeAlias = Annotated[ScalarQuantity, Gt(0), UnitKinds(
    wavelength=('Angstrom', 'nm', 'um'), frequency=('kHz', 'Hz', 'MHz', 'GHz'),
    velocity=('km/s', 'm/s'), pixels=('pix',))]


@dataclasses.dataclass(frozen=True)
class Channel:
    """
    A channel of the spectral axis of data, where an LSF is drawn (see
    LSF.on): the velocity of its centre, its width, and the convention of
    the velocities of the axis, which turns the widths of the LSF into
    km/s of the axis there (see span).

    Attributes
    ----------
    velocity : float, optional
        The velocity of its centre (km/s).
    width : float, optional
        Its width (km/s of the axis), for widths in pixels and the Hanning
        smoothing of the channels (see LSFHanning).
    convention : VelocityConvention, optional
        The convention of the velocities of the axis; by default, one
        without a rest, which takes widths in km/s only.
    """

    velocity: float = 0.0
    width: float | None = None
    convention: VelocityConvention = dataclasses.field(
        default_factory=VelocityConvention)

    @classmethod
    def middle(cls, grid: Grid) -> 'Channel':
        """
        Return the middle channel of the spectral axis of a grid (e.g. of
        data), the one of an LSF drawn for all (e.g. the correlation of
        the noise along the axis).

        Parameters
        ----------
        grid : Grid
            The grid; it must have a spectral axis.

        Returns
        -------
        Channel
            The channel.
        """
        axis = grid.spectral_axis
        step = grid.coords.step[axis]
        middle = grid.zero()[axis] + step * (grid.size[axis] - 1) / 2
        return cls(
            float(middle), abs(step), VelocityConvention.of(grid.coords))

    def span(self, width: u.Quantity) -> float:
        """
        Return the km/s of the axis that a width of the light spans at the
        channel: of a width in pixels, as many widths of the channel; of
        others, see VelocityConvention.span.

        Parameters
        ----------
        width : Quantity
            The width: in km/s of the light, in a unit of wavelength or of
            frequency, or in pixels (channels of the data).

        Returns
        -------
        float
            The km/s of the axis.

        Raises
        ------
        ValueError
            If the width is in a unit of wavelength or of frequency and
            the axis has no rest.
        RuntimeError
            If the width is in pixels and that of the channel is not given.
        """
        if width.unit.is_equivalent(u.pix):
            if self.width is None:
                raise RuntimeError(
                    "a width in pixels is of channels of the data, whose "
                    "width is not given")
            return float(width.to_value(u.pix)) * self.width
        if (self.convention.rest() is None
                and not width.unit.is_equivalent(_KMS)):
            raise ValueError(
                f"a width of the light in "
                f"'{gbkfit.units.format_unit(width.unit)}' spans km/s of the "
                f"spectral axis by its rest (RESTWAV or RESTFRQ in the "
                f"header of the data, or its option rest), which is not "
                f"given; give the width in km/s of the light instead")
        return float(self.convention.span(width, self.velocity))


@dataclasses.dataclass(frozen=True)
class ChannelLSF:
    """
    An LSF at a channel of the spectral axis of data (see LSF.on), drawn
    on arrays of the channels of the axis: its widths in km/s of the axis
    there (see Channel.span).

    Attributes
    ----------
    lsf : LSF
        The LSF.
    channel : Channel
        The channel.
    """

    lsf: 'LSF'
    channel: Channel

    def size(self, step: float) -> int:
        """
        Return the size of the array that holds the LSF.

        Parameters
        ----------
        step : float
            The width of the channels of the array (km/s of the axis).

        Returns
        -------
        int
            The number of channels, odd.

        Raises
        ------
        RuntimeError
            If the wings of the LSF are too heavy for a finite array.
        """
        size = self.lsf._size_impl(step, self.channel)
        _detail.check_finite_size(self.lsf, size)
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
            The width of the channels of the array (km/s of the axis).
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
        return self.lsf._asarray_impl(step, self.channel, size, offset)


class LSF(_detail.SpreadFunction):
    """
    A line spread function, drawn at the channels of the spectral axis of
    data (see on).
    """

    @staticmethod
    def _varying_class() -> type['LSFVarying']:
        return LSFVarying

    def on(self, channel: Channel) -> ChannelLSF:
        """
        Return the LSF at a channel of the spectral axis of data, to draw
        on arrays of the channels of the axis.

        Parameters
        ----------
        channel : Channel
            The channel.

        Returns
        -------
        ChannelLSF
            The LSF at the channel.
        """
        return ChannelLSF(self, channel)

    @abc.abstractmethod
    def _size_impl(self, step: float, channel: Channel) -> float:
        """
        The size of the array that holds the LSF at a channel, before
        rounding (see ChannelLSF.size).
        """

    @abc.abstractmethod
    def _asarray_impl(
            self, step: float, channel: Channel, size: int, offset: int
    ) -> np.ndarray:
        """The LSF on an array at a channel (see ChannelLSF.asarray)."""


def _create_grid_1d(size: int, step: float, offset: int) -> np.ndarray:
    """The velocities of the channels of an array from its centre."""
    center = size // 2 + offset
    return (np.array(range(size)) - center) * step


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

    def _size_impl(self, step: float, channel: Channel) -> float:
        return 1

    def _asarray_impl(
            self, step: float, channel: Channel, size: int, offset: int
    ) -> np.ndarray:
        data = np.zeros(size)
        data[size // 2 + offset] = 1
        return data


class _AnalyticLSF(_detail.GivenOptions, LSF):
    """
    An LSF of a symmetric profile of the velocity, of a scale (e.g. its
    sigma) that is a width of the light (see Channel.span), drawn out to
    its extent on each side (see MIN_EXTENT). Configurations give its
    scale, or its full width at half maximum (see from_fwhm), and dumps
    write the one given.

    Parameters
    ----------
    scale : Quantity
        The scale.
    """

    config_constructors = ('__init__', 'from_fwhm')

    def __init__(self, scale: u.Quantity):
        self._scale = scale

    @abc.abstractmethod
    def _extent(self, scale: float) -> float:
        """
        The half-width out to which it is drawn (see MIN_EXTENT), for its
        scale in km/s of the axis.
        """

    @abc.abstractmethod
    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        """
        Its profile at the velocities v, to any factor, for its scale (both
        in km/s of the axis).
        """

    def _size_impl(self, step: float, channel: Channel) -> float:
        return 2 * self._extent(channel.span(self._scale)) / step

    def _asarray_impl(
            self, step: float, channel: Channel, size: int, offset: int
    ) -> np.ndarray:
        scale = channel.span(self._scale)
        z = _create_grid_1d(size, step, offset)
        data = self._profile(z, scale)
        data[np.abs(z) > self._extent(scale)] = 0
        return data / np.sum(data)


class LSFGauss(_AnalyticLSF):
    """
    A Gaussian, by its sigma or its full width at half maximum.

    Its profile is exp(-v^2 / 2 sigma^2). Configurations give its sigma,
    or its FWHM (see from_fwhm).

    Parameters
    ----------
    sigma : Quantity
        The standard deviation: a width of the light (see Channel.span),
        e.g. '1.7 Angstrom'.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

    def __init__(self, sigma: Annotated[_Width, Varying()]):
        check_options(LSFGauss.__init__, locals())
        super().__init__(sigma)
        self._given_as(sigma=sigma)

    @classmethod
    def from_fwhm(cls, fwhm: Annotated[_Width, Varying()]) -> 'LSFGauss':
        """
        Make the Gaussian of a full width at half maximum: of sigma =
        FWHM / (2 sqrt(2 ln 2)).

        Parameters
        ----------
        fwhm : Quantity
            The full width at half maximum: a width of the light (see
            Channel.span), e.g. '4.0 Angstrom'.

        Returns
        -------
        LSFGauss
            The LSF.
        """
        check_options(LSFGauss.from_fwhm, locals())
        return cls(gbkfit.math.gauss_fwhm_to_sigma(fwhm))._given_as(
            fwhm=fwhm)

    def _extent(self, scale: float) -> float:
        # The flux of a Gaussian within |z| < w is erf(w / sqrt(2) sigma)
        return max(
            MIN_EXTENT * scale,
            scale * np.sqrt(2) * scipy.special.erfinv(1 - WING_FLUX))

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return gbkfit.math.gauss_1d_fun(v, 1, 0, scale)


class LSFGGauss(_AnalyticLSF):
    """
    A Gaussian with a shape parameter beta: flat-topped above 2, peaked
    below 2, exponential at 1; by its scale or its full width at half
    maximum.

    Its profile is exp(-(|v| / alpha)^beta). Configurations give its
    scale, or its FWHM (see from_fwhm).

    Parameters
    ----------
    alpha : Quantity
        The scale: a width of the light (see Channel.span).
    beta : float
        The shape, > 0: 2 is a Gaussian, and 1 an exponential.
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def type_title() -> str:
        return 'Generalised Gaussian'

    def __init__(
            self,
            alpha: Annotated[_Width, Varying()],
            beta: Annotated[float, Gt(0), Varying()]
    ):
        check_options(LSFGGauss.__init__, locals())
        super().__init__(alpha)
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: Annotated[_Width, Varying()],
            beta: Annotated[float, Gt(0), Varying()]
    ) -> 'LSFGGauss':
        """
        Make the generalised Gaussian of a full width at half maximum: of
        alpha = FWHM / (2 (ln 2)^(1 / beta)).

        Parameters
        ----------
        fwhm : Quantity
            The full width at half maximum: a width of the light (see
            Channel.span).
        beta : float
            The shape, > 0.

        Returns
        -------
        LSFGGauss
            The LSF.
        """
        check_options(LSFGGauss.from_fwhm, locals())
        alpha = fwhm / (2 * np.log(2) ** (1 / beta))
        return cls(alpha, beta)._given_as(fwhm=fwhm, beta=beta)

    def _extent(self, scale: float) -> float:
        # The flux of a ggauss within |z| < w is P(1 / beta, (w / alpha)^
        # beta), P the regularized lower incomplete gamma function
        return max(
            MIN_EXTENT * scale,
            scale * scipy.special.gammaincinv(
                1 / self._beta, 1 - WING_FLUX) ** (1 / self._beta))

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return gbkfit.math.ggauss_1d_fun(v, 1, 0, scale, self._beta)


class LSFLorentz(_AnalyticLSF):
    """
    Broad wings, as of a Fabry-Perot interferometer (whose profile about
    each of its peaks is nearly a Lorentzian); by its scale or its full
    width at half maximum.

    Its profile is gamma^2 / (v^2 + gamma^2). Configurations give its
    scale, or its FWHM (see from_fwhm).

    Parameters
    ----------
    gamma : Quantity
        The scale, the half width at half maximum: a width of the light
        (see Channel.span).
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def type_title() -> str:
        return 'Lorentzian'

    def __init__(self, gamma: Annotated[_Width, Varying()]):
        check_options(LSFLorentz.__init__, locals())
        super().__init__(gamma)
        self._given_as(gamma=gamma)

    @classmethod
    def from_fwhm(cls, fwhm: Annotated[_Width, Varying()]) -> 'LSFLorentz':
        """
        Make the Lorentzian of a full width at half maximum: of gamma =
        FWHM / 2.

        Parameters
        ----------
        fwhm : Quantity
            The full width at half maximum: a width of the light (see
            Channel.span).

        Returns
        -------
        LSFLorentz
            The LSF.
        """
        check_options(LSFLorentz.from_fwhm, locals())
        return cls(fwhm / 2)._given_as(fwhm=fwhm)

    def _extent(self, scale: float) -> float:
        # The flux of a Lorentzian within |z| < w is 2 / pi arctan(w /
        # gamma)
        return max(
            MIN_EXTENT * scale,
            scale * np.tan(np.pi / 2 * (1 - WING_FLUX)))

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return gbkfit.math.lorentz_1d_fun(v, 1, 0, scale)


class LSFMoffat(_AnalyticLSF):
    """
    A core with power-law wings; by its scale or its full width at half
    maximum.

    Its profile is (1 + (v / alpha)^2)^(-beta). Configurations give its
    scale, or its FWHM (see from_fwhm).

    Parameters
    ----------
    alpha : Quantity
        The scale of the core: a width of the light (see Channel.span).
    beta : float
        The slope of the wings, > 0.5 (for a finite flux).
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def type_title() -> str:
        return 'Moffat'

    def __init__(
            self,
            alpha: Annotated[_Width, Varying()],
            beta: Annotated[float, Gt(0.5), Varying()]
    ):
        check_options(LSFMoffat.__init__, locals())
        super().__init__(alpha)
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: Annotated[_Width, Varying()],
            beta: Annotated[float, Gt(0.5), Varying()]
    ) -> 'LSFMoffat':
        """
        Make the Moffat of a full width at half maximum: of alpha =
        FWHM / (2 sqrt(2^(1 / beta) - 1)).

        Parameters
        ----------
        fwhm : Quantity
            The full width at half maximum: a width of the light (see
            Channel.span).
        beta : float
            The slope of the wings, > 0.5.

        Returns
        -------
        LSFMoffat
            The LSF.
        """
        check_options(LSFMoffat.from_fwhm, locals())
        alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
        return cls(alpha, beta)._given_as(fwhm=fwhm, beta=beta)

    def _extent(self, scale: float) -> float:
        # The flux of a Moffat within |z| < w is I_u(1/2, beta - 1/2), I
        # the regularized incomplete beta function and u = w^2 / (alpha^2
        # + w^2)
        return max(
            MIN_EXTENT * scale,
            scale * np.sqrt(1 / (1 - scipy.special.betaincinv(
                0.5, self._beta - 0.5, 1 - WING_FLUX)) - 1))

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return gbkfit.math.moffat_1d_fun(v, 1, 0, scale, self._beta)


class LSFImage(LSF):
    """
    A measured LSF, from a 1D image (a profile).

    It is centred on the centre of the image, and resampled to the
    channels it is drawn on (with splines of order 5).

    Configurations give its file (see from_file).

    Parameters
    ----------
    data : ndarray
        The image, finite.
    step : Quantity
        The width of the channels of the image: a width of the light (see
        Channel.span).
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
            cls, file: DataFile, step: _Width | None = None
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
        step : Quantity, optional
            The width of the channels of the image: a width of the light
            (see Channel.span); by default, that of its axis, in its unit.

        Returns
        -------
        LSFImage
            The LSF.

        Raises
        ------
        ValueError
            If the file has no 1D image, its header has invalid world
            coordinates, or step is not given and the axis has no unit of
            velocity, wavelength or frequency.
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
        # (the file has the width of its channels in its axis, which
        # from_file reads)
        filename = f'{prefix}lsf.fits'
        header = astropy.io.fits.Header(dict(
            CUNIT1=gbkfit.units.format_unit(self._step.unit),
            CDELT1=float(self._step.value), CRPIX1=(self._data.size + 1) / 2,
            CRVAL1=0.0))
        write_image(filename, self._data, header, overwrite=overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename))

    def __init__(self, data: np.ndarray, step: _Width):
        check_options(LSFImage.__init__, locals())
        data = np.squeeze(np.asarray(data, dtype=float))
        if data.ndim != 1:
            raise ValueError(
                f"an LSF image has one axis (those of length 1 left out); "
                f"its shape is {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("the pixels of an LSF image must be finite")
        self._data = data
        self._step = step

    def _size_impl(self, step: float, channel: Channel) -> float:
        return channel.span(self._step) / step * self._data.shape[0]

    def _asarray_impl(
            self, step: float, channel: Channel, size: int, offset: int
    ) -> np.ndarray:
        scale = step / channel.span(self._step)
        # The centre of the image, and the centre of the LSF in the array,
        # where the analytic LSFs put it
        old_center = self._data.shape[0] / 2 - 0.5
        new_center = size // 2 + offset
        x = np.arange(size)
        nx = (x - new_center) * scale + old_center
        data = scipy.ndimage.map_coordinates(
            self._data, [nx], order=5, mode='grid-constant')
        return data / np.sum(data)


class LSFSum(_detail.SpreadFunctionTerms, LSF):
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
                lsf_parser, self._terms, prefix, dump_path, overwrite),
            weights=self._weights.tolist())

    def __init__(self, lsfs: Sequence[LSF], weights: Sequence[float]):
        check_options(LSFSum.__init__, locals())
        self._terms = tuple(lsfs)
        self._weights = _detail.sum_weights(
            weights, len(self._terms), "a sum of LSFs")

    def _of_terms(self, terms: Sequence[LSF]) -> 'LSFSum':
        return LSFSum(terms, self._weights)

    def _size_impl(self, step: float, channel: Channel) -> float:
        return max(lsf._size_impl(step, channel) for lsf in self._terms)

    def _asarray_impl(
            self, step: float, channel: Channel, size: int, offset: int
    ) -> np.ndarray:
        # Each term holds its fraction of the light on the same grid
        return sum(
            weight * lsf.on(channel).asarray(step, size, offset)
            for weight, lsf in zip(self._weights, self._terms))


class LSFConvolution(_detail.SpreadFunctionTerms, LSF):
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
                lsf_parser, self._terms, prefix, dump_path, overwrite))

    def __init__(self, lsfs: Sequence[LSF]):
        check_options(LSFConvolution.__init__, locals())
        if not lsfs:
            raise ValueError("a convolution of LSFs needs at least one LSF")
        self._terms = tuple(lsfs)

    def _of_terms(self, terms: Sequence[LSF]) -> 'LSFConvolution':
        return LSFConvolution(terms)

    def _size_impl(self, step: float, channel: Channel) -> float:
        # The sum of the (odd) sizes of the terms, less one for each
        # convolution
        sizes = [lsf.on(channel).size(step) for lsf in self._terms]
        return sum(sizes) - len(sizes) + 1

    def _asarray_impl(
            self, step: float, channel: Channel, size: int, offset: int
    ) -> np.ndarray:
        data = np.ones(1)
        for lsf in self._terms:
            data = np.convolve(data, lsf.on(channel).asarray(step))
        return _detail.embed(data / data.sum(), (size,), (offset,))


class LSFHanning(LSF):
    """
    The smoothing of the channels of the data by a Hanning window, as of
    radio spectrometers.

    1/4, 1/2 and 1/4 of the light of each channel go to the channel
    before, the channel itself and the channel after: channels of the
    width of those of the data (see Channel), or a fraction of it (e.g.
    0.5, of channels smoothed and then averaged in pairs).

    Parameters
    ----------
    channels : float, optional
        The width of the smoothed channels, in channels of the data; a
        multiple of the width of the channels it is drawn on (those of the
        data, with their oversampling).
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
        return dict(type=self.type(), channels=self._channels)

    def __init__(self, channels: Annotated[float, Gt(0)] = 1.0):
        check_options(LSFHanning.__init__, locals())
        self._channels = channels

    def _steps(self, step: float, channel: Channel) -> int:
        """The number of steps in a smoothed channel."""
        if channel.width is None:
            raise RuntimeError(
                "the Hanning smoothing is of the channels of the data, whose "
                "width is not given")
        steps = self._channels * channel.width / step
        if abs(steps - round(steps)) > 1e-6 * steps or round(steps) < 1:
            raise RuntimeError(
                f"the Hanning smoothing of {self._channels:g} channels of "
                f"{channel.width:g} km/s must be a multiple of the channels "
                f"it is drawn on ({step:g} km/s): oversample the spectral "
                f"axis by a multiple of {1 / self._channels:g}")
        return round(steps)

    def _size_impl(self, step: float, channel: Channel) -> float:
        return 2 * self._steps(step, channel) + 1

    def _asarray_impl(
            self, step: float, channel: Channel, size: int, offset: int
    ) -> np.ndarray:
        steps = self._steps(step, channel)
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


def _channel_width(source: str, wcs: astropy.wcs.WCS) -> u.Quantity:
    """
    The width of the channels of LSF profiles, in the unit of the first
    axis of their world coordinates (see _read_profiles).

    Raise ValueError if it is not a unit of velocity, wavelength or
    frequency, or pixels.
    """
    unit = u.Unit(wcs.wcs.cunit[0])
    if not any(unit.is_equivalent(x) for x in (_KMS, u.m, u.Hz, u.pix)):
        raise ValueError(
            f"{source}: the axis of the profiles has no unit of velocity, "
            f"wavelength, frequency or pixels; give the width of their "
            f"channels (step)")
    return abs(float(wcs.pixel_scale_matrix[0, 0])) * unit


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
    step : Quantity
        The width of the channels of the profiles: a width of the light
        (see Channel.span).
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
            step: _Width | None = None
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
        step : Quantity, optional
            The width of the channels of the profiles: a width of the
            light (see Channel.span); by default, that of the axis of the
            columns, in its unit.

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
            step=gbkfit.units.format_quantity(self._step))

    def __init__(
            self,
            data: np.ndarray,
            points: u.Quantity,
            step: _Width
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


class LSFVarying(_detail.VaryingSpreadFunction, LSF):
    """
    An LSF of a type whose options vary along the spectral axis: an LSF of
    the type at each channel (see at). It is not drawn itself.
    It is made by the varying of the class of the type (e.g.
    LSFGauss.varying).
    """


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
