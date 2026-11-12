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
import fractions
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import astropy.units as u
import astropy.wcs
import numpy as np
import scipy.special
from annotated_types import Gt
from astropy.stats import gaussian_fwhm_to_sigma

import gbkfit.units
from gbkfit.config import (
    DataFile, ScalarQuantity, TypedParser, UnitKinds, Varying,
    check_options, split_data_file)
from gbkfit.grid import Grid, VelocityConvention
from gbkfit.io import parse_wcs, read_image, spectral_points
from . import _detail, _profiles
from ._detail import WING_FLUX
from .varying import SpectralTable


__all__ = [
    'LSF',
    'Channel',
    'ChannelLSF',
    'LSFConvolution',
    'LSFGGauss',
    'LSFGauss',
    'LSFHanning',
    'LSFLorentz',
    'LSFMoffat',
    'LSFPoint',
    'LSFProfile',
    'LSFProfiles',
    'LSFSum',
    'LSFTables',
    'lsf_parser'
]


# The unit of velocities
_KMS = u.km / u.s

# A width of the light (see Channel.span): a Quantity in a unit of
# wavelength or of frequency, in km/s of the light, or in pixels (channels
# of the data); in a configuration, its text (e.g. '2.5 Angstrom')
_Width: TypeAlias = Annotated[ScalarQuantity, Gt(0), UnitKinds(
    wavelength=('Angstrom', 'nm', 'um'), frequency=('Hz', 'kHz', 'MHz', 'GHz'),
    velocity=('m/s', 'km/s'), pixels=('pix',))]

# The width of an analytic LSF, which can vary along the spectral axis
_VaryingWidth: TypeAlias = Annotated[_Width, Varying(SpectralTable)]

# The shapes of the generalised Gaussian and of the Moffat (for a finite
# flux, beta > 1/2)
_GGaussBeta: TypeAlias = Annotated[float, Gt(0), Varying(SpectralTable)]
_MoffatBeta: TypeAlias = Annotated[float, Gt(0.5), Varying(SpectralTable)]


@dataclasses.dataclass(frozen=True)
class Channel:
    """
    A channel of the spectral axis of data, where an LSF is drawn (see
    LSF.on): the velocity of its centre, its width, and the convention of
    the velocities of the axis, which turns the widths of the LSF into
    km/s of the axis there (see span). Or a channel of an axis of another
    unit (e.g. Angstrom, or pixels: of channels of the data, each of width
    1), where an LSF is drawn as its widths give it (e.g. to look at it):
    they span as much of the axis as they are.

    Attributes
    ----------
    velocity : float, optional
        The velocity of its centre (km/s; of an axis of velocities).
    width : float, optional
        Its width (in the unit of the axis), for widths in pixels and the
        Hanning smoothing of the channels (see LSFHanning).
    convention : VelocityConvention, optional
        The convention of the velocities of the axis; by default, one
        without a rest, which takes widths in km/s only.
    unit : UnitBase, optional
        The unit of the axis: km/s, of velocities, by default.
    """

    velocity: float = 0.0
    width: float | None = None
    convention: VelocityConvention = dataclasses.field(
        default_factory=VelocityConvention)
    unit: u.UnitBase = _KMS

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
        Return the length of the axis (in its unit) that a width of the
        light spans at the channel: of a width in pixels, as many widths of
        the channel; on an axis of velocities, of others, see
        VelocityConvention.span; on an axis of another unit, as much as the
        width is in it.

        Parameters
        ----------
        width : Quantity
            The width: in km/s of the light, in a unit of wavelength or of
            frequency, or in pixels (channels of the data).

        Returns
        -------
        float
            The length of the axis.

        Raises
        ------
        ValueError
            If the width is in a unit of wavelength or of frequency and
            the axis, of velocities, has no rest; or the axis is of another
            unit, which the width is not of.
        RuntimeError
            If the width is in pixels and that of the channel is not given.
        """
        if width.unit.is_equivalent(u.pix):
            if self.width is None:
                raise RuntimeError(
                    "an LSF of a width in pixels, or a Hanning smoothing, is "
                    "drawn in channels of the data, whose width is not given "
                    "(Channel.width)")
            return float(width.to_value(u.pix)) * self.width
        if self.unit != _KMS:
            if not width.unit.is_equivalent(self.unit):
                raise ValueError(
                    f"a width in '{gbkfit.units.format_unit(width.unit)}' "
                    f"spans no length of an axis of "
                    f"'{gbkfit.units.format_unit(self.unit)}'")
            return float(width.to_value(self.unit))
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
    on arrays of the channels of the axis: its widths in the unit of the
    axis there (e.g. km/s; see Channel.span).

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
        """Return the size of the array that holds it (see LSF.size)."""
        return self.lsf.size(step, self.channel)

    def asarray(
            self,
            step: float,
            size: int | None = None,
            offset: int = 0
    ) -> np.ndarray:
        """Return it on an array of channels (see LSF.asarray)."""
        return self.lsf.asarray(step, self.channel, size, offset)


class LSF(_detail.SpreadFunction):
    """
    A line spread function, drawn at the channels of the spectral axis of
    data (see size and asarray, or on).
    """

    _TERMS_OPTION = 'lsfs'

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

    def size(self, step: float, channel: Channel) -> int:
        """
        Return the size of the array that holds the LSF at a channel.

        Parameters
        ----------
        step : float
            The width of the channels of the array (in the unit of the
            axis, e.g. km/s).
        channel : Channel
            The channel.

        Returns
        -------
        int
            The number of channels, odd.

        Raises
        ------
        RuntimeError
            If the LSF varies along the spectral axis (see at), or its
            wings are too heavy for a finite array.
        """
        self._check_not_varying()
        return _detail.odd_size(self, self._size_impl(step, channel))[0]

    def asarray(
            self,
            step: float,
            channel: Channel,
            size: int | None = None,
            offset: int = 0
    ) -> np.ndarray:
        """
        Return the LSF at a channel on an array of channels, normalised to
        sum to 1.

        Parameters
        ----------
        step : float
            The width of the channels of the array (in the unit of the
            axis, e.g. km/s).
        channel : Channel
            The channel.
        size : int, optional
            The number of channels; by default, size(step, channel).
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
            If the LSF varies along the spectral axis (see at), or size +
            offset is even.
        """
        self._check_not_varying()
        if size is None:
            size = self.size(step, channel)
        _detail.check_odd_centre((size,), (offset,), "LSF")
        return self._asarray_impl(step, size, offset, channel)

    @abc.abstractmethod
    def _size_impl(self, step: float, channel: Channel) -> float:
        """
        The size of the array that holds the LSF at a channel, before
        rounding (see size).
        """

    @abc.abstractmethod
    def _asarray_impl(
            self, step: float, size: int, offset: int, channel: Channel
    ) -> np.ndarray:
        """The LSF on an array at a channel (see asarray)."""


def _offsets(size: int, step: float, offset: int) -> np.ndarray:
    """
    The offsets (in the unit of the axis, e.g. km/s) of the channels of an
    array from its centre (see LSF.asarray).
    """
    center = size // 2 + offset
    return (np.arange(size) - center) * step


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
            self, step: float, size: int, offset: int, channel: Channel
    ) -> np.ndarray:
        return _detail.embed(np.ones(1), (size,), (offset,))


class _AnalyticLSF(_detail.AnalyticSpreadFunction, LSF):
    """
    An LSF of a symmetric profile of the velocity, of a scale (e.g. its
    sigma) that is a width of the light (see Channel.span). Configurations
    give its scale, or its full width at half maximum (see from_fwhm), and
    dumps write the one given.

    Parameters
    ----------
    scale : Quantity
        The scale.
    """

    @staticmethod
    def _tables_class() -> type['LSFTables']:
        return LSFTables

    def _size_impl(self, step: float, channel: Channel) -> float:
        return 2 * self._extent(channel.span(self._scale)) / step

    def _asarray_impl(
            self, step: float, size: int, offset: int, channel: Channel
    ) -> np.ndarray:
        return self._drawn(
            _offsets(size, step, offset), channel.span(self._scale))


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

    def __init__(self, sigma: _VaryingWidth):
        check_options(LSFGauss.__init__, locals())
        super().__init__(sigma)
        self._given_as(sigma=sigma)

    @classmethod
    def from_fwhm(cls, fwhm: _VaryingWidth) -> 'LSFGauss':
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
        return cls(fwhm * gaussian_fwhm_to_sigma)._given_as(fwhm=fwhm)

    def _wings(self) -> float:
        # (the flux of a Gaussian within |v| < w is erf(w / sqrt(2) sigma))
        return np.sqrt(2) * scipy.special.erfinv(1 - WING_FLUX)

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return _profiles.gauss_1d_fun(v, 1, 0, scale)


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
            alpha: _VaryingWidth,
            beta: _GGaussBeta
    ):
        check_options(LSFGGauss.__init__, locals())
        super().__init__(alpha)
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: _VaryingWidth,
            beta: _GGaussBeta
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
        alpha = _profiles.ggauss_fwhm_to_alpha(fwhm, beta)
        return cls(alpha, beta)._given_as(fwhm=fwhm, beta=beta)

    def _wings(self) -> float:
        # (the flux of a ggauss within |v| < w is P(1 / beta, (w / alpha)^
        # beta), P the regularized lower incomplete gamma function)
        return scipy.special.gammaincinv(
            1 / self._beta, 1 - WING_FLUX) ** (1 / self._beta)

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return _profiles.ggauss_1d_fun(v, 1, 0, scale, self._beta)


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

    def __init__(self, gamma: _VaryingWidth):
        check_options(LSFLorentz.__init__, locals())
        super().__init__(gamma)
        self._given_as(gamma=gamma)

    @classmethod
    def from_fwhm(cls, fwhm: _VaryingWidth) -> 'LSFLorentz':
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

    def _wings(self) -> float:
        # (the flux of a Lorentzian within |v| < w is 2 / pi arctan(w /
        # gamma))
        return np.tan(np.pi / 2 * (1 - WING_FLUX))

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return _profiles.lorentz_1d_fun(v, 1, 0, scale)


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
            alpha: _VaryingWidth,
            beta: _MoffatBeta
    ):
        check_options(LSFMoffat.__init__, locals())
        super().__init__(alpha)
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: _VaryingWidth,
            beta: _MoffatBeta
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
        alpha = _profiles.moffat_fwhm_to_alpha(fwhm, beta)
        return cls(alpha, beta)._given_as(fwhm=fwhm, beta=beta)

    def _wings(self) -> float:
        # (the flux of a Moffat within |v| < w is I_u(1/2, beta - 1/2), I
        # the regularized incomplete beta function and u = w^2 / (alpha^2
        # + w^2))
        return np.sqrt(1 / (1 - scipy.special.betaincinv(
            0.5, self._beta - 0.5, 1 - WING_FLUX)) - 1)

    def _profile(self, v: np.ndarray, scale: float) -> np.ndarray:
        return _profiles.moffat_1d_fun(v, 1, 0, scale, self._beta)


class LSFProfile(LSF):
    """
    A measured LSF, from its profile (a 1D image).

    It is centred on the centre of the profile, and resampled to the
    channels it is drawn on (with splines of order 5).

    Configurations give its file (see from_file).

    Parameters
    ----------
    data : ndarray
        The profile, finite; it holds light.
    step : Quantity
        The width of the channels of the profile: a width of the light
        (see Channel.span).
    """

    config_constructors = ('from_file',)

    @staticmethod
    def type() -> str:
        return 'profile'

    @staticmethod
    def type_title() -> str:
        return 'Profile (measured)'

    @classmethod
    def from_file(
            cls, file: DataFile, step: _Width | None = None
    ) -> 'LSFProfile':
        """
        Read an LSF profile (a 1D image) from a file, with its axis as it
        is: the profile is reversed where the axis runs against the
        velocities of the light (velocities or wavelengths that decrease
        along it, or frequencies that increase).

        Parameters
        ----------
        file : DataFile
            The file of the profile, and where it is in it (see
            read_image).
        step : Quantity, optional
            The width of the channels of the profile: a width of the light
            (see Channel.span); by default, that of its axis, in its unit.

        Returns
        -------
        LSFProfile
            The LSF.

        Raises
        ------
        ValueError
            If the file has no 1D image, its header has invalid world
            coordinates, or step is not given and the axis has no unit of
            velocity, wavelength, frequency or pixels.
        """
        check_options(LSFProfile.from_file, locals())
        filename, data, _, step = _read_profiles(file, step)
        if data.ndim != 1:
            raise ValueError(
                f"{filename}: an LSF profile is a 1D image; it has "
                f"{data.ndim} axes"
                + (" (an image of profiles at points of the spectral axis, "
                   "one per row, is an LSF of type profiles)"
                   if data.ndim == 2 else ""))
        return cls(data, step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_image(
            self, self._data, 'lsf', prefix, dump_path, overwrite) | dict(
            step=gbkfit.units.format_quantity(self._step))

    def __init__(self, data: np.ndarray, step: _Width):
        check_options(LSFProfile.__init__, locals())
        data = np.squeeze(np.asarray(data, dtype=float))
        if data.ndim != 1:
            raise ValueError(
                f"an LSF profile has one axis (those of length 1 left out); "
                f"its shape is {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("the pixels of an LSF profile must be finite")
        if not np.sum(data) > 0:
            raise ValueError(
                f"an LSF profile must hold light; the sum of its pixels is "
                f"{np.sum(data):g}")
        self._data = data
        self._step = step

    def _size_impl(self, step: float, channel: Channel) -> float:
        return channel.span(self._step) / step * self._data.shape[0]

    def _asarray_impl(
            self, step: float, size: int, offset: int, channel: Channel
    ) -> np.ndarray:
        step_ratio = step / channel.span(self._step)
        return _detail.resample(
            self._data, (step_ratio,), (size,), (offset,))


class LSFSum(_detail.SpreadFunctionSum, LSF):
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

    def __init__(self, lsfs: Sequence[LSF], weights: Sequence[float]):
        check_options(LSFSum.__init__, locals())
        super().__init__(lsfs, weights, "a sum of LSFs")

    def _size_impl(self, step: float, channel: Channel) -> float:
        return max(lsf.size(step, channel) for lsf in self._terms)

    def _asarray_impl(
            self, step: float, size: int, offset: int, channel: Channel
    ) -> np.ndarray:
        # Each term holds its fraction of the light on the same grid
        return sum(
            weight * lsf.asarray(step, channel, size, offset)
            for weight, lsf in zip(self._weights, self._terms))


class LSFConvolution(_detail.SpreadFunctionConvolution, LSF):
    """
    The convolution of LSFs: e.g. the LSF of an instrument and the
    Hanning smoothing of the channels of the data (hanning).

    Parameters
    ----------
    lsfs : Sequence of LSF
        The LSFs, at least one.
    """

    def __init__(self, lsfs: Sequence[LSF]):
        check_options(LSFConvolution.__init__, locals())
        super().__init__(lsfs, "a convolution of LSFs")

    def _size_impl(self, step: float, channel: Channel) -> float:
        # The sum of the (odd) sizes of the terms, less one for each
        # convolution
        sizes = [lsf.size(step, channel) for lsf in self._terms]
        return sum(sizes) - len(sizes) + 1

    def _asarray_impl(
            self, step: float, size: int, offset: int, channel: Channel
    ) -> np.ndarray:
        data = np.ones(1)
        for lsf in self._terms:
            data = np.convolve(data, lsf.asarray(step, channel))
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

    def _tap_spacing(self, step: float, channel: Channel) -> int:
        """
        The channels of the array (of width step) between the taps of the
        smoothing: as many as span its channels.
        """
        spacing = channel.span(self._channels * u.pix) / step
        if abs(spacing - round(spacing)) > 1e-6 * spacing:
            # (whole on channels of the data divided by a multiple of the
            # denominator of the channels)
            factor = fractions.Fraction(
                self._channels).limit_denominator(1000).denominator
            unit = gbkfit.units.format_unit(channel.unit)
            raise RuntimeError(
                f"a Hanning smoothing of {self._channels:g} x "
                f"{channel.width:g} {unit} (channels of the data) spans "
                f"{spacing:g} of the channels it is drawn on ({step:g} "
                f"{unit}), not a whole number of them: draw it on channels "
                f"of the data divided by a multiple of {factor} (in a model, "
                f"of an oversampling of the spectral axis by a multiple of "
                f"{factor})")
        return round(spacing)

    def _size_impl(self, step: float, channel: Channel) -> float:
        return 2 * self._tap_spacing(step, channel) + 1

    def _asarray_impl(
            self, step: float, size: int, offset: int, channel: Channel
    ) -> np.ndarray:
        spacing = self._tap_spacing(step, channel)
        data = np.zeros(2 * spacing + 1)
        data[[0, spacing, 2 * spacing]] = 0.25, 0.5, 0.25
        return _detail.embed(data, (size,), (offset,))


def _read_profiles(
        file: DataFile, step: _Width | None
) -> tuple[str, np.ndarray, astropy.wcs.WCS, _Width]:
    """
    The LSF profile, or profiles (one per row of an image: nz, nk), of a
    file, along its first axis, with the filename, its world coordinates
    (its axes as they are, unlike those of data: the axis of the profiles
    may be spectral, and so that of the rows), and the width of their
    channels: step, or by default that of their axis, in its unit. The
    profiles run along the velocities of the light: they are reversed
    where their axis runs against them (velocities or wavelengths that
    decrease along it, or frequencies that increase).

    Raise ValueError if step is not given and the axis has no unit of
    velocity, wavelength, frequency or pixels.
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    data = np.asarray(data, dtype=float)
    wcs = parse_wcs(filename, header)
    axis_step = (
        float(wcs.pixel_scale_matrix[0, 0]) * u.Unit(wcs.wcs.cunit[0]))
    unit = axis_step.unit
    if unit.is_equivalent(_KMS) or unit.is_equivalent(u.m):
        against = axis_step.value < 0
    else:
        # (frequencies, energies and wave numbers fall as velocities rise)
        against = axis_step.value > 0 and unit.is_equivalent(
            u.Hz, u.spectral())
    if step is None:
        if not any(unit.is_equivalent(x) for x in (_KMS, u.m, u.Hz, u.pix)):
            raise ValueError(
                f"{filename}: the axis of the profiles has no unit of "
                f"velocity, wavelength, frequency or pixels; give the width "
                f"of their channels (step)")
        step = abs(axis_step)
    return filename, (data[..., ::-1] if against else data), wcs, step


class LSFProfiles(_detail.SpreadFunctionImages, LSF):
    """
    A measured LSF that changes with wavelength, from its profiles.

    It is given as profiles at points of the spectral axis (e.g. measured
    from sky lines or arc lamps). At each channel it is the profile LSF
    (see LSFProfile) of the profiles of the two nearest points,
    interpolated linearly; beyond the points, that of the nearest profile.

    Configurations give its file, an image with one profile per row (see
    from_file).

    Parameters
    ----------
    data : ndarray
        The profiles (nz, nk), finite; each holds light.
    points : Quantity
        The point of each profile: wavelengths, frequencies or velocities.
    step : Quantity
        The width of the channels of the profiles: a width of the light
        (see Channel.span).
    """

    config_constructors = ('from_file',)

    @staticmethod
    def type() -> str:
        return 'profiles'

    @staticmethod
    def type_title() -> str:
        return 'Profile per wavelength (measured)'

    @classmethod
    def from_file(
            cls,
            file: DataFile,
            points: u.Quantity | None = None,
            step: _Width | None = None
    ) -> 'LSFProfiles':
        """
        Read LSF profiles from an image, one per row (nz, nk), with its
        axes as they are: the profiles are reversed where the axis of the
        columns runs against the velocities of the light (see
        LSFProfile.from_file).

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
        LSFProfiles
            The LSF.

        Raises
        ------
        ValueError
            If the file has no 2D image, its header has invalid world
            coordinates, or the points or step are neither given nor in
            its header.
        """
        check_options(LSFProfiles.from_file, locals())
        filename, data, wcs, step = _read_profiles(file, step)
        if data.ndim != 2:
            raise ValueError(
                f"{filename}: LSF profiles must be an image, with one profile "
                f"per row; it has {data.ndim} axes"
                + (" (one profile is an LSF of type profile)"
                   if data.ndim == 1 else ""))
        if points is None:
            points = spectral_points(wcs.sub([2]), data.shape[0])
            if points is None:
                raise ValueError(
                    f"{filename}: the rows have no spectral axis; give the "
                    f"points of the profiles (points)")
        return cls(data, points, step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_image(
            self, self._data, 'lsf_profiles', prefix, dump_path, overwrite
        ) | dict(
            points=self._dump_points(),
            step=gbkfit.units.format_quantity(self._step))

    def __init__(
            self,
            data: np.ndarray,
            points: u.Quantity,
            step: _Width
    ):
        check_options(LSFProfiles.__init__, locals())
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(
                f"LSF profiles are of shape (nz, nk); their shape is "
                f"{data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("the pixels of LSF profiles must be finite")
        super().__init__(data, points, "LSF profiles")
        self._step = step

    def _of_image(self, data: np.ndarray) -> LSFProfile:
        return LSFProfile(data, self._step)


class LSFTables(_detail.SpreadFunctionTables, LSF):
    """
    An LSF of an analytic type whose options vary along the spectral axis,
    as tables of their values: at each point, an LSF of the type (see at).
    It is not drawn itself. It is made by the varying of the class of the
    type (e.g. LSFGauss.varying).
    """


lsf_parser = TypedParser(LSF, [
    LSFPoint,
    LSFGauss,
    LSFGGauss,
    LSFLorentz,
    LSFMoffat,
    LSFProfile,
    LSFSum,
    LSFConvolution,
    LSFHanning,
    LSFProfiles], family='lsf')
