"""
Point spread functions (PSFs): how the atmosphere, the telescope and the
instrument spread the light of a point on the sky.

Widths are in arcsec, and position angles in degrees, from north through
east, like those of the disks. The options of the analytic PSFs can vary
along the spectral axis (see varying).
"""

import abc
import os
from collections.abc import Sequence
from typing import Any

import astropy.io.fits
import astropy.units as u
import astropy.wcs
import astropy.wcs.utils
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.special

import gbkfit.math
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail, varying
from ._detail import MIN_EXTENT, WING_FLUX, check_ratio, check_scale


__all__ = [
    'PSF',
    'PSFConvolution',
    'PSFImages',
    'PSFGauss',
    'PSFGaussBeam',
    'PSFGGauss',
    'PSFImage',
    'PSFMoffat',
    'PSFPoint',
    'PSFSum',
    'PSFVarying',
    'psf_parser'
]


class PSF(parseutils.TypedSerializable, abc.ABC):
    """
    A point spread function, drawn on arrays of pixels.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'PSF':
        """
        Load a PSF from its configuration.

        Options that are tables make a PSF that varies along the spectral
        axis (see varying and PSFVarying).

        Parameters
        ----------
        info : dict
            The options.

        Returns
        -------
        PSF
            The PSF.
        """
        options = varying.load_tables(info)
        if options != info:
            return cls.varying(**options)
        return super().load(info)

    @classmethod
    def varying(cls, **options) -> 'PSF':
        """
        Make a PSF of this class whose options vary along the spectral axis.

        Parameters
        ----------
        **options
            The options of the class; those that vary (see the
            VARYING_OPTIONS of the class) are tables (SpectralTable).

        Returns
        -------
        PSF
            The PSF (see PSFVarying).

        Raises
        ------
        ConfigError
            If an option that cannot vary is a table, or the options are
            invalid.
        """
        return PSFVarying(varying._VaryingOptions(cls, options))

    @abc.abstractmethod
    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the PSF to its configuration.

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

    def size(self, step: tuple[float, float]) -> tuple[int, int]:
        """
        Return the size of the array that holds the PSF.

        Parameters
        ----------
        step : tuple of float
            The size of the pixels along x and y (arcsec).

        Returns
        -------
        tuple of int
            The number of pixels along x and y, each odd.

        Raises
        ------
        RuntimeError
            If the wings of the PSF are too heavy for a finite array.
        """
        base_size = self._size_impl(step)
        _detail.check_finite_size(self, *base_size)
        return (int(gbkfit.math.roundu_odd(base_size[0])),
                int(gbkfit.math.roundu_odd(base_size[1])))

    def asarray(
            self,
            step: tuple[float, float],
            size: tuple[int, int] | None = None,
            offset: tuple[int, int] = (0, 0),
            rota: float = 0
    ) -> np.ndarray:
        """
        Return the PSF on an array of pixels, normalised to sum to 1.

        Parameters
        ----------
        step : tuple of float
            The size of the pixels along x and y (arcsec).
        size : tuple of int, optional
            The number of pixels along x and y; by default, that of size.
        offset : tuple of int, optional
            The offset of the centre of the PSF from the pixel size // 2,
            along x and y. Each size + offset must be odd.
        rota : float, optional
            The rotation of the pixels on the sky (degrees, see
            gridutils.Coords): a PSF of position angle posa is drawn at
            posa - rota.

        Returns
        -------
        ndarray
            The PSF, of shape (ny, nx).

        Raises
        ------
        RuntimeError
            If a size + offset is even.
        """
        if size is None:
            size = self.size(step)
        if (gbkfit.math.is_even(size[0] + offset[0]) or
              gbkfit.math.is_even(size[1] + offset[1])):
            raise RuntimeError(
                f"invalid PSF size: (size + offset) = "
                f"({size[0]} + {offset[0]} = {size[0] + offset[0]}, "
                f"{size[1]} + {offset[1]} = {size[1] + offset[1]}), "
                f"but both values must be odd")
        return self._asarray_impl(step, size, offset, rota)

    def varies(self) -> bool:
        """
        Check whether the PSF varies along the spectral axis.

        Returns
        -------
        bool
            Whether it varies.
        """
        return False

    def at(
            self, points: u.Quantity, rest: u.Quantity | None = None
    ) -> 'PSF | list[PSF]':
        """
        Return the PSF at points of the spectral axis.

        Parameters
        ----------
        points : Quantity
            Wavelengths, frequencies or velocities: one, or a sequence.
        rest : Quantity, optional
            The rest of the spectral axis (see gridutils.Coords); needed
            between velocities and the points of the tables of a PSF
            that varies, if they are of another kind.

        Returns
        -------
        PSF or list of PSF
            The PSF at the point, or at each point: itself, unless it
            varies.
        """
        return varying.per_point(points, lambda points_: [self] * len(points_))

    def velocity_range(
            self, rest: u.Quantity | None = None
    ) -> tuple[float, float]:
        """
        Return the range of velocities of the spectral axis that the PSF
        is known at.

        Parameters
        ----------
        rest : Quantity, optional
            The rest of the spectral axis (see gridutils.Coords).

        Returns
        -------
        tuple of float
            The lowest and highest velocity (km/s); infinite unless it
            varies.
        """
        return -np.inf, np.inf

    @abc.abstractmethod
    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        """The size of the array that holds the PSF, before rounding."""

    @abc.abstractmethod
    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        """The PSF on an array (see asarray)."""


def _create_grid_2d(
        size: tuple[int, int],
        step: tuple[float, float],
        offset: tuple[int, int],
        ratio: float,
        posa: float
) -> np.ndarray:
    """
    The elliptical radius of the pixels of an array from its centre (see
    PSF.asarray), for an axis ratio and the position angle of the major
    axis.
    """
    center_x = size[0] // 2 + offset[0]
    center_y = size[1] // 2 + offset[1]
    x = (np.array(range(size[0])) - center_x) * step[0]
    y = (np.array(range(size[1])) - center_y) * step[1]
    x = x[None, :]
    y = y[:, None]
    x, y = gbkfit.math.transform_lh_rotate_z(x, y, np.radians(posa))
    return np.sqrt(x * x + y * y / (ratio * ratio))


class PSFPoint(PSF):
    """
    A point: all the light stays in its pixel.
    """

    @staticmethod
    def type() -> str:
        return 'point'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type())

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return 1, 1

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        # Like all images, the array has shape (y, x)
        data = np.zeros(size[::-1])
        data[size[1] // 2 + offset[1], size[0] // 2 + offset[0]] = 1
        return data


class PSFGauss(PSF):
    """
    An elliptical Gaussian, exp(-r^2 / 2 sigma^2).

    Parameters
    ----------
    sigma : float
        The standard deviation along the major axis (arcsec).
    ratio : float, optional
        The axis ratio (minor over major), in (0, 1].
    posa : float, optional
        The position angle of the major axis.
    """

    VARYING_OPTIONS = dict(sigma='arcsec', ratio='', posa='deg')

    @staticmethod
    def type() -> str:
        return 'gauss'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            sigma=self._sigma,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(self, sigma: float, ratio: float = 1.0, posa: float = 0.0):
        check_scale('sigma', sigma)
        check_ratio(ratio)
        self._sigma = sigma
        self._ratio = ratio
        self._posa = posa

    def _extent(self):
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        wings of a 2D Gaussian beyond r hold exp(-r^2 / 2 sigma^2).
        """
        return max(
            MIN_EXTENT * self._sigma,
            self._sigma * np.sqrt(-2 * np.log(WING_FLUX)))

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (2 * self._extent() / step[0],
                2 * self._extent() / step[1])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        r = _create_grid_2d(
            size, step, offset, self._ratio, self._posa - rota)
        data = gbkfit.math.gauss_1d_fun(r, 1, 0, self._sigma)
        data[r > self._extent()] = 0
        return data / np.sum(data)


class PSFGGauss(PSF):
    """
    An elliptical generalised Gaussian, exp(-(r / alpha)^beta).

    Parameters
    ----------
    alpha : float
        The scale length along the major axis (arcsec).
    beta : float
        The shape, > 0: 2 is a Gaussian, and 1 an exponential.
    ratio : float, optional
        The axis ratio (minor over major), in (0, 1].
    posa : float, optional
        The position angle of the major axis.
    """

    VARYING_OPTIONS = dict(alpha='arcsec', beta='', ratio='', posa='deg')

    @staticmethod
    def type() -> str:
        return 'ggauss'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(
            self,
            alpha: float,
            beta: float,
            ratio: float = 1.0,
            posa: float = 0.0
    ):
        check_scale('alpha', alpha)
        check_scale('beta', beta)
        check_ratio(ratio)
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _extent(self):
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        flux of a 2D ggauss within r is P(2 / beta, (r / alpha)^beta),
        P the regularized lower incomplete gamma function.
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * scipy.special.gammaincinv(
                2 / self._beta, 1 - WING_FLUX) ** (1 / self._beta))

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (2 * self._extent() / step[0],
                2 * self._extent() / step[1])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        r = _create_grid_2d(
            size, step, offset, self._ratio, self._posa - rota)
        data = gbkfit.math.ggauss_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[r > self._extent()] = 0
        return data / np.sum(data)


class PSFMoffat(PSF):
    """
    An elliptical Moffat, (1 + (r / alpha)^2)^(-beta).

    Parameters
    ----------
    alpha : float
        The scale length of the core along the major axis (arcsec).
    beta : float
        The slope of the wings, > 1 (for a finite flux).
    ratio : float, optional
        The axis ratio (minor over major), in (0, 1].
    posa : float, optional
        The position angle of the major axis.
    """

    VARYING_OPTIONS = dict(alpha='arcsec', beta='', ratio='', posa='deg')

    @staticmethod
    def type():
        return 'moffat'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(
            self,
            alpha: float,
            beta: float,
            ratio: float = 1.0,
            posa: float = 0.0
    ):
        check_scale('alpha', alpha)
        if not beta > 1:
            raise ConfigError(
                "a Moffat PSF has finite flux only for beta > 1; "
                f"beta is {beta}")
        check_ratio(ratio)
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _extent(self):
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        wings of a 2D Moffat beyond r hold (1 + (r / alpha)^2)^(1 - beta).
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * np.sqrt(
                WING_FLUX ** (1 / (1 - self._beta)) - 1))

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (2 * self._extent() / step[0],
                2 * self._extent() / step[1])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        r = _create_grid_2d(
            size, step, offset, self._ratio, self._posa - rota)
        data = gbkfit.math.moffat_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[r > self._extent()] = 0
        return data / np.sum(data)


class PSFImage(PSF):
    """
    A PSF given as an image, centred on the centre of the image, and
    resampled to the pixels it is drawn on (with splines of order 5).

    The image is in the frame of the pixels of the data, so it is not
    rotated with them. Its configuration has its file (a filename, or a
    dict with the filename and the HDU), and the size of its pixels comes
    from the header unless step is given.

    Parameters
    ----------
    data : ndarray
        The image, finite.
    step : Sequence of float, optional
        The size of the pixels of the image along x and y (arcsec).
    """

    @staticmethod
    def type() -> str:
        return 'image'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'PSFImage':
        data, coords = parseutils.load_option(
            _detail.read_image, info, 'file', required=True)
        info = dict(info)
        del info['file']
        info.update(data=data, step=info.get('step', coords.step))
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}psf.fits'
        rpix = tuple(np.array(self._data.shape[::-1]) / 2 - 0.5)
        coords = gridutils.Coords(tuple(self._step), rpix, (0.0, 0.0), 0.0)
        fitsutils.write_data(filename, self._data, coords, None, overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            step=self._step)

    def __init__(
            self, data: np.ndarray, step: Sequence[float] = (1.0, 1.0)
    ):
        data = np.squeeze(data)  # Remove singleton dimensions
        if data.ndim != 2:
            raise ConfigError(
                f"expected a 2D PSF image, but got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ConfigError(
                "non-finite pixels found in the supplied PSF image")
        self._data = data
        self._step = tuple(step)

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (
            (self._step[0] / step[0]) * self._data.shape[1],
            (self._step[1] / step[1]) * self._data.shape[0])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        scale_x = step[0] / self._step[0]
        scale_y = step[1] / self._step[1]
        # The centre of the image, and the centre of the PSF in the array,
        # where the analytic PSFs put it
        old_center_x = self._data.shape[1] / 2 - 0.5
        old_center_y = self._data.shape[0] / 2 - 0.5
        new_center_x = size[0] // 2 + offset[0]
        new_center_y = size[1] // 2 + offset[1]
        # Like all images, the arrays have shape (y, x)
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        nx = (x - new_center_x) * scale_x + old_center_x
        ny = (y - new_center_y) * scale_y + old_center_y
        data = scipy.ndimage.map_coordinates(
            self._data, [ny, nx], order=5, mode='grid-constant')
        return data / np.sum(data)


class PSFSum(PSF):
    """
    A sum of PSFs (e.g. a double Gaussian, or a Gaussian core with Moffat
    wings), each with its fraction of the light.

    Parameters
    ----------
    psfs : Sequence of PSF
        The PSFs.
    weights : Sequence of float
        The weight of each PSF, positive; they are normalised to sum to 1.
    """

    @staticmethod
    def type() -> str:
        return 'sum'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'PSFSum':
        parseutils.load_option_and_update_info(
            psf_parser, info, 'psfs', required=True)
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            psfs=_detail.dump_terms(
                psf_parser, self._psfs, prefix, dump_path, overwrite),
            weights=self._weights.tolist())

    def __init__(self, psfs: Sequence[PSF], weights: Sequence[float]):
        self._psfs = tuple(psfs)
        self._weights = _detail.sum_weights(
            weights, len(self._psfs), "a sum of PSFs")

    def varies(self) -> bool:
        return any(psf.varies() for psf in self._psfs)

    def at(self, points, rest=None):
        if not self.varies():
            return super().at(points, rest)
        return varying.per_point(points, lambda points_: [
            PSFSum(psfs, self._weights) for psfs in
            _detail.terms_at(self._psfs, points_, rest)])

    def velocity_range(self, rest=None):
        return _detail.terms_velocity_range(self._psfs, rest)

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        sizes = np.array([psf._size_impl(step) for psf in self._psfs])
        return tuple(sizes.max(axis=0).tolist())

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        # Each term holds its fraction of the light on the same grid
        return sum(
            weight * psf.asarray(step, size, offset, rota)
            for weight, psf in zip(self._weights, self._psfs))


class PSFConvolution(PSF):
    """
    The convolution of PSFs: e.g. the PSF of an adaptive optics system
    and the seeing, or a PSF and the response of the pixels.

    Parameters
    ----------
    psfs : Sequence of PSF
        The PSFs, at least one.
    """

    @staticmethod
    def type() -> str:
        return 'convolution'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'PSFConvolution':
        parseutils.load_option_and_update_info(
            psf_parser, info, 'psfs', required=True)
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            psfs=_detail.dump_terms(
                psf_parser, self._psfs, prefix, dump_path, overwrite))

    def __init__(self, psfs: Sequence[PSF]):
        if not psfs:
            raise ConfigError("a convolution of PSFs needs at least one PSF")
        self._psfs = tuple(psfs)

    def varies(self) -> bool:
        return any(psf.varies() for psf in self._psfs)

    def at(self, points, rest=None):
        if not self.varies():
            return super().at(points, rest)
        return varying.per_point(points, lambda points_: [
            PSFConvolution(psfs) for psfs in
            _detail.terms_at(self._psfs, points_, rest)])

    def velocity_range(self, rest=None):
        return _detail.terms_velocity_range(self._psfs, rest)

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        # The sum of the (odd) sizes of the terms, less one for each
        # convolution
        sizes = np.array([psf.size(step) for psf in self._psfs])
        return tuple((sizes.sum(axis=0) - len(sizes) + 1).tolist())

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        data = np.ones((1, 1))
        for psf in self._psfs:
            data = scipy.signal.convolve(
                data, psf.asarray(step, rota=rota), method='auto')
        return _detail.embed(data / data.sum(), size, offset)


class PSFGaussBeam(PSF):
    """
    An elliptical Gaussian given as radio data give their beams.

    Its configuration can instead have a FITS file (a filename, or a dict
    with the filename and the HDU), e.g. of the data, whose header has
    BMAJ, BMIN and BPA (degrees): {type: gauss_beam, file: cube.fits}.

    Parameters
    ----------
    bmaj, bmin : float
        The full widths at half maximum of the major and the minor axes
        (arcsec).
    bpa : float, optional
        The position angle of the major axis.
    """

    VARYING_OPTIONS = dict(bmaj='arcsec', bmin='arcsec', bpa='deg')

    @staticmethod
    def type() -> str:
        return 'gauss_beam'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'PSFGaussBeam':
        if (file := info.pop('file', None)) is not None:
            if any(key in info for key in ('bmaj', 'bmin', 'bpa')):
                raise ConfigError(
                    "give either a file, or bmaj, bmin and bpa")
            with parseutils.config_path('file'):
                file, hdu = parseutils.parse_file(file)
                header = astropy.io.fits.getheader(file, hdu)
            missing = [key for key in ('BMAJ', 'BMIN', 'BPA')
                       if key not in header]
            if missing:
                raise ConfigError(f"{file}: the header has no {missing}")
            info.update(
                bmaj=float(header['BMAJ']) * 3600,
                bmin=float(header['BMIN']) * 3600,
                bpa=float(header['BPA']))
        return super().load(info)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(), bmaj=self._bmaj, bmin=self._bmin, bpa=self._bpa)

    def __init__(self, bmaj: float, bmin: float, bpa: float = 0.0):
        check_scale('bmaj', bmaj)
        check_scale('bmin', bmin)
        if bmin > bmaj:
            raise ConfigError(
                f"bmin must be at most bmaj; they are {bmin} and {bmaj}")
        self._bmaj = bmaj
        self._bmin = bmin
        self._bpa = bpa
        self._gauss = PSFGauss(
            gbkfit.math.gauss_fwhm_to_sigma(bmaj), bmin / bmaj, bpa)

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return self._gauss._size_impl(step)

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        return self._gauss._asarray_impl(step, size, offset, rota)


def _read_images(x):
    """
    The PSF images of a FITS cube (nz, ny, nx; in the orientation of the
    model), the point of each image on the spectral axis (or None if the
    file has no spectral axis), and the size of their pixels in arcsec
    (or None if the file has no celestial axes).
    """
    file, hdu = parseutils.parse_file(x)
    with astropy.io.fits.open(file) as hdul:
        data = np.asarray(hdul[hdu].data, dtype=float)
        wcs = astropy.wcs.WCS(hdul[hdu].header)
    if data.ndim != 3:
        raise ConfigError(
            f"{file}: PSF images must be a cube (nz, ny, nx); it has "
            f"{data.ndim} axes")
    data, wcs = fitsutils.to_model_axes(data, wcs)
    points = _detail.spectral_points(wcs, data.shape[0])
    step = None
    if wcs.has_celestial:
        step = tuple(
            (astropy.wcs.utils.proj_plane_pixel_scales(wcs.celestial)
             * 3600).tolist())
    return data, points, step


class PSFImages(PSF):
    """
    A PSF given as images at points of the spectral axis (e.g. simulated
    by STPSF, or of a star in the data). At each channel it is the image
    PSF (see PSFImage) of the images of the two nearest points,
    interpolated linearly; beyond the points, that of the nearest image.

    Its configuration has its file (a filename, or a dict with the
    filename and the HDU): a FITS cube of the images (nz, ny, nx), with
    its spectral axis last. The points come from its spectral axis unless
    given as wavelength, frequency or velocity (see varying.load_points),
    and the size of the pixels comes from its celestial axes unless step
    is given.

    Parameters
    ----------
    data : ndarray
        The images (nz, ny, nx), finite.
    points : Quantity
        The point of each image: wavelengths, frequencies or velocities.
    step : Sequence of float
        The size of the pixels of the images along x and y (arcsec).
    """

    @staticmethod
    def type() -> str:
        return 'images'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'PSFImages':
        info = dict(info)
        data, points, step = parseutils.load_option(
            _read_images, info, 'file', required=True)
        del info['file']
        if (given := varying.load_points(info)) is not None:
            points = given
        if points is None:
            raise ConfigError(
                "the file has no spectral axis: give the point of each "
                "image as wavelength, frequency or velocity")
        if info.get('step') is None:
            if step is None:
                raise ConfigError(
                    "the file has no celestial axes: give the size of its "
                    "pixels as step")
            info['step'] = step
        return cls(**parseutils.parse_options_for_callable(
            info | dict(data=data, points=points), cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}psf_images.fits'
        astropy.io.fits.writeto(filename, self._data, overwrite=overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            step=list(self._step)) | varying.dump_points(self._points)

    def __init__(
            self,
            data: np.ndarray,
            points: u.Quantity,
            step: Sequence[float]
    ):
        data = np.asarray(data, dtype=float)
        if data.ndim != 3 or not np.all(np.isfinite(data)):
            raise ConfigError(
                f"PSF images must be finite, of shape (nz, ny, nx); their "
                f"shape is {data.shape}")
        points = u.Quantity(points, dtype=float)
        if points.shape != data.shape[:1]:
            raise ConfigError(
                f"PSF images need a point for each of the {len(data)} "
                f"images; they have {points.size}")
        if np.unique(points).size != points.size:
            raise ConfigError("the points of the images must be distinct")
        # (each image holds all the light)
        self._data = data / data.sum(axis=(1, 2), keepdims=True)
        self._points = points
        self._step = tuple(step)

    def varies(self) -> bool:
        return True

    def at(self, points, rest=None):
        return varying.per_point(
            points, lambda points_: self._at(points_, rest))

    def _at(self, points, rest):
        """The image of each of the points (1D)."""
        blends = varying.blend(self._points, points, rest)
        psfs = {}
        for i, j, t in set(blends):
            psfs[(i, j, t)] = PSFImage(
                (1 - t) * self._data[i] + t * self._data[j], self._step)
        return [psfs[blend] for blend in blends]

    def velocity_range(self, rest=None):
        points = varying.to_velocities(self._points, rest)
        return float(points.min()), float(points.max())

    def _size_impl(self, step):
        raise RuntimeError(
            "PSF images give an image at each channel (see at)")

    def _asarray_impl(self, step, size, offset, rota):
        raise RuntimeError(
            "PSF images give an image at each channel (see at)")


class PSFVarying(PSF):
    """
    A PSF of a type whose options vary along the spectral axis: a PSF of
    the type at each channel (see at). It is not drawn itself.
    It is made by the varying of the class of the type (e.g.
    PSFMoffat.varying).
    """

    def type(self) -> str:
        return self._options.cls().type()

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return self._options.dump()

    def __init__(self, options: 'varying._VaryingOptions'):
        self._options = options

    def varies(self) -> bool:
        return True

    def at(self, points, rest=None):
        return varying.per_point(
            points, lambda points_: self._options.at(points_, rest))

    def velocity_range(self, rest=None):
        return self._options.velocity_range(rest)

    def _size_impl(self, step):
        raise RuntimeError(
            "a PSF that varies along the spectral axis has a size at each "
            "channel (see at)")

    def _asarray_impl(self, step, size, offset, rota):
        raise RuntimeError(
            "a PSF that varies along the spectral axis is drawn at each "
            "channel (see at)")


psf_parser = parseutils.TypedParser(PSF, [
    PSFPoint,
    PSFGauss,
    PSFGGauss,
    PSFMoffat,
    PSFImage,
    PSFSum,
    PSFConvolution,
    PSFGaussBeam,
    PSFImages])
