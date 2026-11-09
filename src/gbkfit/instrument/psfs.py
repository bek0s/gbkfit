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
from typing import Annotated, Any

import astropy.units as u
import astropy.wcs
import astropy.wcs.utils
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.special
from annotated_types import Gt, Le

from gbkfit.config import (
    DataFile, TypedParser, Unit, Varying, check_options, split_data_file)
from gbkfit.grid import Coords
from gbkfit.io import (
    Beam, header_beam, model_header, parse_wcs, read_header, read_image,
    write_data, write_image)
import gbkfit.math
from . import _detail
from ._detail import MIN_EXTENT, WING_FLUX


__all__ = [
    'PSF',
    'PSFConvolution',
    'PSFGGauss',
    'PSFGauss',
    'PSFImage',
    'PSFImages',
    'PSFMoffat',
    'PSFPoint',
    'PSFSum',
    'PSFVarying',
    'psf_parser'
]


class PSF(_detail.SpreadFunction):
    """
    A point spread function, drawn on arrays of pixels.
    """

    @staticmethod
    def _varying_class() -> type['PSFVarying']:
        return PSFVarying

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
            Coords): a PSF of position angle posa is drawn at
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

    def beam_area(self) -> float:
        """
        Return the solid angle of its beam (arcsec^2), as of the unit of
        light per beam (e.g. Jy/beam): its integral over its peak, of the
        PSF itself (exact for the analytic ones).

        Returns
        -------
        float
            The solid angle.

        Raises
        ------
        ValueError
            If it has no beam: a point, or a PSF that varies (each point
            of the spectral axis has its own; see at).
        """
        if self.varies():
            raise ValueError(
                "a PSF that varies has a beam at each point of the spectral "
                "axis (see PSF.at)")
        return self._beam_area_impl()

    def _beam_area_impl(self) -> float:
        """
        Return the solid angle of its beam (see beam_area): here, of the
        PSF drawn on pixels of about 1/400 of its extent, as the area of a
        pixel over the peak (the PSF sums to 1).
        """
        extent = max(self.size((1.0, 1.0)))
        step = (extent / 400, extent / 400)
        return float(step[0] * step[1] / np.max(self.asarray(step)))

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


class _AnalyticPSF(_detail.GivenOptions, PSF):
    """
    A PSF of a profile of the elliptical radius r, round or elliptical by
    its axis ratio and the position angle of its major axis, drawn out to
    its extent (see MIN_EXTENT). Configurations give its scale, or its
    full width at half maximum along its major axis (see from_fwhm), and
    dumps write the one given.

    Parameters
    ----------
    ratio : float
        The axis ratio (minor over major), in (0, 1].
    posa : float
        The position angle of the major axis.
    """

    config_constructors = ('__init__', 'from_fwhm')

    def __init__(self, ratio: float, posa: float):
        self._ratio = ratio
        self._posa = posa

    @abc.abstractmethod
    def _extent(self) -> float:
        """The radius out to which it is drawn (see MIN_EXTENT)."""

    @abc.abstractmethod
    def _profile(self, r: np.ndarray) -> np.ndarray:
        """Its profile at the radii r (arcsec), to any factor."""

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
        data = self._profile(r)
        data[r > self._extent()] = 0
        return data / np.sum(data)


class PSFPoint(PSF):
    """
    No blurring: the light stays in its pixel.
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

    def _beam_area_impl(self) -> float:
        raise ValueError(
            "a point PSF has no beam: a unit of light per beam needs a PSF "
            "of a width")

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


class PSFGauss(_AnalyticPSF):
    """
    A Gaussian, round or elliptical: by its sigma, its full width at half
    maximum, or those of its axes, as of radio beams (also of the header
    of a file).

    Its profile is exp(-r^2 / 2 sigma^2). Configurations give its sigma,
    its FWHM (see from_fwhm), its beam (see from_beam), or a file whose
    header has its beam (see from_file: e.g. {type: gauss, file:
    cube.fits}).

    Parameters
    ----------
    sigma : float
        The standard deviation along the major axis (arcsec).
    ratio : float, optional
        The axis ratio (minor over major), in (0, 1].
    posa : float, optional
        The position angle of the major axis.
    """

    config_constructors = ('__init__', 'from_fwhm', 'from_beam', 'from_file')

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

    def __init__(
            self,
            sigma: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            ratio: Annotated[float, Gt(0), Le(1), Varying()] = 1.0,
            posa: Annotated[float, Unit('deg'), Varying()] = 0.0
    ):
        check_options(PSFGauss.__init__, locals())
        super().__init__(ratio, posa)
        self._sigma = sigma
        self._given_as(sigma=sigma, ratio=ratio, posa=posa)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            ratio: Annotated[float, Gt(0), Le(1), Varying()] = 1.0,
            posa: Annotated[float, Unit('deg'), Varying()] = 0.0
    ) -> 'PSFGauss':
        """
        Make the Gaussian of a full width at half maximum along its major
        axis: of sigma = FWHM / (2 sqrt(2 ln 2)).

        Parameters
        ----------
        fwhm : float
            The full width at half maximum along the major axis (arcsec).
        ratio : float, optional
            The axis ratio (minor over major), in (0, 1].
        posa : float, optional
            The position angle of the major axis.

        Returns
        -------
        PSFGauss
            The PSF.
        """
        check_options(PSFGauss.from_fwhm, locals())
        return cls(gbkfit.math.gauss_fwhm_to_sigma(fwhm), ratio, posa) \
            ._given_as(fwhm=fwhm, ratio=ratio, posa=posa)

    @classmethod
    def from_beam(
            cls,
            bmaj: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            bmin: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            bpa: Annotated[float, Unit('deg'), Varying()] = 0.0
    ) -> 'PSFGauss':
        """
        Make the Gaussian of a radio beam: by the full widths at half
        maximum of its axes and the angle of its major axis, as in the
        headers of radio FITS files (BMAJ, BMIN and BPA).

        Parameters
        ----------
        bmaj, bmin : float
            The full widths at half maximum of the major and the minor
            axes (arcsec).
        bpa : float, optional
            The position angle of the major axis.

        Returns
        -------
        PSFGauss
            The PSF.

        Raises
        ------
        ValueError
            If bmin is above bmaj.
        """
        check_options(PSFGauss.from_beam, locals())
        if bmin > bmaj:
            raise ValueError(
                f"bmin must be at most bmaj; they are {bmin} and {bmaj}")
        return cls(gbkfit.math.gauss_fwhm_to_sigma(bmaj), bmin / bmaj, bpa) \
            ._given_as(bmaj=bmaj, bmin=bmin, bpa=bpa)

    @classmethod
    def from_file(cls, file: DataFile) -> 'PSFGauss':
        """
        Read the beam of a file, e.g. of the data, as datasets read it (see
        header_beam): that of its header (BMAJ, BMIN and BPA, in degrees),
        or the last that AIPS wrote in its history.

        Parameters
        ----------
        file : DataFile
            The file of the header, and where it is in it (see
            read_header).

        Returns
        -------
        PSFGauss
            The PSF (see from_beam).

        Raises
        ------
        ValueError
            If the header has no beam, or the file has a beam for each
            channel of its cube (a table BEAMS of CASA).
        """
        check_options(PSFGauss.from_file, locals())
        filename = split_data_file(file)[0]
        beam = header_beam(read_header(file), filename)
        if beam is None:
            raise ValueError(
                f"{filename}: the header has no beam (BMAJ and BMIN, or one "
                f"in the history of AIPS)")
        if beam.spectral is not None:
            raise ValueError(
                f"{filename}: the cube has a beam for each channel (its "
                f"table BEAMS): give them as a table of the options (prep "
                f"psf_casa_beams)")
        return cls.from_beam(beam.bmaj, beam.bmin, beam.bpa)

    def beam(self) -> Beam:
        """
        Return it as a radio beam: the full widths at half maximum of its
        axes (arcsec) and the position angle of its major axis.

        Returns
        -------
        Beam
            The beam.
        """
        fwhm = float(self._sigma * np.sqrt(8 * np.log(2)))
        return Beam(fwhm, fwhm * self._ratio, self._posa)

    def _beam_area_impl(self) -> float:
        # (2 pi sigma^2 times the axis ratio)
        return 2 * np.pi * self._sigma ** 2 * self._ratio

    def _extent(self) -> float:
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        wings of a 2D Gaussian beyond r hold exp(-r^2 / 2 sigma^2).
        """
        return max(
            MIN_EXTENT * self._sigma,
            self._sigma * np.sqrt(-2 * np.log(WING_FLUX)))

    def _profile(self, r: np.ndarray) -> np.ndarray:
        return gbkfit.math.gauss_1d_fun(r, 1, 0, self._sigma)


class PSFGGauss(_AnalyticPSF):
    """
    A Gaussian with a shape parameter beta: flat-topped above 2, peaked
    below 2, exponential at 1; by its scale or its full width at half
    maximum.

    Its profile is exp(-(r / alpha)^beta), round or elliptical.
    Configurations give its scale, or its FWHM (see from_fwhm).

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

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def type_title() -> str:
        return 'Generalised Gaussian'

    def __init__(
            self,
            alpha: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            beta: Annotated[float, Gt(0), Varying()],
            ratio: Annotated[float, Gt(0), Le(1), Varying()] = 1.0,
            posa: Annotated[float, Unit('deg'), Varying()] = 0.0
    ):
        check_options(PSFGGauss.__init__, locals())
        super().__init__(ratio, posa)
        self._alpha = alpha
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta, ratio=ratio, posa=posa)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            beta: Annotated[float, Gt(0), Varying()],
            ratio: Annotated[float, Gt(0), Le(1), Varying()] = 1.0,
            posa: Annotated[float, Unit('deg'), Varying()] = 0.0
    ) -> 'PSFGGauss':
        """
        Make the generalised Gaussian of a full width at half maximum
        along its major axis: of alpha = FWHM / (2 (ln 2)^(1 / beta)).

        Parameters
        ----------
        fwhm : float
            The full width at half maximum along the major axis (arcsec).
        beta : float
            The shape, > 0.
        ratio : float, optional
            The axis ratio (minor over major), in (0, 1].
        posa : float, optional
            The position angle of the major axis.

        Returns
        -------
        PSFGGauss
            The PSF.
        """
        check_options(PSFGGauss.from_fwhm, locals())
        alpha = fwhm / (2 * np.log(2) ** (1 / beta))
        return cls(alpha, beta, ratio, posa)._given_as(
            fwhm=fwhm, beta=beta, ratio=ratio, posa=posa)

    def _beam_area_impl(self) -> float:
        # (2 pi alpha^2 Gamma(2 / beta) / beta times the axis ratio)
        return float(
            2 * np.pi * self._alpha ** 2 * scipy.special.gamma(2 / self._beta)
            / self._beta * self._ratio)

    def _extent(self) -> float:
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        flux of a 2D ggauss within r is P(2 / beta, (r / alpha)^beta),
        P the regularized lower incomplete gamma function.
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * scipy.special.gammaincinv(
                2 / self._beta, 1 - WING_FLUX) ** (1 / self._beta))

    def _profile(self, r: np.ndarray) -> np.ndarray:
        return gbkfit.math.ggauss_1d_fun(r, 1, 0, self._alpha, self._beta)


class PSFMoffat(_AnalyticPSF):
    """
    The usual model of atmospheric seeing: a core with power-law wings; by
    its scale or its full width at half maximum.

    Its profile is (1 + (r / alpha)^2)^(-beta), round or elliptical.
    Configurations give its scale, or its FWHM (see from_fwhm).

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

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def type_title() -> str:
        return 'Moffat'

    def __init__(
            self,
            alpha: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            beta: Annotated[float, Gt(1), Varying()],
            ratio: Annotated[float, Gt(0), Le(1), Varying()] = 1.0,
            posa: Annotated[float, Unit('deg'), Varying()] = 0.0
    ):
        check_options(PSFMoffat.__init__, locals())
        super().__init__(ratio, posa)
        self._alpha = alpha
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta, ratio=ratio, posa=posa)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: Annotated[float, Gt(0), Unit('arcsec'), Varying()],
            beta: Annotated[float, Gt(1), Varying()],
            ratio: Annotated[float, Gt(0), Le(1), Varying()] = 1.0,
            posa: Annotated[float, Unit('deg'), Varying()] = 0.0
    ) -> 'PSFMoffat':
        """
        Make the Moffat of a full width at half maximum along its major
        axis (e.g. of the seeing): of alpha = FWHM / (2 sqrt(2^(1 / beta)
        - 1)).

        Parameters
        ----------
        fwhm : float
            The full width at half maximum along the major axis (arcsec).
        beta : float
            The slope of the wings, > 1.
        ratio : float, optional
            The axis ratio (minor over major), in (0, 1].
        posa : float, optional
            The position angle of the major axis.

        Returns
        -------
        PSFMoffat
            The PSF.
        """
        check_options(PSFMoffat.from_fwhm, locals())
        alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
        return cls(alpha, beta, ratio, posa)._given_as(
            fwhm=fwhm, beta=beta, ratio=ratio, posa=posa)

    def _beam_area_impl(self) -> float:
        # (pi alpha^2 / (beta - 1) times the axis ratio)
        return np.pi * self._alpha ** 2 / (self._beta - 1) * self._ratio

    def _extent(self) -> float:
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        wings of a 2D Moffat beyond r hold (1 + (r / alpha)^2)^(1 - beta).
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * np.sqrt(
                WING_FLUX ** (1 / (1 - self._beta)) - 1))

    def _profile(self, r: np.ndarray) -> np.ndarray:
        return gbkfit.math.moffat_1d_fun(r, 1, 0, self._alpha, self._beta)


class PSFImage(PSF):
    """
    A measured PSF, from an image.

    It is centred on the centre of the image, and resampled to the pixels
    it is drawn on (with splines of order 5). Pixels of the image that
    are not finite (e.g. beyond the field of the star it was measured
    on) hold no light.

    The image is in the frame of the pixels of the data, so it is not
    rotated with them. Configurations give its file (see from_file).

    Parameters
    ----------
    data : ndarray
        The image.
    step : Sequence of float, optional
        The size of the pixels of the image along x and y (arcsec).
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
            step: Annotated[Sequence[float], Unit('arcsec')] | None = None
    ) -> 'PSFImage':
        """
        Read a PSF image from a file, in the orientation of the model, as
        data are (see model_header), without its axes of length 1.

        Parameters
        ----------
        file : DataFile
            The file of the image, and where it is in it (see read_image).
        step : Sequence of float, optional
            The size of the pixels of the image along x and y (arcsec); by
            default, that of the celestial axes of the header.

        Returns
        -------
        PSFImage
            The PSF.

        Raises
        ------
        ValueError
            If the file has no image of two axes, its header has invalid
            world coordinates (see model_header), or step is not given and
            the header has no celestial axes.
        """
        check_options(PSFImage.from_file, locals())
        filename = split_data_file(file)[0]
        data, wcs = _read_images(file)
        if data.ndim != 2:
            raise ValueError(
                f"{filename}: a PSF image has two axes; it has {data.ndim}")
        return cls(data, _pixel_size(filename, wcs) if step is None else step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}psf.fits'
        rpix = tuple(np.array(self._data.shape[::-1]) / 2 - 0.5)
        coords = Coords(tuple(self._step), rpix, (0.0, 0.0), 0.0)
        write_data(filename, self._data, coords, overwrite=overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            step=self._step)

    def __init__(
            self,
            data: np.ndarray,
            step: Annotated[Sequence[float], Unit('arcsec')] = (1.0, 1.0)
    ):
        check_options(PSFImage.__init__, locals())
        data = np.squeeze(np.asarray(data, dtype=float))
        if data.ndim != 2:
            raise ValueError(
                f"a PSF image has two axes (those of length 1 left out); its "
                f"shape is {data.shape}")
        step = tuple(float(x) for x in step)
        if len(step) != 2 or not all(x > 0 for x in step):
            raise ValueError(
                f"the size of the pixels of a PSF image (step) must be two "
                f"positive numbers; it is {step}")
        self._data = np.where(np.isfinite(data), data, 0)
        self._step = step

    def _beam_area_impl(self) -> float:
        # (the sum of its pixels times their area, over its peak)
        return float(
            np.sum(self._data) * abs(self._step[0] * self._step[1])
            / np.max(self._data))

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


class PSFSum(_detail.SpreadFunctionTerms, PSF):
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

    @staticmethod
    def type_title() -> str:
        return 'Sum'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            psfs=_detail.dump_terms(
                psf_parser, self._terms, prefix, dump_path, overwrite),
            weights=self._weights.tolist())

    def __init__(self, psfs: Sequence[PSF], weights: Sequence[float]):
        check_options(PSFSum.__init__, locals())
        self._terms = tuple(psfs)
        self._weights = _detail.sum_weights(
            weights, len(self._terms), "a sum of PSFs")

    def _beam_area_impl(self) -> float:
        # (its terms, each of an integral of 1, peak at 1 over their beam
        # areas)
        return float(np.sum(self._weights) / sum(
            weight / psf.beam_area()
            for weight, psf in zip(self._weights, self._terms)))

    def _of_terms(self, terms: Sequence[PSF]) -> 'PSFSum':
        return PSFSum(terms, self._weights)

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        sizes = np.array([psf._size_impl(step) for psf in self._terms])
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
            for weight, psf in zip(self._weights, self._terms))


class PSFConvolution(_detail.SpreadFunctionTerms, PSF):
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

    @staticmethod
    def type_title() -> str:
        return 'Convolution'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            psfs=_detail.dump_terms(
                psf_parser, self._terms, prefix, dump_path, overwrite))

    def __init__(self, psfs: Sequence[PSF]):
        check_options(PSFConvolution.__init__, locals())
        if not psfs:
            raise ValueError("a convolution of PSFs needs at least one PSF")
        self._terms = tuple(psfs)

    def _of_terms(self, terms: Sequence[PSF]) -> 'PSFConvolution':
        return PSFConvolution(terms)

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        # The sum of the (odd) sizes of the terms, less one for each
        # convolution
        sizes = np.array([psf.size(step) for psf in self._terms])
        return tuple((sizes.sum(axis=0) - len(sizes) + 1).tolist())

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        data = np.ones((1, 1))
        for psf in self._terms:
            data = scipy.signal.convolve(
                data, psf.asarray(step, rota=rota), method='auto')
        return _detail.embed(data / data.sum(), size, offset)


def _read_images(
        file: DataFile, spectral_axis: int | None = None
) -> tuple[np.ndarray, astropy.wcs.WCS]:
    """
    The PSF image or images of a file in the orientation of the model, as
    data are read (see model_header), and their world coordinates in it:
    an image without its axes of length 1, or images (nz, ny, nx) whose
    spectral axis, if any, is the axis spectral_axis.
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    header, orientation = model_header(
        filename, header, data.shape, spectral_axis=spectral_axis)
    data = orientation.apply(np.asarray(data, dtype=float))
    return data, parse_wcs(filename, header)


def _pixel_size(source: str, wcs: astropy.wcs.WCS) -> tuple[float, float]:
    """
    The size of the pixels of PSF images along x and y (arcsec), from
    the celestial axes of their world coordinates.

    Raise ValueError if they have none.
    """
    if not wcs.has_celestial:
        raise ValueError(
            f"{source}: the header has no celestial axes; give the size of "
            f"the pixels (step)")
    return tuple(
        (astropy.wcs.utils.proj_plane_pixel_scales(wcs.celestial)
         * 3600).tolist())


class PSFImages(_detail.SpreadFunctionImages, PSF):
    """
    A measured PSF that changes with wavelength, from a cube.

    It is given as images at points of the spectral axis (e.g. simulated
    by STPSF, or of a star in the data). At each channel it is the image
    PSF (see PSFImage) of the images of the two nearest points,
    interpolated linearly; beyond the points, that of the nearest image.
    Pixels of the images that are not finite hold no light.

    Configurations give its file, a cube of the images (see from_file).

    Parameters
    ----------
    data : ndarray
        The images (nz, ny, nx).
    points : Quantity
        The point of each image: wavelengths, frequencies or velocities.
    step : Sequence of float
        The size of the pixels of the images along x and y (arcsec).
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
            step: Annotated[Sequence[float], Unit('arcsec')] | None = None
    ) -> 'PSFImages':
        """
        Read PSF images from a cube (nz, ny, nx), in the orientation of
        the model, as data are (see model_header): its spectral axis last.

        Parameters
        ----------
        file : DataFile
            The file of the cube, and where it is in it (see read_image).
        points : Quantity, optional
            The point of each image: wavelengths, frequencies or velocities;
            by default, those of the spectral axis of the cube.
        step : Sequence of float, optional
            The size of the pixels of the images along x and y (arcsec); by
            default, that of the celestial axes of the cube.

        Returns
        -------
        PSFImages
            The PSF.

        Raises
        ------
        ValueError
            If the file has no cube, its header has invalid world
            coordinates (see model_header), or the points or step are
            neither given nor in its header.
        """
        check_options(PSFImages.from_file, locals())
        filename = split_data_file(file)[0]
        data, wcs = _read_images(file, spectral_axis=2)
        if data.ndim != 3:
            raise ValueError(
                f"{filename}: PSF images must be a cube (nz, ny, nx); it has "
                f"{data.ndim} axes")
        if points is None:
            points = _detail.spectral_points(wcs, data.shape[0])
            if points is None:
                raise ValueError(
                    f"{filename}: the cube has no spectral axis; give the "
                    f"points of its images (points)")
        if step is None:
            step = _pixel_size(filename, wcs)
        return cls(data, points, step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}psf_images.fits'
        write_image(filename, self._data, overwrite=overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            points=self._dump_points(),
            step=list(self._step))

    def __init__(
            self,
            data: np.ndarray,
            points: u.Quantity,
            step: Annotated[Sequence[float], Unit('arcsec')]
    ):
        check_options(PSFImages.__init__, locals())
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            raise ValueError(
                f"PSF images must be of shape (nz, ny, nx); their shape is "
                f"{data.shape}")
        super().__init__(
            np.where(np.isfinite(data), data, 0), points, "PSF images")
        self._step = tuple(step)

    def _image(self, data: np.ndarray) -> PSFImage:
        return PSFImage(data, self._step)


class PSFVarying(_detail.VaryingSpreadFunction, PSF):
    """
    A PSF of a type whose options vary along the spectral axis: a PSF of
    the type at each channel (see at). It is not drawn itself.
    It is made by the varying of the class of the type (e.g.
    PSFMoffat.varying).
    """


psf_parser = TypedParser(PSF, [
    PSFPoint,
    PSFGauss,
    PSFGGauss,
    PSFMoffat,
    PSFImage,
    PSFSum,
    PSFConvolution,
    PSFImages])
