"""
Point spread functions (PSFs): how the atmosphere, the telescope and the
instrument spread the light of a point on the sky.

Widths are in arcsec, and position angles in degrees, from north through
east, like those of the disks. The options of the analytic PSFs can vary
along the spectral axis (see varying).
"""

import abc
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import astropy.units as u
import astropy.wcs
import astropy.wcs.utils
import numpy as np
import scipy.signal
import scipy.special
from annotated_types import Gt, Le

from gbkfit.config import (
    DataFile, TypedParser, Unit, Varying, check_options, split_data_file)
from gbkfit.io import (
    Beam, header_beam, model_header, parse_wcs, read_header, read_image,
    spectral_points)
import gbkfit.math
from . import _detail
from ._detail import WING_FLUX


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
    'PSFTables',
    'psf_parser'
]


# The options of the analytic PSFs: a width (arcsec), an axis ratio
# (minor over major) and a position angle (degrees, of the major axis)
_Width: TypeAlias = Annotated[float, Gt(0), Unit('arcsec'), Varying()]
_Ratio: TypeAlias = Annotated[float, Gt(0), Le(1), Varying()]
_Angle: TypeAlias = Annotated[float, Unit('deg'), Varying()]

# The shapes of the generalised Gaussian and of the Moffat (for a finite
# flux, beta > 1)
_GGaussBeta: TypeAlias = Annotated[float, Gt(0), Varying()]
_MoffatBeta: TypeAlias = Annotated[float, Gt(1), Varying()]


class PSF(_detail.SpreadFunction):
    """
    A point spread function, drawn on arrays of pixels.
    """

    _TERMS_OPTION = 'psfs'

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
            If the PSF varies along the spectral axis (see at), or its
            wings are too heavy for a finite array.
        """
        self._check_not_varying()
        return _detail.odd_size(self, *self._size_impl(step))

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
            The number of pixels along x and y; by default, size(step).
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
            If the PSF varies along the spectral axis (see at), or a size
            + offset is even.
        """
        self._check_not_varying()
        if size is None:
            size = self.size(step)
        _detail.check_odd_centre(size, offset, "PSF")
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
        PSF drawn on pixels of 1/400 of the width of its array of pixels of
        1 arcsec (0.0025 arcsec, for a PSF narrower than that), as the area
        of a pixel over the peak (the PSF sums to 1).
        """
        width = max(self.size((1.0, 1.0)))
        step = (width / 400, width / 400)
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


def _elliptical_radii(
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
    x = (np.arange(size[0]) - center_x) * step[0]
    y = (np.arange(size[1]) - center_y) * step[1]
    x = x[None, :]
    y = y[:, None]
    x, y = gbkfit.math.transform_lh_rotate_z(x, y, np.radians(posa))
    return np.sqrt(x * x + y * y / (ratio * ratio))


class _AnalyticPSF(_detail.AnalyticSpreadFunction, PSF):
    """
    A PSF of a profile of the elliptical radius r, round or elliptical by
    its axis ratio and the position angle of its major axis, of a scale
    along its major axis (arcsec). Configurations give its scale, or its
    full width at half maximum along its major axis (see from_fwhm), and
    dumps write the one given.

    Parameters
    ----------
    scale : float
        The scale along the major axis (arcsec).
    ratio : float
        The axis ratio (minor over major), in (0, 1].
    posa : float
        The position angle of the major axis.
    """

    def __init__(self, scale: float, ratio: float, posa: float):
        super().__init__(scale)
        self._ratio = ratio
        self._posa = posa

    @staticmethod
    def _tables_class() -> type['PSFTables']:
        return PSFTables

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        extent = self._extent(self._scale)
        return 2 * extent / step[0], 2 * extent / step[1]

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        r = _elliptical_radii(
            size, step, offset, self._ratio, self._posa - rota)
        return self._drawn(r, self._scale)


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
        return _detail.embed(np.ones((1, 1)), size, offset)


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
            sigma: _Width,
            ratio: _Ratio = 1.0,
            posa: _Angle = 0.0
    ):
        check_options(PSFGauss.__init__, locals())
        super().__init__(sigma, ratio, posa)
        self._given_as(sigma=sigma, ratio=ratio, posa=posa)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: _Width,
            ratio: _Ratio = 1.0,
            posa: _Angle = 0.0
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
        sigma = gbkfit.math.gauss_fwhm_to_sigma(fwhm)
        return cls(sigma, ratio, posa)._given_as(
            fwhm=fwhm, ratio=ratio, posa=posa)

    @classmethod
    def from_beam(
            cls,
            bmaj: _Width,
            bmin: _Width,
            bpa: _Angle = 0.0
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
        return cls(
            gbkfit.math.gauss_fwhm_to_sigma(bmaj), bmin / bmaj, bpa
        )._given_as(bmaj=bmaj, bmin=bmin, bpa=bpa)

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
        fwhm = float(gbkfit.math.gauss_sigma_to_fwhm(self._scale))
        return Beam(fwhm, fwhm * self._ratio, self._posa)

    def _beam_area_impl(self) -> float:
        # (2 pi sigma^2 times the axis ratio)
        return 2 * np.pi * self._scale ** 2 * self._ratio

    def _wings(self) -> float:
        # (the wings of a 2D Gaussian beyond r hold exp(-r^2 / 2 sigma^2))
        return np.sqrt(-2 * np.log(WING_FLUX))

    def _profile(self, r: np.ndarray, scale: float) -> np.ndarray:
        return gbkfit.math.gauss_1d_fun(r, 1, 0, scale)


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
            alpha: _Width,
            beta: _GGaussBeta,
            ratio: _Ratio = 1.0,
            posa: _Angle = 0.0
    ):
        check_options(PSFGGauss.__init__, locals())
        super().__init__(alpha, ratio, posa)
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta, ratio=ratio, posa=posa)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: _Width,
            beta: _GGaussBeta,
            ratio: _Ratio = 1.0,
            posa: _Angle = 0.0
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
        alpha = gbkfit.math.ggauss_fwhm_to_alpha(fwhm, beta)
        return cls(alpha, beta, ratio, posa)._given_as(
            fwhm=fwhm, beta=beta, ratio=ratio, posa=posa)

    def _beam_area_impl(self) -> float:
        # (2 pi alpha^2 Gamma(2 / beta) / beta times the axis ratio)
        return float(
            2 * np.pi * self._scale ** 2 * scipy.special.gamma(2 / self._beta)
            / self._beta * self._ratio)

    def _wings(self) -> float:
        # (the flux of a 2D ggauss within r is P(2 / beta, (r / alpha)^
        # beta), P the regularized lower incomplete gamma function)
        return scipy.special.gammaincinv(
            2 / self._beta, 1 - WING_FLUX) ** (1 / self._beta)

    def _profile(self, r: np.ndarray, scale: float) -> np.ndarray:
        return gbkfit.math.ggauss_1d_fun(r, 1, 0, scale, self._beta)


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
            alpha: _Width,
            beta: _MoffatBeta,
            ratio: _Ratio = 1.0,
            posa: _Angle = 0.0
    ):
        check_options(PSFMoffat.__init__, locals())
        super().__init__(alpha, ratio, posa)
        self._beta = beta
        self._given_as(alpha=alpha, beta=beta, ratio=ratio, posa=posa)

    @classmethod
    def from_fwhm(
            cls,
            fwhm: _Width,
            beta: _MoffatBeta,
            ratio: _Ratio = 1.0,
            posa: _Angle = 0.0
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
        alpha = gbkfit.math.moffat_fwhm_to_alpha(fwhm, beta)
        return cls(alpha, beta, ratio, posa)._given_as(
            fwhm=fwhm, beta=beta, ratio=ratio, posa=posa)

    def _beam_area_impl(self) -> float:
        # (pi alpha^2 / (beta - 1) times the axis ratio)
        return np.pi * self._scale ** 2 / (self._beta - 1) * self._ratio

    def _wings(self) -> float:
        # (the wings of a 2D Moffat beyond r hold (1 + (r / alpha)^2)^(1 -
        # beta))
        return np.sqrt(WING_FLUX ** (1 / (1 - self._beta)) - 1)

    def _profile(self, r: np.ndarray, scale: float) -> np.ndarray:
        return gbkfit.math.moffat_1d_fun(r, 1, 0, scale, self._beta)


def _read_images(
        file: DataFile,
        step: _detail.Step | None,
        spectral_axis: int | None = None
) -> tuple[str, np.ndarray, astropy.wcs.WCS, _detail.Step]:
    """
    The PSF image or images of a file in the orientation of the model, as
    data are read (see model_header), with the filename, their world
    coordinates in it, and the size of their pixels: step, or by default
    that of their celestial axes (arcsec). An image is without its axes of
    length 1, and images (nz, ny, nx) have their spectral axis, if any, at
    the axis spectral_axis.

    Raise ValueError if step is not given and there are no celestial axes.
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    header, orientation = model_header(
        filename, header, data.shape, spectral_axis=spectral_axis)
    data = orientation.apply(np.asarray(data, dtype=float))
    wcs = parse_wcs(filename, header)
    if step is None:
        if not wcs.has_celestial:
            raise ValueError(
                f"{filename}: the header has no celestial axes; give the "
                f"size of the pixels (step)")
        step = tuple(
            (astropy.wcs.utils.proj_plane_pixel_scales(wcs.celestial)
             * 3600).tolist())
    return filename, data, wcs, step


def _as_step(step: _detail.Step) -> tuple[float, float]:
    """
    The size of the pixels of PSF images along x and y (see Step).

    Raise ValueError unless it is one positive number, or two.
    """
    pair = ((float(step),) * 2 if np.isscalar(step)
            else tuple(float(x) for x in step))
    if len(pair) != 2 or not all(x > 0 for x in pair):
        raise ValueError(
            f"the size of the pixels of PSF images (step) is a positive "
            f"number, or two (along x and y); it is {step}")
    return pair


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
        The image; it holds light.
    step : float or Sequence of float
        The size of its pixels along x and y (arcsec); of one number,
        square pixels.
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
            cls, file: DataFile, step: _detail.Step | None = None
    ) -> 'PSFImage':
        """
        Read a PSF image from a file, in the orientation of the model, as
        data are (see model_header), without its axes of length 1.

        Parameters
        ----------
        file : DataFile
            The file of the image, and where it is in it (see read_image).
        step : float or Sequence of float, optional
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
        filename, data, _, step = _read_images(file, step)
        if data.ndim != 2:
            raise ValueError(
                f"{filename}: a PSF image has two axes; it has {data.ndim}"
                + (" (a cube of images at points of the spectral axis is a "
                   "PSF of type images)" if data.ndim == 3 else ""))
        return cls(data, step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_image(
            self, self._data, 'psf', prefix, dump_path, overwrite) | dict(
            step=list(self._step))

    def __init__(self, data: np.ndarray, step: _detail.Step):
        check_options(PSFImage.__init__, locals())
        data = np.squeeze(np.asarray(data, dtype=float))
        if data.ndim != 2:
            raise ValueError(
                f"a PSF image has two axes (those of length 1 left out); its "
                f"shape is {data.shape}")
        data = np.where(np.isfinite(data), data, 0)
        if not np.sum(data) > 0:
            raise ValueError(
                f"a PSF image must hold light; the sum of its pixels is "
                f"{np.sum(data):g}")
        self._data = data
        self._step = _as_step(step)

    def _beam_area_impl(self) -> float:
        # (the sum of its pixels times their area, over its peak)
        return float(
            np.sum(self._data) * self._step[0] * self._step[1]
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
        step_ratio = (step[0] / self._step[0], step[1] / self._step[1])
        return _detail.resample(self._data, step_ratio, size, offset)


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
        The images (nz, ny, nx); each holds light.
    points : Quantity
        The point of each image: wavelengths, frequencies or velocities.
    step : float or Sequence of float
        The size of the pixels of the images along x and y (arcsec); of
        one number, square pixels.
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
            step: _detail.Step | None = None
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
        step : float or Sequence of float, optional
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
        filename, data, wcs, step = _read_images(file, step, spectral_axis=2)
        if data.ndim != 3:
            raise ValueError(
                f"{filename}: PSF images must be a cube (nz, ny, nx); it has "
                f"{data.ndim} axes"
                + (" (one image is a PSF of type image)" if data.ndim == 2
                   else ""))
        if points is None:
            points = spectral_points(wcs, data.shape[0])
            if points is None:
                raise ValueError(
                    f"{filename}: the cube has no spectral axis; give the "
                    f"points of its images (points)")
        return cls(data, points, step)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_image(
            self, self._data, 'psf_images', prefix, dump_path, overwrite
        ) | dict(points=self._dump_points(), step=list(self._step))

    def __init__(
            self, data: np.ndarray, points: u.Quantity, step: _detail.Step
    ):
        check_options(PSFImages.__init__, locals())
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            raise ValueError(
                f"PSF images must be of shape (nz, ny, nx); their shape is "
                f"{data.shape}")
        super().__init__(
            np.where(np.isfinite(data), data, 0), points, "PSF images")
        self._step = _as_step(step)

    def _of_image(self, data: np.ndarray) -> PSFImage:
        return PSFImage(data, self._step)


class PSFSum(_detail.SpreadFunctionSum, PSF):
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

    def __init__(self, psfs: Sequence[PSF], weights: Sequence[float]):
        check_options(PSFSum.__init__, locals())
        super().__init__(psfs, weights, "a sum of PSFs")

    def _beam_area_impl(self) -> float:
        # (its terms, each of an integral of 1, peak at 1 over their beam
        # areas)
        return float(np.sum(self._weights) / sum(
            weight / psf.beam_area()
            for weight, psf in zip(self._weights, self._terms)))

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        sizes = np.array([psf.size(step) for psf in self._terms])
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


class PSFConvolution(_detail.SpreadFunctionConvolution, PSF):
    """
    The convolution of PSFs: e.g. the PSF of an adaptive optics system
    and the seeing, or a PSF and the response of the pixels.

    Parameters
    ----------
    psfs : Sequence of PSF
        The PSFs, at least one.
    """

    def __init__(self, psfs: Sequence[PSF]):
        check_options(PSFConvolution.__init__, locals())
        super().__init__(psfs, "a convolution of PSFs")

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
            data = scipy.signal.convolve(data, psf.asarray(step, rota=rota))
        return _detail.embed(data / data.sum(), size, offset)


class PSFTables(_detail.SpreadFunctionTables, PSF):
    """
    A PSF of an analytic type whose options vary along the spectral axis,
    as tables of their values: at each point, a PSF of the type (see at).
    It is not drawn itself. It is made by the varying of the class of the
    type (e.g. PSFMoffat.varying).
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
