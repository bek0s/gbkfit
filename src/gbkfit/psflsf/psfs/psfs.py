
import os
from collections.abc import Sequence
from typing import Any

import astropy.io.fits
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.special

import gbkfit.math
from gbkfit.psflsf.base import (
    MIN_EXTENT, PSF, WING_FLUX, check_ratio, check_scale, embed, psf_parser)
from gbkfit.utils import fitsutils, gridutils, parseutils


__all__ = [
    'PSFPoint',
    'PSFGauss',
    'PSFGGauss',
    'PSFMoffat',
    'PSFImage',
    'PSFSum',
    'PSFConvolution',
    'PSFBeam'
]


def _create_grid_2d(
        size: tuple[int, int],
        step: tuple[float, float],
        offset: tuple[int, int],
        ratio: float,
        posa: float
) -> np.ndarray:
    center_x = size[0] // 2 + offset[0]
    center_y = size[1] // 2 + offset[1]
    x = (np.array(range(size[0])) - center_x) * step[0]
    y = (np.array(range(size[1])) - center_y) * step[1]
    x = x[None, :]
    y = y[:, None]
    x, y = gbkfit.math.transform_lh_rotate_z(x, y, np.radians(posa))
    return np.sqrt(x * x + y * y / (ratio * ratio))


def _sum_weights(weights, n, desc):
    """
    The weights of the terms of a sum (one for each of n terms), each
    positive, normalised to sum to 1.
    """
    weights = np.asarray(weights, dtype=float)
    if n < 1:
        raise RuntimeError(f"{desc} needs at least one term")
    if weights.shape != (n,) or not np.all(weights > 0):
        raise RuntimeError(
            f"{desc} needs a positive weight for each of its {n} terms; "
            f"its weights are {weights.tolist()}")
    return weights / weights.sum()


def _load_psf_common(cls, info: dict[str, Any]):
    return parseutils.parse_options_for_callable(info, cls.__init__)


class PSFPoint(PSF):
    """
    A point-like Point Spread Function (PSF).

    Represents a PSF where all the energy is concentrated at a single
    point. The output array contains a single nonzero value (1) at the
    center.
    """

    @staticmethod
    def type() -> str:
        return 'point'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFPoint':
        return cls()

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
    A Gaussian Point Spread Function (PSF).
    """


    @staticmethod
    def type() -> str:
        return 'gauss'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFGauss':
        opts = _load_psf_common(cls, info)
        return cls(**opts)

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
    A Generalized Gaussian Point Spread Function (PSF).
    """


    @staticmethod
    def type() -> str:
        return 'ggauss'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs):
        opts = _load_psf_common(cls, info)
        return cls(**opts)

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
   A Moffat Point Spread Function (PSF).
   """


    @staticmethod
    def type():
        return 'moffat'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFMoffat':
        opts = _load_psf_common(cls, info)
        return cls(**opts)

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
            raise RuntimeError(
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
    An PSF defined by an image, loaded from a FITS file.

    The image is resampled based on the provided step size. It is in the
    frame of the pixels of the data, so it is not rotated with the grid.
    """

    @staticmethod
    def type() -> str:
        return 'image'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFImage':
        # Read the image, and its pixel scale in arcsec
        data, coords = parseutils.load_option(
            fitsutils.read_data, info, 'data', required=True)
        info.update(data=data, step=info.get('step', coords.step))
        return cls(**_load_psf_common(cls, info))

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
            data=filename if dump_path else os.path.basename(filename),
            step=self._step)

    def __init__(
            self, data: np.ndarray, step: Sequence[float] = (1.0, 1.0)
    ):
        data = np.squeeze(data)  # Remove singleton dimensions
        if data.ndim != 2:
            raise RuntimeError(
                f"expected a 2D PSF image, but got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise RuntimeError(
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


def _dump_terms(
        psfs: Sequence[PSF], prefix: str, dump_path: bool, overwrite: bool
) -> list[dict[str, Any]]:
    """
    The options of the PSFs of a sum or a convolution, the files of each
    named with its own prefix (term0_, term1_, ...).
    """
    return [
        psf_parser.dump(
            psf, prefix=f'{prefix}term{i}_', dump_path=dump_path,
            overwrite=overwrite)
        for i, psf in enumerate(psfs)]


class PSFSum(PSF):
    """
    A sum of PSFs (e.g. a double Gaussian, or a Gaussian core with Moffat
    wings), each with its fraction of the light: the weights, normalised
    to sum to 1.
    """

    @staticmethod
    def type() -> str:
        return 'sum'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFSum':
        parseutils.load_option_and_update_info(
            psf_parser, info, 'psfs', required=True)
        return cls(**_load_psf_common(cls, info))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            psfs=_dump_terms(self._psfs, prefix, dump_path, overwrite),
            weights=self._weights.tolist())

    def __init__(self, psfs: Sequence[PSF], weights: Sequence[float]):
        self._psfs = tuple(psfs)
        self._weights = _sum_weights(weights, len(self._psfs), "a sum of PSFs")

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
    """

    @staticmethod
    def type() -> str:
        return 'convolution'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFConvolution':
        parseutils.load_option_and_update_info(
            psf_parser, info, 'psfs', required=True)
        return cls(**_load_psf_common(cls, info))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            psfs=_dump_terms(self._psfs, prefix, dump_path, overwrite))

    def __init__(self, psfs: Sequence[PSF]):
        if not psfs:
            raise RuntimeError("a convolution of PSFs needs at least one PSF")
        self._psfs = tuple(psfs)

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
        return embed(data / data.sum(), size, offset)


class PSFBeam(PSF):
    """
    A Gaussian beam, as radio data give it: the full widths at half
    maximum of its major and minor axes (bmaj and bmin, arcsec) and the
    position angle of its major axis (bpa, degrees, north through east).
    Its load can read them from the header of a FITS file (BMAJ, BMIN and
    BPA, in degrees), e.g. of the data: {type: beam, file: cube.fits}.
    """

    @staticmethod
    def type() -> str:
        return 'beam'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFBeam':
        desc = parseutils.make_typed_desc(cls, 'PSF')
        if (file := info.pop('file', None)) is not None:
            if any(key in info for key in ('bmaj', 'bmin', 'bpa')):
                raise parseutils.ConfigError(
                    f"{desc} takes either a file or bmaj, bmin and bpa")
            with parseutils.config_path('file'):
                file, hdu = parseutils.parse_file(file)
                header = astropy.io.fits.getheader(file, hdu)
            missing = [key for key in ('BMAJ', 'BMIN', 'BPA')
                       if key not in header]
            if missing:
                raise parseutils.ConfigError(
                    f"{file}: the header has no {missing}")
            info.update(
                bmaj=float(header['BMAJ']) * 3600,
                bmin=float(header['BMIN']) * 3600,
                bpa=float(header['BPA']))
        return cls(**_load_psf_common(cls, info))

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
            raise RuntimeError(
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
