"""
The density polar traits of grids of cells in the plane of a disk: a free
map (RPTraitMap), and the light of an observed image (RPTraitImage).
"""

import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.special

from gbkfit.instrument import PSF, psf_parser
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import ImageInput, RPTrait


__all__ = [
    'RPTraitImage',
    'RPTraitMap',
    'map_cell_centres'
]


# Their uid in the native kernels
RPT_UID_MAP = 151


def _check_map_grid(
        size: Sequence[int], step: float
) -> tuple[tuple[int, int], float]:
    """
    Return the numbers of cells (nx, ny) and the side (arcsec) of the grid
    of a map trait (see RPTraitMap), or raise ConfigError if they are not
    valid.
    """
    if not (isinstance(size, Sequence) and len(size) == 2 and all(
            isinstance(n, numbers.Integral) and not isinstance(n, bool)
            and n >= 1 for n in size)):
        raise ConfigError(
            f"size must be the numbers of cells along x and y, [nx, "
            f"ny], each at least 1; it is {size!r}")
    if isinstance(step, bool) or not (
            isinstance(step, numbers.Real) and 0 < step < np.inf):
        raise ConfigError(
            f"step must be a positive length; it is {step!r}")
    return (int(size[0]), int(size[1])), float(step)


def map_cell_centres(
        size: tuple[int, int], step: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the centres (x, y; arcsec in the plane of the disk) of the
    cells of the grid of a map trait of the given numbers of cells (nx,
    ny) and side (see RPTraitMap), each of shape (ny, nx).
    """
    nx, ny = size
    j, i = np.mgrid[0:ny, 0:nx]
    return (i + 0.5 - nx / 2) * step, (j + 0.5 - ny / 2) * step


def _map_azimuthal_edges(
        size: tuple[int, int], step: float, r: np.ndarray
) -> np.ndarray:
    """
    Return the azimuths (degrees) where the circles of the radii r cross
    the edges of the cells of the grid of a map trait (see
    RPTraitMap.azimuthal_edges); NaN where they do not.
    """
    # The circles cross the lines x = xe at the azimuths +-acos(xe / r),
    # and the lines y = ye at asin(ye / r) and 180 - asin(ye / r); jumps
    # where they cross them on the grid
    nx, ny = size
    half_x, half_y = nx * step / 2, ny * step / 2
    xe = np.arange(nx + 1) * step - half_x
    ye = np.arange(ny + 1) * step - half_y
    r = np.asarray(r, dtype=float)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        ax = np.rad2deg(np.arccos(xe / r))
        ay = np.rad2deg(np.arcsin(ye / r))
        cross_x = np.concatenate((ax, -ax), axis=1)
        cross_y = np.concatenate((ay, 180 - ay), axis=1)
        on_x = np.abs(r * np.sin(np.deg2rad(cross_x))) <= half_y
        on_y = np.abs(r * np.cos(np.deg2rad(cross_y))) <= half_x
    return np.concatenate(
        (np.where(on_x, cross_x, np.nan), np.where(on_y, cross_y, np.nan)),
        axis=1)


class RPTraitMap(RPTrait):
    """
    A free map: a surface density of nx by ny square cells of side step
    (arcsec) in the plane of the disk, centred on its centre, x along the
    azimuth 0 and y along the azimuth 90 degrees (see the module
    docstring). The cell (i, j) covers [(i - nx / 2) step, (i + 1 - nx /
    2) step) along x and likewise along y, and has the density a[j nx +
    i], of any sign, all over it; 0 outside the grid. Its parameter: a,
    the nx ny densities, row by row (as a numpy image of shape (ny, nx),
    x along its columns). A warped disk has the map in the plane of each
    of its rings, as it has its blobs.

    Parameters
    ----------
    size : Sequence of int
        The number of cells along x and along y, [nx, ny].
    step : float
        The side of the cells (arcsec).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'map'

    @staticmethod
    def uid() -> int:
        return RPT_UID_MAP

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(size=list(self._size), step=self._step)

    def __init__(
            self,
            size: Sequence[int],
            step: float,
            name: str | None = None
    ):
        super().__init__(name=name)
        self._size, self._step = _check_map_grid(size, step)

    def size(self) -> tuple[int, int]:
        """Return the numbers of cells along x and y, (nx, ny)."""
        return self._size

    def step(self) -> float:
        """Return the side of the cells (arcsec)."""
        return self._step

    def consts(self) -> tuple[float, ...]:
        return (*self._size, self._step)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        nx, ny = self._size
        return dict(a=ParamVectorDesc(
            nx * ny, desc="the density of each cell, row by row"))

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], subrnodes: np.ndarray
    ) -> np.ndarray:
        # The kernel draws the cells by |a| and gives each cloud the sign
        # of the density of its cell
        return np.sum(np.abs(params['a'])) * self._step ** 2

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _map_azimuthal_edges(self._size, self._step, r)


# The widening of the blur of the light of an image trait by the PSF of
# its image, beyond which it warns (see RPTraitImage.check_psf)
_IMAGE_BLUR_WIDENING_WARNING = 0.05


class RPTraitImage(RPTrait):
    """
    The light of an observed image (e.g. of HST or JWST), as the face-on
    light of a map in the plane of the disk: nx by ny square cells of side
    step (arcsec), those of the map trait (see RPTraitMap), whose densities
    its galaxy makes from the image at each evaluation, by the geometry of
    the disk (its centre, position angle and inclination; at the radius of
    each cell in a warped disk): a I(x, y) cos(incl), I the image at the
    position (x, y) on the sky of the centre of the cell, and a, a
    parameter, the light of the model per unit of the image. Seen with the
    same geometry, a thin disk of it is a times the image, whatever its
    inclination: its light says nothing of the geometry, which the
    kinematics give (as KinMS_fitter's skySampler; a photometric
    inclination goes in as a prior on the inclination of the disk). The
    light of a thick disk of it is the image smeared along the minor axis
    by about h sin(incl), h its height.

    The image is placed on the sky by its world coordinates (its RA and
    Dec), as observations are (see ObservationGroup), and moved by the
    offsets dx and dy (arcsec, +x to the west and +y to the north), two
    parameters, if fitted (offsets): e.g. for an error of its astrometry.
    Its pixels that are not finite are 0, and so is the light of the cells
    off the image: mask the image beforehand (e.g. prep.grown_mask) to
    keep its noise out. Its cells take the image at their centres
    (bilinear interpolation), so they should be no larger than its pixels
    (a warning says if they are).

    The image is blurred by its own PSF, and the model of each observation
    blurs its light again by the PSF of its instrument: its psf, if given,
    makes a warning if that widens the blur of the light by more than
    _IMAGE_BLUR_WIDENING_WARNING (see check_psf). Near edge-on, the image
    says little of the face-on light (see ImageMaps).

    Its parameters: m, the densities of the cells, row by row, which its
    galaxy makes (not parameters of configurations), first (the kernels
    of the map read them); a; and dx and dy, if offsets.

    Parameters
    ----------
    file : str
        The FITS file of the image (relative to the configuration).
    size : Sequence of int
        The number of cells along x and along y, [nx, ny].
    step : float
        The side of the cells (arcsec).
    hdu : int or str, optional
        The HDU of the image; the first by default.
    psf : PSF, optional
        The PSF of the image.
    offsets : bool, optional
        Whether the offsets of the image are fitted; False by default.
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid, or the file is not of an image.
    """

    @staticmethod
    def type() -> str:
        return 'image'

    @staticmethod
    def uid() -> int:
        # (the kernels of the map, which read its cells, its first
        # parameters)
        return RPT_UID_MAP

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'RPTraitImage':
        info = dict(info)
        parseutils.load_option_and_update_info(psf_parser, info, 'psf')
        return super().load(info)

    def dump(self) -> dict[str, Any]:
        info = super().dump() | dict(
            file=self._file, hdu=self._hdu, size=list(self._size),
            step=self._step, offsets=self._offsets)
        if self._psf is not None:
            info['psf'] = psf_parser.dump(self._psf)
        return info

    def __init__(
            self,
            file: str,
            size: Sequence[int],
            step: float,
            hdu: int | str = 0,
            psf: PSF | None = None,
            offsets: bool = False,
            name: str | None = None
    ):
        super().__init__(name=name)
        self._size, self._step = _check_map_grid(size, step)
        filename, _ = parseutils.parse_file(dict(file=file, hdu=hdu))
        data, coords = fitsutils.read_data(filename, hdu)
        if data.ndim != 2:
            raise ConfigError(
                f"the file {file} must hold an image; its data have "
                f"{data.ndim} axes")
        self._data = np.where(np.isfinite(data), data, 0).astype(float)
        self._grid = gridutils.make_grid(
            data.shape[::-1], coords.step, coords.rpix, coords.rval,
            coords.rota)
        pixel = min(abs(s) for s in coords.step[:2])
        if self._step > pixel * (1 + 1e-9):
            parseutils.warn(
                f"the cells of the image trait ({self._step:g} arcsec) are "
                f"larger than the pixels of its image ({pixel:g} arcsec): "
                f"they take the image at their centres, and miss its "
                f"light between them")
        self._file = file
        self._hdu = hdu
        self._psf = psf
        self._offsets = bool(offsets)

    def size(self) -> tuple[int, int]:
        """Return the numbers of cells along x and y, (nx, ny)."""
        return self._size

    def step(self) -> float:
        """Return the side of the cells (arcsec)."""
        return self._step

    def offsets(self) -> bool:
        """Check whether the offsets of its image are fitted."""
        return self._offsets

    def consts(self) -> tuple[float, ...]:
        return (*self._size, self._step)

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        nx, ny = self._size
        params = dict(
            m=ParamVectorDesc(
                nx * ny, desc="the face-on light of each cell, row by row "
                              "(made by the galaxy)"),
            a=ParamScalarDesc(desc="the light per unit of the image"))
        if self._offsets:
            params |= dict(
                dx=ParamScalarDesc(
                    desc="the offset of the image to the west (arcsec)"),
                dy=ParamScalarDesc(
                    desc="the offset of the image to the north (arcsec)"))
        return params

    def from_galaxy(self) -> ImageInput:
        return ImageInput(('m',))

    def has_analytical_integral(self) -> bool:
        return True

    def cloud_flux(
            self, params: dict[str, np.ndarray], subrnodes: np.ndarray
    ) -> np.ndarray:
        # (as that of a map, of the cells m)
        return np.sum(np.abs(params['m'])) * self._step ** 2

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return _map_azimuthal_edges(self._size, self._step, r)

    def face_on(
            self,
            params: dict[str, np.ndarray],
            x: np.ndarray,
            y: np.ndarray,
            incl: np.ndarray,
            origin: tuple[float, float]
    ) -> np.ndarray:
        """
        Return the densities of its cells (see the class), row by row.

        Parameters
        ----------
        params : dict
            The values of its parameters a, and dx and dy if offsets.
        x, y : np.ndarray
            The positions on the sky of the centres of the cells (arcsec,
            in the frame of origin; see gridutils.frame_to_pixels), of
            shape (ny, nx).
        incl : np.ndarray
            The inclination of the disk at each cell (degrees), of their
            shape or one.
        origin : tuple of float
            The RA and Dec (degrees) of the origin of the frame.

        Returns
        -------
        np.ndarray
            The densities.
        """
        dx, dy = (params['dx'], params['dy']) if self._offsets else (0, 0)
        px, py = gridutils.frame_to_pixels(
            x - dx, y - dy, origin, self._grid)
        image = scipy.ndimage.map_coordinates(
            self._data, [py, px], order=1, mode='constant', cval=0)
        # (|cos|, as the thin disk kernel divides by: beyond 90 degrees the
        # disk is seen from below, and its light is not negative)
        return (params['a'] * image * np.abs(np.cos(np.radians(incl)))).ravel()

    def check_coverage(
            self,
            x: np.ndarray,
            y: np.ndarray,
            origin: tuple[float, float],
            label: str
    ) -> None:
        """
        Warn (see parseutils.warn) if its image covers none of the given
        positions on the sky (arcsec, in the frame of origin; see
        face_on): e.g. its map about the origin, whose light would then be
        none, because the world coordinates of the image or of the
        observations are wrong.

        Parameters
        ----------
        x, y : np.ndarray
            The positions.
        origin : tuple of float
            The RA and Dec (degrees) of the origin of the frame.
        label : str
            What has the trait, for the message (e.g. "the disk of
            'gas'").
        """
        px, py = gridutils.frame_to_pixels(x, y, origin, self._grid)
        nx, ny = self._grid.size[:2]
        if np.any((px >= -0.5) & (px <= nx - 0.5)
                  & (py >= -0.5) & (py <= ny - 0.5)):
            return
        centre_x, centre_y = gridutils.frame_to_pixels(
            np.zeros(1), np.zeros(1), origin, self._grid)
        parseutils.warn(
            f"the image {self._file!r} of the image trait of {label} "
            f"covers none of its map about the origin of the frame of the "
            f"model (RA {origin[0]:.6f}, Dec {origin[1]:.6f}), which is at "
            f"its pixel ({centre_x[0]:.1f}, {centre_y[0]:.1f}) by their RA "
            f"and Dec: the disk has no light unless its centre is far "
            f"from the origin; check the world coordinates of the image "
            f"and of the observations")

    def check_psf(self, psf: PSF | None) -> None:
        """
        Warn (see parseutils.warn) if the PSF of its image (if given)
        widens the blur of its light in the model of an observation by
        more than _IMAGE_BLUR_WIDENING_WARNING: the model blurs the image
        again by the PSF of the instrument (psf), and the widths of the two
        add in quadrature. Their widths are their rms radii, drawn on the
        pixels of the image; a PSF that varies along the spectral axis is
        not compared.
        """
        if self._psf is None or psf is None or psf.varies():
            return
        step = tuple(abs(s) for s in self._grid.coords.step[:2])
        own, other = (_rms_radius(p, step) for p in (self._psf, psf))
        widening = np.hypot(1, own / other) - 1
        if widening > _IMAGE_BLUR_WIDENING_WARNING:
            parseutils.warn(
                f"the image {self._file} is blurred by its PSF ({own:.3g} "
                f"arcsec rms) by {own / other:.0%} of the PSF of the "
                f"instrument ({other:.3g} arcsec rms): the model blurs its "
                f"light by both, {widening:.0%} more than the data")


def _rms_radius(psf: PSF, step: tuple[float, float]) -> float:
    """
    Return the rms radius (arcsec) of a PSF, drawn on pixels of the given
    size (arcsec).
    """
    image = psf.asarray(step)
    ny, nx = image.shape
    j, i = np.mgrid[0:ny, 0:nx]
    x, y = (i - nx // 2) * step[0], (j - ny // 2) * step[1]
    return float(np.sqrt(np.sum(image * (x ** 2 + y ** 2)) / np.sum(image)))
