import abc
import os.path
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage

from gbkfit.dataset import FitsFile, fits_file
from gbkfit.driver import DeviceArray, Driver
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Foreground',
    'Lens',
    'LensDeflectionMap',
    'LensPlan',
    'foreground_parser',
    'lens_parser'
]


class Lens(parseutils.TypedSerializable, abc.ABC):
    """
    A gravitational lens in front of a gmodel.

    The light of each point of the image plane comes from the point of
    the source plane, where the gmodel is, at its position less the
    deflection there (surface brightness is conserved). The gmodel is
    evaluated on a grid of the source plane, centred on the origin of the
    frame of the model and aligned with the sky (see
    gridutils.sky_positions); each pixel of the image takes the bilinear
    interpolation of the source at its position on the source plane.

    Parameters
    ----------
    source_size : Sequence of int
        The number of pixels of the grid of the source plane (nx, ny).
    source_step : Sequence of float
        The size of its pixels (arcsec).
    """

    def __init__(
            self, source_size: Sequence[int], source_step: Sequence[float]
    ):
        self._source_grid = gridutils.make_grid(
            tuple(source_size), tuple(source_step))

    def source_grid(self) -> gridutils.Grid:
        """Return the grid of the source plane (x and y)."""
        return self._source_grid

    @abc.abstractmethod
    def deflection(
            self, grid: gridutils.Grid
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the deflection at the pixels of a grid of the image plane.

        Parameters
        ----------
        grid : Grid
            The grid; its x and y axes.

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y of the frame of the
            model, each of shape (ny, nx).
        """
        pass

    def plan(
            self, driver: Driver, grid: gridutils.Grid, dtype: np.dtype
    ) -> 'LensPlan':
        """
        Plan the lensing of cubes on a grid of the image plane.

        Parameters
        ----------
        driver : Driver
            The driver the cubes are on.
        grid : Grid
            The grid of the image plane.
        dtype : np.dtype
            The floating type of the cubes.

        Returns
        -------
        LensPlan
            The plan.
        """
        return LensPlan(self, driver, grid, dtype)


class LensPlan:
    """
    The lensing of the cubes of the source plane of a lens to a grid of
    the image plane, on a driver: the position on the source plane of
    each pixel of the image (in pixels of the source).
    """

    def __init__(
            self,
            lens: Lens,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype
    ):
        x, y = gridutils.sky_positions(grid)
        deflection_x, deflection_y = lens.deflection(grid)
        matrix, offset = gridutils.sky_to_pixel(lens.source_grid())
        source_x = x - deflection_x
        source_y = y - deflection_y
        pixel_x = matrix[0, 0] * source_x + matrix[0, 1] * source_y + offset[0]
        pixel_y = matrix[1, 0] * source_x + matrix[1, 1] * source_y + offset[1]
        self._lens = lens
        self._source_x = driver.mem_copy_h2d(pixel_x.astype(dtype))
        self._source_y = driver.mem_copy_h2d(pixel_y.astype(dtype))
        self._backend = driver.native_class('DModel', dtype)()

    def source_grid(self, grid: gridutils.Grid) -> gridutils.Grid:
        """
        Return the grid of the source plane for a gmodel evaluated on the
        given grid of the image plane: the x and y of the source plane, and
        the other axes of the grid (e.g. the spectral axis).
        """
        source = self._lens.source_grid()
        return gridutils.Grid(
            source.size + grid.size[2:],
            gridutils.Coords(
                source.coords.step + grid.coords.step[2:],
                source.coords.rpix + grid.coords.rpix[2:],
                source.coords.rval + grid.coords.rval[2:],
                source.coords.rota, grid.coords.rest),
            grid.spectral_axis)

    def evaluate(self, source: DeviceArray, image: DeviceArray) -> None:
        """Lens the source cube (nz, sy, sx) into the image cube."""
        self._backend.lens_resample(
            self._source_x, self._source_y, source, image)


class LensDeflectionMap(Lens):
    """
    A lens of a given deflection: maps of it on a grid of the image plane
    (e.g. from a lens model), placed by their RA and Dec.

    Between the pixels of the maps the deflection is interpolated
    (bilinearly), and beyond them it is that of their nearest edge (e.g.
    in the padding of the convolution).

    Parameters
    ----------
    alpha_x, alpha_y : np.ndarray
        The deflection (arcsec) along x and y of the frame of the model
        (like xpos and ypos), each of shape (ny, nx).
    source_size, source_step : Sequence
        The grid of the source plane (see Lens).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the maps (see gridutils.make_grid).
    rota : float, optional
        The rotation of the maps on the sky.

    Raises
    ------
    ConfigError
        If the maps are not two finite images of one shape.
    """

    @staticmethod
    def type() -> str:
        return 'deflection_map'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'LensDeflectionMap':
        info = dict(info)
        files = {}
        for key in ('alpha_x', 'alpha_y'):
            files[key] = parseutils.load_option(
                fits_file, info, key, required=True, prefix=prefix)
            del info[key]
        options = parseutils.parse_options_for_callable(
            info, cls.from_files, ignore_params=list(files))
        return cls.from_files(**files, **options)

    @classmethod
    def from_files(
            cls,
            alpha_x: FitsFile,
            alpha_y: FitsFile,
            source_size: Sequence[int],
            source_step: Sequence[float],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ) -> 'LensDeflectionMap':
        """
        Read a lens from FITS files of its deflection maps, with the world
        coordinates of the files unless given; they must agree.

        Parameters
        ----------
        alpha_x, alpha_y : str or tuple
            The files of the maps: each a filename, or a filename and the
            HDU.
        source_size, source_step : Sequence
            The grid of the source plane (see Lens).
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the maps (see gridutils.make_grid); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            fitsutils.read_data).
        rota : float, optional
            The rotation of the maps on the sky; by default, that of the
            headers.

        Returns
        -------
        LensDeflectionMap
            The lens.

        Raises
        ------
        ConfigError
            If the maps have different world coordinates.
        """
        maps = []
        for file in (alpha_x, alpha_y):
            filename, hdu = (file, 0) if isinstance(file, str) else file
            maps.append(fitsutils.read_data(filename, hdu, rpix, rval))
        (data_x, coords), (data_y, coords_y) = maps
        if coords_y != coords:
            raise ConfigError(
                "the deflection maps have different world coordinates")
        return cls(
            data_x, data_y, source_size, source_step,
            coords.step if step is None else step, coords.rpix, coords.rval,
            coords.rota if rota is None else rota)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        coords = self._grid.coords
        info = dict(type=self.type())
        for key, data in (
                ('alpha_x', self._alpha_x), ('alpha_y', self._alpha_y)):
            filename = f'{prefix}lens_{key}.fits'
            fitsutils.write_data(filename, data, coords, None, overwrite)
            info[key] = filename if dump_path else os.path.basename(filename)
        return info | dict(
            source_size=self._source_grid.size,
            source_step=self._source_grid.coords.step)

    def __init__(
            self,
            alpha_x: np.ndarray,
            alpha_y: np.ndarray,
            source_size: Sequence[int],
            source_step: Sequence[float],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        super().__init__(source_size, source_step)
        alpha_x = np.asarray(alpha_x, dtype=float)
        alpha_y = np.asarray(alpha_y, dtype=float)
        if alpha_x.ndim != 2 or alpha_x.shape != alpha_y.shape:
            raise ConfigError(
                f"the deflection maps must be two images of one shape; they "
                f"have the shapes {alpha_x.shape} and {alpha_y.shape}")
        if not (np.all(np.isfinite(alpha_x)) and np.all(np.isfinite(alpha_y))):
            raise ConfigError("the deflection maps must be finite")
        self._alpha_x = alpha_x
        self._alpha_y = alpha_y
        self._grid = gridutils.make_grid(
            alpha_x.shape[::-1], step, rpix, rval, rota)

    def deflection(
            self, grid: gridutils.Grid
    ) -> tuple[np.ndarray, np.ndarray]:
        gridutils.check_overlap(grid, self._grid, "the deflection maps")
        pixel_x, pixel_y = gridutils.pixels_on(grid, self._grid)

        def sample(data):
            return scipy.ndimage.map_coordinates(
                data, [pixel_y, pixel_x], order=1, mode='nearest')
        return sample(self._alpha_x), sample(self._alpha_y)


lens_parser = parseutils.TypedParser(Lens, [LensDeflectionMap])


class Foreground(parseutils.Serializable):
    """
    What happens to the light of a gmodel before it reaches the telescope:
    a gravitational lens. Each has its own slot, in the order the light
    meets them.

    Parameters
    ----------
    lens : Lens, optional
        The gravitational lens, if any.
    """

    @classmethod
    def load(cls, info: dict[str, Any], prefix: str = '') -> 'Foreground':
        parseutils.load_option_and_update_info(
            lens_parser, info, 'lens', prefix=prefix)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(lens=lens_parser.dump(
            self._lens, prefix=prefix, dump_path=dump_path,
            overwrite=overwrite))

    def __init__(self, lens: Lens | None = None):
        self._lens = lens

    def lens(self) -> Lens | None:
        """Return the gravitational lens, if any."""
        return self._lens


foreground_parser = parseutils.BasicParser(Foreground)
