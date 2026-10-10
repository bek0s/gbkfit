import abc
import os.path
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage

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
    A gravitational lens in front of a gmodel: the light of each point of
    the image plane comes from the point of the source plane, where the
    gmodel is, at its position less the deflection there (surface
    brightness is conserved). The gmodel is evaluated on a grid of the
    source plane: source_size pixels of source_step arcsec, centred on the
    origin of the frame of the model, aligned with the sky (see
    gridutils.sky_positions); each pixel of the image takes the bilinear
    interpolation of the source at its position on the source plane.
    """

    def __init__(self, source_size: Sequence[int], source_step: Sequence[float]):
        self._source_grid = gridutils.make_grid(
            tuple(source_size), tuple(source_step))

    def source_grid(self) -> gridutils.Grid:
        """The grid of the source plane (x and y)."""
        return self._source_grid

    @abc.abstractmethod
    def deflection(
            self, grid: gridutils.Grid
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The deflection (arcsec, along x and y of the frame of the model) at
        the pixels of the x and y axes of a grid of the image plane: two
        arrays of shape (ny, nx).
        """
        pass

    def plan(self, driver, grid, dtype) -> 'LensPlan':
        """The lensing of a cube on the given grid of the image plane."""
        return LensPlan(self, driver, grid, dtype)


class LensPlan:
    """
    The lensing of the cubes of the source plane of a lens to a grid of
    the image plane, on a driver: the position on the source plane of
    each pixel of the image (in pixels of the source).
    """

    def __init__(self, lens, driver, grid, dtype):
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
        The grid of the source plane for a gmodel evaluated on the given
        grid of the image plane: the x and y of the source plane, and its
        other axes (e.g. the spectral axis).
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

    def evaluate(self, source, image):
        """Lens the source cube (nz, sy, sx) into the image cube."""
        self._backend.lens_resample(
            self._source_x, self._source_y, source, image)


def _read_map(x, prefix, rpix, rval):
    """
    A map and its world coordinates, from a file (see Data), with the
    given rpix or rval (see fitsutils.read_data).
    """
    file, hdu = parseutils.parse_file(x)
    return fitsutils.read_data(prefix + file, hdu, rpix, rval)


class LensDeflectionMap(Lens):
    """
    A lens of a given deflection: maps of its x and y (arcsec, along x and
    y of the frame of the model, like xpos and ypos) on a grid of the image
    plane (e.g. from a lens model), placed by their RA and Dec. Between the
    pixels of the maps the deflection is interpolated (bilinearly), and
    beyond them it is that of their nearest edge (e.g. in the padding of
    the convolution).
    """

    @staticmethod
    def type():
        return 'deflection_map'

    @classmethod
    def load(cls, info, prefix=''):
        desc = parseutils.make_typed_desc(cls, 'lens')
        for key in ('alpha_x', 'alpha_y'):
            if info.get(key) is None:
                raise ConfigError(f"option '{key}' of {desc} is required")
        # (the world coordinates given are those of the maps, which the
        # others come from)
        rpix, rval = info.get('rpix'), info.get('rval')
        with parseutils.config_path('alpha_x'):
            alpha_x, coords = _read_map(info['alpha_x'], prefix, rpix, rval)
        with parseutils.config_path('alpha_y'):
            alpha_y, coords_y = _read_map(info['alpha_y'], prefix, rpix, rval)
        if coords_y != coords:
            raise ConfigError(
                f"the maps of {desc} have different world coordinates")
        info = dict(info) | dict(
            alpha_x=alpha_x, alpha_y=alpha_y, rpix=coords.rpix,
            rval=coords.rval,
            step=coords.step if info.get('step') is None else info['step'],
            rota=coords.rota if info.get('rota') is None else info['rota'])
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        coords = self._grid.coords
        info = dict(type=self.type())
        for key, data in (('alpha_x', self._alpha_x), ('alpha_y', self._alpha_y)):
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
            step: Any = None,
            rpix: Any = None,
            rval: Any = None,
            rota: float | None = None
    ):
        """
        The maps are on the grid of the given world coordinates (see
        gridutils.make_grid).
        """
        super().__init__(source_size, source_step)
        alpha_x = np.asarray(alpha_x, dtype=float)
        alpha_y = np.asarray(alpha_y, dtype=float)
        if alpha_x.ndim != 2 or alpha_x.shape != alpha_y.shape:
            raise RuntimeError(
                f"the deflection maps must be two images of one shape; they "
                f"have the shapes {alpha_x.shape} and {alpha_y.shape}")
        if not (np.all(np.isfinite(alpha_x)) and np.all(np.isfinite(alpha_y))):
            raise RuntimeError("the deflection maps must be finite")
        self._alpha_x = alpha_x
        self._alpha_y = alpha_y
        self._grid = gridutils.make_grid(
            alpha_x.shape[::-1], step, rpix, rval, rota)

    def deflection(self, grid):
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
    """

    @classmethod
    def load(cls, info, prefix=''):
        parseutils.load_option_and_update_info(
            lens_parser, info, 'lens', prefix=prefix)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        return dict(lens=lens_parser.dump(
            self._lens, prefix=prefix, dump_path=dump_path,
            overwrite=overwrite))

    def __init__(self, lens: Lens | None = None):
        self._lens = lens

    def lens(self) -> Lens | None:
        return self._lens


foreground_parser = parseutils.BasicParser(Foreground)
