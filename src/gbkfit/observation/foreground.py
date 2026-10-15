import abc
import copy
import math
import os.path
from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import scipy.ndimage

from gbkfit.dataset import FitsFile, fits_file
from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamScalarDesc, ParamValues
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Foreground',
    'Lens',
    'LensDeflectionMap',
    'LensPlan',
    'LensSIE',
    'foreground_parser',
    'lens_parser'
]


class Lens(parseutils.TypedSerializable, abc.ABC):
    """
    A gravitational lens in front of a model.

    The light of each point of the image plane comes from the point of
    the source plane, where the model is, at its position less the
    deflection there (surface brightness is conserved). The model is
    evaluated on a grid of the source plane, centred on the origin of the
    frame of the model and aligned with the sky (see
    gridutils.sky_positions); each pixel of the image takes the bilinear
    interpolation of the source at its position on the source plane.

    A lens may have parameters (see pdescs), of the observation it is in
    front of (see Foreground.pdescs): its deflection is then evaluated
    for each evaluation of the model.

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

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name (none by default)."""
        return {}

    @abc.abstractmethod
    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the deflection at the pixels of a grid of the image plane.

        Parameters
        ----------
        grid : Grid
            The grid; its x and y axes.
        params : ParamValues
            The values of its parameters, by their names (see pdescs).

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y of the frame of the
            model, each of shape (ny, nx).
        """
        pass

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype,
            names: Mapping[str, str] | None = None
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
        names : Mapping of str to str, optional
            The names of its parameters among those of the evaluations, by
            their own names (see pdescs); required if it has parameters.

        Returns
        -------
        LensPlan
            The plan.
        """
        return LensPlan(self, driver, grid, dtype, names or {})


class LensPlan:
    """
    The lensing of the cubes of the source plane of a lens to a grid of
    the image plane, on a driver: the position on the source plane of
    each pixel of the image (in pixels of the source), worked out once,
    or for each evaluation if the lens has parameters.
    """

    def __init__(
            self,
            lens: Lens,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype,
            names: Mapping[str, str]
    ):
        if missing := sorted(set(lens.pdescs()) - set(names)):
            raise ValueError(
                f"the names of these parameters of the lens are missing: "
                f"{missing}")
        self._lens = lens
        self._driver = driver
        self._grid = grid
        self._dtype = dtype
        self._names = dict(names)
        self._sky = gridutils.sky_positions(grid)
        self._matrix, self._offset = gridutils.sky_to_pixel(
            lens.source_grid())
        self._source_x = driver.mem_alloc_d(self._sky[0].shape, dtype)
        self._source_y = driver.mem_alloc_d(self._sky[0].shape, dtype)
        if not self._names:
            self._place({})
        self._backend = driver.native_class('DModel', dtype)()

    def _place(self, params: ParamValues) -> None:
        """
        Work out the position on the source plane of each pixel of the
        image, with the parameters of the lens, and copy it to the device.
        """
        x, y = self._sky
        deflection_x, deflection_y = self._lens.deflection(self._grid, params)
        source_x = x - deflection_x
        source_y = y - deflection_y
        matrix, offset = self._matrix, self._offset
        pixel_x = matrix[0, 0] * source_x + matrix[0, 1] * source_y + offset[0]
        pixel_y = matrix[1, 0] * source_x + matrix[1, 1] * source_y + offset[1]
        self._driver.mem_copy_h2d(pixel_x.astype(self._dtype), self._source_x)
        self._driver.mem_copy_h2d(pixel_y.astype(self._dtype), self._source_y)

    def source_grid(self, grid: gridutils.Grid) -> gridutils.Grid:
        """
        Return the grid of the source plane for a model evaluated on the
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
                source.coords.rota, grid.coords.rest, grid.coords.medium),
            grid.spectral_axis)

    def evaluate(
            self,
            params: ParamValues,
            source: DeviceArray,
            image: DeviceArray
    ) -> None:
        """
        Lens the source cube (nz, sy, sx) into the image cube, with the
        values of the parameters of the evaluation (those of the lens, if
        any, by the names of the plan).
        """
        if self._names:
            self._place({
                name: params[full] for name, full in self._names.items()})
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
    scaled : bool, optional
        Whether the deflection is scaled by a parameter, 'scale' (fixed
        or fitted; see pdescs): e.g. D_LS / D_S of the source over that
        of the maps, when they are of a source at another redshift.

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
            rota: float | None = None,
            scaled: bool = False
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
        scaled : bool, optional
            Whether the deflection is scaled by a parameter (see the
            class).

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
            coords.rota if rota is None else rota, scaled)

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
        info |= dict(
            source_size=self._source_grid.size,
            source_step=self._source_grid.coords.step)
        if self._scaled:
            info['scaled'] = True
        return info

    def __init__(
            self,
            alpha_x: np.ndarray,
            alpha_y: np.ndarray,
            source_size: Sequence[int],
            source_step: Sequence[float],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            scaled: bool = False
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
        self._scaled = scaled

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        if not self._scaled:
            return {}
        return dict(scale=ParamScalarDesc(
            "the scale of the deflection (e.g. D_LS / D_S)", minimum=0))

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        gridutils.check_overlap(grid, self._grid, "the deflection maps")
        pixel_x, pixel_y = gridutils.pixels_on(grid, self._grid)
        scale = params['scale'] if self._scaled else 1.0

        def sample(data):
            return scale * scipy.ndimage.map_coordinates(
                data, [pixel_y, pixel_x], order=1, mode='nearest')
        return sample(self._alpha_x), sample(self._alpha_y)


class LensSIE(Lens):
    """
    A singular isothermal ellipsoid (SIE), optionally with an external
    shear: the standard model of a galaxy lens, of convergence

        kappa = theta_e / (2 sqrt(q u^2 + v^2 / q))

    along its major (u) and minor (v) axes, whose deflection is analytic
    (Kormann et al. 1994): with f = sqrt(1 - q^2) and
    psi = sqrt(q^2 u^2 + v^2),

        alpha_u = theta_e sqrt(q) / f * arctan(f u / psi)
        alpha_v = theta_e sqrt(q) / f * arctanh(f v / psi)

    (theta_e u / r and theta_e v / r, a singular isothermal sphere, for
    q = 1). The shear, of strength gamma, stretches the images along the
    position angle gamma_pa: its deflection is linear around the centre.

    Its parameters (see pdescs): theta_e (arcsec, the Einstein radius of
    the sphere of the same mass), q (the axis ratio), pa (the position
    angle of the major axis, north through east, as that of disks), x and
    y (the centre, arcsec, in the frame of the model, as xpos and ypos),
    and with shear, gamma and gamma_pa.

    Parameters
    ----------
    source_size, source_step : Sequence
        The grid of the source plane (see Lens).
    shear : bool, optional
        Whether it has an external shear.
    """

    @staticmethod
    def type() -> str:
        return 'sie'

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return dict(
            type=self.type(), source_size=self._source_grid.size,
            source_step=self._source_grid.coords.step, shear=self._shear)

    def __init__(
            self,
            source_size: Sequence[int],
            source_step: Sequence[float],
            shear: bool = False
    ):
        super().__init__(source_size, source_step)
        self._shear = shear

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        pdescs = dict(
            theta_e=ParamScalarDesc(
                "the Einstein radius (arcsec)", minimum=0),
            # (q > 0 keeps the deflection finite)
            q=ParamScalarDesc("the axis ratio", minimum=1e-3, maximum=1),
            pa=ParamScalarDesc("the position angle (degrees)"),
            x=ParamScalarDesc("the centre along x (arcsec)"),
            y=ParamScalarDesc("the centre along y (arcsec)"))
        if self._shear:
            pdescs |= dict(
                gamma=ParamScalarDesc("the external shear", minimum=0),
                gamma_pa=ParamScalarDesc(
                    "the position angle of the shear (degrees)"))
        return pdescs

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = gridutils.sky_positions(grid)
        x = x - params['x']
        y = y - params['y']
        # Along the major axis (u, at the position angle, from +y towards
        # -x) and the minor axis (v)
        pa = math.radians(params['pa'])
        sin_pa, cos_pa = math.sin(pa), math.cos(pa)
        u = -sin_pa * x + cos_pa * y
        v = cos_pa * x + sin_pa * y
        alpha_u, alpha_v = _sie_deflection(
            u, v, params['theta_e'], params['q'])
        alpha_x = -sin_pa * alpha_u + cos_pa * alpha_v
        alpha_y = cos_pa * alpha_u + sin_pa * alpha_v
        if self._shear:
            # (the angle from +x of the position angle is 90 degrees more)
            angle = 2 * math.radians(params['gamma_pa'] + 90)
            gamma1 = params['gamma'] * math.cos(angle)
            gamma2 = params['gamma'] * math.sin(angle)
            alpha_x = alpha_x + gamma1 * x + gamma2 * y
            alpha_y = alpha_y + gamma2 * x - gamma1 * y
        return alpha_x, alpha_y


def _sie_deflection(
        u: np.ndarray,
        v: np.ndarray,
        theta_e: float,
        q: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an SIE along its major (u) and minor (v)
    axes (see LensSIE); 0 at its centre.
    """
    r = np.hypot(u, v)
    centre = r == 0
    if q > 1 - 1e-8:
        with np.errstate(invalid='ignore', divide='ignore'):
            scale = np.where(centre, 0, theta_e / r)
        return scale * u, scale * v
    f = math.sqrt(1 - q * q)
    psi = np.where(centre, 1, np.sqrt(q * q * u * u + v * v))
    scale = theta_e * math.sqrt(q) / f
    return (scale * np.arctan(f * u / psi),
            scale * np.arctanh(f * v / psi))


lens_parser = parseutils.TypedParser(Lens, [LensDeflectionMap, LensSIE])


class Foreground(parseutils.Serializable):
    """
    What happens to the light of a model before it reaches the telescope:
    a gravitational lens. Each has its own slot, in the order the light
    meets them.

    Its parameters (those of its lens) are those of the observation it is
    in front of, named after it (see with_prefix and pdescs).

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
        self._prefix = ''

    def lens(self) -> Lens | None:
        """Return the gravitational lens, if any."""
        return self._lens

    def with_prefix(self, prefix: str) -> Self:
        """
        Return a copy of it whose parameters have names that start with a
        prefix (e.g. 'muse_', for those of the lens 'muse_lens_theta_e').
        """
        foreground = copy.copy(self)
        foreground._prefix = prefix
        return foreground

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """
        Return its parameters, by name: '{prefix}lens_{name}' for each of
        those of its lens (see with_prefix).
        """
        if self._lens is None:
            return {}
        return {
            f'{self._prefix}lens_{name}': pdesc
            for name, pdesc in self._lens.pdescs().items()}

    def plan_lens(
            self, driver: Driver, grid: gridutils.Grid, dtype: np.dtype
    ) -> LensPlan | None:
        """
        Plan the lensing of cubes on a grid of the image plane (see
        Lens.plan), if it has a lens.
        """
        if self._lens is None:
            return None
        return self._lens.plan(driver, grid, dtype, {
            name: f'{self._prefix}lens_{name}'
            for name in self._lens.pdescs()})


foreground_parser = parseutils.BasicParser(Foreground)
