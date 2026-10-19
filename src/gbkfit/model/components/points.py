from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np
import scipy.special

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamScalarDesc, ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._detail import Fluxes, dump_geometry, dump_name, systemic_velocity
from ._units import DataUnit, band_factors, line_factors
from .base import Component, ComponentPlan, FieldsLayout, NativeGrid
from .geometries import Geometry
from .lines import Band, Bands, LineCatalogue, LinePlacement, Lines
from .roles import BandEmission, Emission, LineEmission, emission_parser


__all__ = [
    'Point'
]


def _check_geometry(geometry: Geometry | None) -> None:
    """Raise ConfigError if the geometry of a point is loose."""
    if geometry is not None and geometry.loose():
        raise ConfigError(
            f"a point has one centre, but its geometry "
            f"{geometry.name()!r} is loose (a centre for each ring)")


def point_pdescs(spectral: bool, depth: bool) -> dict[str, ParamScalarDesc]:
    """
    Return the parameters of a point component: its position (xpos, ypos,
    and depth if it has one) and flux, and for spectral ones its systemic
    velocity and dispersion.
    """
    names = ('xpos', 'ypos') + (('depth',) if depth else ()) + ('flux',) \
        + (('vsys', 'disp') if spectral else ())
    return {name: ParamScalarDesc() for name in names}


def _light_pdescs(
        light: Lines | Bands, spectral: bool, depth: bool
) -> dict[str, ParamScalarDesc]:
    """
    Return the parameters of a point with the given lines (a spectral
    point) or bands. Raise ConfigError if their flux ratios have the names
    of its other parameters.
    """
    pdescs = point_pdescs(spectral=spectral, depth=depth)
    if repeated := sorted(set(light.pdescs()) & set(pdescs)):
        raise ConfigError(
            f"the parameters of the flux ratios have the names of other "
            f"parameters: {repeated}; rename the lines or bands")
    return pdescs | light.pdescs()


def line_light(
        spectral: gridutils.Grid,
        lines: Lines,
        unit: astropy.units.UnitBase | None,
        names: Sequence[str] | None,
        data_unit: DataUnit | None,
        redshift: float
) -> tuple[Fluxes, LinePlacement]:
    """
    Return the fluxes of the rows and the placement of the lines of the
    given names (all if None) of a component with the given lines and
    unit of its light (if any), at the redshift of its galaxy, in the unit
    of the data (see Component.plan).
    """
    selected = lines.select(names)
    rows = lines.values(spectral, redshift)[list(selected)]
    # (with a unit, the frequency at which the galaxy emits each line)
    frequencies = None if unit is None \
        else lines.frequencies(spectral, redshift)[list(selected)]
    ratios = [
        (row, lines.ratio_name(index))
        for row, index in enumerate(selected)
        if lines.ratio_name(index) is not None]
    return Fluxes(
        len(rows), ratios,
        line_factors(unit, data_unit, rows, spectral, frequencies)), \
        LinePlacement(rows, spectral)


def band_light(
        bands: Bands,
        unit: astropy.units.UnitBase | None,
        band: Band | None,
        data_unit: DataUnit | None,
        redshift: float
) -> Fluxes:
    """
    Return the flux of the one row of the light of a component in the
    given band, with its bands and the unit of its light (if any), in the
    unit of the data (see Component.plan).
    """
    ratio = bands.ratio_name(band)
    return Fluxes(
        1, [(0, ratio)] if ratio is not None else [],
        band_factors(unit, data_unit, band, redshift))


def _pixels(
        params: dict[str, float], grid: NativeGrid
) -> list[tuple[int, int, float]]:
    """
    Return the four pixels (x and y indices) around the position of a
    point on the sky, and the share of its light of each (bilinear);
    those outside the grid are left out.
    """
    rota = np.radians(grid['spat_rota'])
    x, y = params['xpos'], params['ypos']
    # From the sky to the pixels of the grid (rotated by rota)
    gx = x * np.cos(rota) + y * np.sin(rota)
    gy = -x * np.sin(rota) + y * np.cos(rota)
    px = (gx - grid['spat_zero'][0]) / grid['spat_step'][0]
    py = (gy - grid['spat_zero'][1]) / grid['spat_step'][1]
    x0, y0 = int(np.floor(px)), int(np.floor(py))
    fx, fy = px - x0, py - y0
    size_x, size_y = grid['spat_size'][:2]
    pixels = []
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            if 0 <= x0 + dx < size_x and 0 <= y0 + dy < size_y \
                    and wx * wy > 0:
                pixels.append((x0 + dx, y0 + dy, wx * wy))
    return pixels


def _optical_depths(
        columns: np.ndarray, depth: float, grid: NativeGrid
) -> np.ndarray:
    """
    Return the optical depth in each cube of opacity from the viewer to
    the given depth on a line of sight (arcsec, away from the viewer), from
    the optical depth from the viewer to the far side of each voxel of the
    line of sight in each cube (columns, of shape (ncubes, nz)), with the
    absorbers of each voxel spread uniformly along it (as the native disk
    kernels do, see gmodel_optical_depth): 0 in front of the grid, and
    that of the whole line of sight behind it.
    """
    if np.isnan(depth):
        return np.full(len(columns), np.nan)
    size, step, zero = (
        grid[key][2] for key in ('spat_size', 'spat_step', 'spat_zero'))
    # The position from the near side of the grid (in voxels), and its voxel
    p = min(max((depth - zero) / step + 0.5, 0), size)
    z = min(int(p), size - 1)
    tau_front = columns[:, z - 1] if z > 0 else np.zeros(len(columns))
    return tau_front + (p - z) * (columns[:, z] - tau_front)


class PointPlan(ComponentPlan):
    """
    The evaluation of a point component (e.g. an unresolved nucleus): its
    flux at its position on the sky, shared by the four pixels around it
    (bilinearly), as surface brightness (per unit area, as the disks), in
    the image, or in the spectral cube as its emission lines (see Lines),
    each a Gaussian of its dispersion, of which each channel gets its
    mean over the channel (as the disks do). On a grid with a depth axis,
    the opacity in front of its depth (its parameter 'depth' if depth, or
    0: the plane of the sky through the centre of its galaxy) dims it,
    with its opacity_ratios (see Component.plan). Its light is that of the
    given lines (where each is on the spectral axis at its systemic
    velocity, see LinePlacement; spectral points), or in an image, that of
    its band, of the given fluxes (see Fluxes).
    """

    def __init__(
            self,
            driver: Driver,
            dtype: np.dtype,
            depth: bool,
            opacity_ratios: np.ndarray | None,
            fluxes: Fluxes,
            lines: LinePlacement | None = None
    ):
        self._driver = driver
        self._dtype = np.dtype(dtype)
        self._depth = depth
        self._opacity_ratios = opacity_ratios
        self._fluxes = fluxes
        # The offset, scale and flux of each line (set at each evaluation)
        self._lines = lines
        self._rows = None if lines is None else np.zeros((len(lines), 3))

    def line_values(self, params: ParamValues) -> np.ndarray:
        if self._lines is None:
            return np.array([[0.0, 0.0, self._fluxes(params)[0]]])
        self._rows[:, :2] = self._lines(systemic_velocity(params))
        self._rows[:, 2] = self._fluxes(params)
        return self._rows

    def _spectrum(
            self,
            params: dict[str, float],
            grid: NativeGrid,
            transmission: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Return the spectrum of the point, per unit flux, with the fraction
        of the light of each line that the opacity lets through
        (transmission), if any.
        """
        size, step, zero = (
            grid['spec_size'], grid['spec_step'], grid['spec_zero'])
        edges = zero + (np.arange(size + 1) - 0.5) * step
        spectrum = np.zeros(size)
        for row, (offset, scale, flux) in enumerate(
                self.line_values(params)):
            if transmission is not None:
                flux = flux * transmission[row]
            centre = offset + scale * params['vsys']
            sigma = scale * abs(params['disp'])
            if sigma > 0:
                cdf = 0.5 * (1 + scipy.special.erf(
                    (edges - centre) / (np.sqrt(2) * sigma)))
                spectrum += flux * np.diff(cdf) / step
            else:
                channel = int(np.floor((centre - edges[0]) / step))
                if 0 <= channel < size:
                    spectrum[channel] += flux / step
        return spectrum

    def fields_layout(self) -> FieldsLayout:
        return FieldsLayout(layer=True, nslots=4)

    def evaluate_fields(
            self,
            params: ParamValues,
            grid: NativeGrid,
            fields: DeviceArray,
            out_extra: dict[str, Any] | None
    ) -> None:
        # A layer at its depth: at the pixels around it, its share of its
        # light, its velocity and dispersion (none in a band), and its
        # position along the depth axis (in voxels)
        layout = self.fields_layout()
        layer = np.zeros(layout.shape(grid), self._dtype)
        depth = params['depth'] if self._depth else 0.0
        brightness = params['flux'] / (
            grid['spat_step'][0] * grid['spat_step'][1])
        velocity, dispersion = (params['vsys'], abs(params['disp'])) \
            if self._lines is not None else (0.0, 0.0)
        position = (depth - grid['spat_zero'][2]) / grid['spat_step'][2]
        for x, y, weight in _pixels(params, grid):
            layer[:, y, x] = (
                brightness * weight, velocity, dispersion, position)
        self._driver.mem_copy_h2d(layer, fields)

    def evaluate_emission(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        driver = self._driver
        image = outputs.get('image')
        scube = outputs.get('scube')
        odata = outputs.get('odata')
        depth = params['depth'] if self._depth else 0.0
        # The surface brightness of all the flux in one pixel
        brightness = params['flux'] / (
            grid['spat_step'][0] * grid['spat_step'][1])
        values = None
        if scube is not None:
            values = brightness * self._spectrum(params, grid)
        for x, y, weight in _pixels(params, grid):
            # The fraction of the light of each line (or of the image) that
            # the opacity in front of the point lets through at the pixel
            transmission = None
            if odata is not None:
                columns = driver.mem_copy_d2h(odata[:, :, y, x])
                transmission = np.exp(-self._opacity_ratios @ _optical_depths(
                    columns, depth, grid))
            if image is not None:
                weight *= self._fluxes(params)[0]
                if transmission is not None:
                    weight *= transmission[0]
                pixel = image.reshape(-1, *image.shape[-2:])[0, y, x:x + 1]
                driver.math_add(pixel, self._dtype.type(
                    brightness * weight), out=pixel)
            if scube is not None:
                if transmission is not None:
                    values = brightness * self._spectrum(
                        params, grid, transmission)
                spectrum = scube[:, y, x]
                driver.math_add(spectrum, driver.mem_copy_h2d(
                    (weight * values).astype(self._dtype)), out=spectrum)


class Point(Component):
    """
    An unresolved component (e.g. a nucleus): its flux at a point (see
    PointPlan). Its parameters: xpos, ypos, depth (if it has one) and
    flux, and with emission lines vsys and disp (the dispersion of its
    lines) and the flux ratios of its lines (see Lines). Seen along a line
    of sight with a depth axis, the opacity of its galaxy in front of it
    dims it.

    Parameters
    ----------
    emission : Emission
        The light it emits.
    depth : bool, optional
        Whether its position along the line of sight is a parameter,
        'depth' (arcsec, away from the viewer, from the plane of the sky
        through the centre of its galaxy); by default, it is in that plane
        (e.g. a nucleus).
    geometry : Geometry, optional
        A geometry whose centre (and systemic velocity) it shares (not a
        loose one).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ConfigError
        If the geometry is loose, or the lines are not valid.
    """

    @staticmethod
    def type() -> str:
        return 'point'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> 'Point':
        parseutils.load_option_and_update_info(
            emission_parser, info, 'emission', required=True,
            catalogue=catalogue)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return (
            dict(type=self.type())
            | dump_name(self)
            | dump_geometry(self)
            | dict(emission=emission_parser.dump(self.emission()),
                   depth=self._depth))

    def __init__(
            self,
            emission: Emission,
            depth: bool = False,
            geometry: Geometry | None = None,
            name: str | None = None
    ):
        _check_geometry(geometry)
        super().__init__(name, geometry, emission, None)
        self._depth = depth
        self._lines = None
        self._bands = None
        if isinstance(emission, LineEmission):
            self._lines = emission.lines()
            self._pdescs = _light_pdescs(self._lines, True, depth)
        if isinstance(emission, BandEmission):
            self._bands = emission.bands()
            self._pdescs = _light_pdescs(self._bands, False, depth)

    def depth(self) -> bool:
        """Check whether its depth along the line of sight is a parameter."""
        return self._depth

    def thick(self) -> bool:
        return False

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return self._pdescs

    def plan(
            self,
            driver: Driver,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            data_unit: DataUnit | None,
            redshift: float
    ) -> PointPlan:
        unit = self.emission().unit()
        if self._lines is None:
            return PointPlan(
                driver, dtype, self._depth, opacity_ratios,
                band_light(self._bands, unit, band, data_unit, redshift))
        return PointPlan(
            driver, dtype, self._depth, opacity_ratios, *line_light(
                spectral, self._lines, unit, lines, data_unit, redshift))
