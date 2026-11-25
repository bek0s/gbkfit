"""
Point components (e.g. an unresolved nucleus), and the parameters and
the light of a point, which particles share.
"""

from typing import Any

import astropy.units
import numpy as np
import scipy.special

from gbkfit.config import TypedParser, TypedSerializable, check_options
from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamScalarDesc, ParamValues
from ._common import (
    Fluxes, dump_name, systemic_velocity)
from .base import (
    Component, ComponentPlan, ComponentState, FieldsLayout, FrameParams,
    NativeGrid, SceneTarget)
from ..geometries import geometry_pdesc
from .lines import Bands, LinePlacement, Lines
from .roles import Light, light_parser
from .tracers import View


__all__ = [
    'GaussKinematics',
    'Point',
    'PointKinematics',
    'point_kinematics_parser'
]


class PointKinematics(TypedSerializable):
    """
    The kinematics of a point component: the distribution of the
    velocities of its light along the line of sight, which a spectral
    axis sees (its lines, or with the tracer losvd, the velocities of its
    stars; see tracers).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())


class GaussKinematics(PointKinematics):
    """
    A Gaussian distribution of velocities, of the systemic velocity of the
    point (vsys) and the dispersion disp: e.g. the stars of a nuclear star
    cluster, or the broad lines of an active nucleus.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

    def __init__(self):
        check_options(GaussKinematics.__init__, locals())


point_kinematics_parser = TypedParser(PointKinematics, [
    GaussKinematics], family='point_kinematics')


def point_pdescs(
        moves: bool,
        depth: bool,
        flux_unit: astropy.units.UnitBase | None
) -> dict[str, ParamScalarDesc]:
    """
    Return the parameters of a point component: its position (xpos, ypos,
    and depth if it has one) and flux (of the given unit, if known), and
    for one that moves (has kinematics) its systemic velocity and
    dispersion.
    """
    others = dict(
        depth=ParamScalarDesc(
            title='Depth',
            desc="the distance behind the plane of the sky through the "
                 "centre of the galaxy", unit='arcsec', default=0),
        flux=ParamScalarDesc(
            title='Flux',
            desc="the flux of the point (of its first line or band)",
            unit=flux_unit, default=1),
        disp=ParamScalarDesc(
            title='Velocity dispersion',
            desc="the dispersion of its velocities along the line of "
                 "sight", min=0,
            unit='km/s', default=20))
    names = ('xpos', 'ypos') + (('depth',) if depth else ()) + ('flux',) \
        + (('vsys', 'disp') if moves else ())
    return {
        name: others[name] if name in others else geometry_pdesc(name)
        for name in names}


def _light_pdescs(
        spectrum: Lines | Bands,
        moves: bool,
        depth: bool,
        unit: astropy.units.UnitBase | None
) -> dict[str, ParamScalarDesc]:
    """
    Return the parameters of a point with the given lines or bands, that
    moves or not (has kinematics), and the unit of its light (if any): its
    flux is in that unit (a density per solid angle) times arcsec^2. Raise
    ConfigError if the flux ratios of its lines or bands have the names
    of its other parameters.
    """
    flux_unit = None if unit is None else unit * astropy.units.arcsec ** 2
    pdescs = point_pdescs(moves, depth, flux_unit)
    if repeated := sorted(set(spectrum.pdescs()) & set(pdescs)):
        raise ValueError(
            f"the parameters of the flux ratios have the names of other "
            f"parameters: {repeated}; rename the lines or bands")
    return pdescs | spectrum.pdescs()






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
            return np.array([[0.0, 0.0, float(np.sum(self._fluxes(params)))]])
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
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
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
            out_extra: dict[str, Any] | None,
            state: ComponentState | None
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
                weight *= float(np.sum(self._fluxes(params)))
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
    An unresolved component (e.g. a nucleus): its flux at a point. It is
    drawn on the pixels around the point (see PointPlan). Its parameters:
    xpos, ypos, depth (if it has one) and flux, the flux ratios of its
    lines or bands (see Lines and Bands), and with kinematics vsys and
    disp (see GaussKinematics), which emission lines need. Seen along a
    line of sight with a depth axis, the opacity of its galaxy in front of
    it dims it.

    Parameters
    ----------
    light : Light
        The light it emits.
    kinematics : PointKinematics, optional
        How its light moves: its distribution of velocities along the line
        of sight; none by default (it does not move).
    depth : bool, optional
        Whether its position along the line of sight is a parameter,
        'depth' (arcsec, away from the viewer, from the plane of the sky
        through the centre of its galaxy); by default, it is in that plane
        (e.g. a nucleus).
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ValueError
        If the lines are not valid, or it has lines and no kinematics.
    """

    config_titles = dict(kinematics='Kinematics', depth='Depth a parameter')

    @staticmethod
    def type() -> str:
        return 'point'

    @staticmethod
    def type_title() -> str:
        return 'Point'

    def dump(self) -> dict[str, Any]:
        return (
            dict(type=self.type())
            | dump_name(self)
            | dict(light=light_parser.dump(self.light()))
            | ({} if self._kinematics is None else dict(
                kinematics=point_kinematics_parser.dump(self._kinematics)))
            | dict(depth=self._depth))

    def __init__(
            self,
            light: Light,
            kinematics: PointKinematics | None = None,
            depth: bool = False,
            name: str | None = None
    ):
        check_options(Point.__init__, locals())
        super().__init__(name, light, None)
        if light.needs_velocities() and kinematics is None:
            raise ValueError(
                "a point of emission lines needs kinematics: give it "
                "kinematics (e.g. kinematics: {type: gauss})")
        self._kinematics = kinematics
        self._depth = depth
        self._pdescs = _light_pdescs(
            light.spectrum(), kinematics is not None, depth, light.unit())

    def depth(self) -> bool:
        """Check whether its depth along the line of sight is a parameter."""
        return self._depth

    def needs_depth(self, view: View) -> bool:
        return False

    def kinematics(self) -> PointKinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def moves(self) -> bool:
        return self._kinematics is not None

    def frame_params(self) -> FrameParams:
        vsys = 'vsys' if 'vsys' in self._pdescs else None
        return FrameParams('xpos', 'ypos', vsys, None, None)

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return self._pdescs

    def plan(
            self,
            target: SceneTarget,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            redshift: float
    ) -> PointPlan:
        # A point is unresolved: on the source plane of a lens, its flux
        # would land in one pixel of the grid, which the rays of the image
        # sample (or miss) at random; its images need the lens equation
        # solved for it instead
        if target.lensed:
            raise ValueError(
                "a point behind a gravitational lens is not supported: its "
                "images need the lens equation solved for it; leave it out "
                "of the components the observation sees")
        # (it always emits: it has no opacity, so that a plan of it that
        # does not emit, which would take no unit of the data, is not made)
        setup = target.view.setup(
            self.light(), target.spectral, target.data_unit, redshift)
        return PointPlan(
            target.driver, target.dtype, self._depth, opacity_ratios,
            setup.fluxes, setup.lines)
