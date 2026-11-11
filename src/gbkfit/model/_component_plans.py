"""
The plans of the components of a galaxy on the grid of an observation,
which the evaluations of the galaxy share (see GalaxyPlan and
TransferPlan): the plans of the components it evaluates, the cubes of
opacity of its absorbers, and the parameters of each component.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.driver import DeviceArray
from gbkfit.grid import Coords, Grid, GridData
from gbkfit.params import ParamValues
from .components.base import (
    ComponentState, DepthAxis, NativeGrid, OpacityCubes, SceneTarget)
from .extras import Extras

if TYPE_CHECKING:
    from .components.base import Component
    from .galaxy import Galaxy


__all__ = [
    'ComponentPlans'
]


_log = logging.getLogger(__name__)

# The share of the light of a component beyond its extent above which a
# warning says that the extent shapes it (see
# ComponentPlans.warn_truncation)
_MAX_CUT = 0.01

# The voxels whose light the resolution of the light checks: those with at
# least this share of the light of the brightest voxel of their component
# (see ComponentPlans.light_resolution)
_BRIGHT = 0.05

# Where the grid of the extra outputs of the components is (see
# ComponentPlans.extra)
_ON_THE_GRID = (
    "On the pixels of the observation (finer by its option oversampling), "
    "along the line of sight if it has a depth axis (z = 0 at the plane of "
    "the galaxy, away from the viewer), or else in an image.")

# The axes of the parts of the velocity as a vector (see vxdata), and what
# they are
_VECTOR_AXES = dict(
    x="the x axis of the grid", y="the y axis of the grid",
    z="the line of sight (away from the viewer)")
_VECTOR_DESCRIPTION = (
    "The part along {axis} of the velocity of {whose} in each voxel, as a "
    "vector (km/s; the mean weighted by {weight} there; without the "
    "systemic velocity, which vdata has: along the line of sight, vzdata "
    "is vdata less it).")

# The units of the extra outputs: of velocities, and of none
_KM_S = astropy.units.km / astropy.units.s
_DIMENSIONLESS = astropy.units.dimensionless_unscaled

# The titles, the descriptions and the units (None for the light, whose
# unit the plans do not know) of the extra outputs of each component, by
# name, and of their totals over the components of a galaxy (see
# ComponentPlans.add_extras)
_COMPONENT_EXTRAS = dict(
    bdata=(
        "Brightness",
        "The light that the component emits in each voxel, before any "
        "opacity.", None),
    odata=(
        "Optical depth of dust at V",
        "The optical depth at V of the dust of the component in each "
        "voxel.", _DIMENSIONLESS),
    vdata=(
        "Velocity",
        "The line-of-sight velocity of the component in each voxel (km/s; "
        "the mean weighted by its density there).", _KM_S),
    ddata=(
        "Velocity dispersion",
        "The velocity dispersion of the component in each voxel (km/s; "
        "the mean weighted by its density there).", _KM_S),
    obdata=(
        "Dimmed brightness",
        "The light of the component in each voxel after the opacity that "
        "dims it there.", None)) | {
    f'v{axis}data': (
        f"Velocity along {axis}", _VECTOR_DESCRIPTION.format(
            axis=desc, whose="the component", weight="its density"), _KM_S)
    for axis, desc in _VECTOR_AXES.items()}
_TOTAL_EXTRAS = dict(
    total_bdata=(
        "Total brightness",
        "The light that the components of the galaxy emit in each voxel, "
        "before any opacity.", None),
    total_obdata=(
        "Total dimmed brightness",
        "The light of the components of the galaxy in each voxel after the "
        "opacity that dims it there.", None),
    total_odata=(
        "Total optical depth of dust at V",
        "The optical depth at V of the dust of the components of the galaxy "
        "in each voxel.", _DIMENSIONLESS)) | {
    f'total_v{axis}data': (
        f"Total velocity along {axis}", _VECTOR_DESCRIPTION.format(
            axis=desc, whose="the components of the galaxy",
            weight="their brightness"), _KM_S)
    for axis, desc in _VECTOR_AXES.items()}


def _total_vector(extras: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """
    Return the velocity as a vector of the given extra outputs of
    components that move (vxdata ...), the mean weighted by their
    brightness (bdata); NaN where none has any.
    """
    weights = [np.abs(extra['bdata']) for extra in extras]
    weight = sum(weights)
    total = {}
    for axis in _VECTOR_AXES:
        weighted = sum(
            np.where(w > 0, w * extra[f'v{axis}data'], 0)
            for w, extra in zip(weights, extras))
        total[f'v{axis}data'] = np.divide(
            weighted, weight, out=np.full_like(weight, np.nan),
            where=weight > 0)
    return total


class ComponentPlans:
    """
    The plans of the components of a galaxy on the grid of an
    observation: the x and y axes of its data, and a depth axis along the
    line of sight (or none: the plane of the sky), with the spectral axis
    of its data (or that of an image). It plans the given emitters, whose
    light is evaluated, and the components with dust (the absorbers: all
    of them, since they absorb whatever the observation sees; the line
    opacity of an emitter is that of its own lines, see LineOpacity). Each
    component has its parameters, and those that its galaxy gives it (see
    GalaxyStage).

    The absorbers add the optical depth of each voxel to the cubes of
    opacity, as the given OpacityCubes hold the dust (see opacity_cubes).

    Parameters
    ----------
    galaxy : Galaxy
        The galaxy: its components, and its redshift, which places the
        lines of the emitters (see Component.plan).
    target : SceneTarget
        What the scene asks of it: the driver, the grids, the floating
        type, the lines or the band, and the unit of the data, which the
        light of the emitters takes (see Component.plan).
    depth : DepthAxis, optional
        The depth axis, if any.
    emitters : Sequence of int
        The indices of the components whose light is evaluated.
    cubes : OpacityCubes
        How the cubes of opacity hold the dust of the absorbers, and that
        of other galaxies, which dims the light of the emitters (see
        opacity_cubes).

    Attributes
    ----------
    driver : Driver
        The driver.
    dtype : np.dtype
        The floating type.
    native_grid : NativeGrid
        The grid of the native evaluation functions.
    emitters : tuple of int
        The indices of the emitters.
    absorbers : tuple of int
        The indices of the absorbers.
    plans : dict of int to ComponentPlan
        The plan of each emitter and absorber, by index.
    ratios : dict of int to np.ndarray
        For each emitter, the ratios of the cubes of opacity at the
        wavelength of each of its lines or its band (an array of shape
        (nrows, ncubes)), or None without dust.
    ncubes : int
        The number of cubes of opacity (0 without dust).

    Raises
    ------
    ValueError
        If the light of an emitter has a unit that cannot become that of
        the data (see Component.plan).
    ValueError
        If there is no depth axis and components have opacity (the
        observations check that first, see LineOfSight).
    """

    def __init__(
            self,
            galaxy: 'Galaxy',
            target: SceneTarget,
            depth: DepthAxis | None,
            emitters: Sequence[int],
            cubes: OpacityCubes
    ):
        components = galaxy.components()
        driver, grid, spectral = target.driver, target.spatial, target.spectral
        redshift = galaxy.redshift()
        absorbers = tuple(cubes.deposits)
        if absorbers and depth is None:
            raise ValueError(
                "components with opacity need a grid with a depth axis")
        self._galaxy = galaxy
        self._grid = grid
        self._spectral = spectral
        self._depth = depth
        self.driver = driver
        self.dtype = target.dtype
        self.native_grid = _native_grid(grid, spectral, depth)
        self.emitters = tuple(emitters)
        self.absorbers = absorbers
        self.ratios = cubes.ratios
        self.ncubes = cubes.ncubes
        self._deposits = cubes.deposits
        # The plans of the emitters, and of the absorbers that do not emit
        # here
        self.plans = {}
        for i in emitters:
            try:
                self.plans[i] = components[i].plan(
                    target, True, self.ratios[i], redshift)
            except ValueError as e:
                raise ValueError(f"{_label(components, i)}: {e}") from e
        self.plans |= {
            i: components[i].plan(target, False, None, redshift)
            for i in absorbers if i not in self.plans}

    def params(self, params: ParamValues) -> dict[int, ParamValues]:
        """
        Return the values of the parameters of each component it evaluates
        (by its own names), by index, from those of the galaxy.
        """
        components = self._galaxy.components()
        return {
            i: {p: params[self._galaxy.component_param_names(i)[p]]
                for p in components[i].pdescs()}
            for i in sorted(self.plans)}

    def evaluate_opacity(
            self,
            params: dict[int, ParamValues],
            states: dict[int, ComponentState],
            grid: NativeGrid,
            odata: DeviceArray,
            extras: dict[int, dict[str, Any]] | None
    ) -> None:
        """
        Add the optical depth of each voxel of the grid of the absorbers
        to the cubes of opacity odata (of shape (ncubes, nz, ny, nx)), each
        with the parameters of its index in params and what its galaxy
        gives it, its state, if any (see GalaxyState). With extras, the
        extra outputs of each absorber go to extras[index].
        """
        for i in self.absorbers:
            cube, scale = self._deposits[i]
            extra = {} if extras is not None else None
            self.plans[i].evaluate_opacity(
                params[i], grid, odata[cube], float(scale[0]), extra,
                states.get(i))
            if extras is not None:
                extras.setdefault(i, {}).update(extra)

    def warn_truncation(self, params: dict[int, ParamValues]) -> None:
        """
        Log a warning for each component it evaluates, with the parameters
        of each by index (see params), whose extent cuts more than 1% of
        its light on a grid with a depth axis (see Component.extent_cut),
        and where an option of a component shapes it rather than holds it
        (see ComponentPlan.warn_truncation).
        """
        components = self._galaxy.components()
        for i, plan in self.plans.items():
            extent = components[i].extent_cut(params[i]) \
                if self._depth is not None else None
            if extent is not None and extent[1] > _MAX_CUT:
                option, cut = extent
                _log.warning(
                    f"{_label(components, i)}: its {option} cuts "
                    f"{cut:.1%} of its light: {option} shapes it; raise it")
            plan.warn_truncation(params[i], _label(components, i))

    def light_resolution(
            self,
            values: dict[int, ParamValues],
            extras: dict[int, dict[str, Any]],
            smoothing: float
    ) -> float:
        """
        Return how finely the depth axis resolves the velocity of the light
        of the emitters, with the values of their parameters and their
        extra outputs (extras, by index: the light, velocity and dispersion
        of each voxel, 'bdata', 'vdata' and 'ddata'): the largest change of
        the velocity between adjacent voxels along a line of sight, where
        both are bright (see _BRIGHT), over the width of the lines there,
        that of their dispersion (the smaller of the two) and the given
        spectral smoothing of the instrument (km/s) in quadrature, on the
        spectral axis (see ComponentPlan.line_values; the widest scale of
        the lines of each emitter). Lines of no width are left out; 0
        without adjacent bright voxels (e.g. without a depth axis).
        """
        worst = 0.0
        for i in self.emitters:
            extra = extras.get(i, {})
            if 'vdata' not in extra or 'ddata' not in extra:
                continue
            j = np.abs(extra['bdata'])
            v, s = extra['vdata'], extra['ddata']
            scale = np.abs(self.plans[i].line_values(values[i])[:, 1]).max()
            bright = j >= _BRIGHT * j.max(initial=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                width = np.hypot(
                    scale * np.minimum(s[1:], s[:-1]), smoothing)
                ratio = scale * np.abs(v[1:] - v[:-1]) / width
            ratio = ratio[bright[1:] & bright[:-1] & (width > 0)]
            worst = max(worst, float(ratio.max(initial=0)))
        return worst

    def add_extras(
            self,
            out_extra: Extras,
            extras: dict[int, dict[str, Any]],
            totals: dict[str, np.ndarray],
            depth_zero: float | None = None
    ) -> None:
        """
        Add the extra outputs of each component (extras, by index: e.g.
        'bdata', see _COMPONENT_EXTRAS) to out_extra, with the id
        'component': its name, or its position if it has none (e.g.
        'component0'); and their totals ('total_*', see _TOTAL_EXTRAS):
        the given ones, the optical depth at V of all the absorbers
        ('total_odata') and the velocity as a vector of the components
        that move, weighted by their brightness ('total_vxdata' ...);
        with the given position of the first voxel of the depth axis, if
        not that of the axis (see extra).
        """
        components = self._galaxy.components()
        for index, extra in extras.items():
            label = _label(components, index)
            for key, value in extra.items():
                title, description, unit = _COMPONENT_EXTRAS[key]
                out_extra.add(
                    key, self.extra(value, depth_zero, unit), title,
                    f"{description} {_ON_THE_GRID}", component=label)
        if self.absorbers:
            totals = dict(odata=sum(
                extras[i]['odata'] for i in self.absorbers)) | totals
        if movers := [extra for extra in extras.values() if 'vxdata' in extra]:
            totals = totals | _total_vector(movers)
        for name, total in totals.items():
            title, description, unit = _TOTAL_EXTRAS[f'total_{name}']
            out_extra.add(
                f'total_{name}', self.extra(total, depth_zero, unit), title,
                f"{description} {_ON_THE_GRID}")

    def extra(
            self,
            data: np.ndarray,
            depth_zero: float | None = None,
            unit: gbkfit.units.AnyUnit | None = None
    ) -> GridData:
        """
        Return an extra output on the grid: an image without a depth axis,
        a cube with one (along the line of sight, from z = 0 at the plane
        of the galaxy), whose first voxel is at the given position (arcsec;
        by default, that of the depth axis; it differs for the galaxies
        that a march evaluates together, see TransferPlan).
        """
        grid = self._grid
        if self._depth is None:
            return GridData(
                data.reshape(data.shape[-2:]), grid.coords, None, unit)
        depth = self._depth
        zero = depth.zero if depth_zero is None else depth_zero
        coords = Coords(
            tuple(grid.coords.step) + (depth.step,),
            tuple(grid.coords.rpix) + (-zero / depth.step,),
            tuple(grid.coords.rval) + (0.0,), grid.coords.rota)
        return GridData(data, coords, None, unit)

    def output_extra(
            self, data: np.ndarray, unit: gbkfit.units.AnyUnit | None = None
    ) -> GridData:
        """
        Return an extra output of the shape of the output on the grid: an
        image (2d data), or a cube along the spectral axis (3d data).
        """
        coords = self._grid.coords
        if data.ndim == 2:
            return GridData(data, coords, None, unit)
        spectral = self._spectral.coords
        return GridData(data, coords._replace(
            step=tuple(coords.step) + tuple(spectral.step),
            rpix=tuple(coords.rpix) + tuple(spectral.rpix),
            rval=tuple(coords.rval) + tuple(spectral.rval),
            rest=spectral.rest, medium=spectral.medium,
            convention=spectral.convention), 2, unit)


def _label(components: Sequence['Component'], index: int) -> str:
    """
    Return the label of a component of the given index: its name, or its
    position if it has none (e.g. 'component0').
    """
    name = components[index].name()
    return name if name is not None else f'component{index}'


def _native_grid(
        grid: Grid,
        spectral: Grid,
        depth: DepthAxis | None
) -> NativeGrid:
    """
    Return the grid of the native evaluation functions: the x and y axes
    of the grid, the depth axis (size 1 and step 0 without one) and the
    spectral axis.
    """
    if depth is None:
        depth = DepthAxis(1, 0, 0)
    return NativeGrid(
        spat_size=tuple(grid.size[:2]) + (depth.size,),
        spat_step=tuple(grid.coords.step[:2]) + (depth.step,),
        spat_zero=tuple(grid.zero()[:2]) + (depth.zero,),
        spat_rota=grid.coords.rota,
        slab_zero=0,
        spec_size=spectral.size[0],
        spec_step=spectral.coords.step[0],
        spec_zero=spectral.zero()[0])
