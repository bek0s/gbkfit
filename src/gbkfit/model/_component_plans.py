"""
The plans of the components of a galaxy on the grid of an observation,
which the evaluations of the galaxy share (see GalaxyPlan and
TransferPlan): the plans of the components it evaluates, the cubes of
opacity of its absorbers, and the parameters of each component.
"""

import collections
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Coords, Grid, GridData, doppler
from gbkfit.params import ParamValues
from .components._units import DataUnit
from .components.base import (
    ComponentState, DepthAxis, NativeGrid, OpacityCubes)
from .components.extinction import ExtinctionLaw
from .components.lines import Band
from .components.roles import BandEmission
from .extras import Extras

if TYPE_CHECKING:
    from .galaxy import Galaxy
    from .components.base import Component
    from .galaxy import Galaxy


__all__ = [
    'ComponentPlans',
    'opacity_cubes'
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
    driver : Driver
        The driver of the evaluation.
    grid : Grid
        The grid of the x and y axes.
    spectral : Grid
        The spectral axis (one axis).
    depth : DepthAxis, optional
        The depth axis, if any.
    dtype : np.dtype
        The floating type of the evaluation.
    emitters : Sequence of int
        The indices of the components whose light is evaluated.
    cubes : OpacityCubes
        How the cubes of opacity hold the dust of the absorbers, and that
        of other galaxies, which dims the light of the emitters (see
        opacity_cubes).
    lines : tuple of str, optional
        The names of the emission lines evaluated; all if None.
    band : Band, optional
        The band of an image, if known.
    data_unit : DataUnit, optional
        The unit of the data of the observation, which the light of the
        emitters takes (see Component.plan); None without data.

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
            driver: Driver,
            grid: Grid,
            spectral: Grid,
            depth: DepthAxis | None,
            dtype: np.dtype,
            emitters: Sequence[int],
            cubes: OpacityCubes,
            lines: tuple[str, ...] | None,
            band: Band | None,
            data_unit: DataUnit | None
    ):
        components = galaxy.components()
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
        self.dtype = dtype
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
                    driver, spectral, dtype, lines, band, True,
                    self.ratios[i], data_unit, redshift)
            except ValueError as e:
                raise ValueError(f"{_label(components, i)}: {e}") from e
        self.plans |= {
            i: components[i].plan(
                driver, spectral, dtype, lines, band, False, None, None,
                redshift)
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


# The most cubes of opacity that the native kernels read (OPACITY_NUM_MAX
# in constants.hpp)
_MAX_OPACITY_CUBES = 4


class _Frame(NamedTuple):
    """
    The frame of the light or the dust of a component, at 1 + z times
    D(vsys) (see Lines.factors): the redshift of its galaxy, and its
    systemic velocity, a parameter (by the index of its galaxy in the
    scene and its name in the parameters of the galaxy; None for none, at
    rest in its galaxy: dust of no vsys, or the band of an image, whose
    wavelength is that observed).
    """
    redshift: float
    vsys: tuple[int, str] | None

    def factor(self, velocities: dict[tuple[int, str], float]) -> float:
        """Return 1 + its redshift, of the systemic velocities."""
        velocity = 0.0 if self.vsys is None else velocities[self.vsys]
        return (1 + self.redshift) * float(doppler(velocity))


class _Light(NamedTuple):
    """
    Light of a rest wavelength (um; None if not known) emitted in a frame.
    """
    wavelength: float | None
    frame: _Frame

    def observed(
            self, velocities: dict[tuple[int, str], float]
    ) -> float | None:
        """Return its observed wavelength (um), if known."""
        return None if self.wavelength is None \
            else self.wavelength * self.frame.factor(velocities)


class _Dust(NamedTuple):
    """
    Dust of an extinction law (None: grey dust, whose optical depth is
    that at V at every wavelength) in a frame (at rest at 0 for grey
    dust).
    """
    law: ExtinctionLaw | None
    frame: _Frame

    def ratio(
            self, light: _Light, velocities: dict[tuple[int, str], float]
    ) -> float:
        """
        Return the ratio of its optical depth at the wavelength of the
        light in its frame to that at V, of the systemic velocities. Raise
        ValueError if it has a law and the wavelength is not known.
        """
        if self.law is None:
            return 1.0
        if light.wavelength is None:
            raise ValueError(
                "dust of an extinction law dims light at its rest "
                "wavelength: the emission of the components it dims must "
                "name its lines (e.g. lines: [halpha]), and an image needs "
                "its band")
        return self.law.ratio(
            light.observed(velocities) / self.frame.factor(velocities))


# Dust whose optical depth is the same at every wavelength
_GREY = _Dust(None, _Frame(0.0, None))


class _DustModel:
    """
    The dust of a scene, in the cubes of opacity of its observation (see
    opacity_cubes): their layout, fixed when it is planned (the kind of
    dust of each cube, and the cube of each absorber), and the numbers of
    each evaluation, of the systemic velocities of the components (see
    update), in the arrays that it gives the plans and updates in place:
    the factor of the optical depth at V that each absorber deposits, and
    for each emitter, the ratios of the optical depths of the cubes to
    theirs at the wavelength of each of its rows of light.
    """

    def __init__(
            self,
            kinds: tuple[_Dust, ...],
            dust: list[_Dust],
            light: list[list[_Light]],
            grey: bool,
            dtype: np.dtype
    ):
        self._kinds = kinds
        self._dust = dust
        self._light = light
        self._grey = grey
        self.scales = np.ones(len(dust))
        self.ratios = [np.ones((len(rows), len(kinds)), dtype)
                       for rows in light]
        # (at rest until the first evaluation; and here the errors of
        # light of no known wavelength)
        self._fill(collections.defaultdict(float))

    def update(self, params: Sequence[ParamValues]) -> None:
        """
        Update its numbers, in place, for the values of the parameters of
        the galaxies of the scene (by their index): of their systemic
        velocities (the mean of their values at the nodes, if node-wise,
        as the lines take them; see systemic_velocity).
        """
        self._fill({
            (g, name): float(np.mean(params[g][name]))
            for g, name in self._parameters()})

    def _fill(self, velocities: dict[tuple[int, str], float]) -> None:
        """Set its numbers, in place, of the systemic velocities."""
        if self._grey:
            first = next(rows[0] for rows in self._light if rows)
            self.scales[:] = [
                dust.ratio(first, velocities) for dust in self._dust]
            return
        for ratios, rows in zip(self.ratios, self._light):
            for i, row in enumerate(rows):
                ratios[i] = [
                    kind.ratio(row, velocities) for kind in self._kinds]

    def _parameters(self) -> set[tuple[int, str]]:
        """Return the systemic velocities that its frames take."""
        frames = [dust.frame for dust in self._dust] + [
            row.frame for rows in self._light for row in rows]
        return {frame.vsys for frame in frames if frame.vsys is not None}


def opacity_cubes(
        galaxies: Sequence['Galaxy'],
        emitters: Sequence[Sequence[int]],
        lines: tuple[str, ...] | None,
        band: Band | None,
        dtype: np.dtype
) -> list[OpacityCubes]:
    """
    Return how the plan of each of the given galaxies holds, in cubes of
    opacity, the dust of all of them, which dims the light of the given
    emitters of each (by index): of each of their lines of the given
    names (all if None), or in the given band of an image (if known). The
    components with dust are the absorbers. The dust of each is that of
    its extinction law in its frame, at 1 + z times D(vsys) of its galaxy
    and its systemic velocity (see _Frame), which dims light at the
    wavelength of that light in its frame: of a line, emitted at its rest
    wavelength in the frame of its component; of the band of an image, at
    the wavelength of the band. The layout of the cubes is fixed here (see
    _layout); their numbers follow the systemic velocities, a parameter,
    at each evaluation (see _DustModel.update, which every OpacityCubes of
    the galaxies shares as its model).

    Raises
    ------
    ValueError
        If dust of an extinction law dims light of a wavelength that is
        not known, or there are too many kinds of dust (see _layout).
    """
    absorbers = [galaxy.absorbers() for galaxy in galaxies]
    if not any(absorbers):
        return [OpacityCubes(0, {}, dict.fromkeys(indices))
                for indices in emitters]
    # The dust of each absorber, and the light of each line (or band) of
    # each emitter of each galaxy
    dust = []
    light = {}
    for g, (galaxy, galaxy_absorbers, galaxy_emitters) in enumerate(zip(
            galaxies, absorbers, emitters)):
        components = galaxy.components()
        redshift = galaxy.redshift()

        def frame(i: int, moves: bool = True) -> _Frame:
            names = galaxy.component_param_names(i)
            vsys = names.get('vsys') if moves else None
            return _Frame(redshift, None if vsys is None else (g, vsys))
        for i in galaxy_absorbers:
            law = components[i].opacity().law()
            dust.append(_GREY if law is None else _Dust(law, frame(i)))
        for i in galaxy_emitters:
            emission = components[i].emission()
            # (the band of an image: its wavelength, as observed)
            at_band = band is not None and isinstance(emission, BandEmission)
            light[g, i] = [
                _Light(wavelength, frame(i, not at_band))
                for wavelength in emission.wavelengths(lines, band, redshift)]
    kinds, cubes, grey = _layout(dust, [
        row for rows in light.values() for row in rows])
    model = _DustModel(kinds, dust, list(light.values()), grey, dtype)
    ratios = dict(zip(light, model.ratios))
    deposits = iter(enumerate(cubes))
    return [
        OpacityCubes(
            len(kinds),
            {i: (cube, model.scales[a:a + 1])
             for i, (a, cube) in zip(galaxy_absorbers, deposits)},
            {i: ratios[g, i] for i in galaxy_emitters}, model)
        for g, (galaxy_absorbers, galaxy_emitters) in enumerate(
            zip(absorbers, emitters))]


def _layout(
        dust: Sequence[_Dust], light: Sequence[_Light]
) -> tuple[tuple[_Dust, ...], tuple[int, ...], bool]:
    """
    Return how the optical depth of absorbers of the given kinds of dust
    is held for an observation of light of the given rows (see _Light):
    the kind of dust of each cube of opacity, the cube of each absorber,
    and whether the cube is one of grey dust at the one wavelength of the
    light. With one kind, one cube of the optical depth at V; with several
    kinds and light of one observed wavelength at every evaluation (see
    _same_light), one cube of the optical depth at that wavelength (of
    grey dust); with several kinds and wavelengths, one cube of the
    optical depth at V of each kind. A kind is a law in a frame: a law
    counts once for each redshift and systemic velocity (a parameter) of
    the components that have it.

    Raises
    ------
    ValueError
        If with several wavelengths there are more kinds of dust than the
        kernels take.
    """
    distinct = tuple(dict.fromkeys(dust))
    if len(distinct) == 1:
        return distinct, (0,) * len(dust), False
    if light and all(_same_light(row, light[0]) for row in light):
        return (_GREY,), (0,) * len(dust), True
    if len(distinct) > _MAX_OPACITY_CUBES:
        raise ValueError(
            f"the dust that an observation of several wavelengths sees can "
            f"have at most {_MAX_OPACITY_CUBES} kinds; it has "
            f"{len(distinct)}: an extinction law counts once for each "
            f"redshift and each systemic velocity (vsys, a parameter) of "
            f"the components that have it, which can tie their vsys")
    return (
        distinct, tuple(distinct.index(kind) for kind in dust), False)


def _same_light(a: _Light, b: _Light) -> bool:
    """
    Check whether two rows of light are observed at one wavelength at
    every evaluation: in frames of one systemic velocity (or none), at one
    wavelength (to the precision of their arithmetic: the rest wavelength
    of a band, times 1 + the redshift, is its wavelength only to
    rounding), or both not known.
    """
    if a.frame.vsys != b.frame.vsys:
        return False
    at_rest = collections.defaultdict(float)
    wa, wb = a.observed(at_rest), b.observed(at_rest)
    if wa is None or wb is None:
        return wa is wb
    return bool(np.isclose(wa, wb, rtol=1e-9, atol=0))
